use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::Bytes;
use axum::extract::multipart::{Multipart, MultipartError, MultipartRejection};
use axum::extract::{ConnectInfo, DefaultBodyLimit, Path, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, extract::multipart::Field};
use serde::Serialize;
use serde_json::json;

use crate::audio::{self, AudioError, DecodeLimits, SAMPLE_RATE};
use crate::auth::{self, ApiKey};
use crate::config::Config;
use crate::engine::{
    DecodeOptions, EngineError, MIN_SAMPLES, ModelRef, TranscribeRequest, Transcriber,
    is_valid_language,
};
use crate::format;
use crate::jobs::JobManager;
use crate::models::{ModelError, ModelInfo, Registry};
use crate::pipeline::Transcription;
use crate::scheduler::Lane;
use crate::vocabulary::{VocabularyError, VocabularyStore};

mod admin_routes;
mod job_routes;

const MAX_TEXT_FIELD_BYTES: usize = 64 * 1024;
/// Most multipart fields one request may carry; the real API has about a dozen.
const MAX_FORM_FIELDS: usize = 64;
const MAX_MODEL_FIELD_CHARS: usize = 128;
const MAX_GRANULARITIES: usize = 8;
/// How long running interactive requests get to finish after SIGTERM.
const DRAIN_SECS: u64 = 30;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub registry: Arc<Registry>,
    pub engine: Arc<dyn Transcriber>,
    pub vocabulary: Arc<VocabularyStore>,
    pub jobs: Arc<JobManager>,
    uploads: Arc<tokio::sync::Semaphore>,
    admin: Arc<admin_routes::ModelLocks>,
}

impl AppState {
    pub fn new(
        config: Arc<Config>,
        registry: Arc<Registry>,
        engine: Arc<dyn Transcriber>,
        vocabulary: Arc<VocabularyStore>,
        jobs: Arc<JobManager>,
    ) -> AppState {
        let uploads = Arc::new(tokio::sync::Semaphore::new(
            config.server.max_concurrent_uploads,
        ));
        AppState {
            config,
            registry,
            engine,
            vocabulary,
            jobs,
            uploads,
            admin: Arc::default(),
        }
    }
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
    param: Option<&'static str>,
    code: &'static str,
    retry_after: Option<u64>,
}

impl ApiError {
    fn invalid(param: &'static str, code: &'static str, message: impl Into<String>) -> ApiError {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: message.into(),
            param: Some(param),
            code,
            retry_after: None,
        }
    }

    fn new(
        status: StatusCode,
        kind: &'static str,
        code: &'static str,
        message: String,
    ) -> ApiError {
        ApiError {
            status,
            kind,
            message,
            param: None,
            code,
            retry_after: None,
        }
    }

    fn model(e: ModelError, param: &'static str) -> ApiError {
        ApiError {
            status: StatusCode::NOT_FOUND,
            kind: "invalid_request_error",
            message: e.to_string(),
            param: Some(param),
            code: "model_not_found",
            retry_after: None,
        }
    }

    fn code(&self) -> &'static str {
        self.code
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({"error": {
            "message": self.message,
            "type": self.kind,
            "param": self.param,
            "code": self.code,
        }});
        let mut resp = (self.status, Json(body)).into_response();
        if let Some(secs) = self.retry_after {
            resp.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(secs));
        }
        resp
    }
}

fn multipart_error(e: MultipartError) -> ApiError {
    if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
        return ApiError::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_request_error",
            "file_too_large",
            "the upload exceeds server.max_upload_mb".into(),
        );
    }
    ApiError::new(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        "invalid_multipart",
        format!("malformed multipart body: {}", e.body_text()),
    )
}

pub fn router(state: AppState) -> Router {
    let limit = (state.config.server.max_upload_mb as usize).saturating_mul(1024 * 1024);
    Router::new()
        .route(
            "/v1/audio/transcriptions",
            post(transcriptions).layer(DefaultBodyLimit::max(limit)),
        )
        .route("/v1/models", get(list_models))
        .route("/v1/models/{id}", get(get_model))
        .route("/health", get(health))
        .route("/hark/v1/status", get(status))
        .route("/hark/v1/models", get(admin_routes::catalog))
        .route(
            "/hark/v1/models/pull",
            post(admin_routes::pull).layer(DefaultBodyLimit::max(admin_routes::MAX_BODY)),
        )
        .route("/hark/v1/models/{id}/load", post(admin_routes::load))
        .route("/hark/v1/models/{id}/unload", post(admin_routes::unload))
        .route("/hark/v1/models/{id}", delete(admin_routes::remove))
        .route(
            "/hark/v1/jobs",
            post(job_routes::submit)
                .layer(DefaultBodyLimit::max(limit))
                .get(job_routes::list),
        )
        .route(
            "/hark/v1/jobs/{id}",
            get(job_routes::get_job).delete(job_routes::remove),
        )
        .route("/hark/v1/jobs/{id}/result", get(job_routes::result))
        .fallback(not_found)
        .layer(middleware::from_fn_with_state(guard_state(&state), guard))
        .with_state(state)
}

#[derive(Clone)]
struct Guard {
    key: Option<Arc<ApiKey>>,
    hosts: Arc<HostPolicy>,
}

/// `Host` values hark answers to: its own loopback names on its port, plus `server.allowed_hosts`.
#[derive(Debug)]
struct HostPolicy {
    allowed: Vec<String>,
}

impl HostPolicy {
    fn new(port: u16, extra: &[String]) -> HostPolicy {
        let mut allowed: Vec<String> = ["127.0.0.1", "localhost", "[::1]"]
            .iter()
            .map(|h| format!("{h}:{port}"))
            .collect();
        if port == 80 {
            allowed.extend(["127.0.0.1", "localhost", "[::1]"].map(String::from));
        }
        allowed.extend(extra.iter().map(|h| h.to_ascii_lowercase()));
        HostPolicy { allowed }
    }

    fn allows(&self, host: &str) -> bool {
        let host = host.to_ascii_lowercase();
        self.allowed.contains(&host)
    }
}

/// An unreadable key file fails closed: the daemon refuses to start on it, and a router built
/// anyway accepts no key at all.
fn guard_state(state: &AppState) -> Guard {
    let key = match auth::load_key(&state.config.server.api_key_file) {
        Ok(k) => k.map(Arc::new),
        Err(e) => {
            tracing::error!(error = %e, "api key unusable; every authenticated route will answer 401");
            Some(Arc::new(ApiKey::unusable()))
        }
    };
    let port = state.config.listen_addr().map_or(8797, |a| a.port());
    Guard {
        key,
        hosts: Arc::new(HostPolicy::new(port, &state.config.server.allowed_hosts)),
    }
}

async fn not_found() -> ApiError {
    ApiError::new(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        "not_found",
        "unknown route".into(),
    )
}

fn forbidden(code: &'static str, message: &str) -> Response {
    ApiError::new(
        StatusCode::FORBIDDEN,
        "invalid_request_error",
        code,
        message.into(),
    )
    .into_response()
}

/// Origin check, then the Host check (DNS rebinding), then the bearer key on every route but `/health`.
/// A request without a Host header (HTTP/1.0 tools) passes only from a loopback peer.
async fn guard(State(guard): State<Guard>, req: Request, next: Next) -> Response {
    if req.headers().contains_key(header::ORIGIN) {
        return forbidden(
            "origin_forbidden",
            "requests carrying an Origin header are refused",
        );
    }
    let host = req
        .headers()
        .get(header::HOST)
        .map(|v| v.to_str().unwrap_or("\u{fffd}").to_string())
        .or_else(|| req.uri().authority().map(|a| a.to_string()));
    let host_ok = match host {
        Some(h) => guard.hosts.allows(&h),
        None => req
            .extensions()
            .get::<ConnectInfo<std::net::SocketAddr>>()
            .is_none_or(|c| c.0.ip().is_loopback()),
    };
    if !host_ok {
        return forbidden(
            "host_forbidden",
            "the Host header is not one hark answers to; see server.allowed_hosts",
        );
    }
    if req.uri().path() != "/health"
        && let Some(key) = &guard.key
    {
        let presented = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(auth::bearer_token);
        if !presented.is_some_and(|t| key.matches(t)) {
            let mut resp = ApiError {
                status: StatusCode::UNAUTHORIZED,
                kind: "invalid_request_error",
                message: "missing or incorrect API key; send `Authorization: Bearer <key>`".into(),
                param: None,
                code: "invalid_api_key",
                retry_after: None,
            }
            .into_response();
            resp.headers_mut()
                .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
            return resp;
        }
    }
    next.run(req).await
}

async fn status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mut v = serde_json::to_value(state.engine.status()).unwrap_or_default();
    let (queued, running) = state.jobs.counts();
    v["jobs"] = json!({"queued": queued, "running": running});
    Json(v)
}

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let degraded = state.engine.status().degraded;
    Json(json!({
        "status": if degraded { "degraded" } else { "ok" },
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

impl From<ModelInfo> for ModelObject {
    fn from(m: ModelInfo) -> Self {
        ModelObject {
            id: m.id,
            object: "model",
            created: m.created,
            owned_by: "hark",
        }
    }
}

async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let data: Vec<ModelObject> = state.registry.list().into_iter().map(Into::into).collect();
    Json(json!({"object": "list", "data": data}))
}

async fn get_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ModelObject>, ApiError> {
    let info = state
        .registry
        .model_info(&id)
        .map_err(|e| ApiError::model(e, "model"))?;
    Ok(Json(info.into()))
}

#[derive(Default)]
struct Form {
    file: Option<Bytes>,
    model: Option<String>,
    language: Option<String>,
    prompt: Option<String>,
    response_format: Option<String>,
    temperature: Option<String>,
    granularities: Vec<String>,
    x_priority: Option<String>,
    x_vad: Option<String>,
    x_vocabulary: Option<String>,
}

async fn text_field(field: &mut Field<'_>) -> Result<String, ApiError> {
    let name = field.name().unwrap_or("field").to_string();
    let mut buf = Vec::new();
    while let Some(chunk) = field.chunk().await.map_err(multipart_error)? {
        if buf.len() + chunk.len() > MAX_TEXT_FIELD_BYTES {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "field_too_large",
                format!("field `{name}` is too large"),
            ));
        }
        buf.extend_from_slice(&chunk);
    }
    String::from_utf8(buf).map_err(|_| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_value",
            format!("field `{name}` is not valid UTF-8"),
        )
    })
}

fn too_many_uploads() -> ApiError {
    let mut e = ApiError::new(
        StatusCode::TOO_MANY_REQUESTS,
        "rate_limit_error",
        "too_many_uploads",
        "too many uploads are being received right now; retry shortly".into(),
    );
    e.retry_after = Some(1);
    e
}

/// Reads the multipart fields; the `file` body is read under an upload permit (429 when none is free).
async fn read_form(mut mp: Multipart, state: &AppState) -> Result<Form, ApiError> {
    let mut form = Form::default();
    let mut fields = 0usize;
    while let Some(mut field) = mp.next_field().await.map_err(multipart_error)? {
        fields += 1;
        if fields > MAX_FORM_FIELDS {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "too_many_fields",
                format!("a request may carry at most {MAX_FORM_FIELDS} form fields"),
            ));
        }
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                let _permit = state
                    .uploads
                    .clone()
                    .try_acquire_owned()
                    .map_err(|_| too_many_uploads())?;
                form.file = Some(field.bytes().await.map_err(multipart_error)?);
            }
            "model" => form.model = Some(text_field(&mut field).await?),
            "language" => form.language = Some(text_field(&mut field).await?),
            "prompt" => form.prompt = Some(text_field(&mut field).await?),
            "response_format" => form.response_format = Some(text_field(&mut field).await?),
            "temperature" => form.temperature = Some(text_field(&mut field).await?),
            "timestamp_granularities[]" | "timestamp_granularities" => {
                form.granularities.push(text_field(&mut field).await?)
            }
            "x_priority" => form.x_priority = Some(text_field(&mut field).await?),
            "x_vad" => form.x_vad = Some(text_field(&mut field).await?),
            "x_vocabulary" => form.x_vocabulary = Some(text_field(&mut field).await?),
            // stream is accepted; unknown fields are ignored.
            _ => {}
        }
    }
    Ok(form)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Json,
    Text,
    Srt,
    Vtt,
    VerboseJson,
}

struct ParsedRequest {
    audio: Bytes,
    model: String,
    format: Format,
    granularities: Vec<String>,
    vocabulary: Option<String>,
    lane: Lane,
    opts: DecodeOptions,
}

fn validate(mut form: Form) -> Result<ParsedRequest, ApiError> {
    let audio = form.file.take().ok_or_else(|| {
        ApiError::invalid(
            "file",
            "missing_required_parameter",
            "you must provide the `file` field",
        )
    })?;
    Ok(ParsedRequest {
        audio,
        ..validate_fields(form)?
    })
}

fn parse_format(raw: Option<&str>) -> Result<Format, ApiError> {
    Ok(match raw.map(str::trim) {
        None | Some("") | Some("json") => Format::Json,
        Some("text") => Format::Text,
        Some("srt") => Format::Srt,
        Some("vtt") => Format::Vtt,
        Some("verbose_json") => Format::VerboseJson,
        Some(other) => {
            return Err(ApiError::invalid(
                "response_format",
                "invalid_value",
                format!("`{other}` is not one of json, text, srt, verbose_json, vtt"),
            ));
        }
    })
}

fn validate_granularities(granularities: &[String], format: Format) -> Result<(), ApiError> {
    if granularities.len() > MAX_GRANULARITIES {
        return Err(ApiError::invalid(
            "timestamp_granularities[]",
            "invalid_value",
            format!("at most {MAX_GRANULARITIES} granularities are accepted"),
        ));
    }
    for g in granularities {
        if g != "segment" && g != "word" {
            return Err(ApiError::invalid(
                "timestamp_granularities[]",
                "invalid_value",
                format!("`{g}` is not one of segment, word"),
            ));
        }
    }
    if !granularities.is_empty() && format != Format::VerboseJson {
        return Err(ApiError::invalid(
            "timestamp_granularities[]",
            "invalid_value",
            "timestamp_granularities[] is only allowed with response_format verbose_json",
        ));
    }
    Ok(())
}

/// Every field check except the file; `audio` comes back empty.
fn validate_fields(form: Form) -> Result<ParsedRequest, ApiError> {
    let model = form.model.filter(|m| !m.is_empty()).ok_or_else(|| {
        ApiError::invalid(
            "model",
            "missing_required_parameter",
            "you must provide the `model` field",
        )
    })?;
    if model.chars().count() > MAX_MODEL_FIELD_CHARS {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "model_not_found",
            "the model id is too long".into(),
        ));
    }
    if form
        .prompt
        .as_ref()
        .is_some_and(|p| p.len() > MAX_TEXT_FIELD_BYTES)
    {
        return Err(ApiError::invalid(
            "prompt",
            "invalid_value",
            "prompt is too long",
        ));
    }
    let format = parse_format(form.response_format.as_deref())?;
    let temperature = match form.temperature.as_deref().map(str::trim) {
        None | Some("") => 0.0,
        Some(t) => t
            .parse::<f32>()
            .ok()
            .filter(|t| (0.0..=1.0).contains(t))
            .ok_or_else(|| {
                ApiError::invalid(
                    "temperature",
                    "invalid_value",
                    "temperature must be a number from 0 to 1",
                )
            })?,
    };
    let language = match form.language.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(l) if is_valid_language(l) => Some(l.to_string()),
        Some(l) => {
            return Err(ApiError::invalid(
                "language",
                "invalid_value",
                format!("`{l}` is not a supported ISO-639-1 language code"),
            ));
        }
    };
    validate_granularities(&form.granularities, format)?;
    let lane = match form.x_priority.as_deref().map(str::trim) {
        None | Some("") | Some("interactive") => Lane::Interactive,
        Some("batch") => Lane::Batch,
        Some("live") => {
            return Err(ApiError::invalid(
                "x_priority",
                "unsupported_value",
                "x_priority `live` is reserved and not available yet",
            ));
        }
        Some(other) => {
            return Err(ApiError::invalid(
                "x_priority",
                "invalid_value",
                format!("`{other}` is not one of interactive, batch"),
            ));
        }
    };
    let vad = match form.x_vad.as_deref().map(str::trim) {
        None | Some("") => None,
        Some("true") => Some(true),
        Some("false") => Some(false),
        Some(other) => {
            return Err(ApiError::invalid(
                "x_vad",
                "invalid_value",
                format!("`{other}` is not true or false"),
            ));
        }
    };
    let prompt = form.prompt.filter(|p| !p.is_empty() && !p.contains('\0'));
    Ok(ParsedRequest {
        audio: Bytes::new(),
        model,
        format,
        granularities: form.granularities,
        vocabulary: form.x_vocabulary,
        lane,
        opts: DecodeOptions {
            language,
            prompt,
            vocabulary: Vec::new(),
            temperature,
            vad,
        },
    })
}

fn audio_error(e: AudioError) -> ApiError {
    match e {
        AudioError::TooLong(_) => ApiError {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            kind: "invalid_request_error",
            message: format!("{e}; submit long audio through the jobs API (/hark/v1/jobs)"),
            param: Some("file"),
            code: "audio_too_long",
            retry_after: None,
        },
        AudioError::FfmpegMissing(_) => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "ffmpeg_unavailable",
            e.to_string(),
        ),
        _ => ApiError::invalid("file", "invalid_audio", e.to_string()),
    }
}

impl From<VocabularyError> for ApiError {
    fn from(e: VocabularyError) -> Self {
        ApiError::invalid("x_vocabulary", "invalid_value", e.to_string())
    }
}

/// Drops every path-like word (a `/` or `\`, or a `C:` drive prefix), so log lines and client messages never carry file system layout.
pub(crate) fn strip_paths(text: &str) -> String {
    text.split_whitespace()
        .map(|w| if is_path_word(w) { "<path>" } else { w })
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_path_word(w: &str) -> bool {
    let b = w.trim_start_matches(['(', '"', '\'', '[']).as_bytes();
    w.contains('/')
        || w.contains('\\')
        || (b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':')
}

/// Client-safe error for an engine failure; the detail goes to the log under `request_id`.
fn engine_error(e: EngineError, request_id: &str) -> ApiError {
    match e {
        EngineError::Load(detail) => {
            tracing::warn!(request_id, detail = %strip_paths(&detail), "model could not be loaded");
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "model_unavailable",
                format!("model could not be loaded (request {request_id})"),
            )
        }
        EngineError::Busy { retry_after_secs } => ApiError {
            status: StatusCode::TOO_MANY_REQUESTS,
            kind: "rate_limit_error",
            message: e.to_string(),
            param: None,
            code: "rate_limit_exceeded",
            retry_after: Some(retry_after_secs),
        },
        EngineError::ModelBusy(_) => ApiError::new(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "model_busy",
            e.to_string(),
        ),
        EngineError::Cancelled => ApiError::new(
            StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
            "invalid_request_error",
            "request_cancelled",
            e.to_string(),
        ),
        EngineError::Run(detail) => {
            let detail = strip_paths(&detail);
            tracing::warn!(request_id, detail = %detail, "transcription failed");
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "transcription_failed",
                format!("transcription failed: {detail}"),
            )
        }
    }
}

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn new_request_id() -> String {
    let n = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or(0);
    format!("req_{:x}{:04x}", t, n & 0xffff)
}

#[derive(Default)]
struct Meta {
    model: String,
    lane: &'static str,
    audio_secs: f64,
    queue_wait_secs: f64,
}

async fn transcriptions(
    State(state): State<AppState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    let started = Instant::now();
    let id = new_request_id();
    let mut meta = Meta::default();
    let result = transcribe(&state, &id, multipart, &mut meta).await;
    let wall = started.elapsed().as_secs_f64();
    let rtf = if meta.audio_secs > 0.0 {
        wall / meta.audio_secs
    } else {
        0.0
    };
    let outcome = match &result {
        Ok(_) => "ok",
        Err(e) => e.code(),
    };
    tracing::info!(
        request_id = %id,
        lane = meta.lane,
        model = ?meta.model,
        audio_secs = format_args!("{:.2}", meta.audio_secs),
        queue_wait_secs = format_args!("{:.3}", meta.queue_wait_secs),
        wall_secs = format_args!("{:.3}", wall),
        rtf = format_args!("{:.3}", rtf),
        outcome,
        "transcription request"
    );
    let mut resp = match result {
        Ok(r) => r,
        Err(e) => e.into_response(),
    };
    if let Ok(v) = HeaderValue::from_str(&id) {
        resp.headers_mut().insert("x-request-id", v);
    }
    resp
}

async fn transcribe(
    state: &AppState,
    id: &str,
    multipart: Result<Multipart, MultipartRejection>,
    meta: &mut Meta,
) -> Result<Response, ApiError> {
    let multipart = multipart.map_err(|e| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_multipart",
            format!("expected a multipart/form-data body: {}", e.body_text()),
        )
    })?;
    let cancel = Arc::new(AtomicBool::new(false));
    let _cancel_on_drop = CancelOnDrop(cancel.clone());
    let mut req = validate(read_form(multipart, state).await?)?;
    req.opts.vocabulary = state
        .vocabulary
        .resolve(req.vocabulary.as_deref())
        .map_err(ApiError::from)?;
    meta.model = req.model.clone();
    meta.lane = req.lane.as_str();
    let (entry, path) = state
        .registry
        .resolve_installed(&req.model)
        .map_err(|e| ApiError::model(e, "model"))?;

    let bytes = req.audio;
    let limits = DecodeLimits {
        ffmpeg_timeout: Duration::from_secs(state.config.server.ffmpeg_timeout_secs),
        max_secs: state.config.server.sync_max_audio_secs,
    };
    let samples = tokio::task::spawn_blocking(move || audio::decode(&bytes, &limits))
        .await
        .map_err(|e| internal_with(e, id))?
        .map_err(audio_error)?;
    let secs = samples.len() as f64 / SAMPLE_RATE as f64;
    meta.audio_secs = secs;

    let result: Transcription = if samples.len() < MIN_SAMPLES {
        Transcription {
            language: req.opts.language.clone(),
            ..Default::default()
        }
    } else {
        let engine = state.engine.clone();
        let job = TranscribeRequest {
            id: id.to_string(),
            lane: req.lane,
            model: ModelRef { id: entry.id, path },
            audio: Arc::new(samples),
            opts: req.opts.clone(),
            cancel,
        };
        tokio::task::spawn_blocking(move || {
            let out = engine.run(&job);
            drop(job);
            crate::scheduler::trim_heap();
            out
        })
        .await
        .map_err(|e| internal_with(e, id))?
        .map_err(|e| engine_error(e, id))?
    };
    meta.queue_wait_secs = result.queue_wait_secs;
    if state.config.server.log_text {
        tracing::debug!(request_id = %id, text = %result.text(), "transcript");
    }

    Ok(match req.format {
        Format::Json => Json(json!({
            "text": result.text(),
            "usage": {"type": "duration", "seconds": secs},
        }))
        .into_response(),
        Format::Text => plain(result.text()),
        Format::Srt => plain(format::render_srt(&result.segments)),
        Format::Vtt => plain(format::render_vtt(&result.segments)),
        Format::VerboseJson => Json(format::verbose_json(
            &result,
            secs,
            req.opts.temperature,
            &req.granularities,
        ))
        .into_response(),
    })
}

/// Dropped with the handler future when the client disconnects; tells the engine to stop.
struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn plain(body: String) -> Response {
    ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], body).into_response()
}

/// A failed or panicked blocking task: the client gets a generic message and a request id, the log the detail.
fn internal(e: tokio::task::JoinError) -> ApiError {
    internal_with(e, &new_request_id())
}

fn internal_with(e: tokio::task::JoinError, request_id: &str) -> ApiError {
    tracing::warn!(request_id, detail = %strip_paths(&e.to_string()), "internal error");
    ApiError::new(
        StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        "internal_error",
        format!("internal error (request {request_id})"),
    )
}

/// SIGTERM and SIGINT handlers, installed before anything slow so an early signal is not lost.
#[cfg(unix)]
pub struct Shutdown {
    term: tokio::signal::unix::Signal,
    int: tokio::signal::unix::Signal,
}

#[cfg(unix)]
impl Shutdown {
    pub fn install() -> std::io::Result<Shutdown> {
        use tokio::signal::unix::{SignalKind, signal};
        Ok(Shutdown {
            term: signal(SignalKind::terminate())?,
            int: signal(SignalKind::interrupt())?,
        })
    }

    async fn wait(mut self) {
        tokio::select! {
            _ = self.int.recv() => {}
            _ = self.term.recv() => {}
        }
        tracing::info!("shutting down");
    }
}

/// Ctrl-C, console close and system shutdown handlers, installed before anything slow.
#[cfg(windows)]
pub struct Shutdown {
    ctrl_c: tokio::signal::windows::CtrlC,
    close: tokio::signal::windows::CtrlClose,
    shutdown: tokio::signal::windows::CtrlShutdown,
}

#[cfg(windows)]
impl Shutdown {
    pub fn install() -> std::io::Result<Shutdown> {
        use tokio::signal::windows::{ctrl_c, ctrl_close, ctrl_shutdown};
        Ok(Shutdown {
            ctrl_c: ctrl_c()?,
            close: ctrl_close()?,
            shutdown: ctrl_shutdown()?,
        })
    }

    async fn wait(mut self) {
        tokio::select! {
            _ = self.ctrl_c.recv() => {}
            _ = self.close.recv() => {}
            _ = self.shutdown.recv() => {}
        }
        tracing::info!("shutting down");
    }
}

/// Binds the listen socket; a taken address is a clear error.
pub async fn bind(addr: std::net::SocketAddr) -> anyhow::Result<tokio::net::TcpListener> {
    tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        if e.kind() == std::io::ErrorKind::AddrInUse {
            anyhow::anyhow!("{addr} is already in use; is another hark running? ({e})")
        } else {
            anyhow::anyhow!("cannot listen on {addr}: {e}")
        }
    })
}

/// Serves on `listener` until SIGTERM or Ctrl-C: sends READY, stops accepting, aborts the running job
/// chunk so it resumes later, gives interactive requests `DRAIN_SECS` to finish, then stops the job runner.
pub async fn serve_on(
    listener: tokio::net::TcpListener,
    state: AppState,
    shutdown: Shutdown,
) -> anyhow::Result<()> {
    tracing::info!(addr = %listener.local_addr()?, "listening");
    crate::notify::ready();
    let jobs = state.jobs.clone();
    let (signalled_tx, signalled_rx) = tokio::sync::oneshot::channel::<()>();
    let jobs_for_signal = jobs.clone();
    let app = router(state).into_make_service_with_connect_info::<std::net::SocketAddr>();
    let server = axum::serve(listener, app).with_graceful_shutdown(async move {
        shutdown.wait().await;
        crate::notify::stopping();
        jobs_for_signal.begin_shutdown();
        let _ = signalled_tx.send(());
    });
    let drained = tokio::select! {
        r = server => {
            r?;
            true
        }
        _ = async {
            let _ = signalled_rx.await;
            tokio::time::sleep(Duration::from_secs(DRAIN_SECS)).await;
        } => false,
    };
    if !drained {
        tracing::warn!(
            secs = DRAIN_SECS,
            "interactive requests still running; exiting anyway"
        );
    }
    let stopper = jobs.clone();
    let _ = tokio::task::spawn_blocking(move || stopper.stop()).await;
    tracing::info!("stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{Segment, Word};
    use crate::scheduler::{
        DeviceStatus, LastError, ModelStatus, QueueDepths, RunningStatus, StatusReport,
    };
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    struct Fake;
    impl Transcriber for Fake {
        fn status(&self) -> StatusReport {
            StatusReport {
                device: DeviceStatus {
                    name: "Test GPU".into(),
                    index: Some(1),
                    fell_back_from: None,
                },
                models: vec![ModelStatus {
                    id: "base".into(),
                    load_ms: 1200,
                    states: 2,
                    vram_per_state_mib: Some(310),
                    realtime_factor: Some(0.04),
                    file_mib: Some(142),
                    active: true,
                    last_load_reason: "preload",
                    chunks_since_recycle: Some(7),
                    recycles: Some(1),
                }],
                queues: QueueDepths {
                    interactive: 1,
                    batch: 2,
                    live: 0,
                },
                running: vec![
                    RunningStatus {
                        state: 0,
                        status: "running",
                        lane: Some("batch"),
                        request_id: Some("req_1".into()),
                        model: Some("base".into()),
                        chunk: Some(3),
                    },
                    RunningStatus {
                        state: 1,
                        status: "idle",
                        lane: None,
                        request_id: None,
                        model: None,
                        chunk: None,
                    },
                ],
                aborted_chunks: 4,
                last_error: Some(LastError {
                    message: "boom".into(),
                    time: "2026-10-27T12:00:00Z".into(),
                }),
                ..StatusReport::default()
            }
        }

        fn transcribe(
            &self,
            _m: &ModelRef,
            audio: &[f32],
            _o: &DecodeOptions,
        ) -> Result<Transcription, EngineError> {
            assert!(audio.len() >= MIN_SAMPLES, "engine called for short audio");
            Ok(Transcription {
                language: Some("en".into()),
                segments: vec![Segment {
                    id: 0,
                    seek_cs: 0,
                    start: 0.0,
                    end: 1.5,
                    text: " hello world".into(),
                    tokens: vec![1, 2],
                    avg_logprob: -0.1,
                    no_speech_prob: 0.01,
                    words: vec![
                        Word {
                            word: "hello".into(),
                            start: 0.0,
                            end: 0.5,
                        },
                        Word {
                            word: "world".into(),
                            start: 0.6,
                            end: 1.5,
                        },
                    ],
                }],
                ..Default::default()
            })
        }
    }

    /// State over fresh temp dirs: models dir with a fake `base`, jobs dir per tag.
    fn test_state(
        tag: &str,
        mut cfg: Config,
        engine: Arc<dyn Transcriber>,
        vocabulary: VocabularyStore,
    ) -> AppState {
        let base =
            std::env::temp_dir().join(format!("hark-server-test-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        std::fs::write(base.join("ggml-base.bin"), b"fake").unwrap();
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        cfg.jobs.dir = base.join(format!("jobs-{n}")).display().to_string();
        let registry = Arc::new(Registry::new(
            vec![base],
            [("whisper-1".to_string(), "base".to_string())].into(),
        ));
        let cfg = Arc::new(cfg);
        let jobs = JobManager::open(cfg.clone(), registry.clone(), engine.clone()).unwrap();
        AppState::new(cfg, registry, engine, Arc::new(vocabulary), jobs)
    }

    fn app(tag: &str) -> Router {
        app_with(tag, Config::default())
    }

    fn app_with(tag: &str, cfg: Config) -> Router {
        router(test_state(
            tag,
            cfg,
            Arc::new(Fake),
            VocabularyStore::new(&Default::default()),
        ))
    }

    fn wav(rate: u32, samples: usize) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut buf = std::io::Cursor::new(Vec::new());
        let mut w = hound::WavWriter::new(&mut buf, spec).unwrap();
        for _ in 0..samples {
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();
        buf.into_inner()
    }

    fn multipart(fields: &[(&str, &[u8])]) -> Request<Body> {
        let mut body = Vec::new();
        for (name, data) in fields {
            body.extend_from_slice(
                format!("--XB\r\nContent-Disposition: form-data; name=\"{name}\"").as_bytes(),
            );
            if *name == "file" {
                body.extend_from_slice(b"; filename=\"a.wav\"\r\nContent-Type: audio/wav");
            }
            body.extend_from_slice(b"\r\n\r\n");
            body.extend_from_slice(data);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(b"--XB--\r\n");
        Request::post("/v1/audio/transcriptions")
            .header("content-type", "multipart/form-data; boundary=XB")
            .body(Body::from(body))
            .unwrap()
    }

    async fn call(app: Router, req: Request<Body>) -> (StatusCode, serde_json::Value, String) {
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let text = String::from_utf8(
            resp.into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        let json = serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
        (status, json, text)
    }

    fn assert_envelope(v: &serde_json::Value, code: &str) {
        let e = &v["error"];
        assert!(e["message"].is_string(), "{v}");
        assert!(e["type"].is_string(), "{v}");
        assert!(e.get("param").is_some(), "{v}");
        assert_eq!(e["code"], code, "{v}");
    }

    #[tokio::test]
    async fn origin_header_is_forbidden() {
        let req = Request::get("/v1/models")
            .header("origin", "http://evil.example")
            .body(Body::empty())
            .unwrap();
        let resp = app("origin").oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        assert!(
            resp.headers()
                .keys()
                .all(|k| !k.as_str().starts_with("access-control"))
        );
        let req = Request::get("/health")
            .header("origin", "null")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app("origin2").oneshot(req).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
    }

    #[tokio::test]
    async fn unknown_model_is_404_envelope() {
        let (s, v, _) = call(
            app("m404"),
            multipart(&[("file", &wav(16000, 16000)), ("model", b"nope")]),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_envelope(&v, "model_not_found");
        let (s, v, _) = call(
            app("m404b"),
            multipart(&[("file", &wav(16000, 16000)), ("model", b"../ggml-base.bin")]),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_envelope(&v, "model_not_found");
    }

    #[tokio::test]
    async fn missing_file_and_model_are_400() {
        let (s, v, _) = call(app("nofile"), multipart(&[("model", b"whisper-1")])).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_envelope(&v, "missing_required_parameter");
        assert_eq!(v["error"]["param"], "file");
        let (s, v, _) = call(app("nomodel"), multipart(&[("file", &wav(16000, 16000))])).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_eq!(v["error"]["param"], "model");
    }

    #[tokio::test]
    async fn live_priority_and_bad_fields_are_400() {
        let f = wav(16000, 16000);
        for (name, val, code) in [
            ("x_priority", &b"live"[..], "unsupported_value"),
            ("x_priority", b"urgent", "invalid_value"),
            ("timestamp_granularities[]", b"word", "invalid_value"),
            ("response_format", b"xml", "invalid_value"),
            ("temperature", b"1.5", "invalid_value"),
            ("temperature", b"abc", "invalid_value"),
            ("language", b"klingon", "invalid_value"),
            ("x_vad", b"maybe", "invalid_value"),
            ("timestamp_granularities[]", b"char", "invalid_value"),
        ] {
            let (s, v, _) = call(
                app("bad"),
                multipart(&[("file", &f), ("model", b"whisper-1"), (name, val)]),
            )
            .await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{name}={val:?}");
            assert_envelope(&v, code);
        }
    }

    struct Recorder(std::sync::Mutex<Vec<Vec<String>>>);
    impl Transcriber for Recorder {
        fn transcribe(
            &self,
            _m: &ModelRef,
            _a: &[f32],
            o: &DecodeOptions,
        ) -> Result<Transcription, EngineError> {
            self.0.lock().unwrap().push(o.vocabulary.clone());
            Ok(Transcription::default())
        }
    }

    fn vocab_app(tag: &str) -> (Router, Arc<Recorder>) {
        let cfg = Config::from_toml(
            "[vocabulary.default]\nterms = [\"Kubernetes\", \"Grafana\"]\n[vocabulary.legal]\nterms = [\"estoppel\"]\n",
        )
        .unwrap();
        let rec = Arc::new(Recorder(Default::default()));
        let vocab = VocabularyStore::new(&cfg.vocabulary);
        let app = router(test_state(tag, cfg, rec.clone(), vocab));
        (app, rec)
    }

    #[tokio::test]
    async fn x_vocabulary_selects_the_profile_the_engine_sees() {
        let (app, rec) = vocab_app("vocab");
        let f = wav(16000, 32000);
        for val in [None, Some(&b"default"[..]), Some(b"legal"), Some(b"none")] {
            let mut fields: Vec<(&str, &[u8])> = vec![("file", &f), ("model", b"whisper-1")];
            if let Some(v) = val {
                fields.push(("x_vocabulary", v));
            }
            let (s, v, _) = call(app.clone(), multipart(&fields)).await;
            assert_eq!(s, StatusCode::OK, "{v}");
        }
        let seen = rec.0.lock().unwrap().clone();
        let want: Vec<Vec<String>> = vec![
            vec!["Kubernetes".into(), "Grafana".into()],
            vec!["Kubernetes".into(), "Grafana".into()],
            vec!["estoppel".into()],
            vec![],
        ];
        assert_eq!(seen, want);
    }

    #[tokio::test]
    async fn unknown_or_malformed_x_vocabulary_is_400() {
        let (app, rec) = vocab_app("vocab400");
        let f = wav(16000, 32000);
        for bad in [&b"nope"[..], b"Bad Name", b"../x", b"NONE"] {
            let (s, v, _) = call(
                app.clone(),
                multipart(&[("file", &f), ("model", b"whisper-1"), ("x_vocabulary", bad)]),
            )
            .await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{bad:?}");
            assert_envelope(&v, "invalid_value");
            assert_eq!(v["error"]["param"], "x_vocabulary");
        }
        assert!(
            rec.0.lock().unwrap().is_empty(),
            "engine ran for a bad profile"
        );
    }

    #[tokio::test]
    async fn garbage_audio_is_400_and_server_survives() {
        let a = app("garbage");
        let (s, v, _) = call(
            a.clone(),
            multipart(&[("file", b"definitely not audio"), ("model", b"whisper-1")]),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_envelope(&v, "invalid_audio");
        let (s, v, _) = call(
            a.clone(),
            multipart(&[("file", b""), ("model", b"whisper-1")]),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_envelope(&v, "invalid_audio");
        let (s, _, _) = call(a, Request::get("/health").body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::OK);
    }

    #[tokio::test]
    async fn playlist_uploads_are_400_invalid_audio() {
        let d = std::env::temp_dir().join(format!("hark-playlist-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let wav_path = d.join("local.wav");
        std::fs::write(&wav_path, wav(44100, 44100)).unwrap();
        let m3u8 = b"#EXTM3U\n#EXTINF:1,\nfile:///etc/hostname\n".to_vec();
        let concat = format!("ffconcat version 1.0\nfile {}\n", wav_path.display()).into_bytes();
        for (name, body) in [("m3u8", m3u8), ("ffconcat", concat)] {
            let (s, v, text) = call(
                app("playlist"),
                multipart(&[("file", &body), ("model", b"whisper-1")]),
            )
            .await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{name}: {text}");
            assert_envelope(&v, "invalid_audio");
            assert!(!text.contains("hark-playlist"), "{name}: {text}");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[tokio::test]
    async fn non_16k_wav_goes_through_ffmpeg() {
        let (s, v, _) = call(
            app("ffmpeg"),
            multipart(&[("file", &wav(44100, 44100 * 2)), ("model", b"whisper-1")]),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["text"], "hello world");
        assert!((v["usage"]["seconds"].as_f64().unwrap() - 2.0).abs() < 0.05);
    }

    #[tokio::test]
    async fn audio_over_sync_limit_is_413_pointing_to_jobs() {
        let mut cfg = Config::default();
        cfg.server.sync_max_audio_secs = 1;
        let a = app_with("toolong", cfg);
        let (s, v, _) = call(
            a,
            multipart(&[("file", &wav(16000, 48000)), ("model", b"whisper-1")]),
        )
        .await;
        assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
        assert_envelope(&v, "audio_too_long");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("/hark/v1/jobs")
        );
    }

    #[tokio::test]
    async fn srt_vtt_and_verbose_json_responses() {
        let f = wav(16000, 32000);
        let get = |fmt: &'static [u8], extra: Option<(&'static str, &'static [u8])>| {
            let mut fields: Vec<(&str, &[u8])> = vec![
                ("file", &f),
                ("model", b"whisper-1"),
                ("response_format", fmt),
            ];
            if let Some(e) = extra {
                fields.push(e);
            }
            call(app("fmt"), multipart(&fields))
        };
        let (s, _, body) = get(b"srt", None).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(body, "1\n00:00:00,000 --> 00:00:01,500\nhello world\n\n");
        let (_, _, body) = get(b"vtt", None).await;
        assert_eq!(
            body,
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nhello world\n\n"
        );
        let (s, v, _) = get(b"verbose_json", None).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["task"], "transcribe");
        assert_eq!(v["language"], "english");
        assert!(v.get("words").is_none());
        assert_eq!(v["segments"][0]["text"], " hello world");
        let (_, v, _) = get(
            b"verbose_json",
            Some(("timestamp_granularities[]", b"word")),
        )
        .await;
        assert!(v.get("segments").is_none());
        assert_eq!(v["words"][1]["word"], "world");
    }

    #[tokio::test]
    async fn json_and_text_responses() {
        let f = wav(16000, 32000);
        let (s, v, _) = call(
            app("ok"),
            multipart(&[("file", &f), ("model", b"whisper-1")]),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["text"], "hello world");
        assert_eq!(v["usage"]["type"], "duration");
        assert_eq!(v["usage"]["seconds"].as_f64().unwrap(), 2.0);
        assert_eq!(v.as_object().unwrap().len(), 2);
        let resp = app("ok2")
            .oneshot(multipart(&[
                ("file", &f),
                ("model", b"whisper-1"),
                ("response_format", b"text"),
            ]))
            .await
            .unwrap();
        assert_eq!(
            resp.headers()[header::CONTENT_TYPE],
            "text/plain; charset=utf-8"
        );
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], b"hello world");
    }

    #[tokio::test]
    async fn short_audio_returns_empty_without_engine() {
        let (s, v, _) = call(
            app("short"),
            multipart(&[("file", &wav(16000, 8000)), ("model", b"whisper-1")]),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["text"], "");
    }

    #[tokio::test]
    async fn upload_over_limit_is_413() {
        let mut cfg = Config::default();
        cfg.server.max_upload_mb = 1;
        let a = router(test_state(
            "413",
            cfg,
            Arc::new(Fake),
            VocabularyStore::new(&Default::default()),
        ));
        let big = vec![0u8; 2 * 1024 * 1024];
        let (s, v, _) = call(a, multipart(&[("file", &big), ("model", b"whisper-1")])).await;
        assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
        assert_envelope(&v, "file_too_large");
    }

    #[tokio::test]
    async fn models_list_and_get() {
        let (s, v, _) = call(
            app("models"),
            Request::get("/v1/models").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["object"], "list");
        let ids: Vec<_> = v["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["base", "whisper-1"]);
        for m in v["data"].as_array().unwrap() {
            assert_eq!(m["object"], "model");
            assert_eq!(m["owned_by"], "hark");
            assert!(m["created"].as_u64().unwrap() > 0);
        }
        let (s, v, _) = call(
            app("models2"),
            Request::get("/v1/models/whisper-1")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["id"], "whisper-1");
        let (s, v, _) = call(
            app("models3"),
            Request::get("/v1/models/small")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_envelope(&v, "model_not_found");
    }

    #[tokio::test]
    async fn health_shape() {
        let (s, v, _) = call(
            app("health"),
            Request::get("/health").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["status"], "ok");
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn status_shape() {
        let (s, v, _) = call(
            app("status"),
            Request::get("/hark/v1/status").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["device"]["name"], "Test GPU");
        assert_eq!(v["device"]["index"], 1);
        assert_eq!(v["degraded"], false);
        let m = &v["models"][0];
        assert_eq!(m["id"], "base");
        assert_eq!(m["load_ms"], 1200);
        assert_eq!(m["states"], 2);
        assert_eq!(m["vram_per_state_mib"], 310);
        assert_eq!(m["realtime_factor"], 0.04);
        assert_eq!(v["queues"]["interactive"], 1);
        assert_eq!(v["queues"]["batch"], 2);
        assert_eq!(v["queues"]["live"], 0);
        assert_eq!(v["running"][0]["lane"], "batch");
        assert_eq!(v["running"][0]["request_id"], "req_1");
        assert_eq!(v["running"][0]["model"], "base");
        assert_eq!(v["running"][0]["chunk"], 3);
        assert!(v["running"][1]["lane"].is_null());
        assert_eq!(v["aborted_chunks"], 4);
        assert_eq!(v["models"][0]["chunks_since_recycle"], 7);
        assert_eq!(v["models"][0]["recycles"], 1);
        assert_eq!(v["recycle_failures"], 0);
        assert_eq!(v["last_error"]["message"], "boom");
        assert!(v["last_error"]["time"].is_string());
    }

    #[tokio::test]
    async fn status_defaults_are_well_formed_and_origin_is_refused() {
        let (_, v, _) = call(
            vocab_app("status2").0,
            Request::get("/hark/v1/status").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(v["device"]["name"], "cpu");
        assert!(v["device"]["index"].is_null());
        assert!(v["models"].as_array().unwrap().is_empty());
        assert!(v["last_error"].is_null());
        let req = Request::get("/hark/v1/status")
            .header("origin", "http://evil.example")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app("status3").oneshot(req).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
    }

    struct Saturated;
    impl Transcriber for Saturated {
        fn transcribe(
            &self,
            _m: &ModelRef,
            _a: &[f32],
            _o: &DecodeOptions,
        ) -> Result<Transcription, EngineError> {
            Err(EngineError::Busy {
                retry_after_secs: 7,
            })
        }
    }

    #[tokio::test]
    async fn full_queue_is_429_with_retry_after() {
        let a = router(test_state(
            "429",
            Config::default(),
            Arc::new(Saturated),
            VocabularyStore::new(&Default::default()),
        ));
        let resp = a
            .oneshot(multipart(&[
                ("file", &wav(16000, 32000)),
                ("model", b"whisper-1"),
            ]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(resp.headers()[header::RETRY_AFTER], "7");
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_envelope(&v, "rate_limit_exceeded");
    }

    struct CountBackend {
        runs: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl crate::scheduler::Backend for CountBackend {
        fn load(
            &self,
            _m: &ModelRef,
            states: usize,
        ) -> Result<crate::scheduler::LoadReport, EngineError> {
            Ok(crate::scheduler::LoadReport {
                load_ms: 1,
                states,
                vram_per_state_mib: None,
                fell_back: None,
            })
        }
        fn unload(&self) {}
        fn run(
            &self,
            _state: usize,
            _task: &crate::scheduler::ChunkTask,
            abort: &AtomicBool,
        ) -> Result<crate::scheduler::ChunkOut, crate::scheduler::ChunkFailure> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            let end = Instant::now() + Duration::from_millis(120);
            while Instant::now() < end {
                if abort.load(Ordering::SeqCst) {
                    return Err(crate::scheduler::ChunkFailure::Aborted);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            Ok(crate::scheduler::ChunkOut::default())
        }
    }

    #[tokio::test]
    async fn a_dropped_connection_stops_the_remaining_chunks() {
        use crate::engine::WhisperEngine;
        use crate::scheduler::SchedulerConfig;
        let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let engine = Arc::new(WhisperEngine::with_backend(
            CountBackend { runs: runs.clone() },
            SchedulerConfig {
                states: 1,
                max_interactive_waiting: 4,
            },
        ));
        let app = router(test_state(
            "cancel",
            Config::default(),
            engine.clone(),
            VocabularyStore::new(&Default::default()),
        ));
        let body = wav(16000, 16000 * 400);
        let make = || multipart(&[("file", &body), ("model", b"whisper-1")]);

        let resp = app.clone().oneshot(make()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let full = runs.swap(0, Ordering::SeqCst);
        assert!(full >= 3, "the clip must take several chunks, got {full}");

        let dropped = tokio::time::timeout(Duration::from_millis(250), app.oneshot(make())).await;
        assert!(dropped.is_err(), "the request should still be running");
        let at_drop = runs.load(Ordering::SeqCst);
        assert!(at_drop >= 1);
        tokio::time::sleep(Duration::from_millis(900)).await;
        let after = runs.load(Ordering::SeqCst);
        assert!(
            after <= at_drop + 1 && after < full,
            "chunks kept running after the disconnect: {at_drop} at drop, {after} after, {full} for a full run"
        );
        let s = engine.status();
        assert_eq!((s.queues.interactive, s.queues.batch), (0, 0));
        assert!(
            s.running.iter().all(|r| r.status == "idle"),
            "{:?}",
            s.running
        );
    }

    struct JobsApp {
        app: Router,
        state: AppState,
        root: std::path::PathBuf,
    }

    /// App with the job runner started; the path root is a fresh temp dir holding `a.wav` (3 s).
    fn jobs_app(tag: &str, run: bool) -> JobsApp {
        jobs_app_with(tag, Config::default(), run)
    }

    fn jobs_app_with(tag: &str, mut cfg: Config, run: bool) -> JobsApp {
        let root =
            std::env::temp_dir().join(format!("hark-jobs-http-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a.wav"), wav(16000, 48000)).unwrap();
        cfg.jobs.path_roots = vec![root.display().to_string()];
        let state = test_state(
            &format!("jobs-{tag}"),
            cfg,
            Arc::new(Fake),
            VocabularyStore::new(&Default::default()),
        );
        if run {
            state.jobs.start();
        }
        JobsApp {
            app: router(state.clone()),
            state,
            root,
        }
    }

    fn json_post(uri: &str, body: serde_json::Value) -> Request<Body> {
        Request::post(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn get(uri: &str) -> Request<Body> {
        Request::get(uri).body(Body::empty()).unwrap()
    }

    async fn wait_job(app: &Router, id: &str, want: &str) -> serde_json::Value {
        for _ in 0..1000 {
            let (_, v, _) = call(app.clone(), get(&format!("/hark/v1/jobs/{id}"))).await;
            if v["status"] == want {
                return v;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("job {id} never reached {want}");
    }

    #[tokio::test]
    async fn json_submit_returns_202_and_the_job_finishes_with_every_format() {
        let j = jobs_app("formats", true);
        let path = j.root.join("a.wav").display().to_string();
        let (s, v, _) = call(
            j.app.clone(),
            json_post(
                "/hark/v1/jobs",
                json!({"path": path, "model": "whisper-1", "x_vocabulary": "none"}),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        let id = v["id"].as_str().unwrap().to_string();
        assert!(crate::jobs::valid_job_id(&id), "{id}");
        assert_eq!(v["model"], "base");
        assert!(v["created_at"].as_str().unwrap().ends_with('Z'));
        assert!(v["finished_at"].is_null());
        let done = wait_job(&j.app, &id, "succeeded").await;
        assert_eq!(done["progress"], 1.0);
        assert_eq!(done["audio_seconds"], 3.0);
        assert!(done["finished_at"].is_string());
        assert!(done["error"].is_null());

        let r = |q: &str| get(&format!("/hark/v1/jobs/{id}/result{q}"));
        let (s, v, _) = call(j.app.clone(), r("")).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["text"], "hello world");
        assert_eq!(v["usage"]["seconds"], 3.0);
        let (_, _, body) = call(j.app.clone(), r("?response_format=text")).await;
        assert_eq!(body, "hello world");
        let (_, _, body) = call(j.app.clone(), r("?response_format=srt")).await;
        assert_eq!(body, "1\n00:00:00,000 --> 00:00:01,500\nhello world\n\n");
        let (_, _, body) = call(j.app.clone(), r("?response_format=vtt")).await;
        assert_eq!(
            body,
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nhello world\n\n"
        );
        let (_, v, _) = call(j.app.clone(), r("?response_format=verbose_json")).await;
        assert_eq!(v["task"], "transcribe");
        assert_eq!(v["language"], "english");
        assert_eq!(v["duration"], 3.0);
        assert_eq!(v["segments"][0]["text"], " hello world");
        assert!(v.get("words").is_none());
        let (_, v, _) = call(
            j.app.clone(),
            r("?response_format=verbose_json&timestamp_granularities%5B%5D=word"),
        )
        .await;
        assert_eq!(v["words"][1]["word"], "world");
        assert!(v.get("segments").is_none());
        let (s, v, _) = call(
            j.app.clone(),
            r("?response_format=json&timestamp_granularities[]=word"),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_envelope(&v, "invalid_value");
        let (s, v, _) = call(j.app.clone(), r("?response_format=xml")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert_envelope(&v, "invalid_value");
        j.state.jobs.stop();
    }

    #[tokio::test]
    async fn multipart_submit_stages_an_upload_and_validates_like_transcriptions() {
        let j = jobs_app("upload", true);
        let mut req = multipart(&[("file", &wav(16000, 32000)), ("model", b"whisper-1")]);
        *req.uri_mut() = "/hark/v1/jobs".parse().unwrap();
        let (s, v, _) = call(j.app.clone(), req).await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        let id = v["id"].as_str().unwrap().to_string();
        wait_job(&j.app, &id, "succeeded").await;

        for (fields, code) in [
            (
                vec![("model", &b"whisper-1"[..])],
                "missing_required_parameter",
            ),
            (vec![("file", &b"x"[..])], "missing_required_parameter"),
            (vec![("file", b"x"), ("model", b"nope")], "model_not_found"),
            (
                vec![
                    ("file", b"x"),
                    ("model", b"whisper-1"),
                    ("language", b"klingon"),
                ],
                "invalid_value",
            ),
            (
                vec![
                    ("file", b"x"),
                    ("model", b"whisper-1"),
                    ("x_vocabulary", b"nope"),
                ],
                "invalid_value",
            ),
        ] {
            let mut req = multipart(&fields);
            *req.uri_mut() = "/hark/v1/jobs".parse().unwrap();
            let (s, v, _) = call(j.app.clone(), req).await;
            assert!(s.is_client_error(), "{code}: {s}");
            assert_envelope(&v, code);
        }
        let (s, v, _) = call(
            j.app.clone(),
            Request::post("/hark/v1/jobs")
                .header("content-type", "text/plain")
                .body(Body::from("x"))
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::UNSUPPORTED_MEDIA_TYPE);
        assert_envelope(&v, "unsupported_media_type");
        let (s, _, _) = call(
            j.app.clone(),
            json_post("/hark/v1/jobs", json!({"model": "whisper-1"})),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        j.state.jobs.stop();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn paths_outside_the_roots_and_symlinks_out_are_403() {
        let j = jobs_app("paths", false);
        let outside =
            std::env::temp_dir().join(format!("hark-jobs-outside-{}.wav", std::process::id()));
        std::fs::write(&outside, wav(16000, 16000)).unwrap();
        let link = j.root.join("link.wav");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&outside, &link).unwrap();
        for p in [
            outside.display().to_string(),
            link.display().to_string(),
            format!(
                "{}/../{}",
                j.root.display(),
                outside.file_name().unwrap().to_string_lossy()
            ),
            j.root.display().to_string(),
            j.root.join("missing.wav").display().to_string(),
            "a.wav".into(),
        ] {
            let (s, v, _) = call(
                j.app.clone(),
                json_post("/hark/v1/jobs", json!({"path": p, "model": "whisper-1"})),
            )
            .await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{p}");
            assert_envelope(&v, "path_not_allowed");
        }
        let (_, v, _) = call(j.app.clone(), get("/hark/v1/jobs")).await;
        assert!(
            v["data"].as_array().unwrap().is_empty(),
            "a refused path created a job"
        );
        let _ = std::fs::remove_file(&outside);
    }

    #[tokio::test]
    async fn list_filters_by_status_newest_first_and_validates() {
        let j = jobs_app("list", false);
        let path = j.root.join("a.wav").display().to_string();
        let mut ids = Vec::new();
        for _ in 0..3 {
            let (s, v, _) = call(
                j.app.clone(),
                json_post("/hark/v1/jobs", json!({"path": path, "model": "whisper-1"})),
            )
            .await;
            assert_eq!(s, StatusCode::ACCEPTED);
            ids.push(v["id"].as_str().unwrap().to_string());
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let list = |q: &'static str| {
            let app = j.app.clone();
            async move { call(app, get(&format!("/hark/v1/jobs{q}"))).await }
        };
        let (s, v, _) = list("").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["object"], "list");
        let got: Vec<&str> = v["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d["id"].as_str().unwrap())
            .collect();
        let want: Vec<&str> = ids.iter().rev().map(String::as_str).collect();
        assert_eq!(got, want, "newest first");
        let (_, v, _) = list("?status=queued&limit=2").await;
        assert_eq!(v["data"].as_array().unwrap().len(), 2);
        assert_eq!(v["data"][0]["id"], ids[2].as_str());
        let (_, v, _) = list("?status=succeeded").await;
        assert!(v["data"].as_array().unwrap().is_empty());
        for bad in [
            "?status=done",
            "?status=",
            "?limit=0",
            "?limit=1001",
            "?limit=x",
        ] {
            let (s, v, _) = list(bad).await;
            assert_eq!(s, StatusCode::BAD_REQUEST, "{bad}");
            assert_envelope(&v, "invalid_value");
        }
        let (_, v, _) = call(j.app.clone(), get("/hark/v1/status")).await;
        assert_eq!(v["jobs"]["queued"], 3);
        assert_eq!(v["jobs"]["running"], 0);
    }

    #[tokio::test]
    async fn unknown_ids_unfinished_results_and_delete_behave() {
        let j = jobs_app("misc", false);
        for uri in [
            "/hark/v1/jobs/job_aaaaaaaaaaaaaaaa",
            "/hark/v1/jobs/job_aaaaaaaaaaaaaaaa/result",
            "/hark/v1/jobs/..%2F..%2Fetc",
            "/hark/v1/jobs/nope/result",
        ] {
            let (s, v, _) = call(j.app.clone(), get(uri)).await;
            assert_eq!(s, StatusCode::NOT_FOUND, "{uri}");
            assert_envelope(&v, "job_not_found");
        }
        let (s, v, _) = call(
            j.app.clone(),
            Request::delete("/hark/v1/jobs/job_aaaaaaaaaaaaaaaa")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert_envelope(&v, "job_not_found");

        let path = j.root.join("a.wav").display().to_string();
        let (_, v, _) = call(
            j.app.clone(),
            json_post("/hark/v1/jobs", json!({"path": path, "model": "whisper-1"})),
        )
        .await;
        let id = v["id"].as_str().unwrap().to_string();
        let (s, v, _) = call(j.app.clone(), get(&format!("/hark/v1/jobs/{id}/result"))).await;
        assert_eq!(s, StatusCode::CONFLICT);
        assert_envelope(&v, "job_not_finished");
        let del = || {
            Request::delete(format!("/hark/v1/jobs/{id}"))
                .body(Body::empty())
                .unwrap()
        };
        let (s, v, _) = call(j.app.clone(), del()).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["status"], "cancelled");
        let (s, v, _) = call(j.app.clone(), get(&format!("/hark/v1/jobs/{id}/result"))).await;
        assert_eq!(s, StatusCode::CONFLICT);
        assert_envelope(&v, "job_cancelled");
        let (s, v, _) = call(j.app.clone(), del()).await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v, json!({"id": id, "deleted": true}));
        let (s, _, _) = call(j.app.clone(), get(&format!("/hark/v1/jobs/{id}"))).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn a_failed_job_reports_job_failed_and_origin_is_refused() {
        let j = jobs_app("failed", true);
        std::fs::write(j.root.join("junk.bin"), b"not audio at all").unwrap();
        let path = j.root.join("junk.bin").display().to_string();
        let (_, v, _) = call(
            j.app.clone(),
            json_post("/hark/v1/jobs", json!({"path": path, "model": "whisper-1"})),
        )
        .await;
        let id = v["id"].as_str().unwrap().to_string();
        let done = wait_job(&j.app, &id, "failed").await;
        assert!(done["error"].as_str().unwrap().contains("could not decode"));
        let (s, v, _) = call(j.app.clone(), get(&format!("/hark/v1/jobs/{id}/result"))).await;
        assert_eq!(s, StatusCode::CONFLICT);
        assert_envelope(&v, "job_failed");
        let req = Request::get("/hark/v1/jobs")
            .header("origin", "http://evil.example")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            j.app.clone().oneshot(req).await.unwrap().status(),
            StatusCode::FORBIDDEN
        );
        j.state.jobs.stop();
    }

    fn key_file(tag: &str, mode: u32) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("hark-server-key-{}-{tag}", std::process::id()));
        std::fs::write(&p, "test-key-0123456789\n").unwrap();
        #[cfg(unix)]
        std::fs::set_permissions(&p, std::os::unix::fs::PermissionsExt::from_mode(mode)).unwrap();
        #[cfg(not(unix))]
        let _ = mode;
        p
    }

    fn keyed_app(tag: &str) -> Router {
        let mut cfg = Config::default();
        cfg.server.api_key_file = key_file(tag, 0o600).display().to_string();
        app_with(tag, cfg)
    }

    const ROUTES: &[(&str, &str)] = &[
        ("GET", "/v1/models"),
        ("GET", "/v1/models/base"),
        ("POST", "/v1/audio/transcriptions"),
        ("GET", "/hark/v1/status"),
        ("GET", "/hark/v1/models"),
        ("POST", "/hark/v1/models/pull"),
        ("POST", "/hark/v1/models/base/load"),
        ("POST", "/hark/v1/models/base/unload"),
        ("DELETE", "/hark/v1/models/base"),
        ("GET", "/hark/v1/jobs"),
        ("POST", "/hark/v1/jobs"),
        ("GET", "/hark/v1/jobs/job_x"),
        ("DELETE", "/hark/v1/jobs/job_x"),
        ("GET", "/hark/v1/jobs/job_x/result"),
        ("GET", "/no/such/route"),
    ];

    fn request(method: &str, path: &str, auth: Option<&str>) -> Request<Body> {
        let mut b = Request::builder().method(method).uri(path);
        if let Some(a) = auth {
            b = b.header("authorization", a);
        }
        b.body(Body::empty()).unwrap()
    }

    #[tokio::test]
    async fn with_a_key_every_route_but_health_needs_the_bearer_token() {
        let app = keyed_app("auth");
        for (method, path) in ROUTES {
            for bad in [
                None,
                Some("Bearer wrong-key"),
                Some("Basic test-key-0123456789"),
                Some("test-key-0123456789"),
            ] {
                let (status, v, _) = call(app.clone(), request(method, path, bad)).await;
                assert_eq!(
                    status,
                    StatusCode::UNAUTHORIZED,
                    "{method} {path} with {bad:?}"
                );
                assert_envelope(&v, "invalid_api_key");
            }
            let (status, _, _) = call(
                app.clone(),
                request(method, path, Some("Bearer test-key-0123456789")),
            )
            .await;
            assert_ne!(
                status,
                StatusCode::UNAUTHORIZED,
                "{method} {path} with the key"
            );
            assert_ne!(status, StatusCode::FORBIDDEN, "{method} {path}");
        }
        let (status, v, _) = call(app.clone(), request("GET", "/health", None)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["status"], "ok");
        let req = Request::get("/health")
            .header("origin", "http://evil.example")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            call(app, req).await.0,
            StatusCode::FORBIDDEN,
            "Origin beats /health"
        );
    }

    #[tokio::test]
    async fn the_401_carries_a_challenge_and_no_cors_headers() {
        let resp = keyed_app("challenge")
            .oneshot(request("GET", "/v1/models", None))
            .await
            .unwrap();
        assert_eq!(resp.headers()[header::WWW_AUTHENTICATE], "Bearer");
        assert!(
            resp.headers()
                .keys()
                .all(|k| !k.as_str().starts_with("access-control"))
        );
    }

    #[tokio::test]
    async fn an_unreadable_key_file_fails_closed() {
        let mut cfg = Config::default();
        cfg.server.api_key_file = key_file("loose", 0o644).display().to_string();
        let app = app_with("loose", cfg);
        let (status, _, _) = call(
            app.clone(),
            request("GET", "/v1/models", Some("Bearer test-key-0123456789")),
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(
            call(app, request("GET", "/health", None)).await.0,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn without_a_key_routes_are_open() {
        let (status, _, _) = call(app("nokey"), request("GET", "/v1/models", None)).await;
        assert_eq!(status, StatusCode::OK);
    }

    struct DegradedEngine;
    impl Transcriber for DegradedEngine {
        fn status(&self) -> StatusReport {
            StatusReport {
                degraded: true,
                device: DeviceStatus {
                    name: "cpu".into(),
                    index: None,
                    fell_back_from: Some("Test GPU".into()),
                },
                ..StatusReport::default()
            }
        }
        fn transcribe(
            &self,
            _m: &ModelRef,
            _a: &[f32],
            _o: &DecodeOptions,
        ) -> Result<Transcription, EngineError> {
            Ok(Transcription::default())
        }
    }

    #[tokio::test]
    async fn health_says_degraded_once_the_engine_is() {
        let app = router(test_state(
            "degraded",
            Config::default(),
            Arc::new(DegradedEngine),
            VocabularyStore::new(&Default::default()),
        ));
        let (status, v, _) = call(app.clone(), request("GET", "/health", None)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(v["status"], "degraded");
        assert!(v["version"].is_string());
        let (_, s, _) = call(app, request("GET", "/hark/v1/status", None)).await;
        assert_eq!(s["degraded"], true);
        assert_eq!(s["device"]["name"], "cpu");
        assert_eq!(s["device"]["fell_back_from"], "Test GPU");
    }

    #[derive(Default)]
    struct Recording {
        calls: std::sync::Mutex<Vec<String>>,
        /// Milliseconds an unload takes, to hold a delete open.
        unload_ms: std::sync::atomic::AtomicU64,
    }
    impl Transcriber for Recording {
        fn load_model(&self, m: &ModelRef) -> Result<(), EngineError> {
            self.calls.lock().unwrap().push(format!("load {}", m.id));
            Ok(())
        }
        fn unload_model(&self, id: &str) -> Result<bool, EngineError> {
            self.calls.lock().unwrap().push(format!("unload {id}"));
            std::thread::sleep(Duration::from_millis(self.unload_ms.load(Ordering::SeqCst)));
            Ok(true)
        }
        fn set_deleting(&self, id: &str, on: bool) {
            self.calls
                .lock()
                .unwrap()
                .push(format!("deleting {id} {on}"));
        }
        fn transcribe(
            &self,
            _m: &ModelRef,
            _a: &[f32],
            _o: &DecodeOptions,
        ) -> Result<Transcription, EngineError> {
            Ok(Transcription::default())
        }
    }

    /// Own models dir holding `base`; extra dir holding `small`.
    fn admin_app(
        tag: &str,
    ) -> (
        Router,
        Arc<Recording>,
        std::path::PathBuf,
        std::path::PathBuf,
    ) {
        let root =
            std::env::temp_dir().join(format!("hark-admin-test-{}-{tag}", std::process::id()));
        let own = root.join("own");
        let extra = root.join("extra");
        std::fs::create_dir_all(&own).unwrap();
        std::fs::create_dir_all(&extra).unwrap();
        std::fs::write(own.join("ggml-base.bin"), b"fake").unwrap();
        std::fs::write(extra.join("ggml-small.bin"), b"fake").unwrap();
        let mut cfg = Config::default();
        cfg.jobs.dir = root.join("jobs").display().to_string();
        let registry = Arc::new(Registry::new(
            vec![own.clone(), extra.clone()],
            [("whisper-1".to_string(), "base".to_string())].into(),
        ));
        let engine = Arc::new(Recording::default());
        let cfg = Arc::new(cfg);
        let state = AppState::new(
            cfg.clone(),
            registry.clone(),
            engine.clone(),
            Arc::new(VocabularyStore::new(&Default::default())),
            JobManager::open(cfg, registry, engine.clone()).unwrap(),
        );
        (router(state), engine, own, extra)
    }

    fn raw_json_post(path: &str, body: &str) -> Request<Body> {
        Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn load_and_unload_go_to_the_engine_and_check_the_model_id() {
        let (app, engine, _, _) = admin_app("loadunload");
        let (st, v, _) = call(
            app.clone(),
            request("POST", "/hark/v1/models/base/load", None),
        )
        .await;
        assert_eq!((st, &v["loaded"]), (StatusCode::OK, &json!(true)));
        let (st, v, _) = call(
            app.clone(),
            request("POST", "/hark/v1/models/whisper-1/unload", None),
        )
        .await;
        assert_eq!((st, &v["unloaded"]), (StatusCode::OK, &json!(true)));
        assert_eq!(
            *engine.calls.lock().unwrap(),
            vec!["load base", "unload base"]
        );
        let (st, v, _) = call(
            app.clone(),
            request("POST", "/hark/v1/models/tiny/load", None),
        )
        .await;
        assert_eq!(st, StatusCode::NOT_FOUND, "not installed: {v}");
        assert_envelope(&v, "model_not_found");
        for bad in ["..%2Fetc%2Fpasswd", "nope", "%2Fetc%2Fpasswd", "base%00"] {
            let (st, v, _) = call(
                app.clone(),
                request("POST", &format!("/hark/v1/models/{bad}/load"), None),
            )
            .await;
            assert_eq!(st, StatusCode::NOT_FOUND, "{bad}");
            assert_envelope(&v, "model_not_found");
        }
        assert_eq!(
            engine.calls.lock().unwrap().len(),
            2,
            "bad ids never reach the engine"
        );
    }

    #[tokio::test]
    async fn delete_unloads_first_and_removes_only_from_models_dir() {
        let (app, engine, own, extra) = admin_app("delete");
        let (st, v, _) = call(app.clone(), request("DELETE", "/hark/v1/models/base", None)).await;
        assert_eq!((st, &v["deleted"]), (StatusCode::OK, &json!(true)));
        assert!(!own.join("ggml-base.bin").exists());
        assert_eq!(
            *engine.calls.lock().unwrap(),
            vec!["deleting base true", "unload base", "deleting base false"],
            "marked as deleting, unloaded, then removed"
        );
        let (st, v, _) = call(
            app.clone(),
            request("DELETE", "/hark/v1/models/small", None),
        )
        .await;
        assert_eq!(st, StatusCode::CONFLICT, "{v}");
        assert_envelope(&v, "model_read_only");
        assert!(extra.join("ggml-small.bin").exists());
        assert_eq!(
            engine.calls.lock().unwrap().len(),
            3,
            "an extra-dir model is not unloaded"
        );
        let (st, v, _) = call(app.clone(), request("DELETE", "/hark/v1/models/base", None)).await;
        assert_eq!(st, StatusCode::NOT_FOUND, "{v}");
        for bad in ["nope", "..%2F..%2Fx", "whisper-1"] {
            let (st, _, _) = call(
                app.clone(),
                request("DELETE", &format!("/hark/v1/models/{bad}"), None),
            )
            .await;
            assert_eq!(st, StatusCode::NOT_FOUND, "{bad}");
        }
    }

    #[tokio::test]
    async fn a_load_during_a_delete_waits_and_then_finds_the_model_gone() {
        let (app, engine, own, _) = admin_app("delete-race");
        engine.unload_ms.store(400, Ordering::SeqCst);
        let deleting = tokio::spawn({
            let app = app.clone();
            async move { call(app, request("DELETE", "/hark/v1/models/base", None)).await }
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        let started = Instant::now();
        let (st, v, _) = call(
            app.clone(),
            request("POST", "/hark/v1/models/base/load", None),
        )
        .await;
        assert!(
            started.elapsed() >= Duration::from_millis(200),
            "the load waited for the delete"
        );
        assert_eq!(st, StatusCode::NOT_FOUND, "{v}");
        assert!(!own.join("ggml-base.bin").exists());
        assert_eq!(deleting.await.unwrap().0, StatusCode::OK);
        assert!(
            !engine
                .calls
                .lock()
                .unwrap()
                .iter()
                .any(|c| c == "load base"),
            "the model was never reloaded in between"
        );
    }

    #[tokio::test]
    async fn pull_validates_the_body_and_skips_installed_models() {
        let (app, _, _, _) = admin_app("pull");
        for body in [
            "",
            "not json",
            "{}",
            "{\"id\": 5}",
            "{\"id\": \"x\"}",
            "{\"id\": \"../base\"}",
            "[]",
        ] {
            let (st, v, _) = call(app.clone(), raw_json_post("/hark/v1/models/pull", body)).await;
            assert!(
                st == StatusCode::BAD_REQUEST || st == StatusCode::NOT_FOUND,
                "{body}: {st}"
            );
            assert!(v["error"]["message"].is_string(), "{body}: {v}");
        }
        let (st, v, _) = call(
            app.clone(),
            raw_json_post("/hark/v1/models/pull", "{\"id\": \"base\"}"),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert_eq!(v["status"], "already_installed");
        assert_eq!(v["downloaded"], false);
        let (st, v, _) = call(
            app.clone(),
            raw_json_post("/hark/v1/models/pull", "{\"id\": \"small\"}"),
        )
        .await;
        assert_eq!(
            (st, &v["status"]),
            (StatusCode::OK, &json!("already_installed")),
            "extra dir counts"
        );
        let (st, v, _) = call(
            app.clone(),
            raw_json_post("/hark/v1/models/pull", "{\"id\": \"whisper-1\"}"),
        )
        .await;
        assert_eq!(st, StatusCode::OK, "{v}");
        assert_eq!(
            (&v["id"], &v["status"]),
            (&json!("base"), &json!("already_installed")),
            "alias resolves"
        );
        let big = format!("{{\"id\": \"{}\"}}", "a".repeat(100_000));
        let (st, _, _) = call(app.clone(), raw_json_post("/hark/v1/models/pull", &big)).await;
        assert_eq!(st, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn the_catalog_lists_install_location_and_state() {
        let (app, _, _, _) = admin_app("catalog");
        let (st, v, _) = call(app, request("GET", "/hark/v1/models", None)).await;
        assert_eq!(st, StatusCode::OK);
        let data = v["data"].as_array().unwrap();
        let find = |id: &str| data.iter().find(|m| m["id"] == id).unwrap();
        assert_eq!(find("base")["location"], "models_dir");
        assert_eq!(find("small")["location"], "extra_dir");
        assert!(find("tiny")["location"].is_null());
        assert_eq!(find("tiny")["installed"], false);
        assert!(data.len() >= 29);
    }

    #[tokio::test]
    async fn too_many_form_fields_and_long_model_ids_are_refused() {
        let mut owned: Vec<(String, Vec<u8>)> = vec![("model".into(), b"base".to_vec())];
        for i in 0..70 {
            owned.push((format!("extra{i}"), b"x".to_vec()));
        }
        let fields: Vec<(&str, &[u8])> = owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_slice()))
            .collect();
        let (st, v, _) = call(app("fields"), multipart(&fields)).await;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        assert_envelope(&v, "too_many_fields");
        let w = wav(16_000, 16_000);
        let long = "m".repeat(200);
        let (st, v, _) = call(
            app("longmodel"),
            multipart(&[("file", &w), ("model", long.as_bytes())]),
        )
        .await;
        assert_eq!(st, StatusCode::NOT_FOUND);
        assert_envelope(&v, "model_not_found");
        let (st, v, _) = call(
            app("manygran"),
            multipart(&[
                ("file", &w),
                ("model", b"base"),
                ("response_format", b"verbose_json"),
                ("timestamp_granularities[]", b"word"),
                ("timestamp_granularities[]", b"word"),
                ("timestamp_granularities[]", b"word"),
                ("timestamp_granularities[]", b"word"),
                ("timestamp_granularities[]", b"word"),
                ("timestamp_granularities[]", b"word"),
                ("timestamp_granularities[]", b"word"),
                ("timestamp_granularities[]", b"word"),
                ("timestamp_granularities[]", b"word"),
            ]),
        )
        .await;
        assert_eq!(st, StatusCode::BAD_REQUEST);
        assert_envelope(&v, "invalid_value");
    }

    fn with_host(host: &str, path: &str) -> Request<Body> {
        Request::get(path)
            .header("host", host)
            .body(Body::empty())
            .unwrap()
    }

    #[tokio::test]
    async fn the_host_header_must_be_a_loopback_name_on_the_listen_port() {
        let app = app("hosts");
        for host in [
            "127.0.0.1:8797",
            "localhost:8797",
            "[::1]:8797",
            "LocalHost:8797",
        ] {
            for path in ["/health", "/hark/v1/jobs", "/hark/v1/status", "/v1/models"] {
                let (s, v, _) = call(app.clone(), with_host(host, path)).await;
                assert_eq!(s, StatusCode::OK, "{host} {path}: {v}");
            }
        }
        for host in [
            "evil.example:8797",
            "evil.example",
            "127.0.0.1",
            "localhost",
            "127.0.0.1:9999",
            "127.0.0.1.evil.example:8797",
            "localhost:8797.evil.example",
            "",
        ] {
            for path in [
                "/health",
                "/hark/v1/jobs",
                "/hark/v1/jobs/job_aaaaaaaaaaaaaaaa/result",
            ] {
                let (s, v, _) = call(app.clone(), with_host(host, path)).await;
                assert_eq!(s, StatusCode::FORBIDDEN, "{host:?} {path}");
                assert_envelope(&v, "host_forbidden");
            }
        }
    }

    #[tokio::test]
    async fn allowed_hosts_extend_the_list_and_the_host_check_runs_before_auth() {
        let mut cfg = Config::default();
        cfg.server.allowed_hosts = vec!["Hark.Lan:8797".into()];
        let app = app_with("hosts-extra", cfg);
        let (s, _, _) = call(app.clone(), with_host("hark.lan:8797", "/health")).await;
        assert_eq!(s, StatusCode::OK);
        let (s, v, _) = call(app, with_host("other.lan:8797", "/health")).await;
        assert_eq!(s, StatusCode::FORBIDDEN);
        assert_envelope(&v, "host_forbidden");
        let keyed = keyed_app("hosts-auth");
        let (s, v, _) = call(keyed, with_host("evil.example:8797", "/hark/v1/status")).await;
        assert_eq!(s, StatusCode::FORBIDDEN, "403 before 401");
        assert_envelope(&v, "host_forbidden");
    }

    #[tokio::test]
    async fn a_missing_host_is_allowed_only_from_a_loopback_peer() {
        let app = app("hosts-missing");
        let from = |ip: &str| {
            let mut req = request("GET", "/hark/v1/jobs", None);
            let addr: std::net::SocketAddr = format!("{ip}:50000").parse().unwrap();
            req.extensions_mut().insert(ConnectInfo(addr));
            req
        };
        for ip in ["127.0.0.1", "[::1]"] {
            let (s, _, _) = call(app.clone(), from(ip)).await;
            assert_eq!(s, StatusCode::OK, "{ip}");
        }
        for ip in ["192.168.1.5", "10.0.0.2"] {
            let (s, v, _) = call(app.clone(), from(ip)).await;
            assert_eq!(s, StatusCode::FORBIDDEN, "{ip}");
            assert_envelope(&v, "host_forbidden");
        }
    }

    #[tokio::test]
    async fn job_uploads_are_staged_inside_the_job_directory_and_capped() {
        let mut cfg = Config::default();
        cfg.jobs.max_queued = 2;
        let j = jobs_app_with("caps", cfg, false);
        let jobs_dir = std::path::PathBuf::from(&j.state.config.jobs.dir);
        let upload = || {
            let mut req = multipart(&[("file", &wav(16000, 32000)), ("model", b"whisper-1")]);
            *req.uri_mut() = "/hark/v1/jobs".parse().unwrap();
            req
        };
        let (s, v, _) = call(j.app.clone(), upload()).await;
        assert_eq!(s, StatusCode::ACCEPTED, "{v}");
        let id = v["id"].as_str().unwrap();
        assert!(
            jobs_dir.join(id).join("upload.bin").is_file(),
            "staged in the job directory on disk"
        );
        let (s, _, _) = call(
            j.app.clone(),
            json_post(
                "/hark/v1/jobs",
                json!({"model": "whisper-1", "path": j.root.join("a.wav")}),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED);
        let before = j.state.jobs.list(None, 100).len();
        let resp = j.app.clone().oneshot(upload()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(resp.headers().contains_key("retry-after"));
        let body = String::from_utf8(
            resp.into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains("queue_full"), "{body}");
        let (s, v, _) = call(
            j.app.clone(),
            json_post(
                "/hark/v1/jobs",
                json!({"model": "whisper-1", "path": j.root.join("a.wav")}),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::TOO_MANY_REQUESTS, "{v}");
        assert_envelope(&v, "queue_full");
        assert_eq!(j.state.jobs.list(None, 100).len(), before);
        assert_eq!(
            std::fs::read_dir(&jobs_dir)
                .unwrap()
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with("job_"))
                .count(),
            before,
            "a refused submit leaves no directory behind"
        );
    }

    #[tokio::test]
    async fn an_upload_that_would_leave_too_little_disk_is_507() {
        let mut cfg = Config::default();
        cfg.jobs.min_free_mb = u64::MAX / (1024 * 1024);
        let j = jobs_app_with("space", cfg, false);
        let mut req = multipart(&[("file", &wav(16000, 32000)), ("model", b"whisper-1")]);
        *req.uri_mut() = "/hark/v1/jobs".parse().unwrap();
        let (s, v, _) = call(j.app.clone(), req).await;
        assert_eq!(s, StatusCode::INSUFFICIENT_STORAGE, "{v}");
        assert_envelope(&v, "insufficient_storage");
        assert!(j.state.jobs.list(None, 100).is_empty());
        let (s, _, _) = call(
            j.app.clone(),
            json_post(
                "/hark/v1/jobs",
                json!({"model": "whisper-1", "path": j.root.join("a.wav")}),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::ACCEPTED, "path jobs stage nothing");
    }

    fn slow_upload_request(rx: tokio::sync::mpsc::Receiver<Bytes>) -> Request<Body> {
        let stream = futures_util::stream::unfold(rx, |mut rx| async move {
            rx.recv().await.map(|c| (Ok::<_, std::io::Error>(c), rx))
        });
        Request::post("/v1/audio/transcriptions")
            .header("content-type", "multipart/form-data; boundary=XB")
            .body(Body::from_stream(stream))
            .unwrap()
    }

    #[tokio::test]
    async fn concurrent_upload_bodies_are_capped_with_429_and_retry_after() {
        let mut cfg = Config::default();
        cfg.server.max_concurrent_uploads = 1;
        let app = app_with("uploads", cfg);
        let w = wav(16000, 32000);
        let head = [
            &b"--XB\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n"[..],
            b"--XB\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n",
            &w[..100],
        ]
        .concat();
        let (tx, rx) = tokio::sync::mpsc::channel::<Bytes>(4);
        tx.send(Bytes::from(head)).await.unwrap();
        let first = tokio::spawn(app.clone().oneshot(slow_upload_request(rx)));
        tokio::time::sleep(Duration::from_millis(300)).await;
        let resp = app
            .clone()
            .oneshot(multipart(&[("file", &w), ("model", b"whisper-1")]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(resp.headers()["retry-after"], "1");
        let rest = [&w[100..], &b"\r\n--XB--\r\n"[..]].concat();
        tx.send(Bytes::from(rest)).await.unwrap();
        drop(tx);
        assert_eq!(first.await.unwrap().unwrap().status(), StatusCode::OK);
        let (s, _, _) = call(app, multipart(&[("file", &w), ("model", b"whisper-1")])).await;
        assert_eq!(s, StatusCode::OK, "the permit is released");
    }

    struct Breaking(&'static str);
    impl Transcriber for Breaking {
        fn transcribe(
            &self,
            _m: &ModelRef,
            _a: &[f32],
            _o: &DecodeOptions,
        ) -> Result<Transcription, EngineError> {
            match self.0 {
                "load" => Err(EngineError::Load(
                    "/home/user/.config/models/ggml-base.bin: bad magic".into(),
                )),
                _ => panic!("secret panic text at /home/user/src/engine.rs:42"),
            }
        }
    }

    #[tokio::test]
    async fn engine_failures_reach_clients_without_paths_or_panic_text() {
        for mode in ["load", "panic"] {
            let app = router(test_state(
                &format!("leaks-{mode}"),
                Config::default(),
                Arc::new(Breaking(mode)),
                VocabularyStore::new(&Default::default()),
            ));
            let w = wav(16000, 32000);
            let (s, v, text) = call(app, multipart(&[("file", &w), ("model", b"whisper-1")])).await;
            assert!(s.is_server_error(), "{mode}: {s}");
            assert!(!text.contains("/home"), "{mode}: {text}");
            assert!(!text.contains("secret"), "{mode}: {text}");
            assert!(!text.contains("bad magic"), "{mode}: {text}");
            let msg = v["error"]["message"].as_str().unwrap();
            assert!(msg.contains("req_"), "{mode}: the request id is in {msg}");
            let want = if mode == "load" {
                "model could not be loaded"
            } else {
                "internal error"
            };
            assert!(msg.contains(want), "{mode}: {msg}");
        }
    }

    #[test]
    fn path_like_words_are_stripped_from_error_text() {
        assert_eq!(
            strip_paths("failed at /home/user/x.rs:4 with oops"),
            "failed at <path> with oops"
        );
        assert_eq!(strip_paths("plain text"), "plain text");
        assert_eq!(
            strip_paths(r"failed at C:\Users\bob\x.rs:4 and (D:\tmp) or ..\up with oops"),
            "failed at <path> and <path> or <path> with oops"
        );
    }
}
