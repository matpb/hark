//! `/hark/v1/jobs`: submit, list, read, result, cancel and delete.

use axum::Json;
use axum::body::Bytes;
use axum::extract::{FromRequest, Path, Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use super::{
    ApiError, AppState, Form, Format, ParsedRequest, internal, parse_format, plain, read_form,
    validate_fields, validate_granularities,
};
use crate::format;
use crate::jobs::{
    self, JobError, JobOptions, Removed, Source, Status, StoredResult, new_job_id, valid_job_id,
};

const DEFAULT_LIMIT: usize = 100;
const MAX_LIMIT: usize = 1000;
const MAX_JSON_BODY: usize = 1024 * 1024;

fn job_error(e: JobError, id: &str) -> ApiError {
    match e {
        JobError::NotFound => not_found(id),
        JobError::NotFinished => ApiError::new(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "job_not_finished",
            format!("job {id} has not finished"),
        ),
        JobError::Failed(msg) => ApiError::new(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "job_failed",
            format!("job {id} failed: {msg}"),
        ),
        JobError::Cancelled => ApiError::new(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "job_cancelled",
            format!("job {id} was cancelled"),
        ),
        JobError::QueueFull(_) => {
            let mut e = ApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "queue_full",
                e.to_string(),
            );
            e.retry_after = Some(30);
            e
        }
        JobError::InsufficientStorage => ApiError::new(
            StatusCode::INSUFFICIENT_STORAGE,
            "server_error",
            "insufficient_storage",
            "the jobs disk is too full to accept this upload".into(),
        ),
        JobError::Io(msg) => ApiError::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "job_store_error",
            msg,
        ),
    }
}

fn not_found(id: &str) -> ApiError {
    ApiError::new(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        "job_not_found",
        format!("no job {id}"),
    )
}

/// Unknown and malformed ids are the same 404, and a malformed id never reaches the filesystem.
fn checked_id(id: &str) -> Result<(), ApiError> {
    if valid_job_id(id) {
        Ok(())
    } else {
        Err(not_found(id))
    }
}

fn path_not_allowed() -> ApiError {
    ApiError::new(
        StatusCode::FORBIDDEN,
        "invalid_request_error",
        "path_not_allowed",
        "path must be an absolute path to an existing regular file under jobs.path_roots".into(),
    )
}

/// Query pairs with `+` and `%XX` decoded; repeated keys stay repeated.
fn parse_query(q: &str) -> Vec<(String, String)> {
    fn decode(s: &str) -> String {
        let b = s.as_bytes();
        let mut out = Vec::with_capacity(b.len());
        let mut i = 0;
        while i < b.len() {
            match b[i] {
                b'+' => out.push(b' '),
                b'%' if i + 2 < b.len() => {
                    let hex = std::str::from_utf8(&b[i + 1..i + 3])
                        .ok()
                        .and_then(|h| u8::from_str_radix(h, 16).ok());
                    match hex {
                        Some(v) => {
                            out.push(v);
                            i += 2;
                        }
                        None => out.push(b'%'),
                    }
                }
                c => out.push(c),
            }
            i += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }
    q.split('&')
        .filter(|p| !p.is_empty())
        .map(|p| match p.split_once('=') {
            Some((k, v)) => (decode(k), decode(v)),
            None => (decode(p), String::new()),
        })
        .collect()
}

fn json_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// A JSON submit body as the same `Form` multipart parsing yields, plus the `path`.
fn form_from_json(body: &[u8]) -> Result<(Form, Option<String>), ApiError> {
    let bad = |m: &str| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_json",
            m.to_string(),
        )
    };
    let v: Value = serde_json::from_slice(body).map_err(|e| bad(&format!("invalid JSON: {e}")))?;
    let obj = v.as_object().ok_or_else(|| bad("expected a JSON object"))?;
    let text = |k: &str| -> Result<Option<String>, ApiError> {
        match obj.get(k) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => json_text(v)
                .map(Some)
                .ok_or_else(|| bad(&format!("`{k}` must be a string, number or boolean"))),
        }
    };
    let mut form = Form {
        model: text("model")?,
        language: text("language")?,
        prompt: text("prompt")?,
        response_format: text("response_format")?,
        temperature: text("temperature")?,
        x_priority: text("x_priority")?,
        x_vad: text("x_vad")?,
        x_vocabulary: text("x_vocabulary")?,
        ..Form::default()
    };
    for key in ["timestamp_granularities", "timestamp_granularities[]"] {
        match obj.get(key) {
            None | Some(Value::Null) => {}
            Some(Value::Array(items)) => {
                for it in items {
                    form.granularities
                        .push(json_text(it).ok_or_else(|| bad("granularities must be strings"))?);
                }
            }
            Some(other) => form
                .granularities
                .push(json_text(other).ok_or_else(|| bad("granularities must be strings"))?),
        }
    }
    let path = match obj.get("path") {
        None | Some(Value::Null) => None,
        Some(Value::String(p)) => Some(p.clone()),
        Some(_) => return Err(bad("`path` must be a string")),
    };
    Ok((form, path))
}

enum Input {
    Upload(Bytes),
    Path(String),
}

fn job_options(
    state: &AppState,
    req: &ParsedRequest,
    raw_format: Option<&str>,
) -> Result<JobOptions, ApiError> {
    let vocabulary = state
        .vocabulary
        .resolve(req.vocabulary.as_deref())
        .map_err(ApiError::from)?;
    let (entry, _) = state
        .registry
        .resolve_installed(&req.model)
        .map_err(|e| ApiError::model(e, "model"))?;
    Ok(JobOptions {
        model: entry.id.to_string(),
        language: req.opts.language.clone(),
        prompt: req.opts.prompt.clone(),
        vocabulary,
        temperature: req.opts.temperature,
        vad: req.opts.vad,
        granularities: req.granularities.clone(),
        response_format: raw_format
            .map(str::trim)
            .filter(|f| !f.is_empty())
            .map(str::to_string),
    })
}

pub(super) async fn submit(State(state): State<AppState>, req: Request) -> Response {
    match submit_inner(&state, req).await {
        Ok(v) => (StatusCode::ACCEPTED, Json(v)).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn submit_inner(state: &AppState, req: Request) -> Result<Value, ApiError> {
    state
        .jobs
        .ensure_queue_room()
        .map_err(|e| job_error(e, ""))?;
    let announced = req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let ctype = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let (mut form, input) = if ctype.starts_with("multipart/form-data") {
        state
            .jobs
            .ensure_space(announced)
            .map_err(|e| job_error(e, ""))?;
        let mp = axum::extract::Multipart::from_request(req, state)
            .await
            .map_err(|e| {
                ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "invalid_multipart",
                    format!("expected a multipart/form-data body: {}", e.body_text()),
                )
            })?;
        let mut form = read_form(mp, state).await?;
        let file = form.file.take().ok_or_else(|| {
            ApiError::invalid(
                "file",
                "missing_required_parameter",
                "you must provide the `file` field or a JSON `path`",
            )
        })?;
        (form, Input::Upload(file))
    } else if ctype.starts_with("application/json") {
        let body = axum::body::to_bytes(req.into_body(), MAX_JSON_BODY)
            .await
            .map_err(|_| {
                ApiError::new(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "invalid_request_error",
                    "body_too_large",
                    "the JSON body is too large".into(),
                )
            })?;
        let (form, path) = form_from_json(&body)?;
        let path = path.ok_or_else(|| {
            ApiError::invalid(
                "path",
                "missing_required_parameter",
                "you must provide the `path` field",
            )
        })?;
        (form, Input::Path(path))
    } else {
        return Err(ApiError::new(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "unsupported_media_type",
            "send multipart/form-data with a file, or application/json with a path".into(),
        ));
    };
    let raw_format = form.response_format.clone();
    form.file = None;
    let parsed = validate_fields(form)?;
    let options = job_options(state, &parsed, raw_format.as_deref())?;
    let id = new_job_id().map_err(|e| job_error(e, ""))?;
    let source = match input {
        Input::Path(raw) => {
            let roots = state.jobs.path_roots().to_vec();
            let resolved =
                tokio::task::spawn_blocking(move || jobs::resolve_job_path(&raw, &roots))
                    .await
                    .map_err(internal)?
                    .ok_or_else(path_not_allowed)?;
            Source::Path {
                path: resolved.display().to_string(),
            }
        }
        Input::Upload(bytes) => {
            if bytes.is_empty() {
                return Err(ApiError::invalid(
                    "file",
                    "invalid_audio",
                    "the file contains no audio",
                ));
            }
            let staged_id = id.clone();
            let jobs = state.jobs.clone();
            let file = tokio::task::spawn_blocking(move || jobs.stage_upload(&staged_id, &bytes))
                .await
                .map_err(internal)?
                .map_err(|e| match e {
                    JobError::Io(msg) => {
                        tracing::warn!(detail = %super::strip_paths(&msg), "could not stage an upload");
                        ApiError::new(
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "server_error",
                            "staging_failed",
                            "could not stage the upload".into(),
                        )
                    }
                    other => job_error(other, &id),
                })?;
            Source::Upload { file }
        }
    };
    let jobs = state.jobs.clone();
    let submit_id = id.clone();
    let rec = tokio::task::spawn_blocking(move || jobs.submit(submit_id, options, source))
        .await
        .map_err(internal)?
        .map_err(|e| job_error(e, &id))?;
    tracing::info!(job = %rec.id, model = %rec.model, "job submitted");
    Ok(rec.view())
}

pub(super) async fn list(
    State(state): State<AppState>,
    req: Request,
) -> Result<Json<Value>, ApiError> {
    let mut status = None;
    let mut limit = DEFAULT_LIMIT;
    for (k, v) in parse_query(req.uri().query().unwrap_or("")) {
        match k.as_str() {
            "status" => {
                status = Some(Status::parse(&v).ok_or_else(|| {
                    ApiError::invalid(
                        "status",
                        "invalid_value",
                        format!(
                            "`{v}` is not one of queued, running, succeeded, failed, cancelled"
                        ),
                    )
                })?);
            }
            "limit" => {
                limit = v
                    .parse::<usize>()
                    .ok()
                    .filter(|n| (1..=MAX_LIMIT).contains(n))
                    .ok_or_else(|| {
                        ApiError::invalid(
                            "limit",
                            "invalid_value",
                            format!("limit must be a number from 1 to {MAX_LIMIT}"),
                        )
                    })?;
            }
            _ => {}
        }
    }
    let data: Vec<Value> = state
        .jobs
        .list(status, limit)
        .iter()
        .map(|j| j.view())
        .collect();
    Ok(Json(json!({"object": "list", "data": data})))
}

pub(super) async fn get_job(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    checked_id(&id)?;
    state
        .jobs
        .get(&id)
        .map(|j| Json(j.view()))
        .ok_or_else(|| not_found(&id))
}

pub(super) async fn remove(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    checked_id(&id)?;
    let jobs = state.jobs.clone();
    let target = id.clone();
    let (kind, rec) = tokio::task::spawn_blocking(move || jobs.remove(&target))
        .await
        .map_err(internal)?
        .map_err(|e| job_error(e, &id))?;
    Ok(Json(match kind {
        Removed::Cancelled => rec.view(),
        Removed::Deleted => json!({"id": id, "deleted": true}),
    }))
}

pub(super) async fn result(
    State(state): State<AppState>,
    Path(id): Path<String>,
    req: Request,
) -> Result<Response, ApiError> {
    checked_id(&id)?;
    let mut raw_format = None;
    let mut granularities = Vec::new();
    for (k, v) in parse_query(req.uri().query().unwrap_or("")) {
        match k.as_str() {
            "response_format" => raw_format = Some(v),
            "timestamp_granularities[]" | "timestamp_granularities" => granularities.push(v),
            _ => {}
        }
    }
    let jobs = state.jobs.clone();
    let target = id.clone();
    let (stored, rec) = tokio::task::spawn_blocking(move || jobs.result(&target))
        .await
        .map_err(internal)?
        .map_err(|e| job_error(e, &id))?;
    let raw_format = raw_format.or_else(|| rec.options.response_format.clone());
    if granularities.is_empty() {
        granularities = rec.options.granularities.clone();
    }
    let fmt = parse_format(raw_format.as_deref())?;
    validate_granularities(&granularities, fmt)?;
    Ok(render(&stored, fmt, &granularities))
}

fn render(stored: &StoredResult, fmt: Format, granularities: &[String]) -> Response {
    let t = stored.transcription();
    match fmt {
        Format::Json => Json(json!({
            "text": t.text(),
            "usage": {"type": "duration", "seconds": stored.duration},
        }))
        .into_response(),
        Format::Text => plain(t.text()),
        Format::Srt => plain(format::render_srt(&t.segments)),
        Format::Vtt => plain(format::render_vtt(&t.segments)),
        Format::VerboseJson => Json(format::verbose_json(
            &t,
            stored.duration,
            stored.temperature,
            granularities,
        ))
        .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_parsing_decodes_and_keeps_repeats() {
        let q = parse_query(
            "status=queued&timestamp_granularities%5B%5D=word&t[]=a+b&x&t[]=%zz&y=%4&z=%41",
        );
        assert_eq!(q[0], ("status".into(), "queued".into()));
        assert_eq!(q[1], ("timestamp_granularities[]".into(), "word".into()));
        assert_eq!(q[2], ("t[]".into(), "a b".into()));
        assert_eq!(q[3], ("x".into(), String::new()));
        assert_eq!(q[4].1, "%zz");
        assert_eq!(q[5].1, "%4");
        assert_eq!(q[6].1, "A");
    }

    #[test]
    fn json_bodies_map_onto_the_multipart_form() {
        let (form, path) = form_from_json(
            br#"{"path":"/a.wav","model":"whisper-1","temperature":0.5,"x_vad":false,"timestamp_granularities":["word","segment"]}"#,
        )
        .unwrap();
        assert_eq!(path.as_deref(), Some("/a.wav"));
        assert_eq!(form.model.as_deref(), Some("whisper-1"));
        assert_eq!(form.temperature.as_deref(), Some("0.5"));
        assert_eq!(form.x_vad.as_deref(), Some("false"));
        assert_eq!(form.granularities, ["word", "segment"]);
        assert!(form_from_json(b"[1]").is_err());
        assert!(form_from_json(b"{").is_err());
        assert!(form_from_json(br#"{"path": 3}"#).is_err());
        assert!(form_from_json(br#"{"model": {"a": 1}}"#).is_err());
    }
}
