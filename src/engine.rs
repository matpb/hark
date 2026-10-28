use std::collections::BTreeMap;
#[cfg(any(feature = "vulkan", feature = "metal"))]
use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use whisper_rs::{
    DtwMode, DtwModelPreset, DtwParameters, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters, WhisperError, WhisperState, WhisperVadContext,
    WhisperVadContextParams, WhisperVadParams,
};

use crate::config::{Config, Device};
use crate::models::Registry;
use crate::pipeline::{
    self, ChunkReport, MAX_CHUNK, MIN_ADVANCE, MIN_CHUNK, Segment, Span, TokenInfo, Transcription,
};
use crate::scheduler::{
    Backend, ChunkFailure, ChunkOut, ChunkRequest, ChunkTask, DeviceStatus, Lane, LoadReport,
    Scheduler, SchedulerConfig, StatusReport,
};

pub const MIN_SAMPLES: usize = MIN_CHUNK;

#[derive(Debug, Clone, Default)]
pub struct DecodeOptions {
    pub language: Option<String>,
    pub prompt: Option<String>,
    /// Terms of the resolved vocabulary profile, in priority order.
    pub vocabulary: Vec<String>,
    pub temperature: f32,
    /// Request `x_vad`; `None` uses the configured default.
    pub vad: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct ModelRef {
    pub id: &'static str,
    pub path: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("could not load model: {0}")]
    Load(String),
    #[error("transcription failed: {0}")]
    Run(String),
    #[error("request cancelled")]
    Cancelled,
    #[error("too many interactive requests are waiting")]
    Busy { retry_after_secs: u64 },
    #[error("model {0} is being deleted")]
    ModelBusy(String),
}

/// One transcription request as the scheduler sees it.
#[derive(Debug, Clone)]
pub struct TranscribeRequest {
    pub id: String,
    pub lane: Lane,
    pub model: ModelRef,
    pub audio: Arc<Vec<f32>>,
    pub opts: DecodeOptions,
    /// Set when the client is gone; the driver stops submitting chunks.
    pub cancel: Arc<AtomicBool>,
}

/// One chunk of a job: runs on the batch lane and yields the segments with absolute times.
#[derive(Debug, Clone)]
pub struct JobChunk {
    pub job_id: String,
    pub index: usize,
    pub last: bool,
    pub model: ModelRef,
    pub audio: Arc<Vec<f32>>,
    pub span: Span,
    pub opts: Arc<DecodeOptions>,
    /// Language to decode with; `None` auto-detects.
    pub language: Option<String>,
    pub cancel: Arc<AtomicBool>,
}

/// Seam between the HTTP layer and whisper; tests plug in a fake.
pub trait Transcriber: Send + Sync + 'static {
    fn transcribe(
        &self,
        model: &ModelRef,
        audio: &[f32],
        opts: &DecodeOptions,
    ) -> Result<Transcription, EngineError>;

    /// Scheduled entry point; fakes that only implement `transcribe` ignore lane and id.
    fn run(&self, req: &TranscribeRequest) -> Result<Transcription, EngineError> {
        self.transcribe(&req.model, &req.audio, &req.opts)
    }

    /// Static chunk plan for a job; the default is fixed 120 s windows. `None` when `cancel` was set meanwhile.
    fn plan_chunks(
        &self,
        audio: &[f32],
        _opts: &DecodeOptions,
        cancel: &AtomicBool,
    ) -> Option<Vec<Span>> {
        (!cancel.load(Ordering::SeqCst)).then(|| pipeline::plan_fixed_chunks(audio.len()))
    }

    /// One scheduled job chunk; fakes that only implement `transcribe` run the slice directly.
    fn run_job_chunk(&self, chunk: &JobChunk) -> Result<ChunkOut, EngineError> {
        if chunk.cancel.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let mut opts = (*chunk.opts).clone();
        opts.language = chunk.language.clone().or(opts.language);
        let slice = &chunk.audio[chunk.span.start..chunk.span.end];
        let mut t = self.transcribe(&chunk.model, slice, &opts)?;
        let offset = chunk.span.start as f64 / f64::from(crate::audio::SAMPLE_RATE);
        for s in &mut t.segments {
            s.start += offset;
            s.end += offset;
            for w in &mut s.words {
                w.start += offset;
                w.end += offset;
            }
        }
        Ok(ChunkOut {
            segments: t.segments,
            detected: t.language,
        })
    }

    fn preload(&self, _model: &ModelRef) -> Result<(), EngineError> {
        Ok(())
    }

    /// Loads `model` now and waits for it; fakes have nothing to load.
    fn load_model(&self, _model: &ModelRef) -> Result<(), EngineError> {
        Ok(())
    }

    /// Unloads `id` once its states are idle; `Ok(false)` when it was not resident.
    fn unload_model(&self, _id: &str) -> Result<bool, EngineError> {
        Ok(false)
    }

    /// Marks `id` as being deleted: new and queued requests for it fail with `ModelBusy` until cleared.
    fn set_deleting(&self, _id: &str, _on: bool) {}

    fn status(&self) -> StatusReport {
        StatusReport::default()
    }
}

/// True when `code` is a language whisper knows (ISO-639-1 and the few three-letter codes it adds).
pub fn is_valid_language(code: &str) -> bool {
    !code.is_empty()
        && code.len() <= 3
        && code.bytes().all(|b| b.is_ascii_lowercase())
        && whisper_rs::get_lang_id(code).is_some()
}

/// DTW alignment-head preset for a catalog model id; quantized variants use their base model's.
pub fn dtw_preset(id: &str) -> Option<DtwModelPreset> {
    Some(match crate::models::base_model_id(id) {
        "tiny" => DtwModelPreset::Tiny,
        "tiny.en" => DtwModelPreset::TinyEn,
        "base" => DtwModelPreset::Base,
        "base.en" => DtwModelPreset::BaseEn,
        "small" => DtwModelPreset::Small,
        "small.en" => DtwModelPreset::SmallEn,
        "medium" => DtwModelPreset::Medium,
        "medium.en" => DtwModelPreset::MediumEn,
        "large-v3" => DtwModelPreset::LargeV3,
        "large-v3-turbo" => DtwModelPreset::LargeV3Turbo,
        _ => return None,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInfo {
    /// Index among GPU and iGPU devices, the numbering whisper.cpp's `gpu_device` uses.
    pub index: i32,
    pub name: String,
    pub discrete: bool,
}

#[cfg(any(feature = "vulkan", feature = "metal"))]
pub fn enumerate_gpus() -> Vec<GpuInfo> {
    use whisper_rs::whisper_rs_sys as sys;
    let mut out = Vec::new();
    // SAFETY: ggml device handles are static registry entries valid for the process; indices come from dev_count.
    unsafe {
        let n = sys::ggml_backend_dev_count();
        for i in 0..n {
            let dev = sys::ggml_backend_dev_get(i);
            let ty = sys::ggml_backend_dev_type(dev);
            let discrete = ty == sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU;
            if !discrete && ty != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU {
                continue;
            }
            let desc = sys::ggml_backend_dev_description(dev);
            let name = if desc.is_null() {
                String::from("unknown")
            } else {
                CStr::from_ptr(desc).to_string_lossy().into_owned()
            };
            let name = if cfg!(all(feature = "metal", not(feature = "vulkan"))) {
                format!("Metal ({name})")
            } else {
                name
            };
            out.push(GpuInfo {
                index: out.len() as i32,
                name,
                discrete,
            });
        }
    }
    out
}

#[cfg(not(any(feature = "vulkan", feature = "metal")))]
pub fn enumerate_gpus() -> Vec<GpuInfo> {
    Vec::new()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DevicePick {
    Cpu,
    Gpu { index: i32, name: String },
}

/// Discrete GPUs win over integrated ones; `auto` is CPU when no GPU is visible.
pub fn pick_device(want: Device, gpus: &[GpuInfo]) -> DevicePick {
    match want {
        Device::Cpu => DevicePick::Cpu,
        Device::Gpu(n) => {
            let name = gpus
                .iter()
                .find(|g| g.index == n as i32)
                .map(|g| g.name.clone())
                .unwrap_or_else(|| format!("gpu:{n}"));
            DevicePick::Gpu {
                index: n as i32,
                name,
            }
        }
        Device::Auto => gpus
            .iter()
            .find(|g| g.discrete)
            .or_else(|| gpus.first())
            .map(|g| DevicePick::Gpu {
                index: g.index,
                name: g.name.clone(),
            })
            .unwrap_or(DevicePick::Cpu),
    }
}

/// Free device memory in MiB as ggml reports it for the GPU numbered like `enumerate_gpus`.
#[cfg(any(feature = "vulkan", feature = "metal"))]
pub fn device_free_mib(index: i32) -> Option<u64> {
    use whisper_rs::whisper_rs_sys as sys;
    let mut n = 0;
    // SAFETY: ggml device handles are static registry entries valid for the process; indices come from dev_count.
    unsafe {
        for i in 0..sys::ggml_backend_dev_count() {
            let dev = sys::ggml_backend_dev_get(i);
            let ty = sys::ggml_backend_dev_type(dev);
            if ty != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU
                && ty != sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU
            {
                continue;
            }
            if n == index {
                let (mut free, mut total) = (0usize, 0usize);
                sys::ggml_backend_dev_memory(dev, &mut free, &mut total);
                return Some(free as u64 / (1024 * 1024));
            }
            n += 1;
        }
    }
    None
}

#[cfg(not(any(feature = "vulkan", feature = "metal")))]
pub fn device_free_mib(_index: i32) -> Option<u64> {
    None
}

/// ggml-vulkan publishes a device before filling its buffer type (ggml-vulkan.cpp:4470 vs 5195),
/// so a parallel load calls a null pointer; serialise device enumeration, model loads and state creation.
static GGML_INIT: Mutex<()> = Mutex::new(());

static VULKAN_RELEASED: AtomicU64 = AtomicU64::new(0);

/// Vulkan devices destroyed by releases since the process started.
pub fn vulkan_devices_released_total() -> u64 {
    VULKAN_RELEASED.load(Ordering::SeqCst)
}

/// Destroys the cached ggml Vulkan devices nothing uses any more; the caller holds `GGML_INIT`.
#[cfg(feature = "vulkan")]
fn release_vulkan_devices() {
    // SAFETY: no model is resident and GGML_INIT serialises every ggml device user.
    let n = unsafe { whisper_rs::whisper_rs_sys::ggml_backend_vk_release_devices() };
    VULKAN_RELEASED.fetch_add(n.max(0) as u64, Ordering::SeqCst);
    tracing::info!("vulkan devices released: {n}");
}

#[cfg(not(feature = "vulkan"))]
fn release_vulkan_devices() {}

struct Loaded {
    id: &'static str,
    uses_dtw: bool,
    eot: i32,
    /// Whether this context was built on a GPU; only those can raise a device fault.
    on_gpu: bool,
    vram_per_state_mib: Option<u64>,
    /// `None` once a state is retired after whisper `-7`.
    states: Vec<Mutex<Option<WhisperState>>>,
    ctx: WhisperContext,
}

/// Idle VAD contexts: a request takes one or builds its own, so none waits for another's run.
struct VadPool<T> {
    idle: Mutex<Vec<T>>,
    max_idle: usize,
}

impl<T> VadPool<T> {
    const MAX_IDLE: usize = 4;

    fn new() -> VadPool<T> {
        VadPool {
            idle: Mutex::new(Vec::new()),
            max_idle: Self::MAX_IDLE,
        }
    }

    fn take(&self) -> Option<T> {
        self.idle.lock().unwrap_or_else(|p| p.into_inner()).pop()
    }

    /// Keeps `item` for reuse, or hands it back when `max_idle` are already idle.
    fn give(&self, item: T) -> Option<T> {
        let mut idle = self.idle.lock().unwrap_or_else(|p| p.into_inner());
        if idle.len() < self.max_idle {
            idle.push(item);
            None
        } else {
            Some(item)
        }
    }
}

/// A model kept resident but inactive (`engine.max_loaded_models` above 1).
struct ParkedModel {
    loaded: Arc<Loaded>,
}

/// whisper.cpp inference: one context and several states per resident model, one of them active.
pub struct WhisperBackend {
    pick: DevicePick,
    /// Set once the GPU failed: every later load runs on the CPU.
    force_cpu: AtomicBool,
    /// The next load tries the original GPU again and falls back to the CPU on any failure.
    retrying: AtomicBool,
    fell_back_from: Mutex<Option<String>>,
    cpu_threads: i32,
    word_timestamps: BTreeMap<String, String>,
    loaded: RwLock<Option<Arc<Loaded>>>,
    parked: Mutex<Vec<ParkedModel>>,
}

/// The patched ggml backend turns a device exception (out of device memory, device lost) into -100.
pub fn is_device_fault_code(code: i32) -> bool {
    code == -100
}

/// Graph failures (-6 encode, -8 and -9 decode) can be one-off: retried once on the same device, never a fallback.
pub fn is_transient_code(code: i32) -> bool {
    matches!(code, -6 | -8 | -9)
}

fn is_device_fault(e: &WhisperError) -> bool {
    matches!(e, WhisperError::GenericError(c) if is_device_fault_code(*c))
}

fn is_transient(e: &WhisperError) -> bool {
    matches!(e, WhisperError::GenericError(c) if is_transient_code(*c))
}

/// True when a failed GPU load ran out of device memory, judged from the error and the backend log lines.
pub fn is_device_memory_failure(error: &str, log_lines: &[String]) -> bool {
    let says = |t: &str| {
        let t = t.to_ascii_lowercase().replace('_', " ");
        t.contains("outofdevicememory")
            || t.contains("out of device memory")
            || t.contains("failed to allocate")
    };
    says(error) || log_lines.iter().any(|l| says(l))
}

const BACKEND_LOG_KEEP: usize = 128;

static BACKEND_LOG: Mutex<(u64, std::collections::VecDeque<(u64, String)>)> =
    Mutex::new((0, std::collections::VecDeque::new()));

/// Position in the backend log; lines logged after it come back from `backend_log_since`.
pub fn backend_log_marker() -> u64 {
    BACKEND_LOG.lock().unwrap_or_else(|p| p.into_inner()).0
}

pub fn backend_log_since(marker: u64) -> Vec<String> {
    let log = BACKEND_LOG.lock().unwrap_or_else(|p| p.into_inner());
    log.1
        .iter()
        .filter(|(n, _)| *n >= marker)
        .map(|(_, l)| l.clone())
        .collect()
}

fn push_backend_log(line: String) {
    let mut log = BACKEND_LOG.lock().unwrap_or_else(|p| p.into_inner());
    let n = log.0;
    log.0 += 1;
    log.1.push_back((n, line));
    while log.1.len() > BACKEND_LOG_KEEP {
        log.1.pop_front();
    }
}

struct MessageVisitor(String);

impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = format!("{value:?}");
        }
    }
}

/// Tracing layer that keeps the last whisper.cpp and ggml log lines, so a failed load can be classified.
pub struct BackendLogLayer;

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for BackendLogLayer {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        let mut v = MessageVisitor(String::new());
        event.record(&mut v);
        push_backend_log(v.0);
    }
}

/// Per-layer filter for `BackendLogLayer`: whisper-rs's log hooks at info and above.
pub fn backend_log_filter(meta: &tracing::Metadata<'_>) -> bool {
    meta.target().starts_with("whisper_rs") && *meta.level() <= tracing::Level::INFO
}

impl WhisperBackend {
    fn new(
        pick: DevicePick,
        cpu_threads: i32,
        word_timestamps: BTreeMap<String, String>,
    ) -> WhisperBackend {
        WhisperBackend {
            pick,
            force_cpu: AtomicBool::new(false),
            retrying: AtomicBool::new(false),
            fell_back_from: Mutex::new(None),
            cpu_threads,
            word_timestamps,
            loaded: RwLock::new(None),
            parked: Mutex::new(Vec::new()),
        }
    }

    fn dtw_for(&self, id: &str) -> Option<DtwModelPreset> {
        let mode = self
            .word_timestamps
            .get(id)
            .or_else(|| self.word_timestamps.get(crate::models::base_model_id(id)));
        match mode.map(String::as_str) {
            Some("dtw") => dtw_preset(id),
            _ => None,
        }
    }

    fn current_pick(&self) -> DevicePick {
        if self.force_cpu.load(Ordering::SeqCst) {
            DevicePick::Cpu
        } else {
            self.pick.clone()
        }
    }

    fn free_mib(pick: &DevicePick) -> Option<u64> {
        match pick {
            DevicePick::Gpu { index, .. } => device_free_mib(*index),
            DevicePick::Cpu => None,
        }
    }

    /// Switches the process to the CPU for good and remembers which GPU it left.
    fn fall_back(&self) {
        if let DevicePick::Gpu { name, .. } = &self.pick
            && !self.force_cpu.swap(true, Ordering::SeqCst)
        {
            *self
                .fell_back_from
                .lock()
                .unwrap_or_else(|p| p.into_inner()) = Some(name.clone());
        }
    }

    /// Builds one context with its states on `pick`; the caller holds `GGML_INIT`.
    fn build(
        &self,
        model: &ModelRef,
        states: usize,
        pick: &DevicePick,
    ) -> Result<Loaded, EngineError> {
        let mut params = WhisperContextParameters::new();
        // whisper.cpp:3741 silently disables DTW under flash attention, so they are exclusive.
        let preset = self.dtw_for(model.id);
        let uses_dtw = preset.is_some();
        match preset {
            Some(model_preset) => {
                params.flash_attn(false).dtw_parameters(DtwParameters {
                    mode: DtwMode::ModelPreset { model_preset },
                    ..DtwParameters::default()
                });
            }
            None => {
                params.flash_attn(true);
            }
        }
        match pick {
            DevicePick::Cpu => {
                params.use_gpu(false);
            }
            DevicePick::Gpu { index, .. } => {
                params.use_gpu(true).gpu_device(*index);
            }
        }
        let path = model.path.to_str().ok_or_else(|| {
            EngineError::Load(format!("model path {} is not UTF-8", model.path.display()))
        })?;
        let ctx = WhisperContext::new_with_params(path, params)
            .map_err(|e| EngineError::Load(format!("{}: {e}", model.id)))?;
        let mut made = Vec::new();
        let mut deltas = Vec::new();
        for i in 0..states.max(1) {
            let before = Self::free_mib(pick);
            match ctx.create_state() {
                Ok(s) => {
                    if let (Some(b), Some(a)) = (before, Self::free_mib(pick)) {
                        deltas.push(b.saturating_sub(a));
                    }
                    made.push(Mutex::new(Some(s)));
                }
                Err(e) if i == 0 => {
                    return Err(EngineError::Load(format!(
                        "{}: create state: {e}",
                        model.id
                    )));
                }
                Err(e) => {
                    tracing::warn!(model = model.id, error = %e, state = i, "extra state not created");
                    break;
                }
            }
        }
        let vram_per_state_mib = (!deltas.is_empty()).then(|| {
            let sum: u64 = deltas.iter().sum();
            sum / deltas.len() as u64
        });
        let eot = ctx.token_eot();
        Ok(Loaded {
            id: model.id,
            uses_dtw,
            eot,
            on_gpu: matches!(pick, DevicePick::Gpu { .. }),
            vram_per_state_mib,
            states: made,
            ctx,
        })
    }

    fn drop_loaded(&self, old: Option<Arc<Loaded>>) {
        let _guard = GGML_INIT.lock().unwrap_or_else(|p| p.into_inner());
        drop(old);
        self.release_if_empty();
    }

    /// Releases the Vulkan devices when no model, active or parked, remains; the caller holds `GGML_INIT`.
    fn release_if_empty(&self) {
        let none_active = self
            .loaded
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .is_none();
        let none_parked = self
            .parked
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_empty();
        if none_active && none_parked {
            release_vulkan_devices();
        }
    }
}

impl Backend for WhisperBackend {
    fn load(&self, model: &ModelRef, states: usize) -> Result<LoadReport, EngineError> {
        let started = Instant::now();
        let reused = {
            let mut parked = self.parked.lock().unwrap_or_else(|p| p.into_inner());
            parked
                .iter()
                .position(|p| p.loaded.id == model.id)
                .map(|i| parked.remove(i).loaded)
        };
        if let Some(loaded) = reused {
            let report = LoadReport {
                load_ms: started.elapsed().as_millis() as u64,
                states: loaded
                    .states
                    .iter()
                    .filter(|s| s.lock().is_ok_and(|g| g.is_some()))
                    .count()
                    .max(1),
                vram_per_state_mib: loaded.vram_per_state_mib,
                fell_back: None,
            };
            *self.loaded.write().unwrap_or_else(|p| p.into_inner()) = Some(loaded);
            tracing::info!(model = model.id, ms = report.load_ms, "model reactivated");
            return Ok(report);
        }
        let _guard = GGML_INIT.lock().unwrap_or_else(|p| p.into_inner());
        let retrying = self.retrying.swap(false, Ordering::SeqCst);
        let pick = if retrying {
            self.pick.clone()
        } else {
            self.current_pick()
        };
        let marker = backend_log_marker();
        let mut fell_back = None;
        let built = match self.build(model, states, &pick) {
            Ok(b) => {
                if retrying && b.on_gpu {
                    self.force_cpu.store(false, Ordering::SeqCst);
                    *self
                        .fell_back_from
                        .lock()
                        .unwrap_or_else(|p| p.into_inner()) = None;
                }
                b
            }
            Err(gpu_err)
                if matches!(pick, DevicePick::Gpu { .. })
                    && (retrying
                        || is_device_memory_failure(
                            &gpu_err.to_string(),
                            &backend_log_since(marker),
                        )) =>
            {
                tracing::warn!(model = model.id, error = %gpu_err, "GPU load failed; retrying on the CPU");
                let gpu_msg = gpu_err.to_string();
                let b = self
                    .build(model, states, &DevicePick::Cpu)
                    .map_err(|_| gpu_err)?;
                self.fall_back();
                self.parked
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .clear();
                fell_back = Some(format!(
                    "GPU load of {} failed ({}); fell back to the CPU",
                    model.id,
                    gpu_msg.trim_start_matches("could not load model: ")
                ));
                b
            }
            Err(e) => return Err(e),
        };
        let report = LoadReport {
            load_ms: started.elapsed().as_millis() as u64,
            states: built.states.len(),
            vram_per_state_mib: built.vram_per_state_mib,
            fell_back,
        };
        tracing::info!(
            model = model.id,
            dtw = built.uses_dtw,
            gpu = built.on_gpu,
            states = report.states,
            vram_per_state_mib = ?report.vram_per_state_mib,
            ms = report.load_ms,
            "model loaded"
        );
        *self.loaded.write().unwrap_or_else(|p| p.into_inner()) = Some(Arc::new(built));
        Ok(report)
    }

    fn try_gpu_again(&self) {
        self.retrying.store(true, Ordering::SeqCst);
    }

    fn devices_released(&self) -> Option<u64> {
        matches!(self.current_pick(), DevicePick::Gpu { .. }).then(vulkan_devices_released_total)
    }

    fn unload(&self) {
        let old = self
            .loaded
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        let parked: Vec<ParkedModel> =
            std::mem::take(&mut *self.parked.lock().unwrap_or_else(|p| p.into_inner()));
        let _guard = GGML_INIT.lock().unwrap_or_else(|p| p.into_inner());
        drop(old);
        drop(parked);
        self.release_if_empty();
    }

    fn park(&self) {
        let old = self
            .loaded
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        if let Some(loaded) = old {
            self.parked
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(ParkedModel { loaded });
        }
    }

    fn evict(&self, id: &str) {
        let gone = {
            let mut parked = self.parked.lock().unwrap_or_else(|p| p.into_inner());
            parked
                .iter()
                .position(|p| p.loaded.id == id)
                .map(|i| parked.remove(i))
        };
        let _guard = GGML_INIT.lock().unwrap_or_else(|p| p.into_inner());
        drop(gone);
        self.release_if_empty();
    }

    fn drop_active(&self) {
        let old = self
            .loaded
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .take();
        self.drop_loaded(old);
    }

    fn device(&self) -> Option<DeviceStatus> {
        let fell_back_from = self
            .fell_back_from
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        Some(match self.current_pick() {
            DevicePick::Cpu => DeviceStatus {
                fell_back_from,
                ..DeviceStatus::default()
            },
            DevicePick::Gpu { index, name } => DeviceStatus {
                name,
                index: Some(index),
                fell_back_from,
            },
        })
    }

    fn run(
        &self,
        state: usize,
        task: &ChunkTask,
        abort: &AtomicBool,
    ) -> Result<ChunkOut, ChunkFailure> {
        let loaded = self
            .loaded
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .ok_or_else(|| ChunkFailure::Failed("no model loaded".into()))?;
        let mut guard = loaded.states[state]
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let Some(st) = guard.as_mut() else {
            return Err(ChunkFailure::Failed("state was retired".into()));
        };
        let audio = &task.audio[task.span.start..task.span.end];
        match decode_chunk(
            &loaded,
            st,
            audio,
            task.span.start,
            &task.opts,
            task.language.as_deref(),
            self.cpu_threads,
            abort,
        ) {
            Ok(out) => Ok(out),
            Err(DecodeFailure::Aborted) => Err(ChunkFailure::Aborted),
            Err(DecodeFailure::Whisper(WhisperError::GenericError(-7))) => {
                // whisper.cpp already freed the KV cache; dropping the state would free it twice.
                if let Some(s) = guard.take() {
                    std::mem::forget(s);
                }
                Err(ChunkFailure::StateLost(
                    "KV cache allocation failed (-7)".into(),
                ))
            }
            Err(DecodeFailure::Whisper(e)) if loaded.on_gpu && is_device_fault(&e) => {
                self.fall_back();
                Err(ChunkFailure::DeviceFault(format!(
                    "{e} while decoding on the GPU"
                )))
            }
            Err(DecodeFailure::Whisper(e)) if loaded.on_gpu && is_transient(&e) => {
                Err(ChunkFailure::Transient(e.to_string()))
            }
            Err(DecodeFailure::Whisper(e)) => Err(ChunkFailure::Failed(e.to_string())),
        }
    }
}

pub struct WhisperEngine<B: Backend = WhisperBackend> {
    pick: DevicePick,
    sched: Scheduler<B>,
    vad_path: Option<PathBuf>,
    vad_default: bool,
    vad: VadPool<Box<WhisperVadContext>>,
    vad_failed: AtomicBool,
}

struct VadConfig {
    path: Option<PathBuf>,
    default: bool,
}

impl WhisperEngine {
    pub fn new(device: Device, cpu_threads: i32, sched_cfg: SchedulerConfig) -> WhisperEngine {
        Self::build(
            device,
            cpu_threads,
            sched_cfg,
            BTreeMap::new(),
            VadConfig {
                path: None,
                default: false,
            },
        )
    }

    fn build(
        device: Device,
        cpu_threads: i32,
        sched_cfg: SchedulerConfig,
        word_timestamps: BTreeMap<String, String>,
        vad: VadConfig,
    ) -> WhisperEngine {
        whisper_rs::install_logging_hooks();
        let gpus = {
            let _guard = GGML_INIT.lock().unwrap_or_else(|p| p.into_inner());
            enumerate_gpus()
        };
        for g in &gpus {
            tracing::debug!(index = g.index, name = %g.name, discrete = g.discrete, "gpu device");
        }
        let pick = pick_device(device, &gpus);
        match &pick {
            DevicePick::Cpu => tracing::info!(device = "cpu", "engine device"),
            DevicePick::Gpu { index, name } => {
                tracing::info!(device = %format!("gpu:{index}"), name = %name, "engine device")
            }
        }
        let backend = WhisperBackend::new(pick.clone(), cpu_threads, word_timestamps);
        WhisperEngine {
            pick,
            sched: Scheduler::new(backend, sched_cfg),
            vad_path: vad.path,
            vad_default: vad.default,
            vad: VadPool::new(),
            vad_failed: AtomicBool::new(false),
        }
    }

    /// Engine wired from the config: device, threads, states, word-timestamp modes and the VAD model.
    pub fn from_config(cfg: &Config, registry: &Registry) -> WhisperEngine {
        let vad_path = registry.locate_file(&cfg.vad.model);
        if cfg.vad.enabled && vad_path.is_none() {
            tracing::warn!(model = %cfg.vad.model, "VAD model not found; VAD is off");
        }
        let engine = Self::build(
            cfg.engine.device,
            cfg.engine.cpu_threads,
            SchedulerConfig {
                states: cfg.engine.states,
                max_interactive_waiting: cfg.server.max_interactive_waiting,
            },
            cfg.models.word_timestamps.clone(),
            VadConfig {
                path: vad_path,
                default: cfg.vad.enabled,
            },
        );
        engine.sched.configure_residency(
            cfg.engine.max_loaded_models,
            (cfg.engine.idle_unload_secs > 0)
                .then(|| std::time::Duration::from_secs(cfg.engine.idle_unload_secs)),
        );
        if matches!(engine.pick(), DevicePick::Gpu { .. }) {
            engine
                .sched
                .configure_recycle(cfg.engine.recycle_after_chunks);
        }
        engine.sched.configure_gpu_retry(
            (cfg.engine.gpu_retry_secs > 0)
                .then(|| std::time::Duration::from_secs(cfg.engine.gpu_retry_secs)),
        );
        engine
    }
}

impl<B: Backend> WhisperEngine<B> {
    /// Engine over any backend, CPU device and no VAD; the seam tests use.
    pub fn with_backend(backend: B, sched_cfg: SchedulerConfig) -> WhisperEngine<B> {
        WhisperEngine {
            pick: DevicePick::Cpu,
            sched: Scheduler::new(backend, sched_cfg),
            vad_path: None,
            vad_default: false,
            vad: VadPool::new(),
            vad_failed: AtomicBool::new(false),
        }
    }

    pub fn pick(&self) -> &DevicePick {
        &self.pick
    }

    fn device_status(&self) -> DeviceStatus {
        if let Some(d) = self.sched.backend().device() {
            return d;
        }
        match &self.pick {
            DevicePick::Cpu => DeviceStatus::default(),
            DevicePick::Gpu { index, name } => DeviceStatus {
                name: name.clone(),
                index: Some(*index),
                fell_back_from: None,
            },
        }
    }

    /// Speech regions in centiseconds, or `None` when VAD is unavailable or failed.
    fn detect_speech(&self, audio: &[f32]) -> Option<Vec<(f32, f32)>> {
        let path = self.vad_path.as_ref()?;
        if self.vad_failed.load(Ordering::SeqCst) {
            return None;
        }
        let mut ctx = match self.vad.take() {
            Some(ctx) => ctx,
            None => {
                let _guard = GGML_INIT.lock().unwrap_or_else(|p| p.into_inner());
                let mut params = WhisperVadContextParams::new();
                params.set_use_gpu(false);
                // Silero's graph is tiny; extra threads spin on ggml barriers and stall concurrent requests.
                params.set_n_threads(1);
                match path
                    .to_str()
                    .filter(|p| !p.contains('\0'))
                    .ok_or(WhisperError::NullPointer)
                    .and_then(|p| WhisperVadContext::new(p, params))
                {
                    Ok(ctx) => Box::new(ctx),
                    Err(e) => {
                        tracing::warn!(error = %e, "VAD model failed to load; VAD is off");
                        self.vad_failed.store(true, Ordering::SeqCst);
                        return None;
                    }
                }
            }
        };
        let level = crate::audio::level_for_vad(audio);
        let regions = match ctx.segments_from_samples(WhisperVadParams::new(), &level) {
            Ok(segs) => Some(
                (0..segs.num_segments())
                    .filter_map(|i| segs.get_segment(i))
                    .map(|s| (s.start, s.end))
                    .collect(),
            ),
            Err(e) => {
                tracing::warn!(error = %e, "VAD run failed; decoding without VAD");
                None
            }
        };
        if regions.is_some() {
            drop(self.vad.give(ctx));
        }
        regions
    }
}

/// Request language wins; otherwise detect per chunk until one yields text, then pin it.
struct LanguagePin {
    request: Option<String>,
    pinned: Option<String>,
}

impl LanguagePin {
    fn for_chunk(&self) -> Option<String> {
        self.request.clone().or_else(|| self.pinned.clone())
    }

    fn record(&mut self, out: &ChunkOut, used: &Option<String>) -> ChunkReport {
        let has_text = out.segments.iter().any(|s| !s.text.trim().is_empty());
        if self.request.is_none() && self.pinned.is_none() && has_text {
            self.pinned = out.detected.clone();
        }
        ChunkReport {
            offset_secs: 0.0,
            duration_secs: 0.0,
            language: used.clone().or_else(|| out.detected.clone()),
            pinned: self.request.is_none() && used.is_some(),
        }
    }
}

impl<B: Backend> Transcriber for WhisperEngine<B> {
    fn preload(&self, model: &ModelRef) -> Result<(), EngineError> {
        self.sched.preload(model.clone())
    }

    fn load_model(&self, model: &ModelRef) -> Result<(), EngineError> {
        self.sched.load_model(model.clone())
    }

    fn unload_model(&self, id: &str) -> Result<bool, EngineError> {
        self.sched.unload_model(id)
    }

    fn set_deleting(&self, id: &str, on: bool) {
        self.sched.set_deleting(id, on);
    }

    fn status(&self) -> StatusReport {
        self.sched.status(self.device_status())
    }

    fn transcribe(
        &self,
        model: &ModelRef,
        audio: &[f32],
        opts: &DecodeOptions,
    ) -> Result<Transcription, EngineError> {
        self.run(&TranscribeRequest {
            id: "direct".into(),
            lane: Lane::Interactive,
            model: model.clone(),
            audio: Arc::new(audio.to_vec()),
            opts: opts.clone(),
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    fn plan_chunks(
        &self,
        audio: &[f32],
        opts: &DecodeOptions,
        cancel: &AtomicBool,
    ) -> Option<Vec<Span>> {
        if cancel.load(Ordering::SeqCst) {
            return None;
        }
        if audio.len() < MIN_SAMPLES {
            return Some(Vec::new());
        }
        let want_vad = opts.vad.unwrap_or(self.vad_default);
        let regions = want_vad.then(|| self.detect_speech(audio)).flatten();
        if cancel.load(Ordering::SeqCst) {
            return None;
        }
        Some(match regions {
            Some(regions) => pipeline::plan_vad_chunks(&regions, audio.len()),
            None => pipeline::plan_fixed_chunks(audio.len()),
        })
    }

    fn run_job_chunk(&self, chunk: &JobChunk) -> Result<ChunkOut, EngineError> {
        let done = self.sched.run_chunk(ChunkRequest {
            request_id: chunk.job_id.clone(),
            lane: Lane::Batch,
            model: chunk.model.clone(),
            chunk: chunk.index,
            task: ChunkTask {
                audio: chunk.audio.clone(),
                span: chunk.span,
                opts: chunk.opts.clone(),
                language: chunk.language.clone(),
            },
            admit: false,
            last: chunk.last,
            cancel: chunk.cancel.clone(),
        })?;
        self.sched.record_request(
            chunk.model.id,
            done.run_secs,
            chunk.span.len() as f64 / f64::from(crate::audio::SAMPLE_RATE),
        );
        Ok(done.out)
    }

    fn run(&self, req: &TranscribeRequest) -> Result<Transcription, EngineError> {
        let audio: &[f32] = &req.audio;
        let opts = &req.opts;
        let mut result = Transcription {
            language: opts.language.clone(),
            ..Default::default()
        };
        if audio.len() < MIN_SAMPLES {
            return Ok(result);
        }
        let want_vad = opts.vad.unwrap_or(self.vad_default);
        let regions = if want_vad {
            self.detect_speech(audio)
        } else {
            None
        };
        if req.cancel.load(Ordering::SeqCst) {
            return Err(EngineError::Cancelled);
        }
        let mut lang = LanguagePin {
            request: opts.language.clone(),
            pinned: None,
        };
        let shared_opts = Arc::new(opts.clone());
        let mut segments: Vec<Segment> = Vec::new();
        let mut chunk_no = 0usize;
        let mut run_secs = 0.0f64;
        let mut queue_wait = 0.0f64;
        let mut run = |span: Span,
                       last: bool,
                       segments: &mut Vec<Segment>|
         -> Result<ChunkOut, EngineError> {
            if req.cancel.load(Ordering::SeqCst) {
                return Err(EngineError::Cancelled);
            }
            let used = lang.for_chunk();
            let done = self.sched.run_chunk(ChunkRequest {
                request_id: req.id.clone(),
                lane: req.lane,
                model: req.model.clone(),
                chunk: chunk_no,
                task: ChunkTask {
                    audio: req.audio.clone(),
                    span,
                    opts: shared_opts.clone(),
                    language: used.clone(),
                },
                admit: chunk_no == 0,
                last,
                cancel: req.cancel.clone(),
            })?;
            chunk_no += 1;
            run_secs += done.run_secs;
            queue_wait += done.queue_wait_secs;
            let out = done.out;
            let mut report = lang.record(&out, &used);
            report.offset_secs = span.start as f64 / f64::from(crate::audio::SAMPLE_RATE);
            report.duration_secs = span.len() as f64 / f64::from(crate::audio::SAMPLE_RATE);
            result.chunks.push(report);
            segments.extend(out.segments.iter().cloned());
            Ok(out)
        };
        match regions {
            Some(regions) => {
                let plan = pipeline::plan_vad_chunks(&regions, audio.len());
                let n = plan.len();
                for (i, span) in plan.into_iter().enumerate() {
                    run(span, i + 1 == n, &mut segments)?;
                }
            }
            None => {
                let mut cursor = 0usize;
                while audio.len() - cursor >= MIN_CHUNK {
                    let is_last = pipeline::window_is_last(audio.len(), cursor);
                    let end = if is_last {
                        audio.len()
                    } else {
                        cursor + MAX_CHUNK
                    };
                    let span = Span { start: cursor, end };
                    let mut chunk_segments = Vec::new();
                    run(span, is_last, &mut chunk_segments)?;
                    let advance = pipeline::plan_advance(
                        &mut chunk_segments,
                        cursor as f64 / f64::from(crate::audio::SAMPLE_RATE),
                        span.len(),
                        is_last,
                        MIN_ADVANCE,
                    );
                    segments.extend(chunk_segments);
                    if is_last {
                        break;
                    }
                    cursor += advance;
                }
            }
        }
        let (segments, repetitions) = pipeline::finalize_segments(segments);
        for r in &repetitions {
            tracing::warn!(
                phrase_words = r.phrase_words,
                count = r.count,
                start = format_args!("{:.1}", r.start),
                end = format_args!("{:.1}", r.end),
                "repetition loop collapsed"
            );
        }
        result.language = lang
            .request
            .clone()
            .or(lang.pinned.clone())
            .or_else(|| result.chunks.iter().find_map(|c| c.language.clone()));
        result.segments = segments;
        result.repetitions = repetitions;
        result.queue_wait_secs = queue_wait;
        self.sched.record_request(
            req.model.id,
            run_secs,
            audio.len() as f64 / f64::from(crate::audio::SAMPLE_RATE),
        );
        Ok(result)
    }
}

enum DecodeFailure {
    Aborted,
    Whisper(WhisperError),
}

impl From<WhisperError> for DecodeFailure {
    fn from(e: WhisperError) -> Self {
        DecodeFailure::Whisper(e)
    }
}

/// Handed to whisper.cpp as the abort callback's user data; lives on the caller's stack for one `full`.
struct AbortCtx<'a> {
    flag: &'a AtomicBool,
    tripped: AtomicBool,
}

/// ggml aborts the graph step when this returns true (whisper.cpp:2463 and :2985).
unsafe extern "C" fn abort_trampoline(user_data: *mut std::ffi::c_void) -> bool {
    // SAFETY: user_data is the &AbortCtx we passed to whisper.cpp, alive for the whole `full` call.
    let ctx = unsafe { &*(user_data as *const AbortCtx) };
    let stop = ctx.flag.load(Ordering::SeqCst);
    if stop {
        ctx.tripped.store(true, Ordering::SeqCst);
    }
    stop
}

/// Start of a repair re-decode relative to the chunk start, 500 ms before a loop's first segment.
const REPAIR_LEAD_SECS: f64 = 0.5;
/// `entropy_thold` of the repair pass: widened so whisper's temperature fallback trips on a loop.
const REPAIR_ENTROPY_THOLD: f32 = 2.9;

/// Where a repetition or collapsed-timestamp run starts in `segments`, minus the lead.
/// The result is clamped to the chunk start.
pub fn repair_start(segments: &[Segment], chunk_start_secs: f64) -> Option<f64> {
    let (_, reports) = pipeline::guard_repetitions(segments.to_vec());
    let repeats = reports.iter().map(|r| r.start);
    let collapsed = pipeline::collapsed_runs(segments, pipeline::COLLAPSED_REPAIR_MIN_WORDS)
        .into_iter()
        .map(|(start, _)| start);
    let earliest = repeats.chain(collapsed).reduce(f64::min)?;
    Some((earliest - REPAIR_LEAD_SECS).max(chunk_start_secs))
}

/// Original segments before `from`, then `repaired`, then the originals that start after the repair
/// ends when it stops more than a second short of `chunk_end` (speech past the looped span).
pub fn splice_repair(
    original: Vec<Segment>,
    repaired: Vec<Segment>,
    from: f64,
    chunk_end: f64,
) -> Vec<Segment> {
    if repaired.is_empty() {
        return original;
    }
    let cut = repaired
        .iter()
        .map(|s| s.start)
        .fold(f64::INFINITY, f64::min);
    let repair_end = repaired.iter().map(|s| s.end).fold(f64::MIN, f64::max);
    let tail: Vec<Segment> = if repair_end < chunk_end - 1.0 {
        original
            .iter()
            .filter(|s| s.start >= repair_end)
            .cloned()
            .collect()
    } else {
        Vec::new()
    };
    let mut out: Vec<Segment> = original.into_iter().filter(|s| s.start < from).collect();
    for s in &mut out {
        if s.end > cut {
            s.end = cut.max(s.start);
            s.words.retain(|w| w.start < cut);
            for w in &mut s.words {
                w.end = w.end.min(cut);
            }
        }
    }
    out.extend(repaired);
    out.extend(tail);
    out
}

/// Segments of the last `full` call on `state`, on the absolute timeline.
fn collect_segments(
    loaded: &Loaded,
    state: &WhisperState,
    offset_samples: usize,
) -> Result<Vec<Segment>, DecodeFailure> {
    let offset_secs = offset_samples as f64 / f64::from(crate::audio::SAMPLE_RATE);
    let mut segments = Vec::new();
    for i in 0..state.full_n_segments() {
        let Some(seg) = state.get_segment(i) else {
            continue;
        };
        let text = seg.to_str_lossy()?.into_owned();
        let mut tokens = Vec::new();
        for j in 0..seg.n_tokens() {
            let Some(tok) = seg.get_token(j) else {
                continue;
            };
            let d = tok.token_data();
            tokens.push(TokenInfo {
                id: d.id,
                bytes: tok.to_bytes().map(<[u8]>::to_vec).unwrap_or_default(),
                t0: d.t0,
                t1: d.t1,
                t_dtw: d.t_dtw,
                plog: d.plog,
            });
        }
        segments.push(Segment {
            id: 0,
            seek_cs: (offset_samples / 160) as i64,
            start: offset_secs + seg.start_timestamp() as f64 * 0.01,
            end: offset_secs + seg.end_timestamp() as f64 * 0.01,
            text,
            tokens: tokens.iter().map(|t| t.id).collect(),
            avg_logprob: pipeline::mean_logprob(&tokens, loaded.eot),
            no_speech_prob: f64::from(seg.no_speech_probability()),
            words: pipeline::words_from_tokens(
                &tokens,
                loaded.eot,
                loaded.uses_dtw,
                offset_secs,
                offset_secs + seg.end_timestamp() as f64 * 0.01,
            ),
        });
    }
    Ok(segments)
}

/// Decode parameters shared by the main pass and the repair pass.
fn make_params<'a>(
    opts: &DecodeOptions,
    prompt: Option<&'a str>,
    threads: i32,
    abort_ctx: &'a AbortCtx<'_>,
    language: Option<&'a str>,
) -> FullParams<'a, 'a> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_token_timestamps(true);
    params.set_suppress_blank(true);
    params.set_no_context(true);
    params.set_temperature(opts.temperature);
    params.set_temperature_inc(0.2);
    params.set_entropy_thold(2.4);
    params.set_logprob_thold(-1.0);
    params.set_no_speech_thold(0.6);
    params.set_n_threads(threads);
    params.set_language(language);
    if let Some(p) = prompt {
        params.set_initial_prompt(p);
    }
    // SAFETY: abort_ctx outlives every `full` of the caller, and the callback only reads atomics.
    unsafe {
        params.set_abort_callback(Some(abort_trampoline));
        params.set_abort_callback_user_data(abort_ctx as *const AbortCtx as *mut std::ffi::c_void);
    }
    params
}

#[allow(clippy::too_many_arguments)]
fn decode_chunk(
    loaded: &Loaded,
    state: &mut WhisperState,
    audio: &[f32],
    offset_samples: usize,
    opts: &DecodeOptions,
    language: Option<&str>,
    threads: i32,
    abort: &AtomicBool,
) -> Result<ChunkOut, DecodeFailure> {
    let limit = (loaded.ctx.n_text_ctx().max(2) / 2) as usize;
    let count = |t: &str| {
        loaded
            .ctx
            .tokenize(t, t.len() + 8)
            .map_or(t.len(), |v| v.len())
    };
    let prompt =
        pipeline::build_initial_prompt(&opts.vocabulary, opts.prompt.as_deref(), limit, &count)
            .filter(|p| !p.contains('\0'));
    let abort_ctx = AbortCtx {
        flag: abort,
        tripped: AtomicBool::new(false),
    };
    let ran = state.full(
        make_params(opts, prompt.as_deref(), threads, &abort_ctx, language),
        audio,
    );
    // A tripped callback means whisper stopped early, so even an Ok result may be truncated.
    if abort_ctx.tripped.load(Ordering::SeqCst) {
        return Err(DecodeFailure::Aborted);
    }
    ran?;

    let detected = whisper_rs::get_lang_str(state.full_lang_id_from_state()).map(str::to_string);
    let mut segments = collect_segments(loaded, state, offset_samples)?;

    let chunk_start = offset_samples as f64 / f64::from(crate::audio::SAMPLE_RATE);
    let chunk_end = chunk_start + audio.len() as f64 / f64::from(crate::audio::SAMPLE_RATE);
    if let Some(from) = repair_start(&segments, chunk_start) {
        let rel = (((from - chunk_start) * f64::from(crate::audio::SAMPLE_RATE)) as usize)
            .min(audio.len());
        if audio.len() - rel >= pipeline::MIN_CHUNK {
            let pinned = language.map(str::to_string).or_else(|| detected.clone());
            let mut repair = make_params(
                opts,
                prompt.as_deref(),
                threads,
                &abort_ctx,
                pinned.as_deref(),
            );
            repair.set_n_max_text_ctx(0);
            repair.set_entropy_thold(REPAIR_ENTROPY_THOLD);
            let ran = state.full(repair, &audio[rel..]);
            if abort_ctx.tripped.load(Ordering::SeqCst) {
                return Err(DecodeFailure::Aborted);
            }
            match ran {
                Ok(()) => {
                    let repaired = collect_segments(loaded, state, offset_samples + rel)?;
                    let original_from: Vec<Segment> =
                        segments.iter().filter(|s| s.end > from).cloned().collect();
                    let before = pipeline::damage(&original_from);
                    let after = pipeline::damage(&repaired);
                    if after < before {
                        tracing::info!(
                            from = format_args!("{from:.1}"),
                            damage_before = before,
                            damage_after = after,
                            "re-decoded a looped span"
                        );
                        segments = splice_repair(segments, repaired, from, chunk_end);
                    } else {
                        tracing::warn!(
                            from = format_args!("{from:.1}"),
                            "repair pass looped too; keeping the original"
                        );
                    }
                }
                Err(e @ WhisperError::GenericError(-7)) => return Err(e.into()),
                Err(e) if is_device_fault(&e) || is_transient(&e) => return Err(e.into()),
                Err(e) => tracing::warn!(error = %e, "repair decode failed; keeping the original"),
            }
        }
    }
    let (segments, dropped) = pipeline::drop_collapsed_tail(segments, chunk_end);
    if dropped > 0 {
        tracing::info!("dropped collapsed tail words={dropped}");
    }
    Ok(ChunkOut { segments, detected })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(index: i32, name: &str, discrete: bool) -> GpuInfo {
        GpuInfo {
            index,
            name: name.into(),
            discrete,
        }
    }

    #[test]
    fn auto_prefers_discrete_over_integrated() {
        let gpus = [gpu(0, "Intel iGPU", false), gpu(1, "Discrete GPU", true)];
        assert_eq!(
            pick_device(Device::Auto, &gpus),
            DevicePick::Gpu {
                index: 1,
                name: "Discrete GPU".into()
            }
        );
    }

    #[test]
    fn auto_uses_igpu_only_when_alone_and_cpu_when_none() {
        let only = [gpu(0, "Intel iGPU", false)];
        assert!(matches!(
            pick_device(Device::Auto, &only),
            DevicePick::Gpu { index: 0, .. }
        ));
        assert_eq!(pick_device(Device::Auto, &[]), DevicePick::Cpu);
    }

    #[test]
    fn explicit_devices() {
        assert_eq!(
            pick_device(Device::Cpu, &[gpu(0, "x", true)]),
            DevicePick::Cpu
        );
        assert_eq!(
            pick_device(Device::Gpu(1), &[gpu(0, "a", true), gpu(1, "b", false)]),
            DevicePick::Gpu {
                index: 1,
                name: "b".into()
            }
        );
    }

    #[test]
    fn vad_pool_hands_out_idle_items_and_caps_what_it_keeps() {
        let pool: VadPool<u32> = VadPool::new();
        assert_eq!(pool.take(), None, "empty pool: the caller builds its own");
        for i in 0..4 {
            assert_eq!(pool.give(i), None);
        }
        assert_eq!(pool.give(9), Some(9), "a fifth idle item is dropped");
        let held = pool.take().unwrap();
        assert!(
            pool.take().is_some(),
            "a second taker never waits for the first"
        );
        assert_eq!(pool.give(held), None);
    }

    #[test]
    fn language_validation() {
        for ok in ["en", "fr", "de", "ja", "yue"] {
            assert!(is_valid_language(ok), "{ok}");
        }
        for bad in ["", "english", "EN", "xx", "e", "en\0", "../"] {
            assert!(!is_valid_language(bad), "{bad:?}");
        }
    }

    fn seg(start: f64, end: f64, text: &str) -> Segment {
        Segment {
            id: 0,
            seek_cs: 0,
            start,
            end,
            text: text.into(),
            tokens: Vec::new(),
            avg_logprob: 0.0,
            no_speech_prob: 0.0,
            words: Vec::new(),
        }
    }

    #[test]
    fn only_the_backend_exception_is_a_device_fault_and_graph_failures_are_transient() {
        assert!(is_device_fault_code(-100));
        assert!(is_device_fault(&WhisperError::GenericError(-100)));
        for code in [-6, -8, -9] {
            assert!(!is_device_fault_code(code), "{code}");
            assert!(
                !is_device_fault(&WhisperError::GenericError(code)),
                "{code}"
            );
            assert!(is_transient_code(code), "{code}");
            assert!(is_transient(&WhisperError::GenericError(code)), "{code}");
        }
        for code in [-1, -2, -3, -4, -5, -7, 0, 7, 8, -101] {
            assert!(!is_device_fault_code(code), "{code}");
            assert!(!is_transient_code(code), "{code}");
        }
        assert!(!is_device_fault(&WhisperError::NoSamples));
        assert!(!is_device_fault(&WhisperError::FailedToDecode));
        assert!(!is_transient(&WhisperError::FailedToDecode));
    }

    #[test]
    fn only_device_memory_load_failures_justify_a_cpu_fallback() {
        let lines = |l: &[&str]| l.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(is_device_memory_failure(
            "could not load model: x",
            &lines(&["ggml_vulkan: failed to allocate the Vulkan0 model buffer"])
        ));
        assert!(is_device_memory_failure(
            "VK_ERROR_OUT_OF_DEVICE_MEMORY",
            &[]
        ));
        assert!(is_device_memory_failure(
            "ErrorOutOfDeviceMemory",
            &lines(&["unrelated"])
        ));
        assert!(!is_device_memory_failure(
            "could not load model: bad magic",
            &lines(&["whisper_init: failed to load model", "invalid model data"])
        ));
        assert!(!is_device_memory_failure("no such file", &[]));
    }

    #[test]
    fn backend_log_lines_are_kept_after_a_marker() {
        let m = backend_log_marker();
        push_backend_log("first".into());
        push_backend_log("second".into());
        assert_eq!(backend_log_since(m), ["first", "second"]);
        let m2 = backend_log_marker();
        push_backend_log("third".into());
        assert_eq!(backend_log_since(m2), ["third"]);
    }

    #[test]
    fn dtw_presets_follow_the_base_model() {
        let name = |id: &str| dtw_preset(id).map(|p| format!("{p:?}"));
        assert_eq!(name("large-v3-turbo").as_deref(), Some("LargeV3Turbo"));
        assert_eq!(name("large-v3-turbo-q5_0"), name("large-v3-turbo"));
        assert_eq!(name("large-v3-q5_0").as_deref(), Some("LargeV3"));
        assert_eq!(name("tiny.en").as_deref(), Some("TinyEn"));
        assert_eq!(name("base.en-q8_0").as_deref(), Some("BaseEn"));
        assert_eq!(name("small.en-q5_1").as_deref(), Some("SmallEn"));
        assert_eq!(name("medium.en-q5_0").as_deref(), Some("MediumEn"));
        assert_eq!(name("small-q8_0").as_deref(), Some("Small"));
        assert!(name("silero-v5.1.2").is_none());
        assert!(name("nope").is_none());
        for e in crate::models::CATALOG
            .iter()
            .filter(|e| e.kind == crate::models::ModelKind::Transcription)
        {
            assert!(dtw_preset(e.id).is_some(), "{} has no DTW preset", e.id);
        }
    }

    #[test]
    fn repair_starts_half_a_second_before_the_loop() {
        let mut segs = vec![seg(0.0, 3.0, " The quick brown fox.")];
        let looped = " one two three four five six seven eight nine";
        for i in 0..14 {
            let t = 300.0 + i as f64 * 2.0;
            segs.push(seg(t, t + 2.0, looped));
        }
        let from = repair_start(&segs, 240.0).expect("a loop");
        assert!((from - 299.5).abs() < 1e-9, "{from}");
        assert_eq!(
            repair_start(&segs, 299.8),
            Some(299.8),
            "clamped to the chunk start"
        );
        assert_eq!(repair_start(&segs[..1], 0.0), None);
        let clean = vec![seg(0.0, 2.0, " a b c"), seg(2.0, 4.0, " d e f")];
        assert_eq!(repair_start(&clean, 0.0), None);
    }

    /// A segment whose words are (text, start, end); its text is the words joined by spaces.
    fn seg_with_words(words: &[(&str, f64, f64)]) -> Segment {
        let start = words.first().map_or(0.0, |w| w.1);
        let end = words.last().map_or(0.0, |w| w.2);
        let text = words.iter().map(|w| w.0).collect::<Vec<_>>().join(" ");
        Segment {
            words: words
                .iter()
                .map(|&(w, s, e)| pipeline::Word {
                    word: w.into(),
                    start: s,
                    end: e,
                })
                .collect(),
            ..seg(start, end, &text)
        }
    }

    #[test]
    fn repair_starts_on_a_collapsed_run_without_a_repetition() {
        let run: Vec<(&str, f64, f64)> = [
            "one", "two", "three", "four", "five", "six", "seven", "eight",
        ]
        .iter()
        .map(|w| (*w, 300.0, 300.0))
        .collect();
        let segs = vec![
            seg_with_words(&[("The", 0.0, 0.5), ("quick", 1.0, 1.5)]),
            seg_with_words(&run),
        ];
        assert!(pipeline::guard_repetitions(segs.clone()).1.is_empty());
        assert_eq!(repair_start(&segs, 240.0), Some(299.5));
        assert_eq!(
            repair_start(&segs, 299.8),
            Some(299.8),
            "clamped to the chunk start"
        );
    }

    #[test]
    fn repair_stays_off_for_normally_timed_words() {
        let words: Vec<(&str, f64, f64)> =
            ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
                .iter()
                .enumerate()
                .map(|(k, w)| (*w, k as f64, k as f64 + 0.5))
                .collect();
        assert_eq!(repair_start(&[seg_with_words(&words)], 0.0), None);
        let two: Vec<(&str, f64, f64)> = (0..2).map(|_| ("x", 300.0, 300.0)).collect();
        assert_eq!(repair_start(&[seg_with_words(&two)], 0.0), None);
    }

    #[test]
    fn repair_fires_on_three_words_at_one_instant() {
        let three = seg_with_words(&[
            ("one", 300.0, 300.0),
            ("two", 300.0, 300.0),
            ("three", 300.0, 300.0),
        ]);
        assert_eq!(repair_start(&[three], 240.0), Some(299.5));
    }

    #[test]
    fn splice_keeps_earlier_segments_and_swaps_in_the_repair() {
        let original = vec![
            seg(0.0, 2.0, " keep one"),
            seg(2.0, 4.0, " keep two"),
            seg(4.0, 6.0, " loop"),
            seg(6.0, 8.0, " loop"),
        ];
        let repaired = vec![seg(4.2, 7.9, " the real words")];
        let out = splice_repair(original.clone(), repaired, 4.0, 8.0);
        let texts: Vec<_> = out.iter().map(|s| s.text.trim().to_string()).collect();
        assert_eq!(texts, ["keep one", "keep two", "the real words"]);
        assert_eq!(out[2].start, 4.2, "absolute times are kept");
        assert_eq!(
            splice_repair(original.clone(), Vec::new(), 4.0, 8.0).len(),
            4
        );
        let overlapping = splice_repair(
            vec![seg(0.0, 4.5, " keep"), seg(4.5, 6.0, " loop")],
            vec![seg(4.0, 6.0, " real")],
            4.2,
            6.0,
        );
        assert!(
            overlapping[0].end <= overlapping[1].start,
            "no overlap after the splice"
        );
    }

    #[test]
    fn speech_after_a_short_repair_is_kept() {
        let original = vec![
            seg(0.0, 2.0, " keep"),
            seg(4.0, 6.0, " loop"),
            seg(6.0, 8.0, " loop"),
            seg(20.0, 23.0, " speech after the loop"),
            seg(23.0, 26.0, " and more"),
        ];
        let repaired = vec![seg(4.2, 9.0, " real words")];
        let out = splice_repair(original.clone(), repaired.clone(), 4.0, 26.0);
        let texts: Vec<_> = out.iter().map(|s| s.text.trim().to_string()).collect();
        assert_eq!(
            texts,
            ["keep", "real words", "speech after the loop", "and more"]
        );
        let full = splice_repair(original, repaired, 4.0, 9.5);
        assert_eq!(
            full.len(),
            2,
            "a repair reaching the chunk end keeps no tail"
        );
    }
}
