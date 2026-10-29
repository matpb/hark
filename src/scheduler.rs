//! Chunk scheduler: lanes, the state pool, batch preemption, admission and model switching.
//! `Core` is pure state (no threads, no model); `Scheduler` drives it from one dispatcher thread.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::engine::{DecodeOptions, EngineError, ModelRef};
use crate::pipeline::{Segment, Span};

/// Rolling window for the per-model realtime factor.
const RTF_WINDOW: usize = 50;
/// Audio length assumed for a typical interactive request when estimating `Retry-After`.
const TYPICAL_REQUEST_SECS: f64 = 10.0;
/// How often the dispatcher looks for cancelled requests while work exists.
const CANCEL_POLL: Duration = Duration::from_millis(50);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Lane {
    Interactive,
    Batch,
    /// Reserved: present in the queues, never scheduled.
    Live,
}

impl Lane {
    pub fn as_str(self) -> &'static str {
        match self {
            Lane::Interactive => "interactive",
            Lane::Batch => "batch",
            Lane::Live => "live",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone)]
pub struct ChunkTask {
    pub audio: Arc<Vec<f32>>,
    pub span: Span,
    /// VAD pieces inside `span`, decoded back to back; empty decodes the whole span.
    pub pieces: Arc<Vec<Span>>,
    pub opts: Arc<DecodeOptions>,
    pub language: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ChunkOut {
    pub segments: Vec<Segment>,
    pub detected: Option<String>,
}

/// Why a model was loaded; status reports it as `last_load_reason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadReason {
    Preload,
    Request,
    Admin,
    ReloadAfterIdle,
    DeviceFallback,
    GpuRetry,
    Recycle,
}

impl LoadReason {
    pub fn as_str(self) -> &'static str {
        match self {
            LoadReason::Preload => "preload",
            LoadReason::Request => "request",
            LoadReason::Admin => "admin",
            LoadReason::ReloadAfterIdle => "reload_after_idle",
            LoadReason::DeviceFallback => "device_fallback",
            LoadReason::GpuRetry => "gpu_retry",
            LoadReason::Recycle => "recycle",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkFailure {
    /// The abort flag was observed by the engine; the result is discarded and the chunk requeued.
    Aborted,
    /// The state cannot be used again (whisper `-7`); only this state is retired.
    StateLost(String),
    /// The GPU ran out of memory or faulted mid-run; the engine falls back to CPU.
    DeviceFault(String),
    /// A graph failure that may be one-off: the chunk is retried once on the same device.
    Transient(String),
    Failed(String),
}

#[derive(Debug, Clone, Default)]
pub struct LoadReport {
    pub load_ms: u64,
    pub states: usize,
    pub vram_per_state_mib: Option<u64>,
    /// Set when the model could only be loaded after falling back from the GPU to the CPU.
    pub fell_back: Option<String>,
}

/// What the scheduler needs from an inference engine; one model resident at a time.
pub trait Backend: Send + Sync + 'static {
    /// Loads `model` with up to `states` states; a parked copy of it is reactivated instead.
    fn load(&self, model: &ModelRef, states: usize) -> Result<LoadReport, EngineError>;
    /// Drops every resident model, parked ones included.
    fn unload(&self);
    /// Keeps the active model resident but inactive. Backends without residency drop it.
    fn park(&self) {
        self.unload();
    }
    /// Drops a parked model.
    fn evict(&self, _id: &str) {}
    /// Drops the active model only.
    fn drop_active(&self) {
        self.unload();
    }
    /// The next `load` tries the original GPU again, falling back to the CPU if that fails.
    fn try_gpu_again(&self) {}
    /// The device inference currently runs on, when the backend knows.
    fn device(&self) -> Option<DeviceStatus> {
        None
    }
    /// Total GPU devices destroyed so far, when the backend counts them.
    fn devices_released(&self) -> Option<u64> {
        None
    }
    fn run(
        &self,
        state: usize,
        task: &ChunkTask,
        abort: &AtomicBool,
    ) -> Result<ChunkOut, ChunkFailure>;
}

#[derive(Debug)]
pub struct ChunkDone {
    pub out: ChunkOut,
    pub run_secs: f64,
    pub queue_wait_secs: f64,
}

type Reply = Result<ChunkDone, EngineError>;

pub struct ChunkRequest {
    pub request_id: String,
    pub lane: Lane,
    pub model: ModelRef,
    pub chunk: usize,
    pub task: ChunkTask,
    /// First chunk of a request: counted against `max_interactive_waiting`.
    pub admit: bool,
    /// More chunks of this request follow; keeps the state reserved briefly.
    pub last: bool,
    /// Set when the client is gone: queued chunks are dropped and a running one is aborted.
    pub cancel: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    pub states: usize,
    pub max_interactive_waiting: usize,
}

enum Work {
    Chunk(ChunkTask),
    Preload,
}

struct Queued {
    request_id: String,
    reason: LoadReason,
    lane: Lane,
    model: ModelRef,
    chunk: usize,
    work: Work,
    enqueued: Instant,
    reply: Sender<Reply>,
    cancel: Arc<AtomicBool>,
    admit: bool,
    last: bool,
    retried: bool,
    device_retried: bool,
    transient_retried: bool,
}

struct Running {
    item: Queued,
    seq: u64,
    abort: Arc<AtomicBool>,
    aborting: bool,
}

/// An idle state held for an interactive request whose next chunk is about to arrive.
struct Reserved {
    request_id: String,
    until: Instant,
    cancel: Arc<AtomicBool>,
}

enum Slot {
    Idle,
    Running(Box<Running>),
    Reserved(Reserved),
    Retired,
}

struct Ready {
    model: ModelRef,
    load_ms: u64,
    vram_per_state_mib: Option<u64>,
    states: usize,
    reason: LoadReason,
    last_used: Instant,
}

/// A model kept resident but inactive so a switch back to it is instant.
struct Parked {
    model: ModelRef,
    load_ms: u64,
    vram_per_state_mib: Option<u64>,
    states: usize,
    reason: LoadReason,
    last_used: Instant,
}

/// What the loader thread does with the resident models before it loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadPrep {
    /// Nothing is active and others may be parked: only `evict` applies.
    Keep,
    /// Park the active model, then evict.
    Park,
    /// Drop everything resident (device fallback).
    PurgeAll,
}

struct AdminUnload {
    id: String,
    reply: Sender<Result<bool, EngineError>>,
}

enum ModelState {
    None,
    Loading,
    Unloading,
    Ready(Ready),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Busy {
    pub retry_after_secs: u64,
}

#[derive(Debug)]
pub enum Action {
    Start {
        state: usize,
        task: ChunkTask,
        abort: Arc<AtomicBool>,
        generation: u64,
    },
    Load {
        model: ModelRef,
        states: usize,
        prep: LoadPrep,
        evict: Vec<&'static str>,
        generation: u64,
        /// Ask the backend to try the original GPU again for this load.
        gpu_retry: bool,
        /// A device recycle: the purge must destroy the GPU device.
        recycle: bool,
    },
    /// Drops the active model.
    Unload { generation: u64, token: u64 },
    /// Drops parked models.
    Evict {
        ids: Vec<&'static str>,
        generation: u64,
        token: u64,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct LastError {
    pub message: String,
    pub time: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DeviceStatus {
    pub name: String,
    pub index: Option<i32>,
    /// The GPU this process fell back from, once it has.
    pub fell_back_from: Option<String>,
}

impl Default for DeviceStatus {
    fn default() -> Self {
        DeviceStatus {
            name: "cpu".into(),
            index: None,
            fell_back_from: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    pub id: String,
    pub load_ms: u64,
    pub states: usize,
    pub vram_per_state_mib: Option<u64>,
    pub realtime_factor: Option<f64>,
    /// Size of the model file on disk.
    pub file_mib: Option<u64>,
    /// `true` for the model serving requests, `false` for one kept resident (`max_loaded_models` above 1).
    pub active: bool,
    /// `preload`, `request`, `admin`, `reload_after_idle`, `device_fallback`, `gpu_retry` or `recycle`.
    pub last_load_reason: &'static str,
    /// Chunks decoded since the GPU device was created; only on the active model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunks_since_recycle: Option<u64>,
    /// Device recycles since start; only on the active model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recycles: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct QueueDepths {
    pub interactive: usize,
    pub batch: usize,
    pub live: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunningStatus {
    pub state: usize,
    /// `idle`, `reserved`, `running` or `aborting`.
    pub status: &'static str,
    pub lane: Option<&'static str>,
    pub request_id: Option<String>,
    pub model: Option<String>,
    pub chunk: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct StatusReport {
    pub device: DeviceStatus,
    pub degraded: bool,
    pub models: Vec<ModelStatus>,
    pub queues: QueueDepths,
    pub running: Vec<RunningStatus>,
    pub aborted_chunks: u64,
    pub last_error: Option<LastError>,
    /// Recycles that failed to load again or released no device.
    pub recycle_failures: u64,
}

/// Unix seconds as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn rfc3339_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn now_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    rfc3339_utc(secs)
}

pub struct Core {
    queues: [VecDeque<Queued>; 3],
    slots: Vec<Slot>,
    model: ModelState,
    want_states: usize,
    max_waiting: usize,
    seq: u64,
    aborted_total: u64,
    last_error: Option<LastError>,
    rtf: Vec<(&'static str, VecDeque<f64>)>,
    reserve_for: Duration,
    /// Bumped when the core is reset after a panic; events from older work are ignored.
    generation: u64,
    pending_unload: bool,
    unload_token: u64,
    parked: Vec<Parked>,
    max_loaded: usize,
    idle: Option<Duration>,
    idle_unloaded: Vec<&'static str>,
    loading_reason: LoadReason,
    /// Set once the GPU failed and the process fell back to the CPU.
    degraded: bool,
    /// A device fault was seen: drain the running chunks, then reload on the CPU.
    reload_pending: bool,
    /// How often an idle degraded daemon tries the GPU again; `None` never.
    gpu_retry: Option<Duration>,
    retry_at: Option<Instant>,
    unload_reqs: Vec<AdminUnload>,
    waiters: Vec<(u64, Sender<Result<bool, EngineError>>)>,
    token_seq: u64,
    /// Recreate the GPU device after this many decoded chunks; 0 = never.
    recycle_after: u64,
    chunks_since_recycle: u64,
    recycles: u64,
    recycle_failures: u64,
    /// Past twice the threshold: no new chunk starts until running ones finish and the device is recycled.
    draining: bool,
    #[cfg(test)]
    panic_next_dispatch: Arc<AtomicBool>,
}

impl Core {
    pub fn new(cfg: SchedulerConfig) -> Core {
        Core {
            queues: Default::default(),
            slots: Vec::new(),
            model: ModelState::None,
            want_states: cfg.states.max(1),
            max_waiting: cfg.max_interactive_waiting,
            seq: 0,
            aborted_total: 0,
            last_error: None,
            rtf: Vec::new(),
            reserve_for: Duration::from_millis(250),
            generation: 0,
            pending_unload: false,
            unload_token: 0,
            parked: Vec::new(),
            max_loaded: 1,
            idle: None,
            idle_unloaded: Vec::new(),
            loading_reason: LoadReason::Request,
            degraded: false,
            reload_pending: false,
            gpu_retry: None,
            retry_at: None,
            unload_reqs: Vec::new(),
            waiters: Vec::new(),
            token_seq: 0,
            recycle_after: 0,
            chunks_since_recycle: 0,
            recycles: 0,
            recycle_failures: 0,
            draining: false,
            #[cfg(test)]
            panic_next_dispatch: Arc::new(AtomicBool::new(false)),
        }
    }

    /// After a fallback, an idle daemon reloads the active model on the original GPU every `every`.
    pub fn set_gpu_retry(&mut self, every: Option<Duration>) {
        self.gpu_retry = every;
        if self.degraded {
            self.retry_at = every.map(|d| Instant::now() + d);
        }
    }

    /// Recreates the GPU device after `after` decoded chunks; 0 turns it off.
    pub fn set_recycle(&mut self, after: u64) {
        self.recycle_after = after;
    }

    fn mark_degraded(&mut self) {
        self.degraded = true;
        self.retry_at = self.gpu_retry.map(|d| Instant::now() + d);
    }

    /// `max_loaded` models stay resident at once; a model unused for `idle` is freed.
    pub fn set_residency(&mut self, max_loaded: usize, idle: Option<Duration>) {
        self.max_loaded = max_loaded.max(1);
        self.idle = idle;
    }

    fn next_token(&mut self) -> u64 {
        self.token_seq += 1;
        self.token_seq
    }

    fn rtf_mean(&self, id: &str) -> Option<f64> {
        let (_, h) = self.rtf.iter().find(|(m, _)| *m == id)?;
        (!h.is_empty()).then(|| h.iter().sum::<f64>() / h.len() as f64)
    }

    pub fn record_request(&mut self, model_id: &'static str, run_secs: f64, audio_secs: f64) {
        if audio_secs <= 0.0 || !run_secs.is_finite() {
            return;
        }
        let idx = match self.rtf.iter().position(|(m, _)| *m == model_id) {
            Some(i) => i,
            None => {
                self.rtf.push((model_id, VecDeque::new()));
                self.rtf.len() - 1
            }
        };
        let h = &mut self.rtf[idx].1;
        if h.len() == RTF_WINDOW {
            h.pop_front();
        }
        h.push_back(run_secs / audio_secs);
    }

    fn live_states(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| !matches!(s, Slot::Retired))
            .count()
    }

    /// States an interactive chunk can take right now: idle or reserved.
    fn free_states(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s, Slot::Idle | Slot::Reserved(_)))
            .count()
    }

    fn running_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s, Slot::Running(_)))
            .count()
    }

    fn record_error(&mut self, message: String) {
        self.last_error = Some(LastError {
            message,
            time: now_rfc3339(),
        });
    }

    /// Seconds a refused client should wait: queued interactive work drained at the rolling rate.
    fn retry_after(&self) -> u64 {
        let rtf = match &self.model {
            ModelState::Ready(m) => self.rtf_mean(m.model.id),
            _ => None,
        }
        .unwrap_or(0.2)
        .max(0.01);
        let states = self.live_states().max(1) as f64;
        let waiting = self.queues[Lane::Interactive.index()].len() as f64;
        let secs = rtf * TYPICAL_REQUEST_SECS * (waiting / states + 1.0);
        (secs.ceil() as u64).max(1)
    }

    pub fn submit(&mut self, req: ChunkRequest, reply: Sender<Reply>) -> Result<(), Busy> {
        if req.lane == Lane::Interactive && req.admit {
            let first_chunks = self.queues[Lane::Interactive.index()]
                .iter()
                .filter(|q| q.admit && matches!(q.work, Work::Chunk(_)))
                .count();
            if (first_chunks + 1).saturating_sub(self.free_states()) > self.max_waiting {
                return Err(Busy {
                    retry_after_secs: self.retry_after(),
                });
            }
        }
        self.queues[req.lane.index()].push_back(Queued {
            request_id: req.request_id,
            reason: LoadReason::Request,
            lane: req.lane,
            model: req.model,
            chunk: req.chunk,
            work: Work::Chunk(req.task),
            enqueued: Instant::now(),
            reply,
            cancel: req.cancel,
            admit: req.admit,
            last: req.last,
            retried: false,
            device_retried: false,
            transient_retried: false,
        });
        Ok(())
    }

    /// Queues a model load that replies when the model is ready.
    pub fn submit_load(&mut self, model: ModelRef, reason: LoadReason, reply: Sender<Reply>) {
        self.queues[Lane::Interactive.index()].push_back(Queued {
            request_id: String::new(),
            reason,
            lane: Lane::Interactive,
            model,
            chunk: 0,
            work: Work::Preload,
            enqueued: Instant::now(),
            reply,
            cancel: Arc::new(AtomicBool::new(false)),
            admit: false,
            last: true,
            retried: false,
            device_retried: false,
            transient_retried: false,
        });
    }

    pub fn submit_preload(&mut self, model: ModelRef, reply: Sender<Reply>) {
        self.submit_load(model, LoadReason::Preload, reply);
    }

    /// Queues an unload of `id` for once its states are idle; the reply says whether it was resident.
    pub fn submit_unload(&mut self, id: String, reply: Sender<Result<bool, EngineError>>) {
        self.unload_reqs.push(AdminUnload { id, reply });
    }

    /// Interactive always first; `live` is never scheduled.
    fn next_lane(&self) -> Option<usize> {
        [Lane::Interactive, Lane::Batch]
            .into_iter()
            .map(Lane::index)
            .find(|&l| !self.queues[l].is_empty())
    }

    /// Drops cancelled work, aborts cancelled running chunks and frees expired reservations.
    fn sweep(&mut self, now: Instant) {
        for q in &mut self.queues {
            if q.iter().any(|i| i.cancel.load(Ordering::SeqCst)) {
                let (gone, keep): (Vec<_>, Vec<_>) =
                    q.drain(..).partition(|i| i.cancel.load(Ordering::SeqCst));
                q.extend(keep);
                for item in gone {
                    let _ = item.reply.send(Err(EngineError::Cancelled));
                }
            }
        }
        for slot in &mut self.slots {
            match slot {
                Slot::Running(r) if r.item.cancel.load(Ordering::SeqCst) && !r.aborting => {
                    r.aborting = true;
                    r.abort.store(true, Ordering::SeqCst);
                }
                Slot::Reserved(r) if now >= r.until || r.cancel.load(Ordering::SeqCst) => {
                    *slot = Slot::Idle;
                }
                _ => {}
            }
        }
    }

    /// How long the dispatcher may sleep before it must look again.
    pub fn next_deadline(&self, now: Instant) -> Option<Duration> {
        let busy = self.queues.iter().any(|q| !q.is_empty())
            || self
                .slots
                .iter()
                .any(|s| matches!(s, Slot::Running(_) | Slot::Reserved(_)));
        let reserved = self
            .slots
            .iter()
            .filter_map(|s| match s {
                Slot::Reserved(r) => Some(r.until.saturating_duration_since(now)),
                _ => None,
            })
            .min();
        let poll = busy.then_some(CANCEL_POLL);
        let idle = self.idle.and_then(|limit| {
            let active = match &self.model {
                ModelState::Ready(r) if !busy => Some(r.last_used),
                _ => None,
            };
            active
                .into_iter()
                .chain(self.parked.iter().map(|p| p.last_used))
                .map(|t| (t + limit).saturating_duration_since(now))
                .min()
        });
        let retry = match (&self.model, self.retry_at) {
            (ModelState::Ready(_), Some(at)) if self.degraded && !busy => {
                Some(at.saturating_duration_since(now))
            }
            _ => None,
        };
        [reserved, poll, idle, retry].into_iter().flatten().min()
    }

    /// Once the retry interval passed and nothing runs, reloads the active model on the original GPU.
    fn start_gpu_retry(&mut self, now: Instant, actions: &mut Vec<Action>) {
        let due = self.degraded && self.retry_at.is_some_and(|at| at <= now);
        let quiet = self.queues.iter().all(VecDeque::is_empty)
            && self.slots.iter().all(|s| matches!(s, Slot::Idle))
            && !self.reload_pending
            && !self.pending_unload
            && self.unload_reqs.is_empty();
        if !due || !quiet || !matches!(self.model, ModelState::Ready(_)) {
            return;
        }
        let ModelState::Ready(r) = std::mem::replace(&mut self.model, ModelState::Loading) else {
            return;
        };
        self.slots.clear();
        self.parked.clear();
        self.retry_at = None;
        self.loading_reason = LoadReason::GpuRetry;
        tracing::info!(model = r.model.id, "trying the GPU again");
        actions.push(Action::Load {
            model: r.model,
            states: self.want_states,
            prep: LoadPrep::PurgeAll,
            evict: Vec::new(),
            generation: self.generation,
            gpu_retry: true,
            recycle: false,
        });
    }

    /// Frees parked models, and the active one, that sat unused for the idle limit.
    fn expire_idle(&mut self, now: Instant, actions: &mut Vec<Action>) {
        let Some(limit) = self.idle else { return };
        let expired: Vec<&'static str> = self
            .parked
            .iter()
            .filter(|p| now.saturating_duration_since(p.last_used) >= limit)
            .map(|p| p.model.id)
            .collect();
        if !expired.is_empty() {
            self.parked.retain(|p| !expired.contains(&p.model.id));
            self.idle_unloaded.extend(expired.iter().copied());
            actions.push(Action::Evict {
                ids: expired,
                generation: self.generation,
                token: 0,
            });
        }
        let quiet = self.queues.iter().all(VecDeque::is_empty)
            && self.slots.iter().all(|s| matches!(s, Slot::Idle))
            && !self.reload_pending
            && !self.pending_unload;
        if quiet
            && let ModelState::Ready(r) = &self.model
            && now.saturating_duration_since(r.last_used) >= limit
        {
            self.idle_unloaded.push(r.model.id);
            self.begin_unload(0);
        }
    }

    /// Answers queued admin unloads: the active model goes once its states are idle, a parked one at once.
    fn process_unload_requests(&mut self, actions: &mut Vec<Action>) {
        for req in std::mem::take(&mut self.unload_reqs) {
            let active = matches!(&self.model, ModelState::Ready(m) if m.model.id == req.id);
            if active {
                if self.running_count() == 0 && !self.reload_pending && !self.pending_unload {
                    let token = self.next_token();
                    self.waiters.push((token, req.reply));
                    self.begin_unload(token);
                } else {
                    self.unload_reqs.push(req);
                }
                continue;
            }
            if let Some(i) = self.parked.iter().position(|p| p.model.id == req.id) {
                let gone = self.parked.remove(i);
                let token = self.next_token();
                self.waiters.push((token, req.reply));
                actions.push(Action::Evict {
                    ids: vec![gone.model.id],
                    generation: self.generation,
                    token,
                });
                continue;
            }
            if matches!(self.model, ModelState::Loading | ModelState::Unloading) {
                self.unload_reqs.push(req);
                continue;
            }
            let _ = req.reply.send(Ok(false));
        }
    }

    /// Reloads the active model after a full purge so the GPU device is recreated: when idle past
    /// `recycle_after` chunks, or once running chunks drained past twice that.
    fn start_recycle(&mut self, actions: &mut Vec<Action>) {
        self.draining = false;
        let n = self.recycle_after;
        if n == 0 || self.degraded || self.chunks_since_recycle < n {
            return;
        }
        if !matches!(self.model, ModelState::Ready(_))
            || self.reload_pending
            || self.pending_unload
            || !self.unload_reqs.is_empty()
        {
            return;
        }
        let idle = self.queues[Lane::Interactive.index()].is_empty()
            && self.queues[Lane::Batch.index()].is_empty()
            && self
                .slots
                .iter()
                .all(|s| matches!(s, Slot::Idle | Slot::Retired));
        let hard = self.chunks_since_recycle >= n.saturating_mul(2);
        let drained = hard && self.running_count() == 0;
        if !idle && !drained {
            self.draining = hard;
            return;
        }
        let ModelState::Ready(r) = std::mem::replace(&mut self.model, ModelState::Loading) else {
            return;
        };
        self.slots.clear();
        self.parked.clear();
        self.loading_reason = LoadReason::Recycle;
        tracing::info!(
            model = r.model.id,
            chunks = self.chunks_since_recycle,
            "recycling the GPU device"
        );
        actions.push(Action::Load {
            model: r.model,
            states: self.want_states,
            prep: LoadPrep::PurgeAll,
            evict: Vec::new(),
            generation: self.generation,
            gpu_retry: false,
            recycle: true,
        });
    }

    /// After a device fault, once no chunk is running, reloads the model on the CPU.
    fn start_fault_reload(&mut self, actions: &mut Vec<Action>) {
        if !self.reload_pending {
            return;
        }
        match &self.model {
            ModelState::Ready(_) => {}
            ModelState::None => {
                self.reload_pending = false;
                return;
            }
            _ => return,
        }
        if self.running_count() > 0 {
            return;
        }
        let ModelState::Ready(r) = std::mem::replace(&mut self.model, ModelState::Loading) else {
            return;
        };
        self.slots.clear();
        self.parked.clear();
        self.reload_pending = false;
        self.loading_reason = LoadReason::DeviceFallback;
        actions.push(Action::Load {
            model: r.model,
            states: self.want_states,
            prep: LoadPrep::PurgeAll,
            evict: Vec::new(),
            generation: self.generation,
            gpu_retry: false,
            recycle: false,
        });
    }

    fn touch_active(&mut self) {
        if let ModelState::Ready(r) = &mut self.model {
            r.last_used = Instant::now();
        }
    }

    pub fn flush_waiters(&mut self, token: u64) {
        if token == 0 {
            return;
        }
        for (t, reply) in std::mem::take(&mut self.waiters) {
            if t == token {
                let _ = reply.send(Ok(true));
            } else {
                self.waiters.push((t, reply));
            }
        }
    }

    /// Slot for `item`: the one reserved for its request, else an idle one, else (interactive
    /// only) a state reserved for another request.
    fn pick_slot(&self, item: &Queued) -> Option<usize> {
        let own = self
            .slots
            .iter()
            .position(|s| matches!(s, Slot::Reserved(r) if r.request_id == item.request_id));
        own.or_else(|| self.slots.iter().position(|s| matches!(s, Slot::Idle)))
            .or_else(|| {
                (item.lane == Lane::Interactive)
                    .then(|| {
                        self.slots
                            .iter()
                            .position(|s| matches!(s, Slot::Reserved(_)))
                    })
                    .flatten()
            })
    }

    pub fn dispatch(&mut self) -> Vec<Action> {
        #[cfg(test)]
        if self.panic_next_dispatch.swap(false, Ordering::SeqCst) {
            panic!("injected dispatch panic");
        }
        let now = Instant::now();
        self.sweep(now);
        let mut actions = Vec::new();
        self.expire_idle(now, &mut actions);
        self.process_unload_requests(&mut actions);
        self.start_fault_reload(&mut actions);
        self.start_gpu_retry(now, &mut actions);
        self.start_recycle(&mut actions);
        if self.pending_unload {
            self.pending_unload = false;
            actions.push(Action::Unload {
                generation: self.generation,
                token: std::mem::take(&mut self.unload_token),
            });
        }
        loop {
            if matches!(self.model, ModelState::Loading | ModelState::Unloading)
                || self.reload_pending
            {
                break;
            }
            let Some(lane) = self.next_lane() else { break };
            let head = self.queues[lane].front().expect("lane is non-empty");
            let matches_loaded =
                matches!(&self.model, ModelState::Ready(m) if m.model.id == head.model.id);
            if matches_loaded {
                if matches!(head.work, Work::Preload) {
                    self.touch_active();
                    let item = self.queues[lane].pop_front().expect("head exists");
                    let _ = item.reply.send(Ok(ChunkDone {
                        out: ChunkOut::default(),
                        run_secs: 0.0,
                        queue_wait_secs: item.enqueued.elapsed().as_secs_f64(),
                    }));
                    continue;
                }
                if self.draining {
                    break;
                }
                let Some(state) = self.pick_slot(head) else {
                    break;
                };
                let item = self.queues[lane].pop_front().expect("head exists");
                self.touch_active();
                let Work::Chunk(task) = &item.work else {
                    unreachable!("preload handled above");
                };
                let task = task.clone();
                let abort = Arc::new(AtomicBool::new(false));
                self.seq += 1;
                self.slots[state] = Slot::Running(Box::new(Running {
                    item,
                    seq: self.seq,
                    abort: abort.clone(),
                    aborting: false,
                }));
                actions.push(Action::Start {
                    state,
                    task,
                    abort,
                    generation: self.generation,
                });
                continue;
            }
            // Another model is wanted: wait for every state to drain, then swap. A batch head
            // never evicts the model an interactive request still holds a state on.
            if self.running_count() > 0 {
                break;
            }
            if lane == Lane::Batch.index()
                && self.slots.iter().any(|s| matches!(s, Slot::Reserved(_)))
            {
                break;
            }
            let model = head.model.clone();
            let mut reason = head.reason;
            let prev = match std::mem::replace(&mut self.model, ModelState::Loading) {
                ModelState::Ready(r) => Some(r),
                _ => None,
            };
            self.slots.clear();
            let prep = if prev.is_some() {
                LoadPrep::Park
            } else if self.max_loaded == 1 {
                LoadPrep::PurgeAll
            } else {
                LoadPrep::Keep
            };
            let mut resident = std::mem::take(&mut self.parked);
            if let Some(r) = prev {
                resident.push(Parked {
                    model: r.model,
                    load_ms: r.load_ms,
                    vram_per_state_mib: r.vram_per_state_mib,
                    states: r.states,
                    reason: r.reason,
                    last_used: r.last_used,
                });
            }
            resident.retain(|p| p.model.id != model.id);
            let mut evict = Vec::new();
            while resident.len() + 1 > self.max_loaded {
                let lru = resident
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, p)| p.last_used)
                    .map(|(i, _)| i)
                    .expect("resident is non-empty");
                evict.push(resident.remove(lru).model.id);
            }
            self.parked = resident;
            if self.idle_unloaded.contains(&model.id) {
                self.idle_unloaded.retain(|id| *id != model.id);
                if reason == LoadReason::Request {
                    reason = LoadReason::ReloadAfterIdle;
                }
            }
            self.loading_reason = reason;
            actions.push(Action::Load {
                model,
                states: self.want_states,
                prep,
                evict,
                generation: self.generation,
                gpu_retry: false,
                recycle: false,
            });
            break;
        }
        self.preempt();
        actions
    }

    /// Aborts the newest batch chunks an interactive head needs; all of them for another model.
    fn preempt(&mut self) {
        if self.draining {
            return;
        }
        let ModelState::Ready(ready) = &self.model else {
            return;
        };
        let interactive = &self.queues[Lane::Interactive.index()];
        let Some(head) = interactive.front() else {
            return;
        };
        let running: Vec<&Running> = self
            .slots
            .iter()
            .filter_map(|s| match s {
                Slot::Running(r) => Some(&**r),
                _ => None,
            })
            .collect();
        if running.is_empty() || running.iter().any(|r| r.item.lane != Lane::Batch) {
            return;
        }
        let in_flight = running.iter().filter(|r| r.aborting).count();
        let mut victims: Vec<(u64, usize)> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                Slot::Running(r) if !r.aborting => Some((r.seq, i)),
                _ => None,
            })
            .collect();
        victims.sort_unstable();
        let count = if head.model.id != ready.model.id {
            victims.len()
        } else {
            let runnable = interactive
                .iter()
                .filter(|q| matches!(q.work, Work::Chunk(_)) && q.model.id == ready.model.id)
                .count();
            runnable.saturating_sub(self.free_states() + in_flight)
        };
        for _ in 0..count {
            let Some((_, i)) = victims.pop() else {
                break;
            };
            if let Slot::Running(r) = &mut self.slots[i] {
                r.aborting = true;
                r.abort.store(true, Ordering::SeqCst);
            }
        }
    }

    fn begin_unload(&mut self, token: u64) {
        self.model = ModelState::Unloading;
        self.slots.clear();
        self.pending_unload = true;
        self.unload_token = token;
    }

    pub fn on_finished(
        &mut self,
        state: usize,
        result: Result<ChunkOut, ChunkFailure>,
        run_secs: f64,
    ) {
        let Some(slot) = self.slots.get_mut(state) else {
            return;
        };
        let Slot::Running(run) = std::mem::replace(slot, Slot::Idle) else {
            *slot = Slot::Idle;
            return;
        };
        let mut item = run.item;
        let cancelled = item.cancel.load(Ordering::SeqCst);
        if !self.degraded {
            self.chunks_since_recycle += 1;
        }
        self.touch_active();
        match result {
            Ok(out) => {
                let _ = item.reply.send(Ok(ChunkDone {
                    out,
                    run_secs,
                    queue_wait_secs: 0.0,
                }));
                if item.lane == Lane::Interactive && !item.last && !cancelled {
                    self.slots[state] = Slot::Reserved(Reserved {
                        request_id: item.request_id,
                        until: Instant::now() + self.reserve_for,
                        cancel: item.cancel,
                    });
                }
            }
            Err(ChunkFailure::Aborted) if cancelled => {
                let _ = item.reply.send(Err(EngineError::Cancelled));
            }
            Err(ChunkFailure::Aborted) => {
                self.aborted_total += 1;
                item.enqueued = Instant::now();
                self.queues[item.lane.index()].push_front(item);
            }
            Err(ChunkFailure::StateLost(msg)) => {
                self.slots[state] = Slot::Retired;
                self.record_error(format!("state {state} retired: {msg}"));
                if cancelled {
                    let _ = item.reply.send(Err(EngineError::Cancelled));
                } else if !item.retried && self.live_states() > 0 {
                    item.retried = true;
                    item.enqueued = Instant::now();
                    self.queues[item.lane.index()].push_front(item);
                } else {
                    let _ = item.reply.send(Err(EngineError::Run(
                        "not enough memory for this model, try a smaller one".into(),
                    )));
                }
                if self.live_states() == 0 {
                    self.begin_unload(0);
                }
            }
            Err(ChunkFailure::DeviceFault(msg)) => {
                if cancelled {
                    self.record_error(msg);
                    let _ = item.reply.send(Err(EngineError::Cancelled));
                } else if item.device_retried || (self.degraded && !self.reload_pending) {
                    self.record_error(msg.clone());
                    let _ = item.reply.send(Err(EngineError::Run(msg)));
                } else {
                    self.record_error(format!("GPU fault, falling back to the CPU: {msg}"));
                    self.mark_degraded();
                    self.reload_pending = true;
                    item.device_retried = true;
                    item.enqueued = Instant::now();
                    self.queues[item.lane.index()].push_front(item);
                }
            }
            Err(ChunkFailure::Transient(msg)) => {
                if cancelled {
                    let _ = item.reply.send(Err(EngineError::Cancelled));
                } else if !item.transient_retried {
                    self.record_error(format!("decode failed, retrying once: {msg}"));
                    item.transient_retried = true;
                    item.enqueued = Instant::now();
                    self.queues[item.lane.index()].push_front(item);
                } else {
                    self.record_error(msg.clone());
                    let _ = item.reply.send(Err(EngineError::Run(msg)));
                }
            }
            Err(ChunkFailure::Failed(msg)) => {
                self.record_error(msg.clone());
                let _ = item.reply.send(Err(EngineError::Run(msg)));
            }
        }
    }

    pub fn on_unloaded(&mut self) {
        if matches!(self.model, ModelState::Unloading) {
            self.model = ModelState::None;
        }
    }

    /// Fails every queued and running waiter and returns to an empty, loadable state.
    pub fn reset_after_panic(&mut self, what: &str) {
        let msg = format!("scheduler panic: {what}");
        self.generation += 1;
        self.record_error(msg.clone());
        let err = || EngineError::Run(format!("internal error: {msg}"));
        for q in &mut self.queues {
            for item in q.drain(..) {
                let _ = item.reply.send(Err(err()));
            }
        }
        for slot in self.slots.drain(..) {
            if let Slot::Running(r) = slot {
                r.abort.store(true, Ordering::SeqCst);
                let _ = r.item.reply.send(Err(err()));
            }
        }
        for req in self.unload_reqs.drain(..) {
            let _ = req.reply.send(Err(err()));
        }
        for (_, reply) in self.waiters.drain(..) {
            let _ = reply.send(Err(err()));
        }
        self.reload_pending = false;
        self.model = if self.pending_unload {
            ModelState::Unloading
        } else {
            ModelState::None
        };
    }

    /// Fails the queued work for `id` with `ModelBusy`; running chunks finish.
    pub fn fail_queued_for(&mut self, id: &str) {
        for q in &mut self.queues {
            let (failed, keep): (Vec<_>, Vec<_>) = q.drain(..).partition(|i| i.model.id == id);
            q.extend(keep);
            for item in failed {
                let _ = item.reply.send(Err(EngineError::ModelBusy(id.to_string())));
            }
        }
    }

    /// Fails everything queued; used when the dispatcher can no longer be reached.
    pub fn fail_all(&mut self, what: &str) {
        for q in &mut self.queues {
            for item in q.drain(..) {
                let _ = item.reply.send(Err(EngineError::Run(what.into())));
            }
        }
    }

    pub fn on_loaded(&mut self, model: ModelRef, result: Result<LoadReport, EngineError>) {
        match result {
            Ok(report) => {
                self.slots = (0..report.states).map(|_| Slot::Idle).collect();
                if self.parked.is_empty() {
                    self.chunks_since_recycle = 0;
                }
                if self.loading_reason == LoadReason::Recycle {
                    self.recycles += 1;
                }
                let retried = self.loading_reason == LoadReason::GpuRetry;
                if let Some(msg) = &report.fell_back {
                    self.mark_degraded();
                    self.record_error(msg.clone());
                } else if retried && self.degraded {
                    self.degraded = false;
                    self.retry_at = None;
                    tracing::info!(model = model.id, "back on the GPU; no longer degraded");
                } else if self.degraded {
                    self.retry_at = self.gpu_retry.map(|d| Instant::now() + d);
                }
                self.model = ModelState::Ready(Ready {
                    model,
                    load_ms: report.load_ms,
                    vram_per_state_mib: report.vram_per_state_mib,
                    states: report.states,
                    reason: self.loading_reason,
                    last_used: Instant::now(),
                });
            }
            Err(e) => {
                self.model = ModelState::None;
                self.slots.clear();
                if self.loading_reason == LoadReason::Recycle {
                    self.recycle_failures += 1;
                }
                let msg = e.to_string();
                self.record_error(msg.clone());
                for q in &mut self.queues {
                    let (failed, keep): (Vec<_>, Vec<_>) =
                        q.drain(..).partition(|i| i.model.id == model.id);
                    q.extend(keep);
                    for item in failed {
                        let _ = item.reply.send(Err(EngineError::Load(
                            msg.trim_start_matches("could not load model: ").to_string(),
                        )));
                    }
                }
            }
        }
    }

    pub fn status(&self, device: DeviceStatus) -> StatusReport {
        let mut running = Vec::new();
        for (i, s) in self.slots.iter().enumerate() {
            match s {
                Slot::Reserved(r) => running.push(RunningStatus {
                    state: i,
                    status: "reserved",
                    lane: Some("interactive"),
                    request_id: Some(r.request_id.clone()),
                    model: None,
                    chunk: None,
                }),
                Slot::Idle => running.push(RunningStatus {
                    state: i,
                    status: "idle",
                    lane: None,
                    request_id: None,
                    model: None,
                    chunk: None,
                }),
                Slot::Running(r) => running.push(RunningStatus {
                    state: i,
                    status: if r.aborting { "aborting" } else { "running" },
                    lane: Some(r.item.lane.as_str()),
                    request_id: Some(r.item.request_id.clone()),
                    model: Some(r.item.model.id.to_string()),
                    chunk: Some(r.item.chunk),
                }),
                Slot::Retired => {}
            }
        }
        let file_mib = |path: &std::path::Path| {
            std::fs::metadata(path)
                .ok()
                .map(|f| f.len() / (1024 * 1024))
        };
        let mut models = match &self.model {
            ModelState::Ready(m) => vec![ModelStatus {
                id: m.model.id.to_string(),
                load_ms: m.load_ms,
                states: self.live_states(),
                vram_per_state_mib: m.vram_per_state_mib,
                realtime_factor: self.rtf_mean(m.model.id),
                file_mib: file_mib(&m.model.path),
                active: true,
                last_load_reason: m.reason.as_str(),
                chunks_since_recycle: Some(self.chunks_since_recycle),
                recycles: Some(self.recycles),
            }],
            _ => Vec::new(),
        };
        models.extend(self.parked.iter().map(|p| ModelStatus {
            id: p.model.id.to_string(),
            load_ms: p.load_ms,
            states: p.states,
            vram_per_state_mib: p.vram_per_state_mib,
            realtime_factor: self.rtf_mean(p.model.id),
            file_mib: file_mib(&p.model.path),
            active: false,
            last_load_reason: p.reason.as_str(),
            chunks_since_recycle: None,
            recycles: None,
        }));
        StatusReport {
            device,
            degraded: self.degraded,
            models,
            queues: QueueDepths {
                interactive: self.queues[Lane::Interactive.index()].len(),
                batch: self.queues[Lane::Batch.index()].len(),
                live: self.queues[Lane::Live.index()].len(),
            },
            running,
            aborted_chunks: self.aborted_total,
            last_error: self.last_error.clone(),
            recycle_failures: self.recycle_failures,
        }
    }
}

enum Event {
    Wake,
    Finished {
        state: usize,
        generation: u64,
        result: Result<ChunkOut, ChunkFailure>,
        run_secs: f64,
    },
    Loaded {
        model: ModelRef,
        generation: u64,
        result: Result<LoadReport, EngineError>,
    },
    Unloaded {
        generation: u64,
        token: u64,
    },
    Evicted {
        generation: u64,
        token: u64,
    },
    Shutdown,
}

struct Shared<B> {
    backend: Arc<B>,
    core: Mutex<Core>,
    /// Model ids an admin delete is working on; requests for them are refused.
    deleting: Mutex<Vec<String>>,
    /// False once the dispatcher thread has exited.
    alive: AtomicBool,
    /// One long-lived inference thread per state index, created on first use.
    workers: Mutex<Vec<Option<Sender<Job>>>>,
    /// Recycles whose purge destroyed no GPU device.
    recycle_leaks: Arc<AtomicU64>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

impl<B> Shared<B> {
    /// Runs `job` on the thread for `state`, respawning it once if it is gone.
    fn run_on_state_thread(&self, state: usize, mut job: Job) -> std::io::Result<()> {
        let mut workers = self.workers.lock().unwrap_or_else(|p| p.into_inner());
        if workers.len() <= state {
            workers.resize_with(state + 1, || None);
        }
        for _ in 0..2 {
            if workers[state].is_none() {
                let (jtx, jrx) = mpsc::channel::<Job>();
                std::thread::Builder::new()
                    .name(format!("hark-state-{state}"))
                    .spawn(move || {
                        for job in jrx {
                            job();
                        }
                    })?;
                workers[state] = Some(jtx);
            }
            match workers[state].as_ref().map(|w| w.send(job)) {
                Some(Ok(())) => return Ok(()),
                Some(Err(mpsc::SendError(back))) => {
                    job = back;
                    workers[state] = None;
                }
                None => unreachable!(),
            }
        }
        Err(std::io::Error::other("inference thread keeps exiting"))
    }

    fn core(&self) -> MutexGuard<'_, Core> {
        self.core.lock().unwrap_or_else(|p| p.into_inner())
    }
}

const UNAVAILABLE: &str = "scheduler unavailable";

/// One dispatcher thread owns every decision; inference and model loads run on short-lived
/// threads that report back through the event channel, so the dispatcher never blocks on them.
pub struct Scheduler<B: Backend> {
    shared: Arc<Shared<B>>,
    tx: Sender<Event>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl<B: Backend> Scheduler<B> {
    pub fn new(backend: B, cfg: SchedulerConfig) -> Scheduler<B> {
        let shared = Arc::new(Shared {
            backend: Arc::new(backend),
            core: Mutex::new(Core::new(cfg)),
            deleting: Mutex::new(Vec::new()),
            alive: AtomicBool::new(true),
            workers: Mutex::new(Vec::new()),
            recycle_leaks: Arc::new(AtomicU64::new(0)),
        });
        let (tx, rx) = mpsc::channel();
        let handle = {
            let shared = shared.clone();
            let tx = tx.clone();
            std::thread::Builder::new()
                .name("hark-scheduler".into())
                .spawn(move || dispatch_loop(shared, rx, tx))
                .expect("spawn scheduler thread")
        };
        Scheduler {
            shared,
            tx,
            thread: Mutex::new(Some(handle)),
        }
    }

    pub fn backend(&self) -> &B {
        &self.shared.backend
    }

    fn unavailable() -> EngineError {
        EngineError::Run(UNAVAILABLE.into())
    }

    fn wake(&self) -> Result<(), EngineError> {
        if self.tx.send(Event::Wake).is_err() {
            self.shared.core().fail_all(UNAVAILABLE);
            return Err(Self::unavailable());
        }
        Ok(())
    }

    fn is_deleting(&self, id: &str) -> bool {
        self.shared
            .deleting
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .any(|d| d == id)
    }

    /// While `on`, new requests for `id` get `ModelBusy` and the ones already queued fail the same way.
    pub fn set_deleting(&self, id: &str, on: bool) {
        {
            let mut list = self
                .shared
                .deleting
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            list.retain(|d| d != id);
            if on {
                list.push(id.to_string());
            }
        }
        if on {
            self.shared.core().fail_queued_for(id);
            let _ = self.tx.send(Event::Wake);
        }
    }

    /// Queues one chunk; the receiver yields its result.
    pub fn submit(&self, req: ChunkRequest) -> Result<Receiver<Reply>, EngineError> {
        if !self.shared.alive.load(Ordering::SeqCst) {
            return Err(Self::unavailable());
        }
        if self.is_deleting(req.model.id) {
            return Err(EngineError::ModelBusy(req.model.id.to_string()));
        }
        let (reply, rx) = mpsc::channel();
        self.shared
            .core()
            .submit(req, reply)
            .map_err(|b| EngineError::Busy {
                retry_after_secs: b.retry_after_secs,
            })?;
        self.wake()?;
        Ok(rx)
    }

    pub fn run_chunk(&self, req: ChunkRequest) -> Result<ChunkDone, EngineError> {
        let queued_at = Instant::now();
        let rx = self.submit(req)?;
        let mut done = rx.recv().map_err(|_| Self::unavailable())??;
        let total = queued_at.elapsed().as_secs_f64();
        done.queue_wait_secs = (total - done.run_secs).max(0.0);
        Ok(done)
    }

    /// Sets how many models stay resident and after how long an unused one is freed.
    pub fn configure_residency(&self, max_loaded: usize, idle: Option<Duration>) {
        self.shared.core().set_residency(max_loaded, idle);
        let _ = self.tx.send(Event::Wake);
    }

    /// After a CPU fallback, an idle daemon retries the GPU every `every`; `None` never.
    pub fn configure_gpu_retry(&self, every: Option<Duration>) {
        self.shared.core().set_gpu_retry(every);
        let _ = self.tx.send(Event::Wake);
    }

    fn load_with(&self, model: ModelRef, reason: LoadReason) -> Result<(), EngineError> {
        if !self.shared.alive.load(Ordering::SeqCst) {
            return Err(Self::unavailable());
        }
        if self.is_deleting(model.id) {
            return Err(EngineError::ModelBusy(model.id.to_string()));
        }
        let (reply, rx) = mpsc::channel();
        self.shared.core().submit_load(model, reason, reply);
        self.wake()?;
        rx.recv().map_err(|_| Self::unavailable())?.map(|_| ())
    }

    pub fn preload(&self, model: ModelRef) -> Result<(), EngineError> {
        self.load_with(model, LoadReason::Preload)
    }

    /// Loads `model` now (waiting for idle states like a switch).
    pub fn load_model(&self, model: ModelRef) -> Result<(), EngineError> {
        self.load_with(model, LoadReason::Admin)
    }

    /// Unloads `id` once its states are idle; `Ok(false)` when it was not resident.
    pub fn unload_model(&self, id: &str) -> Result<bool, EngineError> {
        if !self.shared.alive.load(Ordering::SeqCst) {
            return Err(Self::unavailable());
        }
        let (reply, rx) = mpsc::channel();
        self.shared.core().submit_unload(id.to_string(), reply);
        self.wake()?;
        rx.recv().map_err(|_| Self::unavailable())?
    }

    pub fn record_request(&self, model_id: &'static str, run_secs: f64, audio_secs: f64) {
        self.shared
            .core()
            .record_request(model_id, run_secs, audio_secs);
    }

    /// Recreates the GPU device after `after` decoded chunks; 0 = never.
    pub fn configure_recycle(&self, after: u64) {
        self.shared.core().set_recycle(after);
        let _ = self.tx.send(Event::Wake);
    }

    pub fn status(&self, device: DeviceStatus) -> StatusReport {
        let mut report = self.shared.core().status(device);
        report.recycle_failures += self.shared.recycle_leaks.load(Ordering::SeqCst);
        report.degraded |= !self.shared.alive.load(Ordering::SeqCst);
        report
    }

    #[cfg(test)]
    fn inject_dispatch_panic(&self) {
        let flag = self.shared.core().panic_next_dispatch.clone();
        flag.store(true, Ordering::SeqCst);
    }

    #[cfg(test)]
    fn stop_dispatcher(&self) {
        let _ = self.tx.send(Event::Shutdown);
        let handle = self.thread.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(h) = handle {
            let _ = h.join();
        }
    }
}

impl<B: Backend> Drop for Scheduler<B> {
    fn drop(&mut self) {
        let _ = self.tx.send(Event::Shutdown);
        let handle = self.thread.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(h) = handle {
            let _ = h.join();
        }
    }
}

/// Marks the scheduler unavailable when the dispatcher thread exits for any reason.
struct AliveGuard<'a>(&'a AtomicBool);

impl Drop for AliveGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

fn handle_event<B: Backend>(shared: &Arc<Shared<B>>, tx: &Sender<Event>, ev: Event) {
    let actions = {
        let mut core = shared.core();
        match ev {
            Event::Shutdown | Event::Wake => {}
            Event::Finished {
                state,
                generation,
                result,
                run_secs,
            } => {
                if generation == core.generation {
                    core.on_finished(state, result, run_secs);
                }
            }
            Event::Loaded {
                model,
                generation,
                result,
            } => {
                if generation == core.generation {
                    core.on_loaded(model, result);
                }
            }
            Event::Unloaded { generation, token } => {
                if generation == core.generation {
                    core.on_unloaded();
                    core.flush_waiters(token);
                }
            }
            Event::Evicted { generation, token } => {
                if generation == core.generation {
                    core.flush_waiters(token);
                }
            }
        }
        core.dispatch()
    };
    for action in actions {
        execute(shared, tx, action);
    }
}

fn dispatch_loop<B: Backend>(shared: Arc<Shared<B>>, rx: Receiver<Event>, tx: Sender<Event>) {
    let _alive = AliveGuard(&shared.alive);
    loop {
        let wait = shared.core().next_deadline(Instant::now());
        let ev = match wait {
            Some(d) => match rx.recv_timeout(d) {
                Ok(ev) => ev,
                Err(mpsc::RecvTimeoutError::Timeout) => Event::Wake,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            },
            None => match rx.recv() {
                Ok(ev) => ev,
                Err(_) => break,
            },
        };
        if matches!(ev, Event::Shutdown) {
            break;
        }
        let handled = catch_unwind(AssertUnwindSafe(|| handle_event(&shared, &tx, ev)));
        if let Err(p) = handled {
            let what = panic_text(p.as_ref());
            tracing::error!(panic = %what, "scheduler event panicked; failing waiters and resetting");
            let _ = catch_unwind(AssertUnwindSafe(|| {
                shared.core().reset_after_panic(&what);
            }));
        }
    }
}

/// Returns freed glibc heap to the OS (glibc only); RSS swung about 7 % around its mean without it.
pub fn trim_heap() {
    crate::sys::trim_heap();
}

fn execute<B: Backend>(shared: &Arc<Shared<B>>, tx: &Sender<Event>, action: Action) {
    let backend = shared.backend.clone();
    let tx = tx.clone();
    match action {
        Action::Start {
            state,
            task,
            abort,
            generation,
        } => {
            let job: Job = Box::new({
                let tx = tx.clone();
                move || {
                    let started = Instant::now();
                    let result =
                        catch_unwind(AssertUnwindSafe(|| backend.run(state, &task, &abort)))
                            .unwrap_or_else(|_| {
                                Err(ChunkFailure::Failed("inference thread panicked".into()))
                            });
                    drop(task);
                    trim_heap();
                    let _ = tx.send(Event::Finished {
                        state,
                        generation,
                        result,
                        run_secs: started.elapsed().as_secs_f64(),
                    });
                }
            });
            if let Err(e) = shared.run_on_state_thread(state, job) {
                let _ = tx.send(Event::Finished {
                    state,
                    generation,
                    result: Err(ChunkFailure::Failed(format!("cannot start thread: {e}"))),
                    run_secs: 0.0,
                });
            }
        }
        Action::Load {
            model,
            states,
            prep,
            evict,
            generation,
            gpu_retry,
            recycle,
        } => {
            let tx2 = tx.clone();
            let model2 = model.clone();
            let leaks = shared.recycle_leaks.clone();
            let spawned = std::thread::Builder::new()
                .name("hark-load".into())
                .spawn(move || {
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        let released_before = backend.devices_released();
                        match prep {
                            LoadPrep::PurgeAll => backend.unload(),
                            LoadPrep::Park => backend.park(),
                            LoadPrep::Keep => {}
                        }
                        if recycle
                            && let (Some(b), Some(a)) =
                                (released_before, backend.devices_released())
                            && a <= b
                        {
                            leaks.fetch_add(1, Ordering::SeqCst);
                            tracing::warn!("device recycle released nothing");
                        }
                        for id in &evict {
                            backend.evict(id);
                        }
                        if gpu_retry {
                            backend.try_gpu_again();
                        }
                        backend.load(&model2, states)
                    }))
                    .unwrap_or_else(|_| Err(EngineError::Load("loader panicked".into())));
                    let _ = tx2.send(Event::Loaded {
                        model: model2,
                        generation,
                        result,
                    });
                });
            if let Err(e) = spawned {
                let _ = tx.send(Event::Loaded {
                    model,
                    generation,
                    result: Err(EngineError::Load(format!("cannot start thread: {e}"))),
                });
            }
        }
        Action::Unload { generation, token } => {
            let tx2 = tx.clone();
            let spawned = std::thread::Builder::new()
                .name("hark-unload".into())
                .spawn(move || {
                    let _ = catch_unwind(AssertUnwindSafe(|| backend.drop_active()));
                    let _ = tx2.send(Event::Unloaded { generation, token });
                });
            if spawned.is_err() {
                let _ = tx.send(Event::Unloaded { generation, token });
            }
        }
        Action::Evict {
            ids,
            generation,
            token,
        } => {
            let tx2 = tx.clone();
            let spawned = std::thread::Builder::new()
                .name("hark-evict".into())
                .spawn(move || {
                    let _ = catch_unwind(AssertUnwindSafe(|| {
                        for id in &ids {
                            backend.evict(id);
                        }
                    }));
                    let _ = tx2.send(Event::Evicted { generation, token });
                });
            if spawned.is_err() {
                let _ = tx.send(Event::Evicted { generation, token });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn model(id: &'static str) -> ModelRef {
        ModelRef {
            id,
            path: format!("/models/{id}.bin").into(),
        }
    }

    fn task() -> ChunkTask {
        ChunkTask {
            audio: Arc::new(vec![0.0; 16_000]),
            span: Span {
                start: 0,
                end: 16_000,
            },
            pieces: Arc::new(Vec::new()),
            opts: Arc::new(DecodeOptions::default()),
            language: None,
        }
    }

    fn req(id: &str, lane: Lane, m: &'static str, chunk: usize, admit: bool) -> ChunkRequest {
        ChunkRequest {
            request_id: id.into(),
            lane,
            model: model(m),
            chunk,
            task: task(),
            admit,
            last: true,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    fn ready_core(states: usize, max_waiting: usize) -> Core {
        let mut core = Core::new(SchedulerConfig {
            states,
            max_interactive_waiting: max_waiting,
        });
        core.on_loaded(
            model("m"),
            Ok(LoadReport {
                load_ms: 1,
                states,
                vram_per_state_mib: Some(100),
                fell_back: None,
            }),
        );
        core
    }

    fn put(core: &mut Core, id: &str, lane: Lane, chunk: usize) -> Receiver<Reply> {
        let (tx, rx) = mpsc::channel();
        core.submit(req(id, lane, "m", chunk, chunk == 0), tx)
            .expect("admitted");
        rx
    }

    fn started(actions: &[Action]) -> Vec<usize> {
        actions
            .iter()
            .filter_map(|a| match a {
                Action::Start { state, .. } => Some(*state),
                _ => None,
            })
            .collect()
    }

    fn running_ids(core: &Core) -> Vec<Option<(String, bool)>> {
        core.slots
            .iter()
            .map(|s| match s {
                Slot::Running(r) => Some((r.item.request_id.clone(), r.aborting)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn interactive_is_always_next_and_live_never_runs() {
        let mut core = ready_core(1, 16);
        put(&mut core, "live", Lane::Live, 0);
        put(&mut core, "b1", Lane::Batch, 0);
        put(&mut core, "i1", Lane::Interactive, 0);
        let a = core.dispatch();
        assert_eq!(started(&a), vec![0]);
        assert_eq!(running_ids(&core)[0].as_ref().unwrap().0, "i1");
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        put(&mut core, "i2", Lane::Interactive, 0);
        core.dispatch();
        assert_eq!(running_ids(&core)[0].as_ref().unwrap().0, "i2");
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        core.dispatch();
        assert_eq!(running_ids(&core)[0].as_ref().unwrap().0, "b1");
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        assert!(core.dispatch().is_empty());
        let s = core.status(DeviceStatus::default());
        assert_eq!(
            (s.queues.live, s.queues.interactive, s.queues.batch),
            (1, 0, 0)
        );
        assert!(running_ids(&core)[0].is_none());
    }

    #[test]
    fn preempts_only_when_every_state_runs_batch() {
        let mut core = ready_core(2, 16);
        put(&mut core, "b1", Lane::Batch, 0);
        put(&mut core, "b2", Lane::Batch, 0);
        core.dispatch();
        assert!(
            running_ids(&core)
                .iter()
                .all(|r| r.as_ref().is_some_and(|r| !r.1))
        );
        // An idle state exists for the interactive chunk: no abort.
        let mut core2 = ready_core(2, 16);
        put(&mut core2, "b1", Lane::Batch, 0);
        core2.dispatch();
        put(&mut core2, "i1", Lane::Interactive, 0);
        core2.dispatch();
        assert!(
            running_ids(&core2)
                .iter()
                .all(|r| r.as_ref().is_none_or(|r| !r.1))
        );
        // Every state busy with batch and an interactive waits: exactly one abort.
        put(&mut core, "i1", Lane::Interactive, 0);
        core.dispatch();
        let aborting: Vec<_> = running_ids(&core)
            .into_iter()
            .flatten()
            .filter(|r| r.1)
            .collect();
        assert_eq!(aborting.len(), 1);
        // A state running interactive work blocks preemption.
        let mut core3 = ready_core(2, 16);
        put(&mut core3, "i0", Lane::Interactive, 0);
        put(&mut core3, "b1", Lane::Batch, 0);
        core3.dispatch();
        put(&mut core3, "i1", Lane::Interactive, 0);
        core3.dispatch();
        assert!(
            running_ids(&core3)
                .iter()
                .all(|r| r.as_ref().is_none_or(|r| !r.1))
        );
    }

    #[test]
    fn at_most_one_abort_per_waiting_interactive_item() {
        let mut core = ready_core(2, 16);
        put(&mut core, "b1", Lane::Batch, 0);
        put(&mut core, "b2", Lane::Batch, 0);
        core.dispatch();
        put(&mut core, "i1", Lane::Interactive, 0);
        core.dispatch();
        core.dispatch();
        let n = |c: &Core| running_ids(c).into_iter().flatten().filter(|r| r.1).count();
        assert_eq!(n(&core), 1, "repeated dispatch must not abort again");
        put(&mut core, "i2", Lane::Interactive, 0);
        core.dispatch();
        assert_eq!(n(&core), 2);
        put(&mut core, "i3", Lane::Interactive, 0);
        core.dispatch();
        assert_eq!(n(&core), 2, "no batch state left to abort");
    }

    #[test]
    fn most_recently_started_batch_chunk_is_the_victim() {
        let mut core = ready_core(2, 16);
        put(&mut core, "old", Lane::Batch, 0);
        core.dispatch();
        put(&mut core, "new", Lane::Batch, 0);
        core.dispatch();
        put(&mut core, "i1", Lane::Interactive, 0);
        core.dispatch();
        let ids = running_ids(&core);
        assert_eq!(ids[0], Some(("old".into(), false)));
        assert_eq!(ids[1], Some(("new".into(), true)));
    }

    #[test]
    fn aborted_chunk_returns_to_the_front_of_the_batch_lane() {
        let mut core = ready_core(1, 16);
        put(&mut core, "b1", Lane::Batch, 3);
        core.dispatch();
        put(&mut core, "b2", Lane::Batch, 0);
        put(&mut core, "i1", Lane::Interactive, 0);
        let a = core.dispatch();
        assert!(a.is_empty());
        let flag = match &core.slots[0] {
            Slot::Running(r) => r.abort.load(Ordering::SeqCst),
            _ => unreachable!(),
        };
        assert!(flag);
        core.on_finished(0, Err(ChunkFailure::Aborted), 0.0);
        let a = core.dispatch();
        assert_eq!(started(&a), vec![0]);
        assert_eq!(running_ids(&core)[0].as_ref().unwrap().0, "i1");
        let front = core.queues[Lane::Batch.index()].front().unwrap();
        assert_eq!((front.request_id.as_str(), front.chunk), ("b1", 3));
        match &front.work {
            Work::Chunk(t) => assert_eq!(t.span.start, 0),
            _ => unreachable!(),
        }
        assert_eq!(core.status(DeviceStatus::default()).aborted_chunks, 1);
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        core.dispatch();
        assert_eq!(running_ids(&core)[0].as_ref().unwrap().0, "b1");
    }

    #[test]
    fn full_interactive_queue_is_refused_with_retry_after() {
        let mut core = ready_core(1, 2);
        put(&mut core, "run", Lane::Batch, 0);
        core.dispatch();
        put(&mut core, "i1", Lane::Interactive, 0);
        put(&mut core, "i2", Lane::Interactive, 0);
        let (tx, _rx) = mpsc::channel();
        let err = core
            .submit(req("i3", Lane::Interactive, "m", 0, true), tx)
            .unwrap_err();
        assert!(err.retry_after_secs >= 1);
        let (tx, _rx) = mpsc::channel();
        core.submit(req("i2", Lane::Interactive, "m", 1, false), tx)
            .expect("later chunks of an admitted request are never refused");
        let (tx, _rx) = mpsc::channel();
        core.submit(req("b2", Lane::Batch, "m", 0, true), tx)
            .expect("batch is not capped");
        assert_eq!(core.queues[Lane::Interactive.index()].len(), 3);
    }

    #[test]
    fn model_switch_waits_for_running_chunks_then_loads_once() {
        let mut core = ready_core(2, 16);
        put(&mut core, "a", Lane::Batch, 0);
        core.dispatch();
        let (tx, _rx) = mpsc::channel();
        core.submit(req("other", Lane::Interactive, "n", 0, true), tx)
            .unwrap();
        assert!(core.dispatch().is_empty(), "must drain first");
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        let a = core.dispatch();
        assert!(matches!(a.as_slice(), [Action::Load { model, states: 2, .. }] if model.id == "n"));
        assert!(core.dispatch().is_empty(), "loading blocks dispatch");
        assert!(core.status(DeviceStatus::default()).models.is_empty());
        core.on_loaded(
            model("n"),
            Ok(LoadReport {
                load_ms: 5,
                states: 2,
                vram_per_state_mib: None,
                fell_back: None,
            }),
        );
        assert_eq!(started(&core.dispatch()), vec![0]);
    }

    #[test]
    fn failed_load_errors_every_queued_request_for_that_model() {
        let mut core = Core::new(SchedulerConfig {
            states: 1,
            max_interactive_waiting: 4,
        });
        let rx = put(&mut core, "a", Lane::Interactive, 0);
        assert!(matches!(core.dispatch().as_slice(), [Action::Load { .. }]));
        core.on_loaded(model("m"), Err(EngineError::Load("boom".into())));
        assert!(matches!(rx.recv().unwrap(), Err(EngineError::Load(_))));
        assert!(core.status(DeviceStatus::default()).last_error.is_some());
    }

    #[test]
    fn rtf_is_a_rolling_mean_of_the_last_50() {
        let mut core = ready_core(1, 4);
        for _ in 0..50 {
            core.record_request("m", 1.0, 10.0);
        }
        for _ in 0..50 {
            core.record_request("m", 2.0, 10.0);
        }
        let s = core.status(DeviceStatus::default());
        assert!((s.models[0].realtime_factor.unwrap() - 0.2).abs() < 1e-9);
    }

    #[test]
    fn rfc3339_formats_known_instants() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_utc(1_792_000_000), "2026-10-14T17:46:40Z");
    }

    struct SleepBackend {
        aborts_seen: std::sync::atomic::AtomicUsize,
    }

    impl Backend for SleepBackend {
        fn load(&self, _m: &ModelRef, states: usize) -> Result<LoadReport, EngineError> {
            Ok(LoadReport {
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
            task: &ChunkTask,
            abort: &AtomicBool,
        ) -> Result<ChunkOut, ChunkFailure> {
            let ms = task.span.end as u64 / 16;
            let end = Instant::now() + Duration::from_millis(ms);
            while Instant::now() < end {
                if abort.load(Ordering::SeqCst) {
                    self.aborts_seen.fetch_add(1, Ordering::SeqCst);
                    return Err(ChunkFailure::Aborted);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            Ok(ChunkOut::default())
        }
    }

    #[test]
    fn threaded_scheduler_preempts_batch_and_finishes_everything() {
        let sched = Scheduler::new(
            SleepBackend {
                aborts_seen: Default::default(),
            },
            SchedulerConfig {
                states: 1,
                max_interactive_waiting: 4,
            },
        );
        let mk = |id: &str, lane, ms: usize, chunk, admit| ChunkRequest {
            request_id: id.into(),
            lane,
            model: model("m"),
            chunk,
            task: ChunkTask {
                audio: Arc::new(Vec::new()),
                span: Span {
                    start: 0,
                    end: ms * 16,
                },
                pieces: Arc::new(Vec::new()),
                opts: Arc::new(DecodeOptions::default()),
                language: None,
            },
            admit,
            last: true,
            cancel: Arc::new(AtomicBool::new(false)),
        };
        sched.preload(model("m")).unwrap();
        let batch = sched.submit(mk("b", Lane::Batch, 1500, 0, true)).unwrap();
        std::thread::sleep(Duration::from_millis(100));
        let t = Instant::now();
        let inter = sched
            .run_chunk(mk("i", Lane::Interactive, 50, 0, true))
            .unwrap();
        assert!(
            t.elapsed() < Duration::from_millis(500),
            "{:?}",
            t.elapsed()
        );
        assert!(inter.queue_wait_secs < 0.5);
        assert!(batch.recv_timeout(Duration::from_secs(10)).unwrap().is_ok());
        assert_eq!(sched.backend().aborts_seen.load(Ordering::SeqCst), 1);
        let s = sched.status(DeviceStatus::default());
        assert_eq!(s.aborted_chunks, 1);
        assert_eq!(s.models[0].id, "m");
    }

    fn put_full(
        core: &mut Core,
        id: &str,
        lane: Lane,
        chunk: usize,
        last: bool,
    ) -> Receiver<Reply> {
        let (tx, rx) = mpsc::channel();
        let mut r = req(id, lane, "m", chunk, chunk == 0);
        r.last = last;
        core.submit(r, tx).expect("admitted");
        rx
    }

    fn put_model(
        core: &mut Core,
        id: &str,
        lane: Lane,
        m: &'static str,
        last: bool,
    ) -> Receiver<Reply> {
        let (tx, rx) = mpsc::channel();
        let mut r = req(id, lane, m, 0, true);
        r.last = last;
        core.submit(r, tx).expect("admitted");
        rx
    }

    fn aborting(core: &Core) -> usize {
        running_ids(core)
            .into_iter()
            .flatten()
            .filter(|r| r.1)
            .count()
    }

    #[test]
    fn interactive_for_another_model_aborts_every_running_batch_chunk() {
        let mut core = ready_core(2, 16);
        put(&mut core, "b1", Lane::Batch, 0);
        put(&mut core, "b2", Lane::Batch, 0);
        core.dispatch();
        put_model(&mut core, "i", Lane::Interactive, "n", true);
        assert!(core.dispatch().is_empty());
        assert_eq!(aborting(&core), 2, "both batch chunks abort at once");
        core.on_finished(0, Err(ChunkFailure::Aborted), 0.0);
        assert!(core.dispatch().is_empty(), "one chunk still draining");
        core.on_finished(1, Err(ChunkFailure::Aborted), 0.0);
        let a = core.dispatch();
        assert!(matches!(a.as_slice(), [Action::Load { model, .. }] if model.id == "n"));
        assert_eq!(core.queues[Lane::Batch.index()].len(), 2);
    }

    #[test]
    fn batch_for_the_old_model_does_not_switch_back_while_interactive_is_between_chunks() {
        let mut core = ready_core(1, 16);
        put(&mut core, "b1", Lane::Batch, 0);
        core.dispatch();
        put_model(&mut core, "i", Lane::Interactive, "n", false);
        core.dispatch();
        core.on_finished(0, Err(ChunkFailure::Aborted), 0.0);
        assert!(matches!(core.dispatch().as_slice(), [Action::Load { .. }]));
        core.on_loaded(
            model("n"),
            Ok(LoadReport {
                load_ms: 1,
                states: 1,
                vram_per_state_mib: None,
                fell_back: None,
            }),
        );
        assert_eq!(started(&core.dispatch()), vec![0]);
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        assert!(
            core.dispatch().is_empty(),
            "no switch back to the batch model between interactive chunks"
        );
        let (tx, _rx) = mpsc::channel();
        let mut r = req("i", Lane::Interactive, "n", 1, false);
        r.last = true;
        core.submit(r, tx).unwrap();
        assert_eq!(started(&core.dispatch()), vec![0]);
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        let a = core.dispatch();
        assert!(matches!(a.as_slice(), [Action::Load { model, .. }] if model.id == "m"));
    }

    #[test]
    fn aborts_count_only_interactive_chunks_runnable_on_the_loaded_model() {
        let mut core = ready_core(2, 16);
        put(&mut core, "b1", Lane::Batch, 0);
        put(&mut core, "b2", Lane::Batch, 0);
        core.dispatch();
        put(&mut core, "i1", Lane::Interactive, 0);
        put_model(&mut core, "other", Lane::Interactive, "n", true);
        core.dispatch();
        assert_eq!(aborting(&core), 1, "the other model's chunk needs no abort");
    }

    #[test]
    fn a_three_chunk_interactive_request_keeps_its_state_and_aborts_batch_once() {
        let mut core = ready_core(1, 16);
        put(&mut core, "b1", Lane::Batch, 0);
        put(&mut core, "b2", Lane::Batch, 0);
        core.dispatch();
        put_full(&mut core, "i", Lane::Interactive, 0, false);
        core.dispatch();
        core.on_finished(0, Err(ChunkFailure::Aborted), 0.0);
        assert_eq!(started(&core.dispatch()), vec![0]);
        for chunk in 1..3 {
            core.on_finished(0, Ok(ChunkOut::default()), 0.0);
            assert!(
                core.dispatch().is_empty(),
                "the freed state must not go to batch before chunk {chunk}"
            );
            put_full(&mut core, "i", Lane::Interactive, chunk, chunk == 2);
            assert_eq!(started(&core.dispatch()), vec![0]);
            assert_eq!(running_ids(&core)[0].as_ref().unwrap().0, "i");
        }
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        assert_eq!(started(&core.dispatch()), vec![0]);
        assert_eq!(running_ids(&core)[0].as_ref().unwrap().0, "b1");
        assert_eq!(core.status(DeviceStatus::default()).aborted_chunks, 1);
    }

    #[test]
    fn a_reserved_state_is_released_on_timeout_and_on_cancel() {
        let mut core = ready_core(1, 16);
        core.reserve_for = Duration::from_millis(10);
        put(&mut core, "b1", Lane::Batch, 0);
        put_full(&mut core, "i", Lane::Interactive, 0, false);
        core.dispatch();
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        assert!(core.dispatch().is_empty());
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(
            started(&core.dispatch()),
            vec![0],
            "timeout frees the state"
        );

        let mut core = ready_core(1, 16);
        put(&mut core, "b1", Lane::Batch, 0);
        let (tx, _rx) = mpsc::channel();
        let mut r = req("i", Lane::Interactive, "m", 0, true);
        r.last = false;
        let cancel = r.cancel.clone();
        core.submit(r, tx).unwrap();
        core.dispatch();
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        assert!(core.dispatch().is_empty());
        cancel.store(true, Ordering::SeqCst);
        assert_eq!(started(&core.dispatch()), vec![0], "cancel frees the state");
    }

    #[test]
    fn a_burst_of_first_chunks_with_idle_states_is_admitted() {
        let mut core = ready_core(3, 2);
        for i in 0..3 {
            put(&mut core, &format!("i{i}"), Lane::Interactive, 0);
        }
        // Preload entries and later chunks never count against the cap.
        let mut core = ready_core(1, 1);
        put(&mut core, "run", Lane::Batch, 0);
        core.dispatch();
        for _ in 0..3 {
            core.submit_preload(model("m"), mpsc::channel().0);
        }
        let (tx, _rx) = mpsc::channel();
        core.submit(req("later", Lane::Interactive, "m", 2, false), tx)
            .unwrap();
        put(&mut core, "first", Lane::Interactive, 0);
        let (tx, _rx) = mpsc::channel();
        assert!(
            core.submit(req("second", Lane::Interactive, "m", 0, true), tx)
                .is_err()
        );
    }

    #[test]
    fn cancelled_work_is_dropped_and_a_running_chunk_is_aborted() {
        let mut core = ready_core(1, 16);
        let rx1 = put(&mut core, "b1", Lane::Batch, 0);
        core.dispatch();
        let (tx, rx2) = mpsc::channel();
        let r = req("b2", Lane::Batch, "m", 0, true);
        let c2 = r.cancel.clone();
        core.submit(r, tx).unwrap();
        c2.store(true, Ordering::SeqCst);
        core.dispatch();
        assert_eq!(core.queues[Lane::Batch.index()].len(), 0);
        assert!(matches!(rx2.try_recv(), Ok(Err(EngineError::Cancelled))));
        let flag = match &core.slots[0] {
            Slot::Running(r) => r.item.cancel.clone(),
            _ => unreachable!(),
        };
        flag.store(true, Ordering::SeqCst);
        core.dispatch();
        assert_eq!(aborting(&core), 1);
        core.on_finished(0, Err(ChunkFailure::Aborted), 0.0);
        assert!(matches!(rx1.try_recv(), Ok(Err(EngineError::Cancelled))));
        assert_eq!(core.status(DeviceStatus::default()).aborted_chunks, 0);
        assert!(core.dispatch().is_empty());
    }

    #[test]
    fn a_lost_state_retries_the_chunk_once_on_another_state() {
        let mut core = ready_core(2, 16);
        let rx = put(&mut core, "a", Lane::Interactive, 0);
        core.dispatch();
        core.on_finished(0, Err(ChunkFailure::StateLost("-7".into())), 0.0);
        assert!(
            rx.try_recv().is_err(),
            "no reply yet: the chunk was retried"
        );
        assert_eq!(started(&core.dispatch()), vec![1]);
        core.on_finished(1, Ok(ChunkOut::default()), 0.0);
        assert!(rx.try_recv().unwrap().is_ok());
        let s = core.status(DeviceStatus::default());
        assert_eq!(s.models[0].states, 1);
        assert!(s.last_error.unwrap().message.contains("retired"));

        let mut core = ready_core(2, 16);
        let rx = put(&mut core, "a", Lane::Interactive, 0);
        core.dispatch();
        core.on_finished(0, Err(ChunkFailure::StateLost("-7".into())), 0.0);
        core.dispatch();
        core.on_finished(1, Err(ChunkFailure::StateLost("-7".into())), 0.0);
        assert!(
            rx.try_recv().unwrap().is_err(),
            "second loss fails the chunk"
        );
    }

    struct Fake {
        loads: std::sync::atomic::AtomicUsize,
        unloads: std::sync::atomic::AtomicUsize,
        runs: std::sync::atomic::AtomicUsize,
        panic_unload_once: AtomicBool,
        lose_first_runs: std::sync::atomic::AtomicUsize,
        in_run: std::sync::atomic::AtomicUsize,
        unload_during_run: std::sync::atomic::AtomicUsize,
    }

    struct InRun<'a>(&'a std::sync::atomic::AtomicUsize);

    impl Drop for InRun<'_> {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl Fake {
        fn new() -> Fake {
            Fake {
                loads: Default::default(),
                unloads: Default::default(),
                runs: Default::default(),
                panic_unload_once: AtomicBool::new(false),
                lose_first_runs: Default::default(),
                in_run: Default::default(),
                unload_during_run: Default::default(),
            }
        }
    }

    impl Backend for Fake {
        fn load(&self, _m: &ModelRef, states: usize) -> Result<LoadReport, EngineError> {
            self.loads.fetch_add(1, Ordering::SeqCst);
            Ok(LoadReport {
                load_ms: 1,
                states,
                vram_per_state_mib: None,
                fell_back: None,
            })
        }
        fn unload(&self) {
            self.unloads.fetch_add(1, Ordering::SeqCst);
            if self.in_run.load(Ordering::SeqCst) > 0 {
                self.unload_during_run.fetch_add(1, Ordering::SeqCst);
            }
            if self.panic_unload_once.swap(false, Ordering::SeqCst) {
                panic!("unload exploded");
            }
        }
        fn run(
            &self,
            _state: usize,
            task: &ChunkTask,
            abort: &AtomicBool,
        ) -> Result<ChunkOut, ChunkFailure> {
            self.runs.fetch_add(1, Ordering::SeqCst);
            self.in_run.fetch_add(1, Ordering::SeqCst);
            let _in_run = InRun(&self.in_run);
            let lost = self.lose_first_runs.load(Ordering::SeqCst);
            if lost > 0 {
                self.lose_first_runs.store(lost - 1, Ordering::SeqCst);
                return Err(ChunkFailure::StateLost("-7".into()));
            }
            let end = Instant::now() + Duration::from_millis(task.span.end as u64 / 16);
            while Instant::now() < end {
                if abort.load(Ordering::SeqCst) {
                    return Err(ChunkFailure::Aborted);
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            Ok(ChunkOut::default())
        }
    }

    fn fake_sched(states: usize) -> Scheduler<Fake> {
        Scheduler::new(
            Fake::new(),
            SchedulerConfig {
                states,
                max_interactive_waiting: 8,
            },
        )
    }

    fn chunk_ms(id: &str, lane: Lane, ms: usize) -> ChunkRequest {
        let mut r = req(id, lane, "m", 0, true);
        r.task.span = Span {
            start: 0,
            end: ms * 16,
        };
        r
    }

    const WAIT: Duration = Duration::from_secs(3);

    struct IdBackend {
        seen: Mutex<Vec<(usize, std::thread::ThreadId, String)>>,
        panic_next: AtomicBool,
    }

    impl Backend for IdBackend {
        fn load(&self, _m: &ModelRef, states: usize) -> Result<LoadReport, EngineError> {
            Ok(LoadReport {
                load_ms: 1,
                states,
                vram_per_state_mib: None,
                fell_back: None,
            })
        }
        fn unload(&self) {}
        fn run(
            &self,
            state: usize,
            _task: &ChunkTask,
            _abort: &AtomicBool,
        ) -> Result<ChunkOut, ChunkFailure> {
            let t = std::thread::current();
            self.seen
                .lock()
                .unwrap()
                .push((state, t.id(), t.name().unwrap_or("").to_string()));
            if self.panic_next.swap(false, Ordering::SeqCst) {
                panic!("run exploded");
            }
            std::thread::sleep(Duration::from_millis(10));
            Ok(ChunkOut::default())
        }
    }

    fn id_sched(states: usize) -> Scheduler<IdBackend> {
        Scheduler::new(
            IdBackend {
                seen: Mutex::new(Vec::new()),
                panic_next: AtomicBool::new(false),
            },
            SchedulerConfig {
                states,
                max_interactive_waiting: 8,
            },
        )
    }

    #[test]
    fn inference_threads_are_reused_across_chunks() {
        let sched = id_sched(2);
        sched.preload(model("m")).unwrap();
        for batch in 0..5 {
            let rxs: Vec<_> = (0..4)
                .map(|i| {
                    sched
                        .submit(chunk_ms(&format!("c{batch}-{i}"), Lane::Batch, 10))
                        .unwrap()
                })
                .collect();
            for rx in rxs {
                assert!(rx.recv_timeout(WAIT).unwrap().is_ok());
            }
        }
        let seen = sched.backend().seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 20);
        let ids: std::collections::HashSet<_> = seen.iter().map(|(_, id, _)| *id).collect();
        assert!(ids.len() <= 2, "saw {} distinct threads", ids.len());
        for (state, _, name) in &seen {
            assert_eq!(name, &format!("hark-state-{state}"));
        }
    }

    #[test]
    fn a_panicking_run_does_not_kill_its_state_thread() {
        let sched = id_sched(1);
        sched.preload(model("m")).unwrap();
        let first = sched.run_chunk(chunk_ms("ok", Lane::Batch, 10));
        assert!(first.is_ok());
        sched.backend().panic_next.store(true, Ordering::SeqCst);
        let bad = sched.run_chunk(chunk_ms("boom", Lane::Batch, 10));
        assert!(bad.is_err(), "the panicking chunk fails");
        let after = sched.run_chunk(chunk_ms("after", Lane::Batch, 10));
        assert!(after.is_ok(), "{after:?}");
        let seen = sched.backend().seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 3);
        assert_eq!(seen[0].1, seen[1].1);
        assert_eq!(seen[1].1, seen[2].1);
    }

    #[test]
    fn recycles_when_idle_after_threshold() {
        let sched = fake_sched(2);
        sched.configure_recycle(5);
        sched.preload(model("m")).unwrap();
        for i in 0..6 {
            let done = sched.run_chunk(chunk_ms(&format!("c{i}"), Lane::Batch, 5));
            assert!(done.is_ok(), "chunk {i}: {done:?}");
        }
        let s = sched.status(DeviceStatus::default());
        let m = &s.models[0];
        assert_eq!(m.recycles, Some(1));
        assert_eq!(m.chunks_since_recycle, Some(1));
        assert_eq!(m.last_load_reason, "recycle");
        assert_eq!(s.recycle_failures, 0);
        assert_eq!(sched.backend().loads.load(Ordering::SeqCst), 2);
        assert_eq!(
            sched.backend().unloads.load(Ordering::SeqCst),
            2,
            "preload purge plus one recycle purge"
        );
    }

    #[test]
    fn recycle_drains_under_continuous_load_without_failing_requests() {
        let sched = fake_sched(2);
        sched.configure_recycle(5);
        sched.preload(model("m")).unwrap();
        let rxs: Vec<_> = (0..36)
            .map(|i| {
                let lane = if i % 4 == 0 {
                    Lane::Interactive
                } else {
                    Lane::Batch
                };
                sched
                    .submit(chunk_ms(&format!("c{i}"), lane, 15))
                    .expect("admitted")
            })
            .collect();
        for (i, rx) in rxs.iter().enumerate() {
            let r = rx.recv_timeout(Duration::from_secs(20)).expect("reply");
            assert!(r.is_ok(), "chunk {i}: {:?}", r.err());
        }
        let s = sched.status(DeviceStatus::default());
        assert!(s.models[0].recycles.unwrap() >= 1, "{:?}", s.models[0]);
        assert_eq!(s.recycle_failures, 0);
        assert_eq!(
            sched.backend().unload_during_run.load(Ordering::SeqCst),
            0,
            "a recycle never unloads under a running chunk"
        );
    }

    #[test]
    fn a_hard_drain_stops_new_chunks_without_aborting_running_ones() {
        let mut core = ready_core(2, 16);
        core.set_recycle(1);
        for i in 0..4 {
            put(&mut core, &format!("b{i}"), Lane::Batch, 0);
        }
        assert_eq!(started(&core.dispatch()), vec![0, 1]);
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        assert_eq!(
            started(&core.dispatch()),
            vec![0],
            "below 2N it keeps going"
        );
        core.on_finished(1, Ok(ChunkOut::default()), 0.0);
        put(&mut core, "i", Lane::Interactive, 0);
        let a = core.dispatch();
        assert!(a.is_empty(), "draining starts nothing: {a:?}");
        assert_eq!(aborting(&core), 0, "a drain never aborts");
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        let a = core.dispatch();
        assert!(matches!(
            a.as_slice(),
            [Action::Load {
                prep: LoadPrep::PurgeAll,
                recycle: true,
                ..
            }]
        ));
        assert_eq!(core.queues[Lane::Batch.index()].len(), 1);
        assert_eq!(core.queues[Lane::Interactive.index()].len(), 1);
    }

    #[test]
    fn no_recycle_when_disabled() {
        let sched = fake_sched(2);
        sched.configure_recycle(0);
        sched.preload(model("m")).unwrap();
        for i in 0..12 {
            assert!(
                sched
                    .run_chunk(chunk_ms(&format!("c{i}"), Lane::Batch, 2))
                    .is_ok()
            );
        }
        let s = sched.status(DeviceStatus::default());
        assert_eq!(s.models[0].recycles, Some(0));
        assert_eq!(sched.backend().loads.load(Ordering::SeqCst), 1);
        assert_eq!(sched.backend().unloads.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_degraded_core_never_recycles() {
        let mut core = ready_core(1, 16);
        core.set_recycle(1);
        core.degraded = true;
        for i in 0..3 {
            put(&mut core, &format!("b{i}"), Lane::Batch, 0);
            assert_eq!(started(&core.dispatch()), vec![0]);
            core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        }
        assert!(core.dispatch().is_empty());
        assert_eq!(core.chunks_since_recycle, 0);
    }

    #[test]
    fn a_panic_in_unload_fails_the_load_instead_of_hanging_every_caller() {
        let sched = fake_sched(1);
        sched
            .backend()
            .panic_unload_once
            .store(true, Ordering::SeqCst);
        let rx = sched.submit(chunk_ms("a", Lane::Interactive, 5)).unwrap();
        match rx.recv_timeout(WAIT) {
            Ok(Err(EngineError::Load(_))) => {}
            other => panic!("expected a failed load, got {other:?}"),
        }
        let s = sched.status(DeviceStatus::default());
        assert!(s.models.is_empty());
        assert!(s.last_error.is_some());
        let rx = sched.submit(chunk_ms("b", Lane::Interactive, 5)).unwrap();
        assert!(
            rx.recv_timeout(WAIT).unwrap().is_ok(),
            "the next load works"
        );
    }

    #[test]
    fn a_dispatcher_panic_fails_every_waiter_and_the_loop_keeps_running() {
        let sched = fake_sched(1);
        sched.preload(model("m")).unwrap();
        let running = sched.submit(chunk_ms("run", Lane::Batch, 400)).unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let queued = sched.submit(chunk_ms("queued", Lane::Batch, 10)).unwrap();
        sched.inject_dispatch_panic();
        let _trigger = sched.submit(chunk_ms("trigger", Lane::Batch, 10)).unwrap();
        for (name, rx) in [("running", running), ("queued", queued)] {
            match rx.recv_timeout(WAIT) {
                Ok(Err(EngineError::Run(_))) => {}
                other => panic!("{name}: expected an internal error, got {other:?}"),
            }
        }
        let s = sched.status(DeviceStatus::default());
        assert!(s.last_error.unwrap().message.contains("panic"));
        assert!(!s.degraded, "the dispatcher recovered");
        let rx = sched
            .submit(chunk_ms("after", Lane::Interactive, 5))
            .unwrap();
        assert!(rx.recv_timeout(WAIT).unwrap().is_ok());
    }

    #[test]
    fn a_model_being_deleted_refuses_new_and_queued_requests_with_model_busy() {
        let sched = fake_sched(1);
        sched.preload(model("m")).unwrap();
        let running = sched.submit(chunk_ms("run", Lane::Batch, 300)).unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let queued = sched.submit(chunk_ms("queued", Lane::Batch, 10)).unwrap();
        sched.set_deleting("m", true);
        assert!(matches!(
            sched.submit(chunk_ms("new", Lane::Interactive, 5)),
            Err(EngineError::ModelBusy(id)) if id == "m"
        ));
        assert!(matches!(
            queued.recv_timeout(WAIT).unwrap(),
            Err(EngineError::ModelBusy(_))
        ));
        assert!(
            running.recv_timeout(WAIT).unwrap().is_ok(),
            "running work finishes"
        );
        assert!(matches!(
            sched.load_model(model("m")),
            Err(EngineError::ModelBusy(_))
        ));
        sched.set_deleting("m", false);
        let rx = sched
            .submit(chunk_ms("after", Lane::Interactive, 5))
            .unwrap();
        assert!(rx.recv_timeout(WAIT).unwrap().is_ok());
    }

    #[test]
    fn a_gone_dispatcher_is_an_error_not_a_hang_and_status_is_degraded() {
        let sched = fake_sched(1);
        sched.preload(model("m")).unwrap();
        sched.stop_dispatcher();
        match sched.submit(chunk_ms("a", Lane::Interactive, 5)) {
            Err(EngineError::Run(m)) => assert_eq!(m, "scheduler unavailable"),
            other => panic!("expected an immediate error, got {:?}", other.map(|_| ())),
        }
        assert!(sched.status(DeviceStatus::default()).degraded);
        let (done_tx, done_rx) = mpsc::channel();
        std::thread::scope(|sc| {
            sc.spawn(|| {
                let _ = done_tx.send(sched.preload(model("m")));
            });
            match done_rx.recv_timeout(WAIT) {
                Ok(Err(EngineError::Run(m))) => assert_eq!(m, "scheduler unavailable"),
                other => panic!("preload: {other:?}"),
            }
        });
    }

    #[test]
    fn losing_the_last_state_unloads_the_model_so_the_next_request_reloads_it() {
        let sched = fake_sched(1);
        sched.backend().lose_first_runs.store(1, Ordering::SeqCst);
        let rx = sched.submit(chunk_ms("a", Lane::Interactive, 5)).unwrap();
        assert!(rx.recv_timeout(WAIT).unwrap().is_err());
        let deadline = Instant::now() + WAIT;
        while sched.backend().unloads.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            sched.backend().unloads.load(Ordering::SeqCst),
            2,
            "unloaded after the loss"
        );
        let rx = sched.submit(chunk_ms("b", Lane::Interactive, 5)).unwrap();
        assert!(rx.recv_timeout(WAIT).unwrap().is_ok());
        assert_eq!(sched.backend().loads.load(Ordering::SeqCst), 2);
        assert_eq!(sched.status(DeviceStatus::default()).models[0].states, 1);
    }

    fn put_for(core: &mut Core, id: &str, m: &'static str) -> Receiver<Reply> {
        let (tx, rx) = mpsc::channel();
        core.submit(req(id, Lane::Interactive, m, 0, true), tx)
            .expect("admitted");
        rx
    }

    fn report(states: usize) -> LoadReport {
        LoadReport {
            load_ms: 7,
            states,
            vram_per_state_mib: None,
            fell_back: None,
        }
    }

    fn load_action(actions: &[Action]) -> Option<(&'static str, LoadPrep, Vec<&'static str>)> {
        actions.iter().find_map(|a| match a {
            Action::Load {
                model, prep, evict, ..
            } => Some((model.id, *prep, evict.clone())),
            _ => None,
        })
    }

    /// Serves one request for `m` on a core: load if needed, run, finish.
    fn serve_one(
        core: &mut Core,
        m: &'static str,
    ) -> Option<(&'static str, LoadPrep, Vec<&'static str>)> {
        std::thread::sleep(Duration::from_millis(3));
        let rx = put_for(core, &format!("r-{m}"), m);
        let mut load = None;
        let mut actions = core.dispatch();
        if let Some(l) = load_action(&actions) {
            load = Some(l);
            core.on_loaded(model(m), Ok(report(1)));
            actions = core.dispatch();
        }
        assert_eq!(started(&actions), vec![0], "chunk for {m} starts");
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        assert!(rx.try_recv().unwrap().is_ok());
        load
    }

    fn resident(core: &Core) -> Vec<(String, bool)> {
        core.status(DeviceStatus::default())
            .models
            .into_iter()
            .map(|m| (m.id, m.active))
            .collect()
    }

    #[test]
    fn one_resident_model_by_default_evicts_the_previous_one() {
        let mut core = Core::new(SchedulerConfig {
            states: 1,
            max_interactive_waiting: 16,
        });
        assert_eq!(serve_one(&mut core, "a").unwrap().1, LoadPrep::PurgeAll);
        let (id, prep, evict) = serve_one(&mut core, "b").unwrap();
        assert_eq!((id, prep, evict), ("b", LoadPrep::Park, vec!["a"]));
        assert_eq!(resident(&core), vec![("b".to_string(), true)]);
    }

    #[test]
    fn least_recently_used_model_is_evicted_beyond_max_loaded_models() {
        let mut core = Core::new(SchedulerConfig {
            states: 1,
            max_interactive_waiting: 16,
        });
        core.set_residency(2, None);
        assert_eq!(serve_one(&mut core, "a").unwrap().1, LoadPrep::Keep);
        assert_eq!(
            serve_one(&mut core, "b").unwrap(),
            ("b", LoadPrep::Park, vec![])
        );
        assert_eq!(
            resident(&core),
            vec![("b".to_string(), true), ("a".to_string(), false)]
        );
        let (id, _, evict) = serve_one(&mut core, "c").unwrap();
        assert_eq!(
            (id, evict),
            ("c", vec!["a"]),
            "a is the least recently used"
        );
        assert_eq!(
            resident(&core),
            vec![("c".to_string(), true), ("b".to_string(), false)]
        );
        let (id, _, evict) = serve_one(&mut core, "b").unwrap();
        assert_eq!((id, evict), ("b", vec![]), "reactivating b evicts nothing");
        assert_eq!(
            resident(&core),
            vec![("b".to_string(), true), ("c".to_string(), false)]
        );
    }

    #[test]
    fn a_switch_waits_for_running_states_before_evicting() {
        let mut core = ready_core(1, 16);
        core.set_residency(1, None);
        let _a = put(&mut core, "a", Lane::Interactive, 0);
        assert_eq!(started(&core.dispatch()), vec![0]);
        let _b = put_for(&mut core, "b", "other");
        let actions = core.dispatch();
        assert!(
            load_action(&actions).is_none(),
            "the running chunk finishes first"
        );
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        let actions = core.dispatch();
        assert_eq!(
            load_action(&actions),
            Some(("other", LoadPrep::Park, vec!["m"]))
        );
    }

    #[test]
    fn idle_model_is_unloaded_and_the_next_request_reports_reload_after_idle() {
        let mut core = ready_core(1, 16);
        core.set_residency(1, Some(Duration::from_millis(40)));
        assert!(core.dispatch().is_empty(), "fresh model stays");
        let wait = core
            .next_deadline(Instant::now())
            .expect("an idle deadline");
        assert!(wait <= Duration::from_millis(40), "{wait:?}");
        std::thread::sleep(Duration::from_millis(60));
        let actions = core.dispatch();
        assert!(
            actions.iter().any(|a| matches!(a, Action::Unload { .. })),
            "{actions:?}"
        );
        core.on_unloaded();
        assert!(core.status(DeviceStatus::default()).models.is_empty());
        let rx = put(&mut core, "x", Lane::Interactive, 0);
        let actions = core.dispatch();
        assert!(load_action(&actions).is_some());
        core.on_loaded(model("m"), Ok(report(1)));
        core.dispatch();
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        assert!(rx.try_recv().unwrap().is_ok());
        let m = &core.status(DeviceStatus::default()).models[0];
        assert_eq!(m.last_load_reason, "reload_after_idle");
        assert_eq!(m.load_ms, 7);
    }

    #[test]
    fn a_busy_model_is_never_idle_unloaded() {
        let mut core = ready_core(1, 16);
        core.set_residency(1, Some(Duration::from_millis(20)));
        let _rx = put(&mut core, "a", Lane::Interactive, 0);
        core.dispatch();
        std::thread::sleep(Duration::from_millis(40));
        let actions = core.dispatch();
        assert!(!actions.iter().any(|a| matches!(a, Action::Unload { .. })));
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        assert!(
            !core
                .dispatch()
                .iter()
                .any(|a| matches!(a, Action::Unload { .. })),
            "finishing a chunk restarts the idle clock"
        );
    }

    #[test]
    fn zero_idle_limit_never_unloads() {
        let mut core = ready_core(1, 16);
        core.set_residency(1, None);
        std::thread::sleep(Duration::from_millis(30));
        assert!(core.dispatch().is_empty());
        assert!(core.next_deadline(Instant::now()).is_none());
    }

    #[test]
    fn load_reasons_are_reported_per_model() {
        let mut core = Core::new(SchedulerConfig {
            states: 1,
            max_interactive_waiting: 16,
        });
        let (tx, _rx) = mpsc::channel();
        core.submit_load(model("a"), LoadReason::Preload, tx);
        assert!(load_action(&core.dispatch()).is_some());
        core.on_loaded(model("a"), Ok(report(1)));
        assert_eq!(
            core.status(DeviceStatus::default()).models[0].last_load_reason,
            "preload"
        );
        let (tx, _rx) = mpsc::channel();
        core.submit_load(model("b"), LoadReason::Admin, tx);
        assert!(load_action(&core.dispatch()).is_some());
        core.on_loaded(model("b"), Ok(report(1)));
        assert_eq!(
            core.status(DeviceStatus::default()).models[0].last_load_reason,
            "admin"
        );
        let (tx, _rx) = mpsc::channel();
        core.submit_load(model("a"), LoadReason::Request, tx);
        assert!(load_action(&core.dispatch()).is_some());
        core.on_loaded(model("a"), Ok(report(1)));
        assert_eq!(
            core.status(DeviceStatus::default()).models[0].last_load_reason,
            "request"
        );
    }

    #[test]
    fn admin_unload_waits_for_idle_states_then_answers() {
        let mut core = ready_core(1, 16);
        let _rx = put(&mut core, "a", Lane::Interactive, 0);
        core.dispatch();
        let (tx, answer) = mpsc::channel();
        core.submit_unload("m".into(), tx);
        let actions = core.dispatch();
        assert!(
            !actions.iter().any(|a| matches!(a, Action::Unload { .. })),
            "a running state blocks the unload"
        );
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        let actions = core.dispatch();
        let token = actions
            .iter()
            .find_map(|a| match a {
                Action::Unload { token, .. } => Some(*token),
                _ => None,
            })
            .expect("unload once idle");
        assert!(answer.try_recv().is_err(), "answered only when done");
        core.on_unloaded();
        core.flush_waiters(token);
        assert!(matches!(answer.try_recv(), Ok(Ok(true))));
        assert!(core.status(DeviceStatus::default()).models.is_empty());
    }

    #[test]
    fn admin_unload_of_a_parked_model_evicts_it_and_of_an_unknown_one_says_no() {
        let mut core = Core::new(SchedulerConfig {
            states: 1,
            max_interactive_waiting: 16,
        });
        core.set_residency(2, None);
        serve_one(&mut core, "a");
        serve_one(&mut core, "b");
        let (tx, answer) = mpsc::channel();
        core.submit_unload("a".into(), tx);
        let (tx2, unknown) = mpsc::channel();
        core.submit_unload("zzz".into(), tx2);
        let actions = core.dispatch();
        let (ids, token) = actions
            .iter()
            .find_map(|a| match a {
                Action::Evict { ids, token, .. } => Some((ids.clone(), *token)),
                _ => None,
            })
            .expect("evict action");
        assert_eq!(ids, vec!["a"]);
        core.flush_waiters(token);
        assert!(matches!(answer.try_recv(), Ok(Ok(true))));
        assert!(matches!(unknown.try_recv(), Ok(Ok(false))));
        assert_eq!(resident(&core), vec![("b".to_string(), true)]);
    }

    #[test]
    fn a_device_fault_reloads_on_the_cpu_retries_the_chunk_once_and_degrades() {
        let mut core = ready_core(2, 16);
        let rx = put(&mut core, "a", Lane::Interactive, 0);
        assert_eq!(started(&core.dispatch()), vec![0]);
        core.on_finished(0, Err(ChunkFailure::DeviceFault("-100".into())), 0.0);
        let actions = core.dispatch();
        let (id, prep, _) = load_action(&actions).expect("reload");
        assert_eq!((id, prep), ("m", LoadPrep::PurgeAll));
        assert!(
            started(&actions).is_empty(),
            "nothing starts during the reload"
        );
        let s = core.status(DeviceStatus::default());
        assert!(s.degraded);
        assert!(s.last_error.unwrap().message.contains("falling back"));
        assert!(rx.try_recv().is_err(), "the request is retried, not failed");
        core.on_loaded(model("m"), Ok(report(2)));
        assert_eq!(
            core.status(DeviceStatus::default()).models[0].last_load_reason,
            "device_fallback"
        );
        assert_eq!(
            started(&core.dispatch()),
            vec![0],
            "the failed chunk runs again"
        );
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        assert!(rx.try_recv().unwrap().is_ok());
    }

    #[test]
    fn a_transient_failure_retries_once_on_the_same_device_and_never_falls_back() {
        let mut core = ready_core(1, 16);
        let rx = put(&mut core, "a", Lane::Interactive, 0);
        assert_eq!(started(&core.dispatch()), vec![0]);
        core.on_finished(0, Err(ChunkFailure::Transient("-8".into())), 0.0);
        let actions = core.dispatch();
        assert!(load_action(&actions).is_none(), "no reload on the CPU");
        assert_eq!(started(&actions), vec![0], "the chunk runs again");
        assert!(rx.try_recv().is_err());
        core.on_finished(0, Err(ChunkFailure::Transient("-8 again".into())), 0.0);
        let err = rx.try_recv().unwrap().unwrap_err();
        assert!(matches!(err, EngineError::Run(m) if m.contains("again")));
        let s = core.status(DeviceStatus::default());
        assert!(!s.degraded, "a transient failure never degrades");
        assert!(load_action(&core.dispatch()).is_none());
    }

    #[test]
    fn an_idle_degraded_core_retries_the_gpu_and_clears_degraded_on_success() {
        let mut core = ready_core(1, 16);
        core.set_gpu_retry(Some(Duration::from_millis(20)));
        let rx = put(&mut core, "a", Lane::Interactive, 0);
        core.dispatch();
        core.on_finished(0, Err(ChunkFailure::DeviceFault("-100".into())), 0.0);
        core.dispatch();
        core.on_loaded(model("m"), Ok(report(1)));
        assert_eq!(started(&core.dispatch()), vec![0]);
        std::thread::sleep(Duration::from_millis(40));
        assert!(
            load_action_retry(&core.dispatch()).is_none(),
            "no retry while a chunk runs"
        );
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        assert!(rx.try_recv().unwrap().is_ok());
        let actions = core.dispatch();
        assert_eq!(load_action_retry(&actions), Some(true), "retry once idle");
        assert!(core.degraded, "still degraded until the load succeeds");
        core.on_loaded(model("m"), Ok(report(1)));
        let s = core.status(DeviceStatus::default());
        assert!(!s.degraded);
        assert_eq!(s.models[0].last_load_reason, "gpu_retry");
        std::thread::sleep(Duration::from_millis(40));
        assert!(
            load_action_retry(&core.dispatch()).is_none(),
            "no more retries"
        );
    }

    #[test]
    fn a_failed_gpu_retry_stays_degraded_and_tries_again_later() {
        let mut core = ready_core(1, 16);
        core.set_gpu_retry(Some(Duration::from_millis(20)));
        let rx = put(&mut core, "a", Lane::Interactive, 0);
        core.dispatch();
        core.on_finished(0, Err(ChunkFailure::DeviceFault("-100".into())), 0.0);
        core.dispatch();
        core.on_loaded(model("m"), Ok(report(1)));
        core.dispatch();
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        assert!(rx.try_recv().unwrap().is_ok());
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(load_action_retry(&core.dispatch()), Some(true));
        let mut failed = report(1);
        failed.fell_back = Some("GPU load of m failed; fell back to the CPU".into());
        core.on_loaded(model("m"), Ok(failed));
        assert!(core.status(DeviceStatus::default()).degraded);
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(
            load_action_retry(&core.dispatch()),
            Some(true),
            "the next attempt comes later"
        );
    }

    #[test]
    fn gpu_retry_zero_never_retries() {
        let mut core = ready_core(1, 16);
        core.set_gpu_retry(None);
        let _rx = put(&mut core, "a", Lane::Interactive, 0);
        core.dispatch();
        core.on_finished(0, Err(ChunkFailure::DeviceFault("-100".into())), 0.0);
        core.dispatch();
        core.on_loaded(model("m"), Ok(report(1)));
        core.dispatch();
        core.on_finished(0, Ok(ChunkOut::default()), 0.0);
        std::thread::sleep(Duration::from_millis(30));
        assert!(load_action_retry(&core.dispatch()).is_none());
        assert!(core.next_deadline(Instant::now()).is_none());
    }

    fn load_action_retry(actions: &[Action]) -> Option<bool> {
        actions.iter().find_map(|a| match a {
            Action::Load { gpu_retry, .. } => Some(*gpu_retry),
            _ => None,
        })
    }

    #[test]
    fn a_second_device_fault_on_the_retried_chunk_fails_the_request() {
        let mut core = ready_core(1, 16);
        let rx = put(&mut core, "a", Lane::Interactive, 0);
        core.dispatch();
        core.on_finished(0, Err(ChunkFailure::DeviceFault("first".into())), 0.0);
        assert!(load_action(&core.dispatch()).is_some());
        core.on_loaded(model("m"), Ok(report(1)));
        core.dispatch();
        core.on_finished(0, Err(ChunkFailure::DeviceFault("again".into())), 0.0);
        match rx.try_recv() {
            Ok(Err(EngineError::Run(m))) => assert_eq!(m, "again"),
            other => panic!("{other:?}"),
        }
        assert!(core.status(DeviceStatus::default()).degraded);
    }

    #[test]
    fn the_cpu_reload_waits_for_sibling_chunks_to_finish() {
        let mut core = ready_core(2, 16);
        let _a = put(&mut core, "a", Lane::Interactive, 0);
        let sibling = put(&mut core, "b", Lane::Interactive, 0);
        assert_eq!(started(&core.dispatch()), vec![0, 1]);
        core.on_finished(0, Err(ChunkFailure::DeviceFault("x".into())), 0.0);
        let actions = core.dispatch();
        assert!(load_action(&actions).is_none(), "state 1 still runs");
        core.on_finished(1, Ok(ChunkOut::default()), 0.0);
        assert!(sibling.try_recv().unwrap().is_ok());
        assert!(load_action(&core.dispatch()).is_some());
    }

    #[test]
    fn a_load_that_fell_back_marks_the_scheduler_degraded() {
        let mut core = Core::new(SchedulerConfig {
            states: 1,
            max_interactive_waiting: 16,
        });
        let _rx = put_for(&mut core, "a", "m");
        assert!(load_action(&core.dispatch()).is_some());
        core.on_loaded(
            model("m"),
            Ok(LoadReport {
                fell_back: Some("GPU load of m failed; fell back to the CPU".into()),
                ..report(1)
            }),
        );
        let s = core.status(DeviceStatus::default());
        assert!(s.degraded);
        assert!(s.last_error.unwrap().message.contains("fell back"));
    }
}
