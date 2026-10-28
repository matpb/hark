//! Persistent transcription jobs: the on-disk store, the single-job runner, expiry and restart recovery.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::audio::{self, DecodeLimits, SAMPLE_RATE};
use crate::config::{Config, expand_tilde};
use crate::engine::{DecodeOptions, JobChunk, ModelRef, Transcriber};
use crate::models::Registry;
use crate::pipeline::{self, Segment, Span, Transcription};
use crate::scheduler::rfc3339_utc;

/// Sanity cap on decoded job audio, in seconds (24 h); jobs have no sync limit.
pub const JOB_MAX_AUDIO_SECS: u64 = 24 * 3600;
const SWEEP_EVERY: Duration = Duration::from_secs(3600);
const DAY_MS: u64 = 86_400_000;
/// How long a delete waits for the runner to finish winding down a cancelled job.
const WIND_DOWN_WAIT: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Queued => "queued",
            Status::Running => "running",
            Status::Succeeded => "succeeded",
            Status::Failed => "failed",
            Status::Cancelled => "cancelled",
        }
    }

    pub fn parse(s: &str) -> Option<Status> {
        Some(match s {
            "queued" => Status::Queued,
            "running" => Status::Running,
            "succeeded" => Status::Succeeded,
            "failed" => Status::Failed,
            "cancelled" => Status::Cancelled,
            _ => return None,
        })
    }

    pub fn is_finished(self) -> bool {
        matches!(self, Status::Succeeded | Status::Failed | Status::Cancelled)
    }
}

/// The request options resolved at submit time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobOptions {
    pub model: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub vocabulary: Vec<String>,
    pub temperature: f32,
    pub vad: Option<bool>,
    pub granularities: Vec<String>,
    pub response_format: Option<String>,
}

impl JobOptions {
    pub fn decode_options(&self) -> DecodeOptions {
        DecodeOptions {
            language: self.language.clone(),
            prompt: self.prompt.clone(),
            vocabulary: self.vocabulary.clone(),
            temperature: self.temperature,
            vad: self.vad,
        }
    }
}

/// Where the audio comes from until it is decoded into `audio.f32`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Source {
    Path { path: String },
    Upload { file: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: String,
    pub status: Status,
    pub progress: f64,
    pub audio_seconds: Option<f64>,
    pub model: String,
    pub created_ms: u64,
    pub started_ms: Option<u64>,
    pub finished_ms: Option<u64>,
    pub error: Option<String>,
    pub options: JobOptions,
    pub source: Option<Source>,
    /// Chunk spans in samples; planned once, after decoding.
    pub chunks: Option<Vec<[usize; 2]>>,
    /// Language pinned by the first chunk with text.
    pub language: Option<String>,
    pub resumed_from_chunk: Option<usize>,
}

impl JobRecord {
    pub fn new(id: String, options: JobOptions, source: Source, now_ms: u64) -> JobRecord {
        JobRecord {
            id,
            status: Status::Queued,
            progress: 0.0,
            audio_seconds: None,
            model: options.model.clone(),
            created_ms: now_ms,
            started_ms: None,
            finished_ms: None,
            error: None,
            options,
            source: Some(source),
            chunks: None,
            language: None,
            resumed_from_chunk: None,
        }
    }

    pub fn view(&self) -> Value {
        let at = |ms: Option<u64>| ms.map(|m| rfc3339_utc(m / 1000));
        json!({
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "audio_seconds": self.audio_seconds,
            "model": self.model,
            "created_at": rfc3339_utc(self.created_ms / 1000),
            "started_at": at(self.started_ms),
            "finished_at": at(self.finished_ms),
            "error": self.error,
            "resumed_from_chunk": self.resumed_from_chunk,
        })
    }
}

/// One finished chunk: its segments carry absolute times.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkFile {
    pub index: usize,
    pub span: [usize; 2],
    pub language: Option<String>,
    pub detected: Option<String>,
    pub segments: Vec<Segment>,
}

/// The assembled verbose result after the repetition guard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredResult {
    pub language: Option<String>,
    pub duration: f64,
    pub temperature: f32,
    pub segments: Vec<Segment>,
    pub repetitions: usize,
}

impl StoredResult {
    pub fn transcription(&self) -> Transcription {
        Transcription {
            language: self.language.clone(),
            segments: self.segments.clone(),
            ..Default::default()
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum JobError {
    #[error("no such job")]
    NotFound,
    #[error("the job has not finished")]
    NotFinished,
    #[error("the job failed: {0}")]
    Failed(String),
    #[error("the job was cancelled")]
    Cancelled,
    #[error("too many queued jobs ({0}); wait for some to finish")]
    QueueFull(usize),
    #[error("not enough free disk space for this upload")]
    InsufficientStorage,
    #[error("{0}")]
    Io(String),
}

impl From<std::io::Error> for JobError {
    fn from(e: std::io::Error) -> Self {
        JobError::Io(e.to_string())
    }
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

const B32: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";

/// `job_` plus 16 lowercase RFC 4648 base32 characters from the OS RNG.
pub fn new_job_id() -> Result<String, JobError> {
    let mut rnd = [0u8; 10];
    crate::sys::fill_random(&mut rnd)
        .map_err(|_| JobError::Io("could not read the OS random source".into()))?;
    let mut out = String::from("job_");
    let (mut acc, mut bits) = (0u32, 0u32);
    for b in rnd {
        acc = (acc << 8) | u32::from(b);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            out.push(B32[((acc >> bits) & 31) as usize] as char);
        }
    }
    Ok(out)
}

pub fn valid_job_id(id: &str) -> bool {
    id.strip_prefix("job_")
        .is_some_and(|r| r.len() == 16 && r.bytes().all(|b| B32.contains(&b)))
}

/// Temp file in the target's directory, written and fsynced; `commit_tmp` renames it into place.
pub fn write_tmp(path: &Path, bytes: &[u8]) -> std::io::Result<PathBuf> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.tmp"));
    let mut f =
        crate::sys::private_file(OpenOptions::new().write(true).create(true).truncate(true))
            .open(&tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    Ok(tmp)
}

pub fn commit_tmp(tmp: &Path, path: &Path) -> std::io::Result<()> {
    std::fs::rename(tmp, path)?;
    if let Some(dir) = path.parent() {
        crate::sys::sync_dir(dir)?;
    }
    Ok(())
}

pub fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    commit_tmp(&write_tmp(path, bytes)?, path)
}

/// One directory per job under the jobs dir; holds an exclusive lock on `<dir>/.lock` while it lives.
pub struct JobStore {
    dir: PathBuf,
    _lock: File,
}

impl JobStore {
    pub fn open(dir: &Path) -> std::io::Result<JobStore> {
        if let Some(parent) = dir.parent() {
            std::fs::create_dir_all(parent)?;
        }
        match crate::sys::private_dir_builder().create(dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e),
        }
        let lock = crate::sys::private_file(
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false),
        )
        .open(dir.join(".lock"))?;
        if !crate::sys::try_lock_exclusive(&lock)? {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                format!(
                    "the jobs directory {} is in use by another hark process (it holds {}); stop that process or set a different jobs.dir",
                    dir.display(),
                    dir.join(".lock").display()
                ),
            ));
        }
        Ok(JobStore {
            dir: dir.to_path_buf(),
            _lock: lock,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Writes an upload into the job's own directory and returns its file name.
    pub fn stage_upload(&self, id: &str, bytes: &[u8]) -> std::io::Result<String> {
        const NAME: &str = "upload.bin";
        let job_dir = self.job_dir(id);
        Self::private_dir(&job_dir)?;
        let mut f =
            crate::sys::private_file(OpenOptions::new().write(true).create(true).truncate(true))
                .open(job_dir.join(NAME))?;
        f.write_all(bytes)?;
        f.sync_all()?;
        Ok(NAME.to_string())
    }

    /// Path of a staged upload; a name with separators, a dot prefix or NUL is refused.
    pub fn staged_path(&self, id: &str, file: &str) -> Option<PathBuf> {
        let plain = !file.is_empty()
            && !file.contains('/')
            && !file.contains('\0')
            && !file.starts_with('.');
        (plain && valid_job_id(id)).then(|| self.job_dir(id).join(file))
    }

    pub fn remove_staged(&self, id: &str, source: &Option<Source>) {
        if let Some(Source::Upload { file }) = source
            && let Some(p) = self.staged_path(id, file)
        {
            let _ = std::fs::remove_file(p);
        }
    }

    pub fn job_dir(&self, id: &str) -> PathBuf {
        self.dir.join(id)
    }

    pub fn audio_path(&self, id: &str) -> PathBuf {
        self.job_dir(id).join("audio.f32")
    }

    fn chunk_path(&self, id: &str, index: usize) -> PathBuf {
        self.job_dir(id)
            .join("chunks")
            .join(format!("{index:05}.json"))
    }

    fn private_dir(path: &Path) -> std::io::Result<()> {
        match crate::sys::private_dir_builder().create(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
            Err(e) => Err(e),
        }
    }

    pub fn write_job(&self, rec: &JobRecord) -> std::io::Result<()> {
        Self::private_dir(&self.job_dir(&rec.id))?;
        let bytes = serde_json::to_vec_pretty(rec).map_err(std::io::Error::other)?;
        atomic_write(&self.job_dir(&rec.id).join("job.json"), &bytes)
    }

    pub fn read_job(&self, id: &str) -> Result<JobRecord, String> {
        let bytes = std::fs::read(self.job_dir(id).join("job.json")).map_err(|e| e.to_string())?;
        let rec: JobRecord = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if rec.id != id {
            return Err("job.json names another job".into());
        }
        Ok(rec)
    }

    /// Job ids on disk; directories that are not job ids are ignored.
    pub fn ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| valid_job_id(n))
            .collect();
        ids.sort();
        ids
    }

    pub fn write_chunk(&self, id: &str, chunk: &ChunkFile) -> std::io::Result<()> {
        Self::private_dir(&self.job_dir(id).join("chunks"))?;
        let bytes = serde_json::to_vec(chunk).map_err(std::io::Error::other)?;
        atomic_write(&self.chunk_path(id, chunk.index), &bytes)
    }

    pub fn read_chunk(&self, id: &str, index: usize) -> std::io::Result<ChunkFile> {
        let bytes = std::fs::read(self.chunk_path(id, index))?;
        serde_json::from_slice(&bytes).map_err(std::io::Error::other)
    }

    /// Chunks finished so far: the first index without a readable chunk file.
    pub fn done_chunks(&self, id: &str) -> usize {
        (0..)
            .find(|i| self.read_chunk(id, *i).is_err())
            .unwrap_or(0)
    }

    pub fn write_audio(&self, id: &str, samples: &[f32]) -> std::io::Result<()> {
        Self::private_dir(&self.job_dir(id))?;
        let path = self.audio_path(id);
        let tmp = path.with_file_name(".audio.f32.tmp");
        let mut f = std::io::BufWriter::new(
            crate::sys::private_file(OpenOptions::new().write(true).create(true).truncate(true))
                .open(&tmp)?,
        );
        for block in samples.chunks(64 * 1024) {
            let mut buf = Vec::with_capacity(block.len() * 4);
            for s in block {
                buf.extend_from_slice(&s.to_le_bytes());
            }
            f.write_all(&buf)?;
        }
        f.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        commit_tmp(&tmp, &path)
    }

    pub fn read_audio(&self, id: &str) -> std::io::Result<Vec<f32>> {
        let bytes = std::fs::read(self.audio_path(id))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    pub fn remove_audio(&self, id: &str) {
        let _ = std::fs::remove_file(self.audio_path(id));
    }

    pub fn write_result(&self, id: &str, r: &StoredResult) -> std::io::Result<()> {
        let bytes = serde_json::to_vec(r).map_err(std::io::Error::other)?;
        atomic_write(&self.job_dir(id).join("result.json"), &bytes)
    }

    pub fn read_result(&self, id: &str) -> std::io::Result<StoredResult> {
        let bytes = std::fs::read(self.job_dir(id).join("result.json"))?;
        serde_json::from_slice(&bytes).map_err(std::io::Error::other)
    }

    pub fn delete(&self, id: &str) {
        if valid_job_id(id) {
            let _ = std::fs::remove_dir_all(self.job_dir(id));
        }
    }
}

/// Free bytes for unprivileged writes on the filesystem holding `path`.
pub fn free_bytes(path: &Path) -> std::io::Result<u64> {
    crate::sys::free_bytes(path)
}

/// Canonical path of an existing regular file under one of the canonicalized `roots`.
pub fn resolve_job_path(raw: &str, roots: &[String]) -> Option<PathBuf> {
    if raw.is_empty() || raw.contains('\0') || !Path::new(raw).is_absolute() {
        return None;
    }
    let canon = std::fs::canonicalize(raw).ok()?;
    if !std::fs::metadata(&canon).is_ok_and(|m| m.is_file()) {
        return None;
    }
    roots
        .iter()
        .filter_map(|r| std::fs::canonicalize(expand_tilde(r)).ok())
        .any(|r| canon.starts_with(&r))
        .then_some(canon)
}

struct Current {
    id: String,
    cancel: Arc<AtomicBool>,
}

struct State {
    jobs: HashMap<String, JobRecord>,
    current: Option<Current>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Removed {
    Cancelled,
    Deleted,
}

/// Why a job's run ended without a result.
enum Stop {
    /// Cancelled, deleted or the daemon is stopping; the job keeps whatever state the caller set.
    Halted,
    Failed(String),
}

pub struct JobManager {
    store: JobStore,
    cfg: Arc<Config>,
    registry: Arc<Registry>,
    engine: Arc<dyn Transcriber>,
    state: Mutex<State>,
    shutdown: AtomicBool,
    wake: Sender<()>,
    rx: Mutex<Option<Receiver<()>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl JobManager {
    /// Opens the store, loads every job on disk and requeues the ones a restart interrupted.
    pub fn open(
        cfg: Arc<Config>,
        registry: Arc<Registry>,
        engine: Arc<dyn Transcriber>,
    ) -> std::io::Result<Arc<JobManager>> {
        let store = JobStore::open(&expand_tilde(&cfg.jobs.dir))?;
        let (wake, rx) = mpsc::channel();
        let mgr = Arc::new(JobManager {
            store,
            cfg,
            registry,
            engine,
            state: Mutex::new(State {
                jobs: HashMap::new(),
                current: None,
            }),
            shutdown: AtomicBool::new(false),
            wake,
            rx: Mutex::new(Some(rx)),
            thread: Mutex::new(None),
        });
        mgr.recover();
        mgr.sweep_expired(now_ms());
        Ok(mgr)
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn recover(&self) {
        let mut st = self.state();
        for id in self.store.ids() {
            if !self.store.job_dir(&id).join("job.json").exists() {
                tracing::warn!(job = %id, "removing a job directory that never got a job.json");
                self.store.delete(&id);
                continue;
            }
            let mut rec = match self.store.read_job(&id) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(job = %id, error = %e, "skipping a corrupt job directory");
                    continue;
                }
            };
            if rec.status == Status::Running {
                rec.status = Status::Queued;
                let done = self.store.done_chunks(&id);
                if done > 0 {
                    rec.resumed_from_chunk = Some(done);
                }
                if let Err(e) = self.store.write_job(&rec) {
                    tracing::warn!(job = %id, error = %e, "could not requeue an interrupted job");
                    continue;
                }
                tracing::info!(job = %id, resumed_from_chunk = done, "interrupted job requeued");
            }
            st.jobs.insert(id, rec);
        }
    }

    /// Starts the runner thread: queued jobs one at a time, then the hourly expiry sweep.
    pub fn start(self: &Arc<Self>) {
        let Some(rx) = self.rx.lock().unwrap_or_else(|p| p.into_inner()).take() else {
            return;
        };
        let me = self.clone();
        let handle = std::thread::Builder::new()
            .name("hark-jobs".into())
            .spawn(move || me.runner(rx))
            .expect("spawn job runner");
        *self.thread.lock().unwrap_or_else(|p| p.into_inner()) = Some(handle);
        let _ = self.wake.send(());
    }

    /// Aborts the running chunk and tells the runner to exit; the job resumes at that chunk later.
    pub fn begin_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(cur) = &self.state().current {
            cur.cancel.store(true, Ordering::SeqCst);
        }
        let _ = self.wake.send(());
    }

    pub fn stop(&self) {
        self.begin_shutdown();
        let handle = self.thread.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(h) = handle {
            let _ = h.join();
        }
    }

    pub fn path_roots(&self) -> &[String] {
        &self.cfg.jobs.path_roots
    }

    pub fn get(&self, id: &str) -> Option<JobRecord> {
        self.state().jobs.get(id).cloned()
    }

    /// Newest first, optionally one status.
    pub fn list(&self, status: Option<Status>, limit: usize) -> Vec<JobRecord> {
        let mut all: Vec<JobRecord> = self
            .state()
            .jobs
            .values()
            .filter(|j| status.is_none_or(|s| j.status == s))
            .cloned()
            .collect();
        all.sort_by(|a, b| (b.created_ms, &b.id).cmp(&(a.created_ms, &a.id)));
        all.truncate(limit);
        all
    }

    /// `(queued, running)`.
    pub fn counts(&self) -> (usize, usize) {
        let st = self.state();
        let n = |s| st.jobs.values().filter(|j| j.status == s).count();
        (n(Status::Queued), n(Status::Running))
    }

    pub fn submit(
        &self,
        id: String,
        options: JobOptions,
        source: Source,
    ) -> Result<JobRecord, JobError> {
        let rec = JobRecord::new(id, options, source, now_ms());
        {
            let mut st = self.state();
            if let Err(e) = self.check_queue(&st) {
                drop(st);
                if matches!(rec.source, Some(Source::Upload { .. })) {
                    self.store.delete(&rec.id);
                }
                return Err(e);
            }
            self.store.write_job(&rec)?;
            st.jobs.insert(rec.id.clone(), rec.clone());
        }
        let _ = self.wake.send(());
        Ok(rec)
    }

    fn check_queue(&self, st: &State) -> Result<(), JobError> {
        let queued = st
            .jobs
            .values()
            .filter(|j| j.status == Status::Queued)
            .count();
        if queued >= self.cfg.jobs.max_queued {
            return Err(JobError::QueueFull(self.cfg.jobs.max_queued));
        }
        Ok(())
    }

    /// Refuses early when the queue is already full.
    pub fn ensure_queue_room(&self) -> Result<(), JobError> {
        self.check_queue(&self.state())
    }

    /// Refuses an upload of `len` bytes that would leave less than `jobs.min_free_mb` on the jobs filesystem.
    pub fn ensure_space(&self, len: u64) -> Result<(), JobError> {
        let free = free_bytes(self.store.dir())?;
        let keep = self.cfg.jobs.min_free_mb.saturating_mul(1024 * 1024);
        if free < len.saturating_add(keep) {
            return Err(JobError::InsufficientStorage);
        }
        Ok(())
    }

    /// Checks the free space, then stages the upload inside the job's own directory.
    pub fn stage_upload(&self, id: &str, bytes: &[u8]) -> Result<String, JobError> {
        self.ensure_space(bytes.len() as u64)?;
        Ok(self.store.stage_upload(id, bytes)?)
    }

    /// Cancels a queued or running job; deletes a finished one with its directory.
    pub fn remove(&self, id: &str) -> Result<(Removed, JobRecord), JobError> {
        let mut st = self.state();
        let rec = st.jobs.get_mut(id).ok_or(JobError::NotFound)?;
        if !rec.status.is_finished() {
            let was_running = rec.status == Status::Running;
            rec.status = Status::Cancelled;
            rec.finished_ms = Some(now_ms());
            let snapshot = rec.clone();
            self.store.write_job(&snapshot)?;
            if was_running && let Some(cur) = st.current.as_ref().filter(|c| c.id == id) {
                cur.cancel.store(true, Ordering::SeqCst);
            }
            drop(st);
            self.store.remove_staged(id, &snapshot.source);
            if !was_running {
                self.store.remove_audio(id);
            }
            return Ok((Removed::Cancelled, snapshot));
        }
        let snapshot = rec.clone();
        st.jobs.remove(id);
        drop(st);
        self.wait_until_idle(id);
        self.store.delete(id);
        Ok((Removed::Deleted, snapshot))
    }

    fn wait_until_idle(&self, id: &str) {
        let end = Instant::now() + WIND_DOWN_WAIT;
        while Instant::now() < end && self.state().current.as_ref().is_some_and(|c| c.id == id) {
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    pub fn result(&self, id: &str) -> Result<(StoredResult, JobRecord), JobError> {
        let rec = self.get(id).ok_or(JobError::NotFound)?;
        match rec.status {
            Status::Succeeded => {
                let r = self.store.read_result(id)?;
                Ok((r, rec))
            }
            Status::Failed => Err(JobError::Failed(rec.error.clone().unwrap_or_default())),
            Status::Cancelled => Err(JobError::Cancelled),
            Status::Queued | Status::Running => Err(JobError::NotFinished),
        }
    }

    /// Deletes finished jobs whose `finished_at` is older than `keep_results_days`.
    pub fn sweep_expired(&self, now_ms: u64) -> usize {
        let keep = self.cfg.jobs.keep_results_days.saturating_mul(DAY_MS);
        let expired: Vec<String> = self
            .state()
            .jobs
            .values()
            .filter(|j| j.status.is_finished())
            .filter(|j| {
                j.finished_ms
                    .is_some_and(|f| f.saturating_add(keep) <= now_ms)
            })
            .map(|j| j.id.clone())
            .collect();
        for id in &expired {
            self.state().jobs.remove(id);
            self.store.delete(id);
            tracing::info!(job = %id, "expired job deleted");
        }
        expired.len()
    }

    fn runner(self: Arc<Self>, rx: Receiver<()>) {
        let mut next_sweep = Instant::now() + SWEEP_EVERY;
        while !self.shutdown.load(Ordering::SeqCst) {
            if let Some(id) = self.claim_next() {
                self.run_job(&id);
                continue;
            }
            let wait = next_sweep.saturating_duration_since(Instant::now());
            match rx.recv_timeout(wait) {
                Ok(()) | Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            if Instant::now() >= next_sweep {
                self.sweep_expired(now_ms());
                next_sweep = Instant::now() + SWEEP_EVERY;
            }
        }
    }

    /// Oldest queued job, marked running and made current in one step.
    fn claim_next(&self) -> Option<String> {
        let mut st = self.state();
        let id = st
            .jobs
            .values()
            .filter(|j| j.status == Status::Queued)
            .min_by(|a, b| (a.created_ms, &a.id).cmp(&(b.created_ms, &b.id)))?
            .id
            .clone();
        let rec = st.jobs.get_mut(&id)?;
        rec.status = Status::Running;
        rec.started_ms.get_or_insert_with(now_ms);
        let snapshot = rec.clone();
        if let Err(e) = self.store.write_job(&snapshot) {
            tracing::error!(job = %id, error = %e, "could not persist a job start");
        }
        st.current = Some(Current {
            id: id.clone(),
            cancel: Arc::new(AtomicBool::new(false)),
        });
        Some(id)
    }

    /// Applies `f` to a job that is still running, then persists it; false when it is not.
    fn update_running(&self, id: &str, f: impl FnOnce(&mut JobRecord)) -> bool {
        let mut st = self.state();
        let Some(rec) = st.jobs.get_mut(id).filter(|r| r.status == Status::Running) else {
            return false;
        };
        f(rec);
        let snapshot = rec.clone();
        if let Err(e) = self.store.write_job(&snapshot) {
            tracing::error!(job = %id, error = %e, "could not persist job progress");
        }
        true
    }

    fn run_job(&self, id: &str) {
        let cancel = self
            .state()
            .current
            .as_ref()
            .map(|c| c.cancel.clone())
            .unwrap_or_default();
        tracing::info!(job = %id, "job started");
        let outcome = self.run_job_inner(id, &cancel);
        match outcome {
            Ok(()) => tracing::info!(job = %id, "job succeeded"),
            Err(Stop::Failed(msg)) => {
                tracing::warn!(job = %id, error = %msg, "job failed");
                let message = msg.clone();
                if self.update_running(id, |r| {
                    r.status = Status::Failed;
                    r.error = Some(message);
                    r.finished_ms = Some(now_ms());
                }) {
                    self.store.remove_audio(id);
                    self.store
                        .remove_staged(id, &self.get(id).and_then(|r| r.source));
                }
            }
            Err(Stop::Halted) => {
                if self.shutdown.load(Ordering::SeqCst) {
                    tracing::info!(job = %id, "job stopped for shutdown; it resumes at its next chunk");
                } else {
                    tracing::info!(job = %id, "job cancelled");
                    self.store.remove_audio(id);
                }
            }
        }
        self.state().current = None;
    }

    fn halted(&self, cancel: &AtomicBool) -> bool {
        cancel.load(Ordering::SeqCst) || self.shutdown.load(Ordering::SeqCst)
    }

    fn run_job_inner(&self, id: &str, cancel: &Arc<AtomicBool>) -> Result<(), Stop> {
        let rec = self.get(id).ok_or(Stop::Halted)?;
        let fail = |what: &str, e: &dyn std::fmt::Display| Stop::Failed(format!("{what}: {e}"));

        let samples = if self.store.audio_path(id).exists() {
            self.store
                .read_audio(id)
                .map_err(|e| fail("could not read the decoded audio", &e))?
        } else {
            let samples = self.decode_source(&rec, cancel)?;
            if self.halted(cancel) {
                return Err(Stop::Halted);
            }
            self.store
                .write_audio(id, &samples)
                .map_err(|e| fail("could not store the decoded audio", &e))?;
            samples
        };
        let secs = samples.len() as f64 / f64::from(SAMPLE_RATE);
        self.store.remove_staged(id, &rec.source);
        if !self.update_running(id, |r| {
            r.audio_seconds = Some(secs);
            r.source = None;
        }) {
            return Err(Stop::Halted);
        }
        if self.halted(cancel) {
            return Err(Stop::Halted);
        }

        let audio = Arc::new(samples);
        let opts = Arc::new(rec.options.decode_options());
        let plan: Vec<[usize; 2]> = match rec.chunks.clone() {
            Some(p) => p,
            None => {
                let p: Vec<[usize; 2]> = self
                    .engine
                    .plan_chunks(&audio, &opts, cancel)
                    .ok_or(Stop::Halted)?
                    .into_iter()
                    .map(|s| [s.start, s.end])
                    .collect();
                let saved = p.clone();
                if !self.update_running(id, |r| r.chunks = Some(saved)) {
                    return Err(Stop::Halted);
                }
                p
            }
        };
        if self.halted(cancel) {
            return Err(Stop::Halted);
        }

        let n = plan.len();
        let mut pinned = rec.language.clone();
        let model = if n > 0 {
            let (entry, path) = self
                .registry
                .resolve_installed(&rec.options.model)
                .map_err(|e| fail("model unavailable", &e))?;
            Some(ModelRef { id: entry.id, path })
        } else {
            None
        };
        let first = self.store.done_chunks(id);
        for (index, bounds) in plan.iter().enumerate().skip(first) {
            if self.halted(cancel) {
                return Err(Stop::Halted);
            }
            let used = rec.options.language.clone().or_else(|| pinned.clone());
            let chunk = JobChunk {
                job_id: id.to_string(),
                index,
                last: index + 1 == n,
                model: model.clone().expect("a model exists when chunks do"),
                audio: audio.clone(),
                span: Span {
                    start: bounds[0],
                    end: bounds[1],
                },
                opts: opts.clone(),
                language: used.clone(),
                cancel: cancel.clone(),
            };
            let out = match self.engine.run_job_chunk(&chunk) {
                Ok(out) => out,
                Err(crate::engine::EngineError::Cancelled) => return Err(Stop::Halted),
                Err(e) => return Err(Stop::Failed(e.to_string())),
            };
            let has_text = out.segments.iter().any(|s| !s.text.trim().is_empty());
            if rec.options.language.is_none() && pinned.is_none() && has_text {
                pinned = out.detected.clone();
            }
            let file = ChunkFile {
                index,
                span: *bounds,
                language: used.or_else(|| out.detected.clone()),
                detected: out.detected,
                segments: out.segments,
            };
            self.store
                .write_chunk(id, &file)
                .map_err(|e| fail("could not store a finished chunk", &e))?;
            let pin = pinned.clone();
            let progress = (index + 1) as f64 / n as f64;
            if !self.update_running(id, |r| {
                r.progress = progress;
                r.language = pin;
            }) {
                return Err(Stop::Halted);
            }
        }

        let result = self
            .assemble(id, &rec, n, pinned, secs)
            .map_err(|e| fail("could not assemble the result", &e))?;
        self.store
            .write_result(id, &result)
            .map_err(|e| fail("could not store the result", &e))?;
        if !self.update_running(id, |r| {
            r.status = Status::Succeeded;
            r.progress = 1.0;
            r.finished_ms = Some(now_ms());
        }) {
            return Err(Stop::Halted);
        }
        self.store.remove_audio(id);
        Ok(())
    }

    fn decode_source(&self, rec: &JobRecord, cancel: &AtomicBool) -> Result<Vec<f32>, Stop> {
        let limits = DecodeLimits {
            ffmpeg_timeout: Duration::from_secs(self.cfg.jobs.ffmpeg_timeout_secs),
            max_secs: JOB_MAX_AUDIO_SECS,
        };
        let path = match &rec.source {
            Some(Source::Path { path }) => resolve_job_path(path, &self.cfg.jobs.path_roots)
                .ok_or_else(|| {
                    Stop::Failed("the audio file is gone or no longer under jobs.path_roots".into())
                })?,
            Some(Source::Upload { file }) => self
                .store
                .staged_path(&rec.id, file)
                .filter(|p| p.is_file())
                .ok_or_else(|| Stop::Failed("the staged upload is gone".into()))?,
            None => return Err(Stop::Failed("the job has no audio".into())),
        };
        audio::decode_file(&path, &limits, cancel).map_err(|e| match e {
            audio::AudioError::Cancelled => Stop::Halted,
            e => Stop::Failed(e.to_string()),
        })
    }

    fn assemble(
        &self,
        id: &str,
        rec: &JobRecord,
        n: usize,
        pinned: Option<String>,
        secs: f64,
    ) -> std::io::Result<StoredResult> {
        let mut segments = Vec::new();
        let mut first_language = None;
        for i in 0..n {
            let chunk = self.store.read_chunk(id, i)?;
            if first_language.is_none() {
                first_language = chunk.language.clone();
            }
            segments.extend(chunk.segments);
        }
        let (segments, repetitions) = pipeline::finalize_segments(segments);
        for r in &repetitions {
            tracing::warn!(
                job = %id,
                phrase_words = r.phrase_words,
                count = r.count,
                "repetition loop collapsed"
            );
        }
        Ok(StoredResult {
            language: rec.options.language.clone().or(pinned).or(first_language),
            duration: secs,
            temperature: rec.options.temperature,
            segments,
            repetitions: repetitions.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineError, Transcriber};
    use crate::pipeline::Word;
    use std::sync::atomic::AtomicUsize;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hark-jobs-test-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    pub(crate) fn opts() -> JobOptions {
        JobOptions {
            model: "base".into(),
            language: None,
            prompt: None,
            vocabulary: vec!["Kubernetes".into()],
            temperature: 0.0,
            vad: None,
            granularities: Vec::new(),
            response_format: None,
        }
    }

    fn seg(id: usize, start: f64, end: f64, text: &str) -> Segment {
        Segment {
            id,
            seek_cs: 0,
            start,
            end,
            text: text.into(),
            tokens: vec![1],
            avg_logprob: -0.1,
            no_speech_prob: 0.0,
            words: vec![Word {
                word: text.trim().into(),
                start,
                end,
            }],
        }
    }

    /// Fake engine: every chunk yields one segment named after its index.
    struct Fake {
        runs: AtomicUsize,
        gate: Option<Arc<AtomicBool>>,
    }

    impl Transcriber for Fake {
        fn transcribe(
            &self,
            _m: &ModelRef,
            _a: &[f32],
            _o: &DecodeOptions,
        ) -> Result<Transcription, EngineError> {
            let n = self.runs.fetch_add(1, Ordering::SeqCst);
            if let Some(g) = &self.gate {
                while !g.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            Ok(Transcription {
                language: Some("en".into()),
                segments: vec![seg(0, 0.0, 1.0, &format!(" chunk{n}"))],
                ..Default::default()
            })
        }
    }

    fn manager(tag: &str, engine: Arc<dyn Transcriber>) -> (Arc<JobManager>, PathBuf) {
        let base = tmp(tag);
        let models = base.join("models");
        std::fs::create_dir_all(&models).unwrap();
        std::fs::write(models.join("ggml-base.bin"), b"fake").unwrap();
        let mut cfg = Config::default();
        cfg.jobs.dir = base.join("jobs").display().to_string();
        cfg.jobs.path_roots = vec![base.display().to_string()];
        let registry = Arc::new(Registry::new(vec![models], Default::default()));
        let m = JobManager::open(Arc::new(cfg), registry, engine).unwrap();
        (m, base)
    }

    /// Reopens a store once the previous manager is gone; a child forked by another test may hold the lock for a moment.
    fn reopen(
        cfg: Arc<Config>,
        registry: Arc<Registry>,
        engine: Arc<dyn Transcriber>,
    ) -> Arc<JobManager> {
        for _ in 0..100 {
            match JobManager::open(cfg.clone(), registry.clone(), engine.clone()) {
                Ok(m) => return m,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => panic!("{e}"),
            }
        }
        panic!("the jobs directory stayed locked");
    }

    fn fake() -> Arc<Fake> {
        Arc::new(Fake {
            runs: AtomicUsize::new(0),
            gate: None,
        })
    }

    fn wait_for(m: &JobManager, id: &str, want: Status) -> JobRecord {
        let end = Instant::now() + Duration::from_secs(20);
        loop {
            let r = m.get(id).unwrap();
            if r.status == want {
                return r;
            }
            assert!(Instant::now() < end, "job stuck in {:?}", r.status);
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn finished_job(m: &JobManager, id: &str, status: Status, finished_ms: u64) {
        let mut rec = JobRecord::new(id.into(), opts(), Source::Path { path: "/x".into() }, 1);
        rec.status = status;
        rec.finished_ms = Some(finished_ms);
        m.store.write_job(&rec).unwrap();
        m.state().jobs.insert(id.into(), rec);
    }

    #[test]
    fn a_second_open_on_the_same_jobs_dir_is_refused_and_touches_nothing() {
        let (m, base) = manager("lock", fake());
        let mut rec = JobRecord::new(
            "job_aaaaaaaaaaaaaaaa".into(),
            opts(),
            Source::Path { path: "/x".into() },
            1,
        );
        rec.status = Status::Running;
        m.store.write_job(&rec).unwrap();
        let mut cfg = Config::default();
        cfg.jobs.dir = base.join("jobs").display().to_string();
        let registry = Arc::new(Registry::new(vec![base.join("models")], Default::default()));
        let err = JobManager::open(Arc::new(cfg.clone()), registry.clone(), fake())
            .err()
            .expect("the second manager is refused");
        assert_eq!(err.kind(), std::io::ErrorKind::WouldBlock);
        assert!(err.to_string().contains("in use by another hark"), "{err}");
        let on_disk = m.store.read_job("job_aaaaaaaaaaaaaaaa").unwrap();
        assert_eq!(on_disk.status, Status::Running, "not recovered or requeued");
        drop(m);
        let again = reopen(Arc::new(cfg), registry, fake());
        assert_eq!(
            again.get("job_aaaaaaaaaaaaaaaa").unwrap().status,
            Status::Queued,
            "the lock is released with the first manager"
        );
    }

    #[test]
    fn a_job_directory_without_a_job_json_is_removed_at_startup() {
        let (m, base) = manager("orphan", fake());
        let dir = base.join("jobs").join("job_bbbbbbbbbbbbbbbb");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("upload.bin"), b"left behind").unwrap();
        drop(m);
        let mut cfg = Config::default();
        cfg.jobs.dir = base.join("jobs").display().to_string();
        let registry = Arc::new(Registry::new(vec![base.join("models")], Default::default()));
        let again = reopen(Arc::new(cfg), registry, fake());
        assert!(!dir.exists());
        assert!(again.list(None, 10).is_empty());
    }

    #[test]
    fn job_ids_are_random_base32_and_valid() {
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            let id = new_job_id().unwrap();
            assert_eq!(id.len(), 20, "{id}");
            assert!(id.starts_with("job_"));
            assert!(
                id[4..]
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || (b'2'..=b'7').contains(&b)),
                "{id}"
            );
            assert!(valid_job_id(&id));
            assert!(seen.insert(id), "duplicate id");
        }
        for bad in [
            "",
            "job_",
            "job_ABCDEFGHIJKLMNOP",
            "job_abcdefghijklmnop1",
            "job_abcdefghijklmno1",
            "job_../etc/passwd..",
            "xxx_abcdefghijklmnop",
        ] {
            assert!(!valid_job_id(bad), "{bad:?}");
        }
    }

    #[test]
    fn a_crash_before_the_rename_keeps_the_previous_job_json() {
        let d = tmp("atomic");
        let store = JobStore::open(&d.join("jobs")).unwrap();
        let mut rec = JobRecord::new(
            "job_aaaaaaaaaaaaaaaa".into(),
            opts(),
            Source::Path { path: "/x".into() },
            5,
        );
        store.write_job(&rec).unwrap();
        rec.progress = 0.5;
        let path = store.job_dir(&rec.id).join("job.json");
        let tmp_file = write_tmp(&path, &serde_json::to_vec(&rec).unwrap()).unwrap();
        assert!(tmp_file.exists(), "the new content sits in the temp file");
        let read = store.read_job(&rec.id).unwrap();
        assert_eq!(read.progress, 0.0, "the old job.json is untouched");
        assert!(store.ids().contains(&rec.id));
        commit_tmp(&tmp_file, &path).unwrap();
        assert_eq!(store.read_job(&rec.id).unwrap().progress, 0.5);
        assert!(!tmp_file.exists());
    }

    #[cfg(unix)]
    #[test]
    fn the_store_is_private_and_chunks_count_from_zero() {
        use std::os::unix::fs::MetadataExt;
        let d = tmp("store");
        let store = JobStore::open(&d.join("jobs")).unwrap();
        assert_eq!(
            std::fs::metadata(d.join("jobs")).unwrap().mode() & 0o777,
            0o700
        );
        let id = "job_aaaaaaaaaaaaaaaa";
        store
            .write_job(&JobRecord::new(
                id.into(),
                opts(),
                Source::Path { path: "/x".into() },
                1,
            ))
            .unwrap();
        assert_eq!(store.done_chunks(id), 0);
        for index in [0usize, 1] {
            store
                .write_chunk(
                    id,
                    &ChunkFile {
                        index,
                        span: [0, 1],
                        language: None,
                        detected: None,
                        segments: vec![],
                    },
                )
                .unwrap();
        }
        store
            .write_chunk(
                id,
                &ChunkFile {
                    index: 3,
                    span: [0, 1],
                    language: None,
                    detected: None,
                    segments: vec![],
                },
            )
            .unwrap();
        assert_eq!(store.done_chunks(id), 2, "a gap ends the finished run");
        store.write_audio(id, &[0.5, -0.25]).unwrap();
        assert_eq!(store.read_audio(id).unwrap(), vec![0.5, -0.25]);
    }

    #[test]
    fn running_jobs_requeue_at_their_checkpoint_and_corrupt_dirs_are_skipped() {
        let (m, base) = manager("recover", fake());
        let id = "job_bbbbbbbbbbbbbbbb";
        let mut rec = JobRecord::new(id.into(), opts(), Source::Path { path: "/x".into() }, 10);
        rec.status = Status::Running;
        rec.progress = 0.4;
        m.store.write_job(&rec).unwrap();
        for index in 0..2 {
            m.store
                .write_chunk(
                    id,
                    &ChunkFile {
                        index,
                        span: [0, 1],
                        language: None,
                        detected: None,
                        segments: vec![],
                    },
                )
                .unwrap();
        }
        let bad = base.join("jobs").join("job_cccccccccccccccc");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join("job.json"), b"{ not json").unwrap();
        let empty = base.join("jobs").join("job_dddddddddddddddd");
        std::fs::create_dir_all(&empty).unwrap();
        std::fs::create_dir_all(base.join("jobs").join("not-a-job")).unwrap();
        let (cfg, registry, engine) = (m.cfg.clone(), m.registry.clone(), m.engine.clone());
        drop(m);

        let m2 = reopen(cfg, registry, engine);
        let r = m2.get(id).unwrap();
        assert_eq!(r.status, Status::Queued);
        assert_eq!(r.resumed_from_chunk, Some(2));
        assert_eq!(m2.store.read_job(id).unwrap().status, Status::Queued);
        assert!(m2.get("job_cccccccccccccccc").is_none());
        assert!(m2.get("job_dddddddddddddddd").is_none());
        assert_eq!(m2.list(None, 100).len(), 1);
    }

    #[test]
    fn expiry_deletes_old_finished_jobs_only() {
        let (m, _) = manager("expiry", fake());
        let day = DAY_MS;
        let now = 100 * day;
        finished_job(&m, "job_aaaaaaaaaaaaaaaa", Status::Succeeded, now - 8 * day);
        finished_job(&m, "job_bbbbbbbbbbbbbbbb", Status::Failed, now - 7 * day);
        finished_job(&m, "job_cccccccccccccccc", Status::Cancelled, now - 6 * day);
        finished_job(&m, "job_dddddddddddddddd", Status::Succeeded, now - 1);
        for (id, status) in [
            ("job_eeeeeeeeeeeeeeee", Status::Queued),
            ("job_ffffffffffffffff", Status::Running),
        ] {
            let mut rec = JobRecord::new(id.into(), opts(), Source::Path { path: "/x".into() }, 1);
            rec.status = status;
            m.store.write_job(&rec).unwrap();
            m.state().jobs.insert(id.into(), rec);
        }
        assert_eq!(m.sweep_expired(now), 2);
        for gone in ["job_aaaaaaaaaaaaaaaa", "job_bbbbbbbbbbbbbbbb"] {
            assert!(m.get(gone).is_none());
            assert!(!m.store.job_dir(gone).exists());
        }
        for kept in [
            "job_cccccccccccccccc",
            "job_dddddddddddddddd",
            "job_eeeeeeeeeeeeeeee",
            "job_ffffffffffffffff",
        ] {
            assert!(m.get(kept).is_some(), "{kept}");
            assert!(m.store.job_dir(kept).exists(), "{kept}");
        }
        assert_eq!(
            m.sweep_expired(now + 365 * day),
            2,
            "only finished jobs ever expire"
        );
    }

    #[test]
    fn list_is_newest_first_and_filters_by_status() {
        let (m, _) = manager("list", fake());
        for (i, (id, status)) in [
            ("job_aaaaaaaaaaaaaaaa", Status::Succeeded),
            ("job_bbbbbbbbbbbbbbbb", Status::Failed),
            ("job_cccccccccccccccc", Status::Succeeded),
        ]
        .into_iter()
        .enumerate()
        {
            let mut rec = JobRecord::new(
                id.into(),
                opts(),
                Source::Path { path: "/x".into() },
                1000 + i as u64,
            );
            rec.status = status;
            rec.finished_ms = Some(2000);
            m.state().jobs.insert(id.into(), rec);
        }
        let ids = |s: Option<Status>, n| -> Vec<String> {
            m.list(s, n).into_iter().map(|r| r.id).collect()
        };
        assert_eq!(
            ids(None, 10),
            [
                "job_cccccccccccccccc",
                "job_bbbbbbbbbbbbbbbb",
                "job_aaaaaaaaaaaaaaaa"
            ]
        );
        assert_eq!(
            ids(Some(Status::Succeeded), 10),
            ["job_cccccccccccccccc", "job_aaaaaaaaaaaaaaaa"]
        );
        assert_eq!(ids(None, 1), ["job_cccccccccccccccc"]);
        assert!(ids(Some(Status::Running), 10).is_empty());
    }

    #[test]
    fn a_job_runs_to_a_stored_result_and_drops_its_audio() {
        let (m, base) = manager("run", fake());
        let wav = base.join("a.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&wav, spec).unwrap();
        for _ in 0..(16000 * 3) {
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();
        m.start();
        let rec = m
            .submit(
                "job_gggggggggggggggg".into(),
                opts(),
                Source::Path {
                    path: wav.display().to_string(),
                },
            )
            .unwrap();
        let done = wait_for(&m, &rec.id, Status::Succeeded);
        assert_eq!(done.progress, 1.0);
        assert_eq!(done.audio_seconds, Some(3.0));
        assert!(done.finished_ms.is_some() && done.started_ms.is_some());
        let end = Instant::now() + Duration::from_secs(5);
        while m.store.audio_path(&rec.id).exists() && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !m.store.audio_path(&rec.id).exists(),
            "audio.f32 is deleted on success"
        );
        let (r, _) = m.result(&rec.id).unwrap();
        assert_eq!(r.language.as_deref(), Some("en"));
        assert_eq!(r.segments.len(), 1);
        assert_eq!(r.segments[0].text, " chunk0");
        m.stop();
    }

    #[test]
    fn a_job_that_cannot_decode_fails_with_a_message() {
        let (m, base) = manager("fail", fake());
        let junk = base.join("junk.bin");
        std::fs::write(&junk, b"definitely not audio").unwrap();
        m.start();
        let rec = m
            .submit(
                "job_hhhhhhhhhhhhhhhh".into(),
                opts(),
                Source::Path {
                    path: junk.display().to_string(),
                },
            )
            .unwrap();
        let done = wait_for(&m, &rec.id, Status::Failed);
        assert!(done.error.unwrap().contains("could not decode"));
        assert!(matches!(m.result(&rec.id), Err(JobError::Failed(_))));
        m.stop();
    }

    #[test]
    fn cancel_then_delete_and_result_of_unfinished_jobs() {
        let gate = Arc::new(AtomicBool::new(false));
        let eng = Arc::new(Fake {
            runs: AtomicUsize::new(0),
            gate: Some(gate.clone()),
        });
        let (m, base) = manager("cancel", eng);
        let wav = base.join("b.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&wav, spec).unwrap();
        for _ in 0..(16000 * 3) {
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();
        let src = Source::Path {
            path: wav.display().to_string(),
        };
        m.start();
        let rec = m
            .submit("job_iiiiiiiiiiiiiiii".into(), opts(), src)
            .unwrap();
        wait_for(&m, &rec.id, Status::Running);
        assert!(matches!(m.result(&rec.id), Err(JobError::NotFinished)));
        let (kind, snap) = m.remove(&rec.id).unwrap();
        assert_eq!(kind, Removed::Cancelled);
        assert_eq!(snap.status, Status::Cancelled);
        assert_eq!(m.get(&rec.id).unwrap().status, Status::Cancelled);
        gate.store(true, Ordering::SeqCst);
        let (kind, _) = m.remove(&rec.id).unwrap();
        assert_eq!(kind, Removed::Deleted);
        assert!(m.get(&rec.id).is_none());
        assert!(!m.store.job_dir(&rec.id).exists());
        assert!(matches!(m.remove(&rec.id), Err(JobError::NotFound)));
        m.stop();
    }

    #[cfg(unix)]
    #[test]
    fn job_paths_must_resolve_inside_a_root() {
        let base = tmp("paths");
        let root = base.join("root");
        let outside = base.join("outside");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(root.join("in.wav"), b"x").unwrap();
        std::fs::write(outside.join("out.wav"), b"x").unwrap();
        std::os::unix::fs::symlink(outside.join("out.wav"), root.join("link.wav")).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("dirlink")).unwrap();
        let roots = vec![root.display().to_string()];
        let ok = resolve_job_path(&root.join("in.wav").display().to_string(), &roots);
        assert_eq!(
            ok,
            Some(std::fs::canonicalize(root.join("in.wav")).unwrap())
        );
        for bad in [
            outside.join("out.wav").display().to_string(),
            root.join("link.wav").display().to_string(),
            root.join("dirlink/out.wav").display().to_string(),
            format!("{}/../outside/out.wav", root.display()),
            root.display().to_string(),
            root.join("missing.wav").display().to_string(),
            "relative.wav".into(),
            String::new(),
        ] {
            assert!(resolve_job_path(&bad, &roots).is_none(), "{bad:?}");
        }
        assert!(resolve_job_path(&root.join("in.wav").display().to_string(), &[]).is_none());
    }

    #[test]
    fn stored_result_round_trips() {
        let r = StoredResult {
            language: Some("en".into()),
            duration: 4.5,
            temperature: 0.2,
            segments: vec![seg(0, 0.0, 1.0, " hi")],
            repetitions: 0,
        };
        let back: StoredResult = serde_json::from_slice(&serde_json::to_vec(&r).unwrap()).unwrap();
        assert_eq!(back, r);
        assert_eq!(back.transcription().text(), "hi");
    }

    /// Plans four 1 s chunks and records which indexes the runner asks for.
    struct Planned {
        seen: Mutex<Vec<usize>>,
    }

    impl Transcriber for Planned {
        fn transcribe(
            &self,
            _m: &ModelRef,
            _a: &[f32],
            _o: &DecodeOptions,
        ) -> Result<Transcription, EngineError> {
            unreachable!("run_job_chunk is overridden")
        }

        fn plan_chunks(
            &self,
            audio: &[f32],
            _o: &DecodeOptions,
            _c: &AtomicBool,
        ) -> Option<Vec<Span>> {
            let n = audio.len() / 16000;
            Some(
                (0..n)
                    .map(|i| Span {
                        start: i * 16000,
                        end: (i + 1) * 16000,
                    })
                    .collect(),
            )
        }

        fn run_job_chunk(&self, c: &JobChunk) -> Result<crate::scheduler::ChunkOut, EngineError> {
            self.seen.lock().unwrap().push(c.index);
            let t = c.index as f64;
            Ok(crate::scheduler::ChunkOut {
                segments: vec![seg(0, t, t + 1.0, &format!(" new{}", c.index))],
                detected: Some("fr".into()),
            })
        }
    }

    #[test]
    fn a_restarted_job_resumes_at_its_first_unfinished_chunk() {
        let eng = Arc::new(Planned {
            seen: Mutex::new(Vec::new()),
        });
        let (m, _) = manager("resume", eng.clone());
        let id = "job_jjjjjjjjjjjjjjjj";
        let mut rec = JobRecord::new(
            id.into(),
            opts(),
            Source::Path {
                path: "/gone".into(),
            },
            10,
        );
        rec.status = Status::Running;
        rec.chunks = Some((0..4).map(|i| [i * 16000, (i + 1) * 16000]).collect());
        rec.language = Some("fr".into());
        m.store.write_job(&rec).unwrap();
        m.store.write_audio(id, &vec![0.0; 4 * 16000]).unwrap();
        for index in 0..2 {
            let t = index as f64;
            m.store
                .write_chunk(
                    id,
                    &ChunkFile {
                        index,
                        span: [index * 16000, (index + 1) * 16000],
                        language: Some("fr".into()),
                        detected: Some("fr".into()),
                        segments: vec![seg(0, t, t + 1.0, &format!(" old{index}"))],
                    },
                )
                .unwrap();
        }
        let (cfg, registry) = (m.cfg.clone(), m.registry.clone());
        drop(m);

        let m2 = reopen(cfg, registry, eng.clone());
        assert_eq!(m2.get(id).unwrap().resumed_from_chunk, Some(2));
        m2.start();
        let done = wait_for(&m2, id, Status::Succeeded);
        assert_eq!(done.resumed_from_chunk, Some(2));
        assert_eq!(
            *eng.seen.lock().unwrap(),
            vec![2, 3],
            "chunks 0 and 1 were not redone"
        );
        let (r, _) = m2.result(id).unwrap();
        let texts: Vec<&str> = r.segments.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(texts, [" old0", " old1", " new2", " new3"]);
        assert_eq!(r.language.as_deref(), Some("fr"));
        assert!(r.segments.windows(2).all(|w| w[0].end <= w[1].start));
        assert_eq!(
            r.segments.iter().map(|s| s.id).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        m2.stop();
    }
}
