//! Model downloads: HTTPS to huggingface.co (and the CDN it redirects to) only, streamed to
//! `<file>.part` while hashing, checked against the catalog pin, fsynced and renamed into place.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::models::CatalogEntry;

const MAX_REDIRECTS: usize = 5;
/// Slowest rate a pull may average before it is aborted.
const MIN_PULL_BYTES_PER_SEC: u64 = 1024 * 1024;
const MIN_PULL_DEADLINE: Duration = Duration::from_secs(600);

/// Overall time a download of `size` bytes may take: at least 600 s, else size at 1 MiB/s.
pub fn pull_deadline(size: u64) -> Duration {
    MIN_PULL_DEADLINE.max(Duration::from_secs(size / MIN_PULL_BYTES_PER_SEC))
}

/// Takes the exclusive lock a running pull holds on its part file; false when someone else holds it.
fn lock_part(file: &std::fs::File) -> bool {
    crate::sys::try_lock_exclusive(file).unwrap_or(false)
}

/// Deletes `*.part` files no pull holds (a crashed one left them) in `dir`; returns how many.
/// A part file another running hark is downloading into stays: that pull holds its lock.
pub fn remove_stale_parts(dir: &Path) -> usize {
    let mut n = 0;
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "part") || !path.is_file() {
            continue;
        }
        let Ok(file) = std::fs::OpenOptions::new().write(true).open(&path) else {
            continue;
        };
        if lock_part(&file) && std::fs::remove_file(&path).is_ok() {
            n += 1;
        }
    }
    n
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PullError {
    #[error("{0}")]
    Network(String),
    #[error("{0}")]
    Mismatch(String),
    #[error("{0}")]
    Io(String),
    #[error("{0}")]
    Timeout(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PullOutcome {
    AlreadyInstalled,
    Downloaded { bytes: u64 },
}

#[derive(Debug, Clone)]
pub struct PullSpec {
    pub file: String,
    pub url: String,
    pub sha256: String,
    pub size: u64,
}

impl PullSpec {
    pub fn from_entry(e: &CatalogEntry) -> PullSpec {
        PullSpec {
            file: e.file.to_string(),
            url: e.url(),
            sha256: e.sha256.to_string(),
            size: e.size,
        }
    }
}

/// An opened download: the body and the Content-Length when the server sent one.
pub struct Download {
    pub body: Box<dyn Read + Send>,
    pub length: Option<u64>,
}

pub trait Fetch: Send + Sync {
    fn open(&self, url: &str) -> Result<Download, PullError>;
}

/// True for `https://` URLs on huggingface.co (first hop) or on its CDN hosts (redirect hops).
/// The host comes from the same `url` parser the HTTP client uses, so the check and the connection agree.
pub fn url_allowed(url: &str, redirect: bool) -> bool {
    let Ok(parsed) = url::Url::parse(url) else {
        return false;
    };
    if parsed.scheme() != "https"
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.port().is_some()
    {
        return false;
    }
    let Some(host) = parsed.host_str() else {
        return false;
    };
    if host == "huggingface.co" {
        return true;
    }
    redirect && (host.ends_with(".huggingface.co") || host == "hf.co" || host.ends_with(".hf.co"))
}

/// Resolves a `Location` header against the URL it came from; unparsable input stays as is and fails the host check.
fn resolve_location(from: &str, location: &str) -> String {
    url::Url::parse(from)
        .and_then(|base| base.join(location))
        .map_or_else(|_| location.to_string(), String::from)
}

pub struct HfFetch {
    agent: ureq::Agent,
}

impl Default for HfFetch {
    fn default() -> Self {
        HfFetch {
            agent: ureq::AgentBuilder::new()
                .redirects(0)
                .timeout_connect(Duration::from_secs(15))
                .timeout_read(Duration::from_secs(60))
                .build(),
        }
    }
}

impl Fetch for HfFetch {
    fn open(&self, url: &str) -> Result<Download, PullError> {
        let mut current = url.to_string();
        for hop in 0..=MAX_REDIRECTS {
            if !url_allowed(&current, hop > 0) {
                return Err(PullError::Network(format!(
                    "refusing to download from {}",
                    current.split('?').next().unwrap_or("")
                )));
            }
            match self.agent.get(&current).call() {
                Ok(resp) if (300..400).contains(&resp.status()) => {
                    let loc = resp.header("location").ok_or_else(|| {
                        PullError::Network("redirect without a Location header".into())
                    })?;
                    current = resolve_location(&current, loc);
                }
                Ok(resp) => {
                    let length = resp.header("content-length").and_then(|v| v.parse().ok());
                    return Ok(Download {
                        body: Box::new(resp.into_reader()),
                        length,
                    });
                }
                Err(ureq::Error::Status(code, resp)) if (300..400).contains(&code) => {
                    let loc = resp.header("location").ok_or_else(|| {
                        PullError::Network("redirect without a Location header".into())
                    })?;
                    current = resolve_location(&current, loc);
                }
                Err(ureq::Error::Status(code, _)) => {
                    return Err(PullError::Network(format!("download failed: HTTP {code}")));
                }
                Err(e) => return Err(PullError::Network(format!("download failed: {e}"))),
            }
        }
        Err(PullError::Network("too many redirects".into()))
    }
}

fn io_err(what: &str, e: std::io::Error) -> PullError {
    PullError::Io(format!("{what}: {e}"))
}

/// Downloads `spec` into `dir` within `pull_deadline`; a file already there is left alone.
pub fn pull(dir: &Path, spec: &PullSpec, fetch: &dyn Fetch) -> Result<PullOutcome, PullError> {
    pull_within(dir, spec, fetch, pull_deadline(spec.size))
}

pub fn pull_within(
    dir: &Path,
    spec: &PullSpec,
    fetch: &dyn Fetch,
    deadline: Duration,
) -> Result<PullOutcome, PullError> {
    let target = dir.join(&spec.file);
    if target.is_file() {
        return Ok(PullOutcome::AlreadyInstalled);
    }
    std::fs::create_dir_all(dir).map_err(|e| io_err("creating the models directory", e))?;
    let part = dir.join(format!("{}.part", spec.file));
    let result = download_to(&part, spec, fetch, Instant::now() + deadline);
    match result {
        Ok(bytes) => {
            std::fs::rename(&part, &target).map_err(|e| {
                let _ = std::fs::remove_file(&part);
                io_err("moving the model into place", e)
            })?;
            if let Ok(d) = std::fs::File::open(dir) {
                let _ = d.sync_all();
            }
            Ok(PullOutcome::Downloaded { bytes })
        }
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

fn download_to(
    part: &Path,
    spec: &PullSpec,
    fetch: &dyn Fetch,
    deadline: Instant,
) -> Result<u64, PullError> {
    let mut dl = fetch.open(&spec.url)?;
    if let Some(len) = dl.length
        && len != spec.size
    {
        return Err(PullError::Mismatch(format!(
            "{}: the server announces {len} bytes, expected {}",
            spec.file, spec.size
        )));
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(part)
        .map_err(|e| io_err("creating the part file", e))?;
    if !lock_part(&file) {
        return Err(PullError::Io(format!(
            "{}: another process is downloading this model",
            spec.file
        )));
    }
    file.set_len(0)
        .map_err(|e| io_err("resetting the part file", e))?;
    let mut hasher = Sha256::new();
    let mut total = 0u64;
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        if Instant::now() >= deadline {
            return Err(PullError::Timeout(format!(
                "{}: the download did not finish in time and was aborted",
                spec.file
            )));
        }
        let n = dl
            .body
            .read(&mut buf)
            .map_err(|e| PullError::Network(format!("download interrupted: {e}")))?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > spec.size {
            return Err(PullError::Mismatch(format!(
                "{}: the download is larger than the expected {} bytes",
                spec.file, spec.size
            )));
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n])
            .map_err(|e| io_err("writing the part file", e))?;
    }
    if total != spec.size {
        return Err(PullError::Mismatch(format!(
            "{}: got {total} bytes, expected {}",
            spec.file, spec.size
        )));
    }
    let digest = hex(&hasher.finalize());
    if digest != spec.sha256 {
        return Err(PullError::Mismatch(format!(
            "{}: SHA-256 {digest} does not match the pinned {}",
            spec.file, spec.sha256
        )));
    }
    file.sync_all()
        .map_err(|e| io_err("syncing the part file", e))?;
    Ok(total)
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

struct Cell {
    done: Mutex<Option<Result<PullOutcome, PullError>>>,
    cv: Condvar,
}

/// Runs one download per key; concurrent callers wait for it and share its result.
#[derive(Default)]
pub struct Pulls {
    inflight: Mutex<HashMap<String, Arc<Cell>>>,
}

impl Pulls {
    /// True for the caller that actually ran `work`.
    pub fn run(
        &self,
        key: &str,
        work: impl FnOnce() -> Result<PullOutcome, PullError>,
    ) -> (Result<PullOutcome, PullError>, bool) {
        let (cell, leader) = {
            let mut map = self.inflight.lock().unwrap_or_else(|p| p.into_inner());
            match map.get(key) {
                Some(c) => (c.clone(), false),
                None => {
                    let c = Arc::new(Cell {
                        done: Mutex::new(None),
                        cv: Condvar::new(),
                    });
                    map.insert(key.to_string(), c.clone());
                    (c, true)
                }
            }
        };
        if leader {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
                .unwrap_or_else(|_| Err(PullError::Io("the download thread panicked".into())));
            self.inflight
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .remove(key);
            *cell.done.lock().unwrap_or_else(|p| p.into_inner()) = Some(result.clone());
            cell.cv.notify_all();
            return (result, true);
        }
        let mut guard = cell.done.lock().unwrap_or_else(|p| p.into_inner());
        while guard.is_none() {
            guard = cell.cv.wait(guard).unwrap_or_else(|p| p.into_inner());
        }
        (guard.clone().expect("set before notify"), false)
    }
}

/// Process-wide coordinator, so every route shares one download per file.
pub fn global() -> &'static Pulls {
    static P: OnceLock<Pulls> = OnceLock::new();
    P.get_or_init(Pulls::default)
}

pub fn pull_key(dir: &Path, file: &str) -> String {
    let p: PathBuf = dir.join(file);
    p.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Mem {
        data: Vec<u8>,
        announce: Option<u64>,
        opens: AtomicUsize,
    }

    impl Fetch for Mem {
        fn open(&self, _url: &str) -> Result<Download, PullError> {
            self.opens.fetch_add(1, Ordering::SeqCst);
            Ok(Download {
                body: Box::new(Cursor::new(self.data.clone())),
                length: self.announce,
            })
        }
    }

    fn spec_for(data: &[u8]) -> PullSpec {
        PullSpec {
            file: "ggml-test.bin".into(),
            url: "https://huggingface.co/x/ggml-test.bin".into(),
            sha256: hex(&Sha256::digest(data)),
            size: data.len() as u64,
        }
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hark-pull-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn good_download_lands_renamed_without_a_part_file() {
        let dir = tmp("good");
        let data = vec![7u8; 700_000];
        let f = Mem {
            data: data.clone(),
            announce: Some(data.len() as u64),
            opens: AtomicUsize::new(0),
        };
        let out = pull(&dir, &spec_for(&data), &f).unwrap();
        assert_eq!(out, PullOutcome::Downloaded { bytes: 700_000 });
        assert_eq!(std::fs::read(dir.join("ggml-test.bin")).unwrap(), data);
        assert!(!dir.join("ggml-test.bin.part").exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn hash_mismatch_deletes_the_part_file_and_installs_nothing() {
        let dir = tmp("badhash");
        let data = vec![1u8; 1000];
        let mut spec = spec_for(&data);
        spec.sha256 = "0".repeat(64);
        let f = Mem {
            data,
            announce: None,
            opens: AtomicUsize::new(0),
        };
        let err = pull(&dir, &spec, &f).unwrap_err();
        assert!(matches!(err, PullError::Mismatch(_)), "{err}");
        assert!(err.to_string().contains("SHA-256"), "{err}");
        assert!(!dir.join("ggml-test.bin").exists());
        assert!(!dir.join("ggml-test.bin.part").exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn wrong_size_is_refused_before_and_while_streaming() {
        let dir = tmp("size");
        let data = vec![2u8; 500];
        let spec = spec_for(&data);
        let announced = Mem {
            data: data.clone(),
            announce: Some(9999),
            opens: AtomicUsize::new(0),
        };
        assert!(matches!(
            pull(&dir, &spec, &announced),
            Err(PullError::Mismatch(_))
        ));
        let long = Mem {
            data: vec![2u8; 600],
            announce: None,
            opens: AtomicUsize::new(0),
        };
        assert!(matches!(
            pull(&dir, &spec, &long),
            Err(PullError::Mismatch(_))
        ));
        let short = Mem {
            data: vec![2u8; 400],
            announce: None,
            opens: AtomicUsize::new(0),
        };
        assert!(matches!(
            pull(&dir, &spec, &short),
            Err(PullError::Mismatch(_))
        ));
        assert!(!dir.join("ggml-test.bin.part").exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn an_installed_file_is_not_downloaded_again() {
        let dir = tmp("again");
        let data = vec![3u8; 100];
        let f = Mem {
            data: data.clone(),
            announce: None,
            opens: AtomicUsize::new(0),
        };
        let spec = spec_for(&data);
        pull(&dir, &spec, &f).unwrap();
        assert_eq!(
            pull(&dir, &spec, &f).unwrap(),
            PullOutcome::AlreadyInstalled
        );
        assert_eq!(f.opens.load(Ordering::SeqCst), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn only_https_huggingface_hosts_are_allowed() {
        let ok = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin";
        assert!(url_allowed(ok, false));
        assert!(!url_allowed("http://huggingface.co/x", false));
        assert!(!url_allowed("https://evil.example/x", false));
        assert!(!url_allowed("https://huggingface.co.evil.example/x", false));
        assert!(!url_allowed("https://huggingface.co@evil.example/x", true));
        assert!(!url_allowed("https://us.aws.cdn.hf.co/x", false));
        assert!(url_allowed("https://us.aws.cdn.hf.co/xet/abc?X=1", true));
        assert!(url_allowed("https://cdn-lfs.huggingface.co/x", true));
        assert!(!url_allowed("https://nothf.co/x", true));
        assert!(!url_allowed("http://us.aws.cdn.hf.co/x", true));
    }

    #[test]
    fn the_host_check_agrees_with_the_url_parser() {
        for bad in [
            "https://evil.com\\.huggingface.co/x",
            "https://huggingface.co@evil.com/",
            "https://huggingface.co:pw@huggingface.co/x",
            "https://evil.com/huggingface.co",
            "https://evil.com/?h=huggingface.co",
            "https://huggingface.co./x",
            "https://huggingface.co:8443/x",
            "https://evil.com#@huggingface.co/",
            "https://x.hf.co./y",
            "ftp://huggingface.co/x",
            "not a url",
        ] {
            assert!(!url_allowed(bad, true), "{bad}");
            assert!(!url_allowed(bad, false), "{bad}");
        }
        assert!(url_allowed("https://HuggingFace.CO/x", false));
        assert!(url_allowed("https://US.AWS.CDN.HF.CO/x", true));
        assert!(url_allowed("https://hf.co/x", true));
        let parsed = url::Url::parse("https://evil.com\\.huggingface.co/x").unwrap();
        assert_eq!(parsed.host_str(), Some("evil.com"));
    }

    #[test]
    fn redirect_locations_resolve_like_the_url_parser() {
        assert_eq!(
            resolve_location("https://huggingface.co/a/b", "//evil.com/x"),
            "https://evil.com/x"
        );
        assert_eq!(
            resolve_location("https://huggingface.co/a/b", "c?d=1"),
            "https://huggingface.co/a/c?d=1"
        );
        assert!(!url_allowed(
            &resolve_location("https://huggingface.co/a", "https://evil.com\\.hf.co/x"),
            true
        ));
    }

    #[test]
    fn the_pull_deadline_is_ten_minutes_or_one_mib_per_second() {
        assert_eq!(pull_deadline(0), Duration::from_secs(600));
        assert_eq!(pull_deadline(32 * 1024 * 1024), Duration::from_secs(600));
        assert_eq!(
            pull_deadline(3_095_033_483),
            Duration::from_secs(3_095_033_483 / (1024 * 1024))
        );
    }

    struct Slow;
    impl Fetch for Slow {
        fn open(&self, _url: &str) -> Result<Download, PullError> {
            struct Drip;
            impl Read for Drip {
                fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                    std::thread::sleep(Duration::from_millis(50));
                    buf[0] = 1;
                    Ok(1)
                }
            }
            Ok(Download {
                body: Box::new(Drip),
                length: None,
            })
        }
    }

    #[test]
    fn a_pull_past_its_deadline_aborts_and_removes_the_part_file() {
        let dir = tmp("deadline");
        let spec = PullSpec {
            size: 1_000_000,
            ..spec_for(b"x")
        };
        let started = Instant::now();
        let err = pull_within(&dir, &spec, &Slow, Duration::from_millis(300)).unwrap_err();
        assert!(matches!(err, PullError::Timeout(_)), "{err}");
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(!dir.join("ggml-test.bin.part").exists());
        assert!(!dir.join("ggml-test.bin").exists());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_part_file_a_running_pull_holds_is_not_stale() {
        let dir = tmp("held");
        std::fs::create_dir_all(&dir).unwrap();
        let held = dir.join("ggml-a.bin.part");
        let f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&held)
            .unwrap();
        assert!(lock_part(&f));
        std::fs::write(dir.join("ggml-b.bin.part"), b"x").unwrap();
        assert_eq!(remove_stale_parts(&dir), 1);
        assert!(held.exists(), "a held part file is left alone");
        drop(f);
        // A child forked by another test can hold the lock for a moment after the close.
        let mut removed = 0;
        for _ in 0..100 {
            removed = remove_stale_parts(&dir);
            if removed == 1 {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(removed, 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn stale_part_files_are_removed_and_counted() {
        let dir = tmp("stale");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("ggml-a.bin.part"), b"x").unwrap();
        std::fs::write(dir.join("ggml-b.bin.part"), b"x").unwrap();
        std::fs::write(dir.join("ggml-c.bin"), b"keep").unwrap();
        assert_eq!(remove_stale_parts(&dir), 2);
        assert!(dir.join("ggml-c.bin").exists());
        assert_eq!(remove_stale_parts(&dir), 0);
        assert_eq!(remove_stale_parts(&dir.join("missing")), 0);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn relative_locations_resolve_against_the_current_host() {
        assert_eq!(
            resolve_location("https://huggingface.co/a/b", "/c/d"),
            "https://huggingface.co/c/d"
        );
        assert_eq!(
            resolve_location("https://huggingface.co/a/b", "https://x.hf.co/z"),
            "https://x.hf.co/z"
        );
    }

    #[test]
    fn concurrent_pulls_of_one_key_share_one_download() {
        let pulls = Arc::new(Pulls::default());
        let ran = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let pulls = pulls.clone();
            let ran = ran.clone();
            handles.push(std::thread::spawn(move || {
                pulls.run("k", || {
                    ran.fetch_add(1, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(300));
                    Ok(PullOutcome::Downloaded { bytes: 1 })
                })
            }));
        }
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(ran.load(Ordering::SeqCst), 1);
        assert_eq!(results.iter().filter(|(_, leader)| *leader).count(), 1);
        assert!(results.iter().all(|(r, _)| r.is_ok()));
        let (again, leader) = pulls.run("k", || Ok(PullOutcome::AlreadyInstalled));
        assert!(leader && again.is_ok(), "the key is free once done");
    }
}
