//! Security and operations hardening: each test starts the real `hark` binary.
//! Tests that need weights or a GPU run with `HARK_MODEL_TESTS=1`; the rest always run.

#![cfg(target_os = "linux")]

use std::io::{Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixDatagram;
use std::path::Path;
use std::time::{Duration, Instant};

mod support;
use support::*;

const JFK: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/jfk.wav");

/// Starts a daemon that must exit; returns (exited non-zero, log).
fn refuses(dir: &Path, config: &str) -> (bool, String) {
    let mut child = spawn(dir, config, &[]);
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break Some(s);
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let log = std::fs::read_to_string(dir.join("daemon.log")).unwrap_or_default();
    (status.is_some_and(|s| !s.success()), log)
}

fn job_json(id: &str, status: &str) -> String {
    let mut rec = hark::jobs::JobRecord::new(
        id.to_string(),
        hark::jobs::JobOptions {
            model: "base".into(),
            language: None,
            prompt: None,
            vocabulary: Vec::new(),
            keywords: Vec::new(),
            temperature: 0.0,
            vad: None,
            granularities: Vec::new(),
            response_format: None,
        },
        hark::jobs::Source::Path { path: "/x".into() },
        1,
    );
    rec.status = match status {
        "running" => hark::jobs::Status::Running,
        _ => hark::jobs::Status::Queued,
    };
    serde_json::to_string(&rec).unwrap()
}

fn on_disk_status(jobs: &Path, id: &str) -> String {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(jobs.join(id).join("job.json")).unwrap()).unwrap();
    v["status"].as_str().unwrap().to_string()
}

#[test]
fn a_second_serve_on_the_same_jobs_dir_refuses_and_leaves_a_running_job_untouched() {
    let a = start("lock-a", "[engine]\npreload = false\n", &[]);
    let jobs = a.dir.join("jobs");
    let id = "job_aaaaaaaaaaaaaaaa";
    std::fs::create_dir_all(jobs.join(id)).unwrap();
    std::fs::write(jobs.join(id).join("job.json"), job_json(id, "running")).unwrap();

    let dir = scratch("lock-b");
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let cfg = config_text(
        &dir,
        &format!("127.0.0.1:{}", free_port()),
        "",
        "[engine]\npreload = false\n",
        "[]",
    )
    .replace(
        &format!("dir = \"{}/jobs\"", dir.display()),
        &format!("dir = \"{}\"", jobs.display()),
    );
    let (refused, log) = refuses(&dir, &cfg);
    assert!(refused, "the second daemon started:\n{log}");
    assert!(log.contains("in use by another hark process"), "{log}");
    assert_eq!(
        on_disk_status(&jobs, id),
        "running",
        "the second process must not recover or requeue the job:\n{log}"
    );
    assert!(!log.contains("interrupted job requeued"), "{log}");
}

#[test]
fn a_taken_port_exits_before_the_jobs_store_is_touched() {
    let a = start("port-a", "[engine]\npreload = false\n", &[]);
    let dir = scratch("port-b");
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let cfg = config_text(
        &dir,
        &format!("127.0.0.1:{}", a.port),
        "",
        "[engine]\npreload = false\n",
        "[]",
    );
    let (refused, log) = refuses(&dir, &cfg);
    assert!(refused, "{log}");
    assert!(log.contains("already in use"), "{log}");
    assert!(
        !dir.join("jobs").exists(),
        "the jobs directory must not be created:\n{log}"
    );
}

fn raw_http(port: u16, request: &str) -> String {
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    s.write_all(request.as_bytes()).unwrap();
    let mut out = String::new();
    let _ = s.read_to_string(&mut out);
    out
}

#[test]
fn a_rebinding_host_header_cannot_read_anything() {
    let d = start("rebind", "[engine]\npreload = false\n", &[]);
    for path in ["/hark/v1/jobs", "/hark/v1/status", "/v1/models", "/health"] {
        let r = d.get_with(path, "Host", &format!("evil.example:{}", d.port));
        assert_eq!(r.status, 403, "{path}: {}", r.body);
        assert_eq!(r.json()["error"]["code"], "host_forbidden", "{path}");
    }
    let r = d.get_with("/hark/v1/jobs", "Host", &format!("localhost:{}", d.port));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(d.get("/hark/v1/jobs").status, 200);
    let no_host = raw_http(d.port, "GET /hark/v1/jobs HTTP/1.0\r\n\r\n");
    assert!(
        no_host.starts_with("HTTP/1.0 200") || no_host.starts_with("HTTP/1.1 200"),
        "{no_host}"
    );
}

#[test]
fn a_model_value_with_a_newline_is_one_escaped_log_field() {
    let d = start("logfield", "[engine]\npreload = false\n", &[]);
    let wav = std::fs::read(JFK).unwrap();
    let r = d.transcribe(&wav, "evil\nINJECTED level=ERROR forged line");
    assert_eq!(r.status, 404, "{}", r.body);
    let log = d.log();
    assert!(
        !log.lines().any(|l| l.trim_start().starts_with("INJECTED")),
        "the newline broke the log line:\n{log}"
    );
    assert!(log.contains("evil\\nINJECTED"), "not escaped:\n{log}");
    assert_eq!(
        log.lines()
            .filter(|l| l.contains("transcription request"))
            .count(),
        1,
        "{log}"
    );
}

#[test]
fn stale_part_files_are_removed_at_startup() {
    let dir = scratch("parts");
    let models = dir.join("models");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::write(models.join("ggml-tiny.bin.part"), b"half").unwrap();
    std::fs::write(models.join("ggml-keep.bin"), b"keep").unwrap();
    let port = free_port();
    let cfg = config_text(
        &dir,
        &format!("127.0.0.1:{port}"),
        "",
        "[engine]\npreload = false\n",
        "[]",
    );
    let d = start_in(dir, port, &cfg, &[]);
    assert!(!models.join("ggml-tiny.bin.part").exists());
    assert!(models.join("ggml-keep.bin").exists());
    assert!(
        d.log().contains("removed stale partial model downloads"),
        "{}",
        d.log()
    );
}

#[test]
fn unsafe_configs_are_refused_at_start() {
    let (refused, log) =
        start_refused("remote-nokey", "0.0.0.0:18803", "allow_non_loopback = true");
    assert!(refused, "{log}");
    assert!(log.contains("allow_unauthenticated_remote"), "{log}");

    let dir = scratch("models-in-extra");
    let extra = dir.join("extra");
    std::fs::create_dir_all(&extra).unwrap();
    let cfg = format!(
        "[server]\nlisten = \"127.0.0.1:{}\"\n[jobs]\ndir = \"{d}/jobs\"\n[models]\ndir = \"{e}/mine\"\nextra_dirs = [\"{e}\"]\n",
        free_port(),
        d = dir.display(),
        e = extra.display()
    );
    let (refused, log) = refuses(&dir, &cfg);
    assert!(refused, "{log}");
    assert!(log.contains("models.extra_dirs"), "{log}");
}

#[test]
fn a_symlinked_key_file_is_refused_at_start() {
    let keys = scratch("keylink-keys");
    let key = keys.join("real.key");
    std::fs::write(&key, "a-secret-key\n").unwrap();
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
    let link = keys.join("link.key");
    std::os::unix::fs::symlink(&key, &link).unwrap();
    let dir = scratch("keylink");
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let cfg = config_text(
        &dir,
        &format!("127.0.0.1:{}", free_port()),
        &format!("api_key_file = \"{}\"", link.display()),
        "[engine]\npreload = false\n",
        "[]",
    );
    let (refused, log) = refuses(&dir, &cfg);
    assert!(refused, "{log}");
    assert!(log.contains("symbolic link"), "{log}");
}

/// A directory with an `ffmpeg` that sleeps forever, standing in for a slow decode.
fn slow_ffmpeg_path(marker: &str) -> String {
    let bin = scratch(&format!("slowbin-{marker}"));
    let script = bin.join("ffmpeg");
    std::fs::write(&script, format!("#!/bin/sh\nexec sleep {marker}\n")).unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    format!(
        "{}:{}",
        bin.display(),
        std::env::var("PATH").unwrap_or_default()
    )
}

fn sleeping(marker: &str) -> bool {
    std::fs::read_dir("/proc")
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| {
            std::fs::read(e.path().join("cmdline")).is_ok_and(|c| {
                let text = String::from_utf8_lossy(&c).replace('\0', " ");
                text.starts_with("sleep ") && text.contains(marker)
            })
        })
}

fn submit_slow_job(d: &Daemon, media: &Path) -> String {
    let clip = media.join("clip.bin");
    std::fs::write(&clip, b"not a wav, so ffmpeg is asked to decode it").unwrap();
    let body = format!("{{\"path\":\"{}\",\"model\":\"base\"}}", clip.display());
    let r = finish(
        agent()
            .post(&d.url("/hark/v1/jobs"))
            .set("Content-Type", "application/json")
            .send_bytes(body.as_bytes()),
    );
    assert_eq!(r.status, 202, "{}", r.body);
    r.json()["id"].as_str().unwrap().to_string()
}

fn wait_status(d: &Daemon, id: &str, want: &str) {
    let end = Instant::now() + Duration::from_secs(30);
    loop {
        let v = d.get(&format!("/hark/v1/jobs/{id}")).json();
        if v["status"] == want {
            return;
        }
        assert!(Instant::now() < end, "job stuck: {v}\n{}", d.log());
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Each test sleeps for its own number of seconds, so its decode child is told apart from the others'.
fn slow_decode_daemon(tag: &str, media: &Path, marker: &str) -> Daemon {
    let path = slow_ffmpeg_path(marker);
    let dir = scratch(tag);
    let models = dir.join("models");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::write(models.join("ggml-base.bin"), b"fake weights").unwrap();
    std::fs::create_dir_all(media).unwrap();
    let port = free_port();
    let cfg = config_text(
        &dir,
        &format!("127.0.0.1:{port}"),
        "",
        "[engine]\npreload = false\n",
        "[]",
    )
    .replace(
        "[jobs]\n",
        &format!("[jobs]\npath_roots = [\"{}\"]\n", media.display()),
    );
    start_in(dir, port, &cfg, &[("PATH", path)])
}

#[test]
fn sigterm_during_a_slow_decode_exits_promptly_and_the_job_stays_recoverable() {
    let media = scratch("slowdecode-media");
    let mut d = slow_decode_daemon("slowdecode", &media, "6173");
    let id = submit_slow_job(&d, &media);
    wait_status(&d, &id, "running");
    std::thread::sleep(Duration::from_millis(500));
    assert!(sleeping("6173"), "the decode is not running");
    // SAFETY: signalling the child this test spawned.
    unsafe { libc::kill(d.child.id() as i32, libc::SIGTERM) };
    let started = Instant::now();
    let status = loop {
        if let Some(s) = d.child.try_wait().unwrap() {
            break s;
        }
        assert!(
            started.elapsed() < Duration::from_secs(35),
            "still running 35 s after SIGTERM:\n{}",
            d.log()
        );
        std::thread::sleep(Duration::from_millis(100));
    };
    assert!(status.success(), "{status:?}\n{}", d.log());
    assert!(!sleeping("6173"), "the decode child was left running");
    let jobs = d.dir.join("jobs");
    assert_eq!(
        on_disk_status(&jobs, &id),
        "running",
        "recoverable, not failed"
    );
    let path = slow_ffmpeg_path("6173");
    let port = free_port();
    let cfg = std::fs::read_to_string(d.dir.join("config.toml"))
        .unwrap()
        .replace(
            &format!("127.0.0.1:{}", d.port),
            &format!("127.0.0.1:{port}"),
        );
    let dir = d.dir.clone();
    std::mem::forget(d);
    let again = start_in(dir, port, &cfg, &[("PATH", path)]);
    assert!(
        again.log().contains("interrupted job requeued"),
        "{}",
        again.log()
    );
    let r = finish(
        agent()
            .delete(&again.url(&format!("/hark/v1/jobs/{id}")))
            .call(),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let end = Instant::now() + Duration::from_secs(10);
    while sleeping("6173") {
        assert!(
            Instant::now() < end,
            "the restarted decode was not cancelled"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
fn deleting_a_job_in_the_decode_step_kills_ffmpeg() {
    let media = scratch("slowdelete-media");
    let d = slow_decode_daemon("slowdelete", &media, "6174");
    let id = submit_slow_job(&d, &media);
    wait_status(&d, &id, "running");
    std::thread::sleep(Duration::from_millis(500));
    assert!(sleeping("6174"));
    let r = finish(
        agent()
            .delete(&d.url(&format!("/hark/v1/jobs/{id}")))
            .call(),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let end = Instant::now() + Duration::from_secs(10);
    while sleeping("6174") {
        assert!(Instant::now() < end, "ffmpeg survived the cancel");
        std::thread::sleep(Duration::from_millis(100));
    }
    assert_eq!(
        d.get(&format!("/hark/v1/jobs/{id}")).json()["status"],
        "cancelled"
    );
}

/// The port accepts connections and SIGTERM is handled while the model is still loading; READY comes last.
#[test]
fn the_socket_is_bound_before_the_preload_and_ready_follows_the_model() {
    if !model_tests_enabled() {
        return;
    }
    let dir = scratch("ready");
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let sock_path = dir.join("notify.sock");
    let sock = UnixDatagram::bind(&sock_path).unwrap();
    sock.set_read_timeout(Some(Duration::from_secs(120)))
        .unwrap();
    let port = free_port();
    let cfg = config_text(
        &dir,
        &format!("127.0.0.1:{port}"),
        "",
        "[engine]\ndefault_model = \"base\"\nstates = 1\npreload = true\n",
        &models_extra_dirs(),
    );
    let child = spawn(
        &dir,
        &cfg,
        &[("NOTIFY_SOCKET", sock_path.display().to_string())],
    );
    let d = Daemon {
        child,
        port,
        dir: dir.clone(),
    };
    let log_of = |d: &Daemon| d.log();
    let bound_at = Instant::now();
    loop {
        if std::net::TcpStream::connect(("127.0.0.1", port)).is_ok() {
            break;
        }
        assert!(
            bound_at.elapsed() < Duration::from_secs(30),
            "never bound:\n{}",
            d.log()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(
        !log_of(&d).contains("model loaded"),
        "the port only opened after the model loaded:\n{}",
        log_of(&d)
    );
    let mut buf = [0u8; 64];
    let n = sock.recv(&mut buf).expect("READY");
    assert_eq!(&buf[..n], b"READY=1");
    assert!(
        log_of(&d).contains("model loaded"),
        "READY came before the preload finished:\n{}",
        log_of(&d)
    );
    let r = d.get("/health");
    assert_eq!(r.status, 200);
}

#[test]
fn ready_is_still_sent_when_the_default_model_is_missing() {
    let dir = scratch("ready-missing");
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let sock_path = dir.join("notify.sock");
    let sock = UnixDatagram::bind(&sock_path).unwrap();
    sock.set_read_timeout(Some(Duration::from_secs(60)))
        .unwrap();
    let port = free_port();
    let cfg = config_text(&dir, &format!("127.0.0.1:{port}"), "", "", "[]");
    let child = spawn(
        &dir,
        &cfg,
        &[("NOTIFY_SOCKET", sock_path.display().to_string())],
    );
    let d = Daemon { child, port, dir };
    let mut buf = [0u8; 64];
    let n = sock.recv(&mut buf).expect("READY even without a model");
    assert_eq!(&buf[..n], b"READY=1");
    assert!(d.log().contains("skipping preload"), "{}", d.log());
    assert_eq!(d.get("/health").status, 200);
}
