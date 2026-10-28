//! Failure handling ("fails safe"): each test starts the real `hark` binary and drives it over HTTP.
//! Tests that need weights or a Vulkan GPU run with `HARK_MODEL_TESTS=1`; the rest always run.

#![cfg_attr(not(unix), allow(unused_imports, dead_code))]

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

mod support;
use support::*;

const JFK: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/jfk.wav");
const SENTENCE: &str = "ask not what your country can do for you";
const KEY: &str = "failure-test-key-0123456789";

fn jfk() -> Vec<u8> {
    std::fs::read(JFK).unwrap()
}

fn normalized(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn have_discrete_gpu() -> bool {
    hark::engine::enumerate_gpus().iter().any(|g| g.discrete)
}

/// Builds the LD_PRELOAD shim that denies device-local `vkAllocateMemory` while a file exists.
#[cfg(target_os = "linux")]
fn build_shim() -> PathBuf {
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("libvkoom_failure.so");
    let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/support/vkoom.c");
    let o = Command::new("cc")
        .args(["-shared", "-fPIC", "-O1", "-Wall", "-o"])
        .arg(&out)
        .arg(src)
        .args(["-ldl", "-lpthread"])
        .output()
        .expect("cc");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    out
}

#[cfg(target_os = "linux")]
fn shim_env(deny_file: &Path) -> Vec<(&'static str, String)> {
    vec![
        ("LD_PRELOAD", build_shim().display().to_string()),
        ("VKOOM_DENY_FILE", deny_file.display().to_string()),
    ]
}

#[cfg(target_os = "linux")]
fn assert_fell_back_to_cpu(d: &Daemon) {
    let health = d.get("/health").json();
    assert_eq!(health["status"], "degraded", "{health}");
    assert!(health["version"].is_string());
    let status = d.get("/hark/v1/status").json();
    assert_eq!(status["degraded"], true, "{status}");
    assert_eq!(status["device"]["name"], "cpu", "{status}");
    assert!(status["device"]["fell_back_from"].is_string(), "{status}");
    let err = status["last_error"]["message"].as_str().unwrap_or("");
    assert!(err.contains("CPU"), "last_error: {status}");
}

/// Statement 6(a): the allocation of the model buffer fails at load; the request still succeeds.
#[cfg(target_os = "linux")]
#[test]
fn gpu_oom_at_load_falls_back_to_cpu_once_and_the_request_succeeds() {
    if !model_tests_enabled() {
        return;
    }
    if !have_discrete_gpu() {
        eprintln!("skipped: no Vulkan GPU");
        return;
    }
    let deny = scratch("oom-load-flag").join("deny");
    std::fs::write(&deny, b"").unwrap();
    let d = start_with_models(
        "oom-load",
        "[engine]\ndefault_model = \"base\"\nstates = 1\npreload = false\n",
        &shim_env(&deny),
    );
    assert_eq!(
        d.get("/health").json()["status"],
        "ok",
        "healthy before the first load"
    );
    let r = d.transcribe(&jfk(), "base");
    assert_eq!(r.status, 200, "{}\n{}", r.body, d.log());
    assert!(
        normalized(r.json()["text"].as_str().unwrap()).contains(SENTENCE),
        "{}",
        r.body
    );
    assert_fell_back_to_cpu(&d);
    let log = d.log();
    assert!(
        log.contains("vkoom: denying device-local allocation"),
        "the shim never fired:\n{log}"
    );
    assert!(
        log.contains("failed to allocate the Vulkan0 model buffer"),
        "{log}"
    );
    assert_eq!(
        log.matches("GPU load failed; retrying on the CPU").count(),
        1,
        "{log}"
    );
    let again = d.transcribe(&jfk(), "base");
    assert_eq!(again.status, 200, "{}", again.body);
    assert_eq!(
        d.log()
            .matches("GPU load failed; retrying on the CPU")
            .count(),
        1,
        "the process falls back once, not per request"
    );
    let status = d.get("/hark/v1/status").json();
    assert_eq!(status["models"][0]["id"], "base");
}

/// Statement 6(b): the load succeeds on the GPU, then a device-local allocation fails during `full`.
#[cfg(target_os = "linux")]
#[test]
fn gpu_oom_during_full_falls_back_to_cpu_for_that_chunk_and_the_request_succeeds() {
    if !model_tests_enabled() {
        return;
    }
    if !have_discrete_gpu() {
        eprintln!("skipped: no Vulkan GPU");
        return;
    }
    let deny = scratch("oom-run-flag").join("deny");
    let d = start_with_models(
        "oom-run",
        "[engine]\ndefault_model = \"base\"\nstates = 1\npreload = true\n",
        &shim_env(&deny),
    );
    let status = d.get("/hark/v1/status").json();
    assert_eq!(status["degraded"], false, "{status}");
    assert_ne!(
        status["device"]["name"], "cpu",
        "the preload must be on the GPU: {status}"
    );
    assert_eq!(status["models"][0]["id"], "base");
    let loaded_at = d.log().find("model loaded").expect("model loaded line");
    assert!(d.log().contains("gpu=true"), "{}", d.log());
    std::fs::write(&deny, b"").unwrap();
    let r = d.transcribe(&jfk(), "base");
    assert_eq!(r.status, 200, "{}\n{}", r.body, d.log());
    assert!(
        normalized(r.json()["text"].as_str().unwrap()).contains(SENTENCE),
        "{}",
        r.body
    );
    assert_fell_back_to_cpu(&d);
    let log = d.log();
    let denied_at = log
        .find("vkoom: denying device-local allocation")
        .expect("the shim fired");
    assert!(
        denied_at > loaded_at,
        "the denial came after the load, so it hit the run"
    );
    assert!(log.contains("ggml backend threw, returning -100"), "{log}");
    assert!(
        !log.contains("GPU load failed"),
        "this was not a load failure:\n{log}"
    );
    let status = d.get("/hark/v1/status").json();
    assert_eq!(
        status["models"][0]["last_load_reason"], "device_fallback",
        "{status}"
    );
    let again = d.transcribe(&jfk(), "base");
    assert_eq!(again.status, 200, "{}", again.body);
}

/// Fallback is not permanent: an idle degraded daemon retries the GPU every `gpu_retry_secs`.
#[cfg(target_os = "linux")]
#[test]
fn the_cpu_fallback_is_undone_once_the_gpu_works_again() {
    if !model_tests_enabled() {
        return;
    }
    if !have_discrete_gpu() {
        eprintln!("skipped: no Vulkan GPU");
        return;
    }
    let deny = scratch("gpu-retry-flag").join("deny");
    let d = start_with_models(
        "gpu-retry",
        "[engine]\ndefault_model = \"base\"\nstates = 1\npreload = true\ngpu_retry_secs = 2\n",
        &shim_env(&deny),
    );
    assert_eq!(d.get("/hark/v1/status").json()["degraded"], false);
    std::fs::write(&deny, b"").unwrap();
    let r = d.transcribe(&jfk(), "base");
    assert_eq!(r.status, 200, "{}\n{}", r.body, d.log());
    assert_fell_back_to_cpu(&d);
    std::thread::sleep(std::time::Duration::from_secs(6));
    let still = d.get("/hark/v1/status").json();
    assert_eq!(still["degraded"], true, "the GPU is still denied: {still}");
    assert_eq!(d.get("/health").json()["status"], "degraded");
    std::fs::remove_file(&deny).unwrap();
    let end = std::time::Instant::now() + std::time::Duration::from_secs(90);
    let status = loop {
        let s = d.get("/hark/v1/status").json();
        if s["degraded"] == false && s["device"]["name"] != "cpu" {
            break s;
        }
        assert!(
            std::time::Instant::now() < end,
            "never returned to the GPU: {s}\n{}",
            d.log()
        );
        std::thread::sleep(std::time::Duration::from_millis(500));
    };
    assert_eq!(
        status["models"][0]["last_load_reason"], "gpu_retry",
        "{status}"
    );
    assert_eq!(status["device"]["fell_back_from"], serde_json::Value::Null);
    assert_eq!(d.get("/health").json()["status"], "ok");
    assert!(d.log().contains("back on the GPU"), "{}", d.log());
    let again = d.transcribe(&jfk(), "base");
    assert_eq!(again.status, 200, "{}", again.body);
    assert!(
        normalized(again.json()["text"].as_str().unwrap()).contains(SENTENCE),
        "{}",
        again.body
    );
}

/// Statement 6(c): garbage audio is a 400 envelope and the same server then transcribes real audio.
#[test]
fn undecodable_audio_returns_a_400_envelope_and_the_next_request_works() {
    if !model_tests_enabled() {
        return;
    }
    let mut d = start_with_models(
        "undecodable",
        "[engine]\ndefault_model = \"base\"\nstates = 1\ndevice = \"cpu\"\npreload = false\n",
        &[],
    );
    let wav = jfk();
    let samples: [(&str, Vec<u8>); 4] = [
        ("text", b"this is not audio at all".to_vec()),
        ("zeros", vec![0u8; 4096]),
        ("truncated wav header", wav[..30].to_vec()),
        ("html", b"<html><body>404</body></html>".to_vec()),
    ];
    for (name, bytes) in samples {
        let r = d.transcribe(&bytes, "whisper-1");
        assert_eq!(r.status, 400, "{name}: {}", r.body);
        let e = &r.json()["error"];
        assert_eq!(e["type"], "invalid_request_error", "{name}: {}", r.body);
        assert_eq!(e["code"], "invalid_audio", "{name}: {}", r.body);
        assert_eq!(e["param"], "file", "{name}: {}", r.body);
        assert!(e["message"].is_string());
        assert!(d.alive(), "{name}: the daemon died:\n{}", d.log());
    }
    assert_eq!(d.get("/health").json()["status"], "ok");
    let r = d.transcribe(&wav, "whisper-1");
    assert_eq!(r.status, 200, "{}\n{}", r.body, d.log());
    assert!(
        normalized(r.json()["text"].as_str().unwrap()).contains(SENTENCE),
        "{}",
        r.body
    );
}

/// Statement 6(d): a non-loopback listen address is refused at start unless explicitly allowed.
#[test]
fn non_loopback_listen_is_refused_unless_allowed() {
    for addr in ["0.0.0.0:18800", "[::]:18800", "192.168.1.5:18800"] {
        let (refused, log) = start_refused("nonloop", addr, "");
        assert!(refused, "hark started on {addr}:\n{log}");
        assert!(log.contains("not a loopback address"), "{addr}: {log}");
        assert!(log.contains("allow_non_loopback"), "{addr}: {log}");
    }
    let dir = scratch("nonloop-ok");
    std::fs::create_dir_all(dir.join("models")).unwrap();
    let port = free_port();
    let cfg = config_text(
        &dir,
        &format!("0.0.0.0:{port}"),
        "allow_non_loopback = true\nallow_unauthenticated_remote = true",
        "[engine]\npreload = false\n",
        "[]",
    );
    let d = start_in(dir, port, &cfg, &[]);
    assert_eq!(d.get("/health").status, 200, "allowed non-loopback serves");
}

/// Statement 6(e): any request carrying an Origin header is a 403, health included, with no CORS headers.
#[test]
fn origin_requests_get_403_on_every_route() {
    let d = start("origin", "[engine]\npreload = false\n", &[]);
    for (method, path) in [
        ("GET", "/health"),
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
        ("OPTIONS", "/v1/audio/transcriptions"),
        ("GET", "/no/such/route"),
    ] {
        let res = agent()
            .request(method, &d.url(path))
            .set("Origin", "http://evil.example")
            .call();
        let resp = match res {
            Err(ureq::Error::Status(_, r)) => r,
            Ok(r) => r,
            Err(e) => panic!("{e}"),
        };
        assert_eq!(resp.status(), 403, "{method} {path}");
        assert!(
            resp.headers_names()
                .iter()
                .all(|h| !h.starts_with("access-control")),
            "{method} {path}: CORS header present"
        );
        let body: Value = serde_json::from_str(&resp.into_string().unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "origin_forbidden", "{method} {path}");
    }
    assert_eq!(
        d.get("/health").status,
        200,
        "the same routes work without Origin"
    );
}

/// Statement 6(f): job paths outside `jobs.path_roots` are refused, symlinks out included.
#[cfg(unix)]
#[test]
fn out_of_root_job_paths_get_403() {
    let dir = scratch("roots-seed");
    let root = dir.join("root");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::copy(JFK, root.join("inside.wav")).unwrap();
    std::fs::copy(JFK, dir.join("outside.wav")).unwrap();
    std::os::unix::fs::symlink(dir.join("outside.wav"), root.join("link.wav")).unwrap();
    let roots_cfg = |d: &Path, port: u16| {
        config_text(
            d,
            &format!("127.0.0.1:{port}"),
            "",
            "[engine]\npreload = false\n",
            "[]",
        )
        .replace(
            "[jobs]\n",
            &format!("[jobs]\npath_roots = [\"{}\"]\n", root.display()),
        )
    };
    let d = {
        let dd = scratch("roots");
        std::fs::create_dir_all(dd.join("models")).unwrap();
        let port = free_port();
        let cfg = roots_cfg(&dd, port);
        start_in(dd, port, &cfg, &[])
    };
    std::fs::write(d.dir.join("models").join("ggml-base.bin"), b"fake model").unwrap();
    let submit = |path: &str| {
        let body = serde_json::json!({"model": "base", "path": path}).to_string();
        finish(
            agent()
                .post(&d.url("/hark/v1/jobs"))
                .set("Content-Type", "application/json")
                .send_bytes(body.as_bytes()),
        )
    };
    for path in [
        dir.join("outside.wav").display().to_string(),
        root.join("link.wav").display().to_string(),
        format!("{}/../outside.wav", root.display()),
        "/etc/hostname".to_string(),
        "relative/path.wav".to_string(),
        root.display().to_string(),
        "/proc/self/environ".to_string(),
    ] {
        let r = submit(&path);
        assert_eq!(r.status, 403, "{path}: {}", r.body);
        assert_eq!(r.json()["error"]["code"], "path_not_allowed", "{path}");
    }
    let ok = submit(&root.join("inside.wav").display().to_string());
    assert_eq!(ok.status, 202, "an in-root file is accepted: {}", ok.body);
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
fn key_file(dir: &Path, mode: u32) -> PathBuf {
    let p = dir.join("api.key");
    std::fs::write(&p, format!("{KEY}\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
    p
}

/// Statement 6(g): with `api_key_file`, every route but `/health` needs the key; a loose key file refuses to start.
#[cfg(unix)]
#[test]
fn api_key_guards_every_route_but_health_and_a_loose_key_file_refuses_to_start() {
    let seed = scratch("key-seed");
    for mode in [0o644, 0o640, 0o604, 0o666, 0o700] {
        let loose = key_file(&seed, mode);
        let (refused, log) = start_refused(
            "key-loose",
            "127.0.0.1:18801",
            &format!("api_key_file = \"{}\"", loose.display()),
        );
        assert!(refused, "a {mode:04o} key file was accepted:\n{log}");
        assert!(log.contains("0600"), "{mode:04o}: {log}");
    }
    let (refused, log) = start_refused(
        "key-missing",
        "127.0.0.1:18802",
        "api_key_file = \"/nonexistent/hark.key\"",
    );
    assert!(refused && log.contains("api_key_file"), "{log}");
    let (refused, log) = start_refused(
        "key-dir",
        "127.0.0.1:18803",
        &format!("api_key_file = \"{}\"", seed.display()),
    );
    assert!(refused && log.contains("not a regular file"), "{log}");

    let key = key_file(&seed, 0o600);
    let d = start_with(
        "key-ok",
        &format!("api_key_file = \"{}\"", key.display()),
        "[engine]\npreload = false\n",
        "[]",
        &[],
    );
    let routes = [
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
        ("DELETE", "/hark/v1/jobs/job_x"),
        ("GET", "/no/such/route"),
    ];
    for (method, path) in routes {
        for header in [None, Some("Bearer wrong"), Some("Bearer "), Some(KEY)] {
            let mut req = agent().request(method, &d.url(path));
            if let Some(h) = header {
                req = req.set("Authorization", h);
            }
            let r = finish(req.call());
            assert_eq!(r.status, 401, "{method} {path} with {header:?}: {}", r.body);
            assert_eq!(
                r.json()["error"]["code"],
                "invalid_api_key",
                "{method} {path}"
            );
        }
        let r = finish(
            agent()
                .request(method, &d.url(path))
                .set("Authorization", &format!("Bearer {KEY}"))
                .call(),
        );
        assert_ne!(
            r.status, 401,
            "{method} {path} with the right key: {}",
            r.body
        );
    }
    let health = d.get("/health");
    assert_eq!(health.status, 200);
    assert_eq!(health.json()["status"], "ok");
    let ok = d.get_with("/hark/v1/status", "Authorization", &format!("Bearer {KEY}"));
    assert_eq!(ok.status, 200, "{}", ok.body);
    let _ = std::fs::remove_dir_all(&seed);
    assert!(!d.log().contains(KEY), "the key must never be logged");
}
