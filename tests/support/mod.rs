//! Shared helpers for the daemon-level tests: start the real `hark` binary and talk to it over HTTP.
#![allow(dead_code)]

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

/// Read-only model directory as a TOML array: `HARK_TEST_MODELS_DIR`, else `~/.config/Wspr/models`.
pub fn models_extra_dirs() -> String {
    let dir = test_models_dir();
    format!("[{:?}]", dir)
}

pub fn test_models_dir() -> String {
    std::env::var("HARK_TEST_MODELS_DIR").unwrap_or_else(|_| "~/.config/Wspr/models".to_string())
}

pub fn model_tests_enabled() -> bool {
    let on = std::env::var("HARK_MODEL_TESTS").is_ok_and(|v| v == "1");
    if !on {
        eprintln!("skipped: set HARK_MODEL_TESTS=1 to run model-backed tests");
    }
    on
}

pub fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

pub fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("hark-failure-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

pub struct Daemon {
    pub child: Child,
    pub port: u16,
    pub dir: PathBuf,
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub struct Reply {
    pub status: u16,
    pub body: String,
}

impl Reply {
    pub fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or(Value::Null)
    }
}

pub fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(120))
        .build()
}

pub fn finish(res: Result<ureq::Response, ureq::Error>) -> Reply {
    match res {
        Ok(r) => Reply {
            status: r.status(),
            body: r.into_string().unwrap_or_default(),
        },
        Err(ureq::Error::Status(code, r)) => Reply {
            status: code,
            body: r.into_string().unwrap_or_default(),
        },
        Err(e) => panic!("transport error: {e}"),
    }
}

pub fn multipart(fields: &[(&str, &str)], file: &[u8]) -> (String, Vec<u8>) {
    let boundary = "failure-test-boundary";
    let mut body = Vec::new();
    for (k, v) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n")
                .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.bin\"\r\nContent-Type: application/octet-stream\r\n\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

impl Daemon {
    pub fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }

    pub fn get(&self, path: &str) -> Reply {
        finish(agent().get(&self.url(path)).call())
    }

    pub fn get_with(&self, path: &str, header: &str, value: &str) -> Reply {
        finish(agent().get(&self.url(path)).set(header, value).call())
    }

    pub fn transcribe(&self, file: &[u8], model: &str) -> Reply {
        let (ctype, body) = multipart(&[("model", model), ("language", "en")], file);
        finish(
            agent()
                .post(&self.url("/v1/audio/transcriptions"))
                .set("Content-Type", &ctype)
                .send_bytes(&body),
        )
    }

    pub fn log(&self) -> String {
        std::fs::read_to_string(self.dir.join("daemon.log")).unwrap_or_default()
    }

    pub fn alive(&mut self) -> bool {
        self.child.try_wait().unwrap().is_none()
    }
}

/// Config text for a daemon under `dir`. `server_extra` lands inside `[server]`, `extra` is
/// appended verbatim, `extra_dirs` is the TOML array of read-only model directories.
pub fn config_text(
    dir: &Path,
    listen: &str,
    server_extra: &str,
    extra: &str,
    extra_dirs: &str,
) -> String {
    format!(
        "[server]\nlisten = \"{listen}\"\n{server_extra}\n[jobs]\ndir = \"{d}/jobs\"\n[models]\ndir = \"{d}/models\"\nextra_dirs = {extra_dirs}\n{extra}\n",
        d = dir.display()
    )
}

pub fn spawn(dir: &Path, config: &str, env: &[(&str, String)]) -> Child {
    std::fs::write(dir.join("config.toml"), config).unwrap();
    let log = std::fs::File::create(dir.join("daemon.log")).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_hark"));
    cmd.args(["serve", "--config"])
        .arg(dir.join("config.toml"))
        .stdout(Stdio::from(log.try_clone().unwrap()))
        .stderr(Stdio::from(log))
        .env_remove("RUST_LOG")
        .env("NO_COLOR", "1");
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.spawn().expect("spawn hark")
}

/// Starts a daemon and waits for `/health` to answer.
pub fn start(tag: &str, extra: &str, env: &[(&str, String)]) -> Daemon {
    start_with(tag, "", extra, "[]", env)
}

/// Like `start`, with the test model folder readable so `base` resolves to real weights.
pub fn start_with_models(tag: &str, extra: &str, env: &[(&str, String)]) -> Daemon {
    start_with(tag, "", extra, &models_extra_dirs(), env)
}

pub fn start_with(
    tag: &str,
    server_extra: &str,
    extra: &str,
    extra_dirs: &str,
    env: &[(&str, String)],
) -> Daemon {
    // free_port can race other processes for the port; retry on a lost race.
    for attempt in 1.. {
        let dir = scratch(tag);
        std::fs::create_dir_all(dir.join("models")).unwrap();
        let port = free_port();
        let cfg = config_text(
            &dir,
            &format!("127.0.0.1:{port}"),
            server_extra,
            extra,
            extra_dirs,
        );
        match try_start_in(dir, port, &cfg, env) {
            Ok(d) => return d,
            Err(log) if attempt < 5 && log.contains("is already in use") => continue,
            Err(log) => panic!("daemon exited early:\n{log}"),
        }
    }
    unreachable!()
}

pub fn start_in(dir: PathBuf, port: u16, config: &str, env: &[(&str, String)]) -> Daemon {
    try_start_in(dir, port, config, env).unwrap_or_else(|log| panic!("daemon exited early:\n{log}"))
}

/// Starts the daemon and waits for /health; `Err` holds the log when it exits first.
pub fn try_start_in(
    dir: PathBuf,
    port: u16,
    config: &str,
    env: &[(&str, String)],
) -> Result<Daemon, String> {
    let child = spawn(&dir, config, env);
    let mut d = Daemon { child, port, dir };
    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        if !d.alive() {
            return Err(d.log());
        }
        if let Ok(r) = agent().get(&d.url("/health")).call()
            && r.status() == 200
        {
            return Ok(d);
        }
        assert!(Instant::now() < deadline, "no /health:\n{}", d.log());
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Runs a daemon that must refuse to start; returns its exit status and output.
pub fn start_refused(tag: &str, listen: &str, server_extra: &str) -> (bool, String) {
    let dir = scratch(tag);
    let mut child = spawn(
        &dir,
        &config_text(&dir, listen, server_extra, "", "[]"),
        &[],
    );
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
        std::thread::sleep(Duration::from_millis(100));
    };
    let mut log = String::new();
    std::fs::File::open(dir.join("daemon.log"))
        .unwrap()
        .read_to_string(&mut log)
        .unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    (status.is_some_and(|s| !s.success()), log)
}
