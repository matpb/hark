//! Model management against the real daemon: pull (a real download of `tiny-q5_1`), load, unload,
//! delete, LRU residency and the CLI. Runs with `HARK_MODEL_TESTS=1` (needs network and weights).

use std::path::Path;
use std::process::Command;

use serde_json::Value;
use sha2::{Digest, Sha256};

mod support;
use support::*;

const JFK: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/jfk.wav");
const TINY: &str = "tiny-q5_1";
const TINY_SHA: &str = "818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7";
const TINY_SIZE: u64 = 32_152_673;

fn post(d: &Daemon, path: &str, body: Option<&str>) -> Reply {
    let req = agent().post(&d.url(path));
    finish(match body {
        Some(b) => req
            .set("Content-Type", "application/json")
            .send_bytes(b.as_bytes()),
        None => req.call(),
    })
}

fn delete(d: &Daemon, path: &str) -> Reply {
    finish(agent().delete(&d.url(path)).call())
}

fn status(d: &Daemon) -> Value {
    d.get("/hark/v1/status").json()
}

fn model_ids(status: &Value) -> Vec<String> {
    status["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect()
}

fn sha256_file(p: &Path) -> String {
    let mut h = Sha256::new();
    h.update(std::fs::read(p).unwrap());
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

struct Cli {
    config: std::path::PathBuf,
}

impl Cli {
    fn run(&self, args: &[&str]) -> (bool, String, String) {
        let out = Command::new(env!("CARGO_BIN_EXE_hark"))
            .args(args)
            .arg("--config")
            .arg(&self.config)
            .env_remove("RUST_LOG")
            .env("NO_COLOR", "1")
            .output()
            .expect("run hark");
        let (stdout, stderr) = (
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        );
        eprintln!("$ hark {}\n{stdout}{stderr}", args.join(" "));
        (out.status.success(), stdout, stderr)
    }
}

#[test]
fn pull_load_unload_and_delete_a_real_model_and_drive_it_from_the_cli() {
    if !model_tests_enabled() {
        return;
    }
    let dir = scratch("admin-extra");
    let extra = dir.join("extra");
    std::fs::create_dir_all(&extra).unwrap();
    std::fs::write(extra.join("ggml-base.bin"), b"read-only fake").unwrap();
    let cfg_extra = format!("[\"{}\"]", extra.display());
    let d = start_with(
        "admin",
        "",
        "[engine]\ndefault_model = \"tiny-q5_1\"\nstates = 1\npreload = false\nidle_unload_secs = 0\n",
        &cfg_extra,
        &[],
    );
    let own = d.dir.join("models");
    let cli = Cli {
        config: d.dir.join("config.toml"),
    };

    let list = d.get("/hark/v1/models").json();
    let entry = |id: &str| {
        list["data"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["id"] == id)
            .cloned()
            .unwrap()
    };
    assert_eq!(entry(TINY)["installed"], false);
    assert_eq!(entry("base")["location"], "extra_dir");

    // Two concurrent pulls of one model share a single download.
    let replies: Vec<Reply> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..2)
            .map(|_| {
                s.spawn(|| {
                    post(
                        &d,
                        "/hark/v1/models/pull",
                        Some(&format!("{{\"id\":\"{TINY}\"}}")),
                    )
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for r in &replies {
        assert_eq!(r.status, 200, "{}\n{}", r.body, d.log());
    }
    let leaders = replies
        .iter()
        .filter(|r| r.json()["status"] == "installed" && r.json()["shared_download"] == false)
        .count();
    assert_eq!(
        leaders,
        1,
        "exactly one request downloaded: {:?}",
        replies.iter().map(|r| &r.body).collect::<Vec<_>>()
    );
    assert_eq!(
        d.log().matches("model download started").count(),
        1,
        "{}",
        d.log()
    );
    let file = own.join(format!("ggml-{TINY}.bin"));
    assert_eq!(std::fs::metadata(&file).unwrap().len(), TINY_SIZE);
    assert_eq!(sha256_file(&file), TINY_SHA);
    assert!(!own.join(format!("ggml-{TINY}.bin.part")).exists());

    let again = post(
        &d,
        "/hark/v1/models/pull",
        Some(&format!("{{\"id\":\"{TINY}\"}}")),
    );
    assert_eq!(
        (again.status, again.json()["status"].clone()),
        (200, "already_installed".into()),
        "{}",
        again.body
    );
    assert_eq!(
        d.log().matches("model download started").count(),
        1,
        "no second download"
    );

    // Load, transcribe, unload.
    let r = post(&d, &format!("/hark/v1/models/{TINY}/load"), None);
    assert_eq!(r.status, 200, "{}\n{}", r.body, d.log());
    let s = status(&d);
    assert_eq!(model_ids(&s), vec![TINY.to_string()]);
    assert_eq!(s["models"][0]["last_load_reason"], "admin");
    let wav = std::fs::read(JFK).unwrap();
    let t = d.transcribe(&wav, TINY);
    assert_eq!(t.status, 200, "{}\n{}", t.body, d.log());
    assert!(
        t.json()["text"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("country"),
        "{}",
        t.body
    );
    let (ok, out, _) = cli.run(&["models", "unload", TINY]);
    assert!(ok && out.contains("unloaded"), "{out}");
    assert!(model_ids(&status(&d)).is_empty());
    let (ok, out, _) = cli.run(&["models", "unload", TINY]);
    assert!(ok && out.contains("was not loaded"), "{out}");

    // CLI: list, load, status, transcribe through the daemon and in-process.
    let (ok, out, _) = cli.run(&["models", "list"]);
    assert!(
        ok && out.contains(TINY) && out.contains("yes (extra)"),
        "{out}"
    );
    let (ok, out, _) = cli.run(&["models", "pull", TINY]);
    assert!(ok && out.contains("already installed"), "{out}");
    let (ok, out, _) = cli.run(&["models", "load", TINY]);
    assert!(ok && out.contains("loaded"), "{out}");
    let (ok, out, _) = cli.run(&["status"]);
    assert!(
        ok && out.contains(&format!("model: {TINY}")) && out.contains("status: ok"),
        "{out}"
    );
    let (ok, out, _) = cli.run(&[
        "transcribe",
        JFK,
        "--model",
        TINY,
        "--language",
        "en",
        "--format",
        "text",
        "--priority",
        "interactive",
    ]);
    assert!(ok && out.to_lowercase().contains("country"), "{out}");
    let (ok, out, _) = cli.run(&["transcribe", JFK, "--model", TINY, "--format", "srt"]);
    assert!(ok && out.contains("-->"), "{out}");
    let (ok, out, err) = cli.run(&["transcribe", JFK, "--model", "nope"]);
    assert!(
        !ok && out.is_empty() && err.contains("does not exist"),
        "{out}{err}"
    );
    let (ok, out, _) = cli.run(&[
        "transcribe",
        JFK,
        "--model",
        TINY,
        "--local",
        "--format",
        "text",
    ]);
    assert!(ok && out.to_lowercase().contains("country"), "local: {out}");
    let (ok, _, err) = cli.run(&["models", "rm", "base"]);
    assert!(!ok && err.contains("read-only"), "{err}");
    assert!(
        extra.join("ggml-base.bin").exists(),
        "the extra dir is untouched"
    );
    let (ok, _, err) = cli.run(&["models", "load", "nope"]);
    assert!(!ok && err.contains("does not exist"), "{err}");
    let (ok, _, err) = cli.run(&["models", "load", "small"]);
    assert!(!ok && err.contains("not installed"), "{err}");

    // Delete unloads first and removes the file from models.dir.
    let r = delete(&d, &format!("/hark/v1/models/{TINY}"));
    assert_eq!(r.status, 200, "{}", r.body);
    assert!(!file.exists());
    assert!(
        model_ids(&status(&d)).is_empty(),
        "the loaded model was unloaded first"
    );
    let r = delete(&d, &format!("/hark/v1/models/{TINY}"));
    assert_eq!(r.status, 404, "{}", r.body);
    let (ok, out, _) = cli.run(&["models", "pull", TINY]);
    assert!(ok && out.contains("downloaded"), "{out}");
    let (ok, out, _) = cli.run(&["models", "rm", TINY]);
    assert!(ok && out.contains("deleted"), "{out}");
    assert!(!file.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn max_loaded_models_keeps_the_most_recently_used_models_resident() {
    if !model_tests_enabled() {
        return;
    }
    let wav = std::fs::read(JFK).unwrap();
    let d = start_with_models(
        "lru",
        "[engine]\ndefault_model = \"base\"\nstates = 1\npreload = false\nmax_loaded_models = 2\nidle_unload_secs = 0\n",
        &[],
    );
    for id in ["base", "small"] {
        let r = post(&d, &format!("/hark/v1/models/{id}/load"), None);
        assert_eq!(r.status, 200, "{id}: {}\n{}", r.body, d.log());
    }
    let s = status(&d);
    let mut ids = model_ids(&s);
    ids.sort();
    assert_eq!(ids, ["base", "small"], "two models resident: {s}");
    let active: Vec<_> = s["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|m| m["active"] == true)
        .collect();
    assert_eq!(active.len(), 1);
    assert_eq!(active[0]["id"], "small");

    let r = d.transcribe(&wav, "base");
    assert_eq!(r.status, 200, "{}", r.body);
    let s = status(&d);
    assert_eq!(s["models"][0]["id"], "base", "base is active again: {s}");
    assert!(
        s["models"][0]["load_ms"].as_u64().unwrap() < 100,
        "reactivation is not a reload: {s}"
    );

    // base is now the most recently used, so loading medium evicts small.
    let r = post(&d, "/hark/v1/models/medium/load", None);
    assert_eq!(r.status, 200, "{}\n{}", r.body, d.log());
    let mut ids = model_ids(&status(&d));
    ids.sort();
    assert_eq!(ids, ["base", "medium"], "small was the least recently used");

    let r = post(&d, "/hark/v1/models/base/unload", None);
    assert_eq!(r.json()["unloaded"], true, "{}", r.body);
    assert_eq!(model_ids(&status(&d)), ["medium"]);
}

#[test]
fn one_resident_model_by_default() {
    if !model_tests_enabled() {
        return;
    }
    let d = start_with_models(
        "one",
        "[engine]\ndefault_model = \"base\"\nstates = 1\npreload = false\nidle_unload_secs = 0\n",
        &[],
    );
    for id in ["base", "small"] {
        assert_eq!(
            post(&d, &format!("/hark/v1/models/{id}/load"), None).status,
            200
        );
    }
    assert_eq!(
        model_ids(&status(&d)),
        ["small"],
        "loading another model unloads the previous one"
    );
}

#[test]
fn idle_unload_frees_the_model_and_the_next_request_reports_the_reload() {
    if !model_tests_enabled() {
        return;
    }
    let wav = std::fs::read(JFK).unwrap();
    let d = start_with_models(
        "idle",
        "[engine]\ndefault_model = \"base\"\nstates = 1\npreload = true\nidle_unload_secs = 2\n",
        &[],
    );
    assert_eq!(model_ids(&status(&d)), ["base"]);
    assert_eq!(d.transcribe(&wav, "base").status, 200);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(7);
    while !model_ids(&status(&d)).is_empty() {
        assert!(
            std::time::Instant::now() < deadline,
            "still loaded: {}",
            status(&d)
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    assert_eq!(d.transcribe(&wav, "base").status, 200);
    let s = status(&d);
    assert_eq!(
        s["models"][0]["last_load_reason"], "reload_after_idle",
        "{s}"
    );
    assert!(s["models"][0]["load_ms"].as_u64().unwrap() > 0, "{s}");
}

#[test]
fn the_cli_sends_the_bearer_key_and_exits_non_zero_without_it() {
    let dir = scratch("cli-key-seed");
    let key = dir.join("api.key");
    std::fs::write(&key, "cli-test-key-0123456789\n").unwrap();
    #[cfg(unix)]
    std::fs::set_permissions(&key, std::os::unix::fs::PermissionsExt::from_mode(0o600)).unwrap();
    let d = start_with(
        "cli-key",
        &format!("api_key_file = \"{}\"", key.display()),
        "[engine]\npreload = false\n",
        "[]",
        &[],
    );
    let cli = Cli {
        config: d.dir.join("config.toml"),
    };
    let (ok, out, _) = cli.run(&["status"]);
    assert!(ok && out.contains("status: ok"), "{out}");
    let (ok, out, _) = cli.run(&["models", "list"]);
    assert!(ok && out.contains("tiny"), "{out}");
    let wrong = dir.join("wrong.key");
    std::fs::write(&wrong, "not-the-key\n").unwrap();
    let text = std::fs::read_to_string(d.dir.join("config.toml"))
        .unwrap()
        .replace(&key.display().to_string(), &wrong.display().to_string());
    std::fs::write(d.dir.join("wrong.toml"), text).unwrap();
    let bad = Cli {
        config: d.dir.join("wrong.toml"),
    };
    let (ok, out, err) = bad.run(&["status"]);
    assert!(
        !ok && out.is_empty() && err.contains("HTTP 401"),
        "{out}{err}"
    );
    let unreachable = scratch("cli-down");
    std::fs::write(
        unreachable.join("c.toml"),
        "[server]\nlisten = \"127.0.0.1:1\"\n",
    )
    .unwrap();
    let down = Cli {
        config: unreachable.join("c.toml"),
    };
    let (ok, _, err) = down.run(&["status"]);
    assert!(!ok && err.contains("could not reach hark"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&unreachable);
}
