//! `hark jobs`, `hark models`, `hark status` and `hark transcribe`: a small blocking client for the
//! running daemon, plus the in-process `transcribe --local`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail};
use serde_json::{Value, json};

use crate::config::{Config, expand_tilde};

pub struct SubmitArgs {
    pub file: PathBuf,
    pub model: Option<String>,
    pub language: Option<String>,
    pub format: Option<String>,
    pub vocabulary: Option<String>,
    /// Request `keywords`, one per `--keyword`.
    pub keywords: Vec<String>,
}

pub struct TranscribeArgs {
    pub file: PathBuf,
    pub model: Option<String>,
    pub language: Option<String>,
    /// `json`, `text`, `srt`, `vtt` or `verbose_json`; `text` when unset.
    pub format: Option<String>,
    pub vocabulary: Option<String>,
    /// Request `keywords`, one per `--keyword`.
    pub keywords: Vec<String>,
    /// `interactive` or `batch`.
    pub priority: Option<String>,
}

pub struct Client {
    base: String,
    key_file: String,
    roots: Vec<String>,
    default_model: String,
    vad_model: Option<String>,
    agent: ureq::Agent,
}

impl Client {
    pub fn new(cfg: &Config) -> Client {
        Client {
            base: format!("http://{}", cfg.client_addr()),
            key_file: cfg.server.api_key_file.clone(),
            roots: cfg.jobs.path_roots.clone(),
            default_model: cfg.engine.default_model.clone(),
            vad_model: cfg.vad.enabled.then(|| cfg.vad.model.clone()),
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(5))
                .timeout(Duration::from_secs(3600))
                .build(),
        }
    }

    fn request(&self, method: &str, path: &str) -> anyhow::Result<ureq::Request> {
        let mut req = self.agent.request(method, &format!("{}{path}", self.base));
        if !self.key_file.is_empty() {
            let key = std::fs::read_to_string(expand_tilde(&self.key_file))
                .context("reading server.api_key_file")?;
            req = req.set("Authorization", &format!("Bearer {}", key.trim()));
        }
        Ok(req)
    }

    fn finish(&self, res: Result<ureq::Response, ureq::Error>) -> anyhow::Result<String> {
        match res {
            Ok(r) => r.into_string().context("reading the response"),
            Err(ureq::Error::Status(code, r)) => {
                let body = r.into_string().unwrap_or_default();
                bail!("{} (HTTP {code})", error_message(&body))
            }
            Err(e) => bail!("could not reach hark at {}: {e}", self.base),
        }
    }

    pub fn submit(&self, a: &SubmitArgs) -> anyhow::Result<String> {
        let file = std::fs::canonicalize(&a.file)
            .with_context(|| format!("cannot read {}", a.file.display()))?;
        let model = a
            .model
            .clone()
            .unwrap_or_else(|| self.default_model.clone());
        let mut fields: Vec<(&str, String)> = vec![("model", model)];
        for (name, v) in [
            ("language", &a.language),
            ("response_format", &a.format),
            ("x_vocabulary", &a.vocabulary),
        ] {
            if let Some(v) = v {
                fields.push((name, v.clone()));
            }
        }
        let req = self.request("POST", "/hark/v1/jobs")?;
        let res = if under_roots(&file, &self.roots) {
            let body = job_json_body(&file.display().to_string(), &fields, &a.keywords);
            req.set("Content-Type", "application/json")
                .send_bytes(body.to_string().as_bytes())
        } else {
            let bytes = std::fs::read(&file).context("reading the file")?;
            let name = file
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "audio".into());
            let boundary = "hark-boundary-7f3a9c1e5b2d";
            fields.extend(keyword_fields(&a.keywords));
            req.set(
                "Content-Type",
                &format!("multipart/form-data; boundary={boundary}"),
            )
            .send_bytes(&multipart_body(boundary, &fields, &name, &bytes))
        };
        let body = self.finish(res)?;
        job_text(&body)
    }

    /// Uploads `a.file` to the running daemon and returns the transcript in the requested format.
    pub fn transcribe(&self, a: &TranscribeArgs) -> anyhow::Result<String> {
        let bytes =
            std::fs::read(&a.file).with_context(|| format!("cannot read {}", a.file.display()))?;
        let mut fields: Vec<(&str, String)> = vec![
            (
                "model",
                a.model
                    .clone()
                    .unwrap_or_else(|| self.default_model.clone()),
            ),
            (
                "response_format",
                a.format.clone().unwrap_or_else(|| "text".into()),
            ),
        ];
        for (name, v) in [
            ("language", &a.language),
            ("x_vocabulary", &a.vocabulary),
            ("x_priority", &a.priority),
        ] {
            if let Some(v) = v {
                fields.push((name, v.clone()));
            }
        }
        fields.extend(keyword_fields(&a.keywords));
        let name = a
            .file
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "audio".into());
        let boundary = "hark-boundary-7f3a9c1e5b2d";
        let res = self
            .request("POST", "/v1/audio/transcriptions")?
            .set(
                "Content-Type",
                &format!("multipart/form-data; boundary={boundary}"),
            )
            .send_bytes(&multipart_body(boundary, &fields, &name, &bytes));
        self.finish(res)
    }

    pub fn status(&self) -> anyhow::Result<String> {
        let body = self.finish(self.request("GET", "/hark/v1/status")?.call())?;
        let v: Value = serde_json::from_str(&body).context("parsing the status")?;
        Ok(status_text(&v))
    }

    pub fn models_list(&self) -> anyhow::Result<String> {
        let body = self.finish(self.request("GET", "/hark/v1/models")?.call())?;
        let v: Value = serde_json::from_str(&body).context("parsing the model list")?;
        Ok(models_text(&v))
    }

    /// The ids `models pull --defaults` fetches: the default model (alias resolved by the daemon) and the VAD catalog entry.
    pub fn default_pull_ids(&self) -> Vec<String> {
        let mut ids = vec![self.default_model.clone()];
        if let Some(name) = &self.vad_model
            && let Some(e) = crate::models::CATALOG.iter().find(|e| e.file == name)
        {
            ids.push(e.id.to_string());
        }
        ids
    }

    pub fn models_pull_defaults(&self) -> anyhow::Result<String> {
        let mut out = String::new();
        for id in self.default_pull_ids() {
            out.push_str(&self.models_pull(&id)?);
        }
        Ok(out)
    }

    pub fn models_pull(&self, id: &str) -> anyhow::Result<String> {
        let body = self.finish(
            self.request("POST", "/hark/v1/models/pull")?
                .set("Content-Type", "application/json")
                .send_bytes(json!({"id": id}).to_string().as_bytes()),
        )?;
        let v: Value = serde_json::from_str(&body).context("parsing the response")?;
        Ok(match v["status"].as_str() {
            Some("already_installed") => format!("{id}: already installed\n"),
            _ => format!("{id}: downloaded and verified\n"),
        })
    }

    pub fn models_rm(&self, id: &str) -> anyhow::Result<String> {
        self.finish(
            self.request("DELETE", &format!("/hark/v1/models/{}", pct(id)))?
                .call(),
        )?;
        Ok(format!("{id}: deleted\n"))
    }

    pub fn models_load(&self, id: &str) -> anyhow::Result<String> {
        self.finish(
            self.request("POST", &format!("/hark/v1/models/{}/load", pct(id)))?
                .call(),
        )?;
        Ok(format!("{id}: loaded\n"))
    }

    pub fn models_unload(&self, id: &str) -> anyhow::Result<String> {
        let body = self.finish(
            self.request("POST", &format!("/hark/v1/models/{}/unload", pct(id)))?
                .call(),
        )?;
        let v: Value = serde_json::from_str(&body).context("parsing the response")?;
        Ok(if v["unloaded"] == json!(true) {
            format!("{id}: unloaded\n")
        } else {
            format!("{id}: was not loaded\n")
        })
    }

    pub fn list(&self, status: Option<&str>) -> anyhow::Result<String> {
        let path = match status {
            Some(s) => format!("/hark/v1/jobs?status={}", pct(s)),
            None => "/hark/v1/jobs".into(),
        };
        let body = self.finish(self.request("GET", &path)?.call())?;
        let v: Value = serde_json::from_str(&body).context("parsing the job list")?;
        let mut out = String::new();
        for j in v["data"].as_array().into_iter().flatten() {
            out.push_str(&format!(
                "{}  {:<9}  {:>5.1}%  {}\n",
                j["id"].as_str().unwrap_or("?"),
                j["status"].as_str().unwrap_or("?"),
                j["progress"].as_f64().unwrap_or(0.0) * 100.0,
                j["created_at"].as_str().unwrap_or("?"),
            ));
        }
        Ok(out)
    }

    pub fn get(&self, id: &str) -> anyhow::Result<String> {
        let body = self.finish(
            self.request("GET", &format!("/hark/v1/jobs/{}", pct(id)))?
                .call(),
        )?;
        job_text(&body)
    }

    pub fn result(&self, id: &str, format: Option<&str>) -> anyhow::Result<String> {
        let mut path = format!("/hark/v1/jobs/{}/result", pct(id));
        if let Some(f) = format {
            path.push_str(&format!("?response_format={}", pct(f)));
        }
        self.finish(self.request("GET", &path)?.call())
    }

    pub fn cancel(&self, id: &str) -> anyhow::Result<String> {
        let body = self.finish(
            self.request("DELETE", &format!("/hark/v1/jobs/{}", pct(id)))?
                .call(),
        )?;
        let v: Value = serde_json::from_str(&body).context("parsing the response")?;
        if v["deleted"] == json!(true) {
            return Ok(format!("deleted {id}\n"));
        }
        job_text(&body)
    }
}

/// Runs one transcription in this process, without a daemon, through the same pipeline.
pub fn transcribe_local(cfg: &Config, a: &TranscribeArgs) -> anyhow::Result<String> {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use crate::audio::{self, DecodeLimits, SAMPLE_RATE};
    use crate::engine::{
        DecodeOptions, ModelRef, TranscribeRequest, Transcriber, WhisperEngine, is_valid_language,
    };
    use crate::models::Registry;
    use crate::scheduler::Lane;
    use crate::vocabulary::{VocabularyStore, request_keywords};

    let format = a.format.as_deref().unwrap_or("text");
    if !["json", "text", "srt", "vtt", "verbose_json"].contains(&format) {
        bail!("--format must be one of json, text, srt, vtt, verbose_json");
    }
    let lane = match a.priority.as_deref() {
        None | Some("interactive") => Lane::Interactive,
        Some("batch") => Lane::Batch,
        Some(other) => bail!("--priority must be interactive or batch, not `{other}`"),
    };
    if let Some(l) = &a.language
        && !is_valid_language(l)
    {
        bail!("`{l}` is not a supported ISO-639-1 language code");
    }
    let keywords = request_keywords(&a.keywords).map_err(|e| anyhow::anyhow!("{e}"))?;
    let registry = Registry::from_config(cfg);
    let model_id = a
        .model
        .clone()
        .unwrap_or_else(|| cfg.engine.default_model.clone());
    let (entry, path) = registry
        .resolve_installed(&model_id)
        .map_err(|e| anyhow::anyhow!("{e}; run `hark models pull {model_id}`"))?;
    let vocabulary = VocabularyStore::new(&cfg.vocabulary)
        .resolve(a.vocabulary.as_deref())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let bytes =
        std::fs::read(&a.file).with_context(|| format!("cannot read {}", a.file.display()))?;
    let limits = DecodeLimits {
        ffmpeg_timeout: Duration::from_secs(cfg.jobs.ffmpeg_timeout_secs),
        max_secs: u64::MAX,
    };
    let samples = audio::decode(&bytes, &limits).map_err(|e| anyhow::anyhow!("{e}"))?;
    let secs = samples.len() as f64 / f64::from(SAMPLE_RATE);
    let opts = DecodeOptions {
        language: a.language.clone(),
        vocabulary,
        keywords,
        ..DecodeOptions::default()
    };
    let engine = WhisperEngine::from_config(cfg, &registry);
    let result = engine
        .run(&TranscribeRequest {
            id: "local".into(),
            lane,
            model: ModelRef { id: entry.id, path },
            audio: Arc::new(samples),
            opts: opts.clone(),
            cancel: Arc::new(AtomicBool::new(false)),
        })
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(match format {
        "json" => json!({"text": result.text(), "usage": {"type": "duration", "seconds": secs}})
            .to_string(),
        "srt" => crate::format::render_srt(&result.segments),
        "vtt" => crate::format::render_vtt(&result.segments),
        "verbose_json" => {
            crate::format::verbose_json(&result, secs, opts.temperature, &[]).to_string()
        }
        _ => result.text(),
    })
}

/// The status document as readable lines.
pub fn status_text(v: &Value) -> String {
    let mut out = String::new();
    let dev = &v["device"];
    out.push_str(&format!(
        "status: {}\ndevice: {}",
        if v["degraded"] == json!(true) {
            "degraded"
        } else {
            "ok"
        },
        dev["name"].as_str().unwrap_or("?"),
    ));
    if let Some(from) = dev["fell_back_from"].as_str() {
        out.push_str(&format!(" (fell back from {from})"));
    }
    out.push('\n');
    let models = v["models"].as_array().map(Vec::as_slice).unwrap_or(&[]);
    if models.is_empty() {
        out.push_str("models: none loaded\n");
    }
    for m in models {
        out.push_str(&format!(
            "model: {}{}  load {} ms ({})  states {}  rtf {}\n",
            m["id"].as_str().unwrap_or("?"),
            if m["active"] == json!(false) {
                " (resident, inactive)"
            } else {
                ""
            },
            m["load_ms"],
            m["last_load_reason"].as_str().unwrap_or("?"),
            m["states"],
            m["realtime_factor"]
                .as_f64()
                .map_or("-".to_string(), |r| format!("{r:.3}")),
        ));
    }
    out.push_str(&format!(
        "queues: interactive {}  batch {}  live {}\n",
        v["queues"]["interactive"], v["queues"]["batch"], v["queues"]["live"]
    ));
    out.push_str(&format!(
        "jobs: {} queued, {} running\naborted chunks: {}\n",
        v["jobs"]["queued"], v["jobs"]["running"], v["aborted_chunks"]
    ));
    match v["last_error"]["message"].as_str() {
        Some(m) => out.push_str(&format!(
            "last error: {m} ({})\n",
            v["last_error"]["time"].as_str().unwrap_or("?")
        )),
        None => out.push_str("last error: none\n"),
    }
    out
}

/// The catalog as a table.
pub fn models_text(v: &Value) -> String {
    let mut out = format!(
        "{:<22} {:>9}  {:<10} {}\n",
        "ID", "SIZE", "INSTALLED", "STATE"
    );
    for m in v["data"].as_array().into_iter().flatten() {
        let mib = m["size_bytes"].as_u64().unwrap_or(0) / (1024 * 1024);
        let installed = match m["location"].as_str() {
            Some("models_dir") => "yes",
            Some(_) => "yes (extra)",
            None => "no",
        };
        let state = match (m["loaded"] == json!(true), m["active"] == json!(true)) {
            (true, true) => "loaded",
            (true, false) => "resident",
            _ => "",
        };
        out.push_str(&format!(
            "{:<22} {:>6} MiB  {:<10} {}\n",
            m["id"].as_str().unwrap_or("?"),
            mib,
            installed,
            state
        ));
    }
    out
}

/// True when `file` sits under a canonicalized path root.
fn under_roots(file: &Path, roots: &[String]) -> bool {
    roots
        .iter()
        .filter_map(|r| std::fs::canonicalize(expand_tilde(r)).ok())
        .any(|r| file.starts_with(r))
}

fn pct(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// The `message` of an OpenAI error envelope, else the raw body.
pub fn error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
        .unwrap_or_else(|| body.trim().to_string())
}

/// The JSON jobs body: the path, the text fields, and `keywords` as an array when there are any.
fn job_json_body(path: &str, fields: &[(&str, String)], keywords: &[String]) -> Value {
    let mut body = json!({"path": path});
    for (k, v) in fields {
        body[*k] = json!(v);
    }
    if !keywords.is_empty() {
        body["keywords"] = json!(keywords);
    }
    body
}

/// One `keywords[]` multipart field per term.
fn keyword_fields(keywords: &[String]) -> impl Iterator<Item = (&'static str, String)> + '_ {
    keywords.iter().map(|k| ("keywords[]", k.clone()))
}

pub fn multipart_body(
    boundary: &str,
    fields: &[(&str, String)],
    filename: &str,
    file: &[u8],
) -> Vec<u8> {
    let mut body = Vec::with_capacity(file.len() + 512);
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    let safe: String = filename
        .chars()
        .map(|c| if c == '"' || c.is_control() { '_' } else { c })
        .collect();
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{safe}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

/// A job object as `key: value` lines.
pub fn job_text(body: &str) -> anyhow::Result<String> {
    let v: Value = serde_json::from_str(body).context("parsing the job")?;
    let mut out = String::new();
    for key in [
        "id",
        "status",
        "progress",
        "audio_seconds",
        "model",
        "created_at",
        "started_at",
        "finished_at",
        "error",
        "resumed_from_chunk",
    ] {
        let val = match &v[key] {
            Value::Null => continue,
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        out.push_str(&format!("{key}: {val}\n"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_body_has_fields_then_the_file() {
        let b = multipart_body("B", &[("model", "whisper-1".into())], "a\"b.wav", b"DATA");
        let s = String::from_utf8(b).unwrap();
        assert!(s.starts_with(
            "--B\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n"
        ));
        assert!(s.contains("name=\"file\"; filename=\"a_b.wav\""));
        assert!(s.ends_with("DATA\r\n--B--\r\n"));
    }

    #[test]
    fn keywords_go_out_as_one_keywords_field_per_term() {
        let terms = vec!["Ghidra".to_string(), "Zstandard".to_string()];
        let mut fields: Vec<(&str, String)> = vec![("model", "whisper-1".into())];
        fields.extend(keyword_fields(&terms));
        let s = String::from_utf8(multipart_body("B", &fields, "a.wav", b"D")).unwrap();
        assert!(s.contains("name=\"keywords[]\"\r\n\r\nGhidra\r\n"));
        assert!(s.contains("name=\"keywords[]\"\r\n\r\nZstandard\r\n"));
        assert!(s.find("Ghidra").unwrap() < s.find("Zstandard").unwrap());
        assert_eq!(keyword_fields(&[]).count(), 0);
    }

    #[test]
    fn the_json_jobs_body_carries_keywords_as_an_array() {
        let fields = vec![("model", "whisper-1".to_string())];
        let terms = vec!["Ghidra".to_string(), "Zstandard".to_string()];
        assert_eq!(
            job_json_body("/a.wav", &fields, &terms),
            json!({"path": "/a.wav", "model": "whisper-1", "keywords": ["Ghidra", "Zstandard"]})
        );
        assert_eq!(
            job_json_body("/a.wav", &fields, &[]),
            json!({"path": "/a.wav", "model": "whisper-1"})
        );
    }

    #[test]
    fn error_messages_come_from_the_envelope() {
        assert_eq!(
            error_message(r#"{"error":{"message":"nope","type":"x","param":null,"code":"y"}}"#),
            "nope"
        );
        assert_eq!(error_message("plain"), "plain");
    }

    #[test]
    fn default_pull_ids_are_the_default_model_and_the_vad_entry() {
        let mut cfg = Config::default();
        cfg.engine.default_model = "whisper-1".into();
        assert_eq!(
            Client::new(&cfg).default_pull_ids(),
            ["whisper-1", "silero-v5.1.2"]
        );
        cfg.vad.enabled = false;
        assert_eq!(Client::new(&cfg).default_pull_ids(), ["whisper-1"]);
    }

    #[test]
    fn job_text_skips_nulls() {
        let t =
            job_text(r#"{"id":"job_x","status":"queued","progress":0.0,"error":null}"#).unwrap();
        assert_eq!(t, "id: job_x\nstatus: queued\nprogress: 0.0\n");
    }

    #[test]
    fn status_text_reports_device_change_and_error() {
        let v = json!({
            "degraded": true,
            "device": {"name": "cpu", "index": null, "fell_back_from": "RTX"},
            "models": [{"id": "base", "load_ms": 12, "last_load_reason": "request", "states": 2,
                        "realtime_factor": 0.05, "active": true}],
            "queues": {"interactive": 0, "batch": 1, "live": 0},
            "jobs": {"queued": 0, "running": 1},
            "aborted_chunks": 3,
            "last_error": {"message": "GPU fault", "time": "2026-10-27T12:00:00Z"}
        });
        let t = status_text(&v);
        assert!(t.contains("status: degraded"), "{t}");
        assert!(t.contains("device: cpu (fell back from RTX)"), "{t}");
        assert!(
            t.contains("model: base  load 12 ms (request)  states 2  rtf 0.050"),
            "{t}"
        );
        assert!(t.contains("last error: GPU fault"), "{t}");
        assert!(status_text(&json!({})).contains("models: none loaded"));
    }

    #[test]
    fn models_table_marks_install_and_state() {
        let v = json!({"data": [
            {"id": "tiny", "size_bytes": 77691713, "location": "models_dir", "loaded": true, "active": true},
            {"id": "base", "size_bytes": 147951465, "location": "extra_dir", "loaded": false, "active": false},
            {"id": "small", "size_bytes": 487601967, "location": null, "loaded": false, "active": false}
        ]});
        let t = models_text(&v);
        assert!(
            t.contains("tiny") && t.contains("yes ") && t.contains("loaded"),
            "{t}"
        );
        assert!(t.contains("yes (extra)"), "{t}");
        assert!(t.lines().last().unwrap().contains(" no "), "{t}");
    }

    #[test]
    fn percent_encoding_keeps_unreserved() {
        assert_eq!(pct("a b/../c"), "a%20b%2F..%2Fc");
    }

    #[test]
    fn files_under_a_root_are_detected() {
        let d = std::env::temp_dir().join(format!("hark-cli-test-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let f = std::fs::canonicalize(&d).unwrap().join("a.wav");
        std::fs::write(&f, b"x").unwrap();
        assert!(under_roots(&f, &[d.display().to_string()]));
        assert!(!under_roots(&f, &["/nonexistent-root".to_string()]));
        assert!(!under_roots(&f, &[]));
    }
}
