use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::Deserialize;

use crate::pipeline::PromptStyle;

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("cannot read config {0}: {1}")]
    Read(PathBuf, std::io::Error),
    #[error("invalid config {0}: {1}")]
    Parse(PathBuf, toml::de::Error),
    #[error("{0}")]
    Invalid(String),
}

#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: ServerConfig,
    pub engine: EngineConfig,
    pub models: ModelsConfig,
    pub aliases: BTreeMap<String, String>,
    pub vad: VadConfig,
    pub jobs: JobsConfig,
    pub vocabulary: BTreeMap<String, VocabularyProfile>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub listen: String,
    pub allow_non_loopback: bool,
    /// Non-loopback bind with no `api_key_file` is refused unless this is set.
    pub allow_unauthenticated_remote: bool,
    /// Extra `Host` header values accepted besides the loopback names, for non-loopback setups.
    pub allowed_hosts: Vec<String>,
    pub api_key_file: String,
    pub max_upload_mb: u64,
    /// Upload bodies read at once, interactive and job routes together; more get 429.
    pub max_concurrent_uploads: usize,
    pub sync_max_audio_secs: u64,
    pub max_interactive_waiting: usize,
    pub ffmpeg_timeout_secs: u64,
    /// Debug-level logging of transcript text; off unless asked for.
    pub log_text: bool,
    /// Append logs to this file instead of stderr; empty keeps stderr.
    pub log_file: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: "127.0.0.1:8797".into(),
            allow_non_loopback: false,
            allow_unauthenticated_remote: false,
            allowed_hosts: Vec::new(),
            api_key_file: String::new(),
            max_upload_mb: 500,
            max_concurrent_uploads: 4,
            sync_max_audio_secs: 1800,
            max_interactive_waiting: 16,
            ffmpeg_timeout_secs: 300,
            log_text: false,
            log_file: String::new(),
        }
    }
}

/// `auto`, `cpu` or `gpu:N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "String")]
pub enum Device {
    Auto,
    Cpu,
    Gpu(u32),
}

impl FromStr for Device {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Device::Auto),
            "cpu" => Ok(Device::Cpu),
            _ => s
                .strip_prefix("gpu:")
                .and_then(|n| n.parse::<u32>().ok())
                .map(Device::Gpu)
                .ok_or_else(|| format!("invalid engine.device {s:?}, expected auto, cpu or gpu:N")),
        }
    }
}

impl TryFrom<String> for Device {
    type Error = String;
    fn try_from(s: String) -> Result<Self, String> {
        s.parse()
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineConfig {
    pub device: Device,
    pub default_model: String,
    pub preload: bool,
    pub states: usize,
    pub max_loaded_models: usize,
    pub idle_unload_secs: u64,
    pub cpu_threads: i32,
    /// After a GPU fallback, how often an idle daemon tries to go back to the GPU; 0 = never.
    pub gpu_retry_secs: u64,
    /// Decoded chunks after which the GPU device is recreated to free driver bookkeeping; 0 = never.
    pub recycle_after_chunks: u64,
    /// How vocabulary terms are written into the initial prompt: `sentence` or `list`.
    pub vocabulary_prompt: PromptStyle,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            device: Device::Auto,
            default_model: "large-v3-turbo".into(),
            preload: true,
            states: 2,
            max_loaded_models: 1,
            idle_unload_secs: 900,
            cpu_threads: 8,
            gpu_retry_secs: 600,
            recycle_after_chunks: 500,
            vocabulary_prompt: PromptStyle::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelsConfig {
    pub dir: String,
    pub extra_dirs: Vec<String>,
    pub word_timestamps: BTreeMap<String, String>,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            dir: default_models_dir(),
            extra_dirs: vec![format!("{}/models", wspr_dir())],
            word_timestamps: BTreeMap::from([("large-v3-turbo".to_string(), "dtw".to_string())]),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VadConfig {
    pub enabled: bool,
    pub model: String,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: "ggml-silero-v5.1.2.bin".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct JobsConfig {
    pub dir: String,
    pub path_roots: Vec<String>,
    pub keep_results_days: u64,
    pub ffmpeg_timeout_secs: u64,
    /// A job upload is refused (507) when it would leave less than this much free on the jobs filesystem.
    pub min_free_mb: u64,
    /// Queued jobs beyond this are refused with 429.
    pub max_queued: usize,
}

impl Default for JobsConfig {
    fn default() -> Self {
        Self {
            dir: default_jobs_dir(),
            path_roots: vec!["~".into()],
            keep_results_days: 7,
            ffmpeg_timeout_secs: 3600,
            min_free_mb: 2048,
            max_queued: 100,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VocabularyProfile {
    pub terms: Vec<String>,
    pub files: Vec<String>,
    pub wspr_config: String,
}

/// Profile names are `[a-z0-9_-]{1,64}`.
pub fn valid_profile_name(name: &str) -> bool {
    (1..=64).contains(&name.len())
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

/// A directory under a `dirs` base as a string, or `fallback` when the OS gives no base.
#[cfg(not(target_os = "linux"))]
fn under(base: Option<PathBuf>, rest: &str, fallback: &str) -> String {
    base.map(|b| b.join(rest).to_string_lossy().into_owned())
        .unwrap_or_else(|| fallback.to_string())
}

/// Default `models.dir`: `~/.local/share/hark/models` on Linux, `dirs::data_dir()/hark/models` elsewhere.
pub fn default_models_dir() -> String {
    #[cfg(target_os = "linux")]
    {
        "~/.local/share/hark/models".into()
    }
    #[cfg(not(target_os = "linux"))]
    {
        under(dirs::data_dir(), "hark/models", "~/hark/models")
    }
}

/// Default `jobs.dir`: `~/.local/state/hark/jobs` on Linux, `dirs::data_local_dir()/hark/jobs` elsewhere.
pub fn default_jobs_dir() -> String {
    #[cfg(target_os = "linux")]
    {
        "~/.local/state/hark/jobs".into()
    }
    #[cfg(not(target_os = "linux"))]
    {
        under(dirs::data_local_dir(), "hark/jobs", "~/hark/jobs")
    }
}

/// Wspr's own folder (its `paths.rs` rule): `~/.config/Wspr` on Linux, `dirs::data_dir()/Wspr` elsewhere.
pub fn wspr_dir() -> String {
    #[cfg(target_os = "linux")]
    {
        "~/.config/Wspr".into()
    }
    #[cfg(not(target_os = "linux"))]
    {
        under(dirs::data_dir(), "Wspr", "~/Wspr")
    }
}

pub fn default_config_path() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        expand_tilde("~/.config/hark/config.toml")
    }
    #[cfg(not(target_os = "linux"))]
    {
        dirs::config_dir()
            .map(|d| d.join("hark").join("config.toml"))
            .unwrap_or_else(|| expand_tilde("~/hark/config.toml"))
    }
}

/// Expands a leading `~` or `~/` to the home directory; other input is returned as is.
pub fn expand_tilde(p: &str) -> PathBuf {
    let home = dirs::home_dir;
    if p == "~" {
        return home().unwrap_or_else(|| PathBuf::from(p));
    }
    match (p.strip_prefix("~/"), home()) {
        (Some(rest), Some(h)) => h.join(rest),
        _ => PathBuf::from(p),
    }
}

/// Canonical form of a path that may not exist yet: the deepest existing ancestor is resolved
/// through symlinks and the missing tail is appended as written.
fn canonical_lenient(path: &Path) -> PathBuf {
    let mut tail = Vec::new();
    let mut cur = path.to_path_buf();
    loop {
        if let Ok(c) = std::fs::canonicalize(&cur) {
            return tail.iter().rev().fold(c, |acc, part| acc.join(part));
        }
        match (cur.file_name().map(|n| n.to_os_string()), cur.parent()) {
            (Some(name), Some(parent)) => {
                tail.push(name);
                cur = parent.to_path_buf();
            }
            _ => return path.to_path_buf(),
        }
    }
}

/// The first `extra_dirs` entry that `models_dir` equals or sits inside, compared canonically.
fn models_dir_inside_extra(models_dir: &str, extra_dirs: &[String]) -> Option<String> {
    let own = canonical_lenient(&expand_tilde(models_dir));
    extra_dirs
        .iter()
        .find(|e| own.starts_with(canonical_lenient(&expand_tilde(e))))
        .cloned()
}

impl Config {
    /// A missing file yields the defaults; a present file must parse and validate.
    pub fn load(path: &Path) -> Result<Config, ConfigError> {
        let cfg = match std::fs::read_to_string(path) {
            Ok(text) => Self::from_toml(&text).map_err(|e| match e {
                ConfigError::Parse(_, err) => ConfigError::Parse(path.to_path_buf(), err),
                other => other,
            })?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let cfg = Config::default();
                cfg.validate()?;
                cfg
            }
            Err(e) => return Err(ConfigError::Read(path.to_path_buf(), e)),
        };
        Ok(cfg)
    }

    pub fn from_toml(text: &str) -> Result<Config, ConfigError> {
        let cfg: Config =
            toml::from_str(text).map_err(|e| ConfigError::Parse(PathBuf::new(), e))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let addr = self.listen_addr()?;
        if !addr.ip().is_loopback() && !self.server.allow_non_loopback {
            return Err(ConfigError::Invalid(format!(
                "server.listen {addr} is not a loopback address; set server.allow_non_loopback = true to expose hark"
            )));
        }
        if self.server.allow_non_loopback
            && !addr.ip().is_loopback()
            && self.server.api_key_file.is_empty()
            && !self.server.allow_unauthenticated_remote
        {
            return Err(ConfigError::Invalid(
                "server.allow_non_loopback needs server.api_key_file; set server.allow_unauthenticated_remote = true to expose hark without a key".into(),
            ));
        }
        if self.server.max_concurrent_uploads == 0 {
            return Err(ConfigError::Invalid(
                "server.max_concurrent_uploads must be at least 1".into(),
            ));
        }
        if self.jobs.max_queued == 0 {
            return Err(ConfigError::Invalid(
                "jobs.max_queued must be at least 1".into(),
            ));
        }
        if let Some(extra) = models_dir_inside_extra(&self.models.dir, &self.models.extra_dirs) {
            return Err(ConfigError::Invalid(format!(
                "models.dir {} is the same as, or inside, models.extra_dirs entry {extra}; extra_dirs are read-only, so hark would delete from them",
                self.models.dir
            )));
        }
        if self.server.max_upload_mb == 0 {
            return Err(ConfigError::Invalid(
                "server.max_upload_mb must be at least 1".into(),
            ));
        }
        if self.server.max_interactive_waiting == 0 {
            return Err(ConfigError::Invalid(
                "server.max_interactive_waiting must be at least 1".into(),
            ));
        }
        if self.server.ffmpeg_timeout_secs == 0 {
            return Err(ConfigError::Invalid(
                "server.ffmpeg_timeout_secs must be at least 1".into(),
            ));
        }
        if self.jobs.ffmpeg_timeout_secs == 0 {
            return Err(ConfigError::Invalid(
                "jobs.ffmpeg_timeout_secs must be at least 1".into(),
            ));
        }
        if self.jobs.keep_results_days == 0 {
            return Err(ConfigError::Invalid(
                "jobs.keep_results_days must be at least 1".into(),
            ));
        }
        for (id, mode) in &self.models.word_timestamps {
            match (mode.as_str(), crate::engine::dtw_preset(id)) {
                ("heuristic", _) | ("dtw", Some(_)) => {}
                ("dtw", None) => {
                    return Err(ConfigError::Invalid(format!(
                        "models.word_timestamps.{id}: no DTW alignment-head preset for this model"
                    )));
                }
                _ => {
                    return Err(ConfigError::Invalid(format!(
                        "models.word_timestamps.{id} must be \"dtw\" or \"heuristic\""
                    )));
                }
            }
        }
        for name in self.vocabulary.keys() {
            if !valid_profile_name(name) || name == "none" {
                return Err(ConfigError::Invalid(format!(
                    "vocabulary profile name {name:?} must match [a-z0-9_-]{{1,64}} and not be \"none\""
                )));
            }
        }
        if self.engine.recycle_after_chunks > 1_000_000_000 {
            return Err(ConfigError::Invalid(
                "engine.recycle_after_chunks must be 0 or at most 1000000000".into(),
            ));
        }
        if !(1..=8).contains(&self.engine.max_loaded_models) {
            return Err(ConfigError::Invalid(
                "engine.max_loaded_models must be between 1 and 8".into(),
            ));
        }
        if self.server.max_upload_mb > 16 * 1024 {
            return Err(ConfigError::Invalid(
                "server.max_upload_mb must be at most 16384".into(),
            ));
        }
        if !(1..=8).contains(&self.engine.states) {
            return Err(ConfigError::Invalid(
                "engine.states must be between 1 and 8".into(),
            ));
        }
        if self.engine.cpu_threads < 1 {
            return Err(ConfigError::Invalid(
                "engine.cpu_threads must be at least 1".into(),
            ));
        }
        Ok(())
    }

    /// Address clients dial: `server.listen`, with a wildcard bind (0.0.0.0, [::]) replaced by loopback.
    pub fn client_addr(&self) -> String {
        match self.listen_addr() {
            Ok(a) if a.ip().is_unspecified() => {
                let ip: std::net::IpAddr = if a.is_ipv4() {
                    std::net::Ipv4Addr::LOCALHOST.into()
                } else {
                    std::net::Ipv6Addr::LOCALHOST.into()
                };
                SocketAddr::new(ip, a.port()).to_string()
            }
            _ => self.server.listen.clone(),
        }
    }

    pub fn listen_addr(&self) -> Result<SocketAddr, ConfigError> {
        self.server.listen.parse().map_err(|e| {
            ConfigError::Invalid(format!(
                "server.listen {:?} is not a socket address: {e}",
                self.server.listen
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_interactive_waiting_is_rejected() {
        let e = Config::from_toml("[server]\nmax_interactive_waiting = 0").unwrap_err();
        assert!(e.to_string().contains("max_interactive_waiting"), "{e}");
        assert!(Config::from_toml("[server]\nmax_interactive_waiting = 1").is_ok());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn example_config_equals_the_defaults() {
        let c = Config::from_toml(include_str!("../scripts/config.example.toml")).unwrap();
        assert_eq!(c, Config::default());
    }

    #[test]
    fn empty_file_gives_documented_defaults() {
        let c = Config::from_toml("").unwrap();
        assert_eq!(c.server.listen, "127.0.0.1:8797");
        assert!(!c.server.allow_non_loopback);
        assert_eq!(c.server.api_key_file, "");
        assert_eq!(c.server.max_upload_mb, 500);
        assert_eq!(c.server.sync_max_audio_secs, 1800);
        assert_eq!(c.server.max_interactive_waiting, 16);
        assert_eq!(c.server.ffmpeg_timeout_secs, 300);
        assert!(!c.server.log_text);
        assert!(c.server.log_file.is_empty());
        assert_eq!(c.engine.device, Device::Auto);
        assert_eq!(c.engine.default_model, "large-v3-turbo");
        assert!(c.engine.preload);
        assert_eq!(c.engine.states, 2);
        assert_eq!(c.engine.max_loaded_models, 1);
        assert_eq!(c.engine.idle_unload_secs, 900);
        assert_eq!(c.engine.cpu_threads, 8);
        assert_eq!(c.models.dir, default_models_dir());
        assert_eq!(c.models.extra_dirs, vec![format!("{}/models", wspr_dir())]);
        assert_eq!(c.models.word_timestamps["large-v3-turbo"], "dtw");
        assert!(c.vad.enabled);
        assert_eq!(c.vad.model, "ggml-silero-v5.1.2.bin");
        assert_eq!(c.jobs.dir, default_jobs_dir());
        assert_eq!(c.jobs.path_roots, vec!["~"]);
        assert_eq!(c.jobs.keep_results_days, 7);
        assert_eq!(c.jobs.ffmpeg_timeout_secs, 3600);
        assert_eq!(c, Config::default());
    }

    #[test]
    fn non_loopback_refused_unless_allowed() {
        let err = Config::from_toml("[server]\nlisten = \"0.0.0.0:8797\"").unwrap_err();
        assert!(err.to_string().contains("loopback"), "{err}");
        let ok = Config::from_toml(
            "[server]\nlisten = \"0.0.0.0:8797\"\nallow_non_loopback = true\napi_key_file = \"/k\"",
        );
        assert!(ok.is_ok());
        assert!(Config::from_toml("[server]\nlisten = \"[::1]:9000\"").is_ok());
    }

    #[test]
    fn remote_bind_without_a_key_needs_an_explicit_opt_in() {
        let base = "[server]\nlisten = \"0.0.0.0:8797\"\nallow_non_loopback = true\n";
        let e = Config::from_toml(base).unwrap_err();
        assert!(e.to_string().contains("api_key_file"), "{e}");
        assert!(
            e.to_string().contains("allow_unauthenticated_remote"),
            "{e}"
        );
        assert!(Config::from_toml(&format!("{base}allow_unauthenticated_remote = true")).is_ok());
        assert!(Config::from_toml(&format!("{base}api_key_file = \"/some/key\"")).is_ok());
        let loopback = "[server]\nallow_non_loopback = true\n";
        assert!(Config::from_toml(loopback).is_ok(), "loopback needs no key");
    }

    #[cfg(unix)]
    #[test]
    fn models_dir_may_not_be_or_sit_inside_an_extra_dir() {
        let base = std::env::temp_dir().join(format!("hark-cfg-{}", std::process::id()));
        let extra = base.join("extra");
        std::fs::create_dir_all(&extra).unwrap();
        let link = base.join("link");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&extra, &link).unwrap();
        let cfg = |dir: &Path| {
            format!(
                "[models]\ndir = \"{}\"\nextra_dirs = [\"{}\"]",
                dir.display(),
                extra.display()
            )
        };
        for bad in [
            extra.clone(),
            extra.join("sub"),
            extra.join("a/b"),
            link.clone(),
            link.join("sub"),
            extra.join("../extra"),
        ] {
            let e = Config::from_toml(&cfg(&bad)).unwrap_err();
            assert!(
                e.to_string().contains("extra_dirs"),
                "{}: {e}",
                bad.display()
            );
        }
        assert!(Config::from_toml(&cfg(&base.join("own"))).is_ok());
        assert!(Config::from_toml(&cfg(&base.join("extra2"))).is_ok());
        std::fs::remove_dir_all(base).ok();
    }

    #[test]
    fn new_limits_are_validated_and_defaulted() {
        let c = Config::default();
        assert_eq!(c.server.max_concurrent_uploads, 4);
        assert!(c.server.allowed_hosts.is_empty());
        assert_eq!(c.jobs.min_free_mb, 2048);
        assert_eq!(c.jobs.max_queued, 100);
        assert_eq!(c.engine.gpu_retry_secs, 600);
        assert!(Config::from_toml("[server]\nmax_concurrent_uploads = 0").is_err());
        assert!(Config::from_toml("[jobs]\nmax_queued = 0").is_err());
        assert!(Config::from_toml("[engine]\ngpu_retry_secs = 0").is_ok());
        assert_eq!(c.engine.recycle_after_chunks, 500);
        assert!(Config::from_toml("[engine]\nrecycle_after_chunks = 0").is_ok());
        assert!(Config::from_toml("[engine]\nrecycle_after_chunks = 2000000000").is_err());
    }

    #[test]
    fn clients_dial_loopback_when_the_server_binds_a_wildcard() {
        let addr = |l: &str| {
            Config::from_toml(&format!(
                "[server]\nlisten = \"{l}\"\nallow_non_loopback = true\nallow_unauthenticated_remote = true"
            ))
            .unwrap()
            .client_addr()
        };
        assert_eq!(addr("0.0.0.0:8797"), "127.0.0.1:8797");
        assert_eq!(addr("[::]:9000"), "[::1]:9000");
        assert_eq!(addr("127.0.0.1:8797"), "127.0.0.1:8797");
        assert_eq!(addr("192.168.1.5:8797"), "192.168.1.5:8797");
    }

    #[test]
    fn bad_device_refused() {
        for bad in ["gpu", "gpu:x", "gpu:-1", "cuda", "GPU:0", ""] {
            let toml = format!("[engine]\ndevice = \"{bad}\"");
            assert!(Config::from_toml(&toml).is_err(), "{bad:?} accepted");
        }
        let c = Config::from_toml("[engine]\ndevice = \"gpu:1\"").unwrap();
        assert_eq!(c.engine.device, Device::Gpu(1));
        let c = Config::from_toml("[engine]\ndevice = \"cpu\"").unwrap();
        assert_eq!(c.engine.device, Device::Cpu);
    }

    #[test]
    fn unknown_keys_are_errors() {
        assert!(Config::from_toml("[server]\nlisten_on = \"x\"").is_err());
        assert!(Config::from_toml("[nope]\nx = 1").is_err());
    }

    #[test]
    fn word_timestamp_modes_are_validated() {
        assert!(Config::from_toml("[models.word_timestamps]\n\"small\" = \"heuristic\"").is_ok());
        assert!(Config::from_toml("[models.word_timestamps]\n\"small\" = \"dtw\"").is_ok());
        assert!(Config::from_toml("[models.word_timestamps]\n\"small\" = \"fast\"").is_err());
        assert!(Config::from_toml("[models.word_timestamps]\n\"custom\" = \"dtw\"").is_err());
        assert!(Config::from_toml("[server]\nffmpeg_timeout_secs = 0").is_err());
    }

    #[test]
    fn job_limits_are_validated() {
        assert!(Config::from_toml("[jobs]\nffmpeg_timeout_secs = 0").is_err());
        assert!(Config::from_toml("[jobs]\nkeep_results_days = 0").is_err());
        assert!(Config::from_toml("[jobs]\nkeep_results_days = 1").is_ok());
    }

    #[test]
    fn vocabulary_prompt_defaults_to_sentence_and_rejects_unknown_values() {
        assert_eq!(
            Config::default().engine.vocabulary_prompt,
            PromptStyle::Sentence
        );
        let c = Config::from_toml("[engine]\nvocabulary_prompt = \"list\"").unwrap();
        assert_eq!(c.engine.vocabulary_prompt, PromptStyle::List);
        let c = Config::from_toml("[engine]\nvocabulary_prompt = \"sentence\"").unwrap();
        assert_eq!(c.engine.vocabulary_prompt, PromptStyle::Sentence);
        let e = Config::from_toml("[engine]\nvocabulary_prompt = \"csv\"").unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("vocabulary_prompt"), "{msg}");
        assert!(msg.contains("unknown variant `csv`"), "{msg}");
    }

    #[test]
    fn model_residency_limits_are_validated() {
        assert!(Config::from_toml("[engine]\nmax_loaded_models = 0").is_err());
        assert!(Config::from_toml("[engine]\nmax_loaded_models = 9").is_err());
        assert!(Config::from_toml("[engine]\nmax_loaded_models = 2\nidle_unload_secs = 0").is_ok());
        assert!(Config::from_toml("[server]\nmax_upload_mb = 99999").is_err());
    }

    #[test]
    fn hostname_listen_refused() {
        assert!(Config::from_toml("[server]\nlisten = \"localhost:8797\"").is_err());
    }

    #[test]
    fn tilde_expansion() {
        let home = dirs::home_dir().unwrap();
        assert_eq!(expand_tilde("~/a/b"), PathBuf::from(&home).join("a/b"));
        assert_eq!(expand_tilde("~"), PathBuf::from(&home));
        assert_eq!(expand_tilde("/abs"), PathBuf::from("/abs"));
        assert_eq!(expand_tilde("~other/x"), PathBuf::from("~other/x"));
    }

    #[test]
    fn vocabulary_profile_names_are_validated() {
        assert!(Config::from_toml("[vocabulary.my-prof_1]\nterms = [\"a\"]").is_ok());
        for bad in ["Upper", "has space", "none", "a.b", &"x".repeat(65)] {
            let toml = format!("[vocabulary.\"{bad}\"]\nterms = []");
            assert!(Config::from_toml(&toml).is_err(), "{bad:?} accepted");
        }
        assert!(valid_profile_name("default"));
        assert!(!valid_profile_name(""));
    }

    #[test]
    fn missing_file_is_defaults() {
        let c = Config::load(Path::new("/nonexistent/hark/config.toml")).unwrap();
        assert_eq!(c, Config::default());
    }
}
