//! `hark doctor`: one line per check (ok, warn or FAIL); the exit code is non-zero on any FAIL.

use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::config::{Config, Device, default_config_path, expand_tilde};
use crate::engine::{DevicePick, GpuInfo, enumerate_gpus, pick_device};
use crate::models::{CATALOG, CatalogEntry, Registry};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Ok(String),
    Warn(String),
    Fail(String),
}

impl Verdict {
    fn label(&self) -> &'static str {
        match self {
            Verdict::Ok(_) => "ok  ",
            Verdict::Warn(_) => "warn",
            Verdict::Fail(_) => "FAIL",
        }
    }

    fn message(&self) -> &str {
        match self {
            Verdict::Ok(m) | Verdict::Warn(m) | Verdict::Fail(m) => m,
        }
    }
}

pub const LIBEGL_FIX: &str = "export LD_PRELOAD=<system libEGL.so.1> before starting hark (/usr/lib64/libEGL.so.1 on Fedora, /usr/lib/x86_64-linux-gnu/libEGL.so.1 on Debian and Ubuntu; systemd: Environment=LD_PRELOAD=... in a drop-in)";

fn describe_gpus(gpus: &[GpuInfo]) -> String {
    gpus.iter()
        .map(|g| {
            format!(
                "gpu:{} {} ({})",
                g.index,
                g.name,
                if g.discrete { "discrete" } else { "integrated" }
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Verdict on the device `engine.device` would pick, given the visible Vulkan devices and whether
/// `nvidia-smi` reports an NVIDIA card.
pub fn device_verdict(want: Device, gpus: &[GpuInfo], nvidia_card: bool) -> Verdict {
    let listing = if gpus.is_empty() {
        "no Vulkan devices".to_string()
    } else {
        describe_gpus(gpus)
    };
    let pick = pick_device(want, gpus);
    let any_discrete = gpus.iter().any(|g| g.discrete);
    match pick {
        DevicePick::Cpu if want == Device::Cpu => {
            Verdict::Ok(format!("engine.device = cpu; Vulkan: {listing}"))
        }
        DevicePick::Cpu => {
            if nvidia_card {
                Verdict::Warn(format!(
                    "no Vulkan device visible but nvidia-smi reports an NVIDIA card, hark would run on CPU; fix: {LIBEGL_FIX}"
                ))
            } else {
                Verdict::Warn(format!("{listing}; hark would run on CPU"))
            }
        }
        DevicePick::Gpu { index, name } => {
            let picked_discrete = gpus.iter().find(|g| g.index == index).map(|g| g.discrete);
            match picked_discrete {
                Some(false) if any_discrete => Verdict::Fail(format!(
                    "engine.device would pick the integrated GPU gpu:{index} {name} while a discrete GPU exists; set engine.device to its gpu:N. Vulkan: {listing}"
                )),
                Some(false) if nvidia_card => Verdict::Warn(format!(
                    "only the integrated GPU {name} is visible but nvidia-smi reports an NVIDIA card; fix: {LIBEGL_FIX}. Vulkan: {listing}"
                )),
                Some(false) => Verdict::Ok(format!(
                    "picks gpu:{index} {name} (integrated, the only GPU). Vulkan: {listing}"
                )),
                Some(true) => Verdict::Ok(format!(
                    "picks gpu:{index} {name} (discrete). Vulkan: {listing}"
                )),
                None => Verdict::Fail(format!(
                    "engine.device = gpu:{index} but no such Vulkan device. Vulkan: {listing}"
                )),
            }
        }
    }
}

/// What answers on the configured listen address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortState {
    Free,
    Hark { status: String, version: String },
    Other(String),
}

pub fn port_verdict(addr: &str, state: &PortState) -> Verdict {
    match state {
        PortState::Free => Verdict::Ok(format!("{addr} is free")),
        PortState::Hark { status, version } if status == "ok" => {
            Verdict::Ok(format!("{addr} is answered by hark {version} (status ok)"))
        }
        PortState::Hark { status, version } => Verdict::Warn(format!(
            "{addr} is answered by hark {version} (status {status})"
        )),
        PortState::Other(why) => Verdict::Fail(format!(
            "{addr} is in use and /health is not hark's: {why}; change server.listen or stop the other program"
        )),
    }
}

pub fn permission_verdict(problem: Option<String>, path: &Path) -> Verdict {
    match problem {
        None => Verdict::Ok(format!("{} is a private regular file", path.display())),
        Some(p) => Verdict::Fail(p),
    }
}

/// `systemctl --user show` properties of the unit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnitState {
    pub load_state: String,
    pub unit_file_state: String,
    pub active_state: String,
    pub sub_state: String,
    pub private_tmp: bool,
}

pub fn parse_unit_show(out: &str) -> UnitState {
    let mut u = UnitState::default();
    for line in out.lines() {
        if let Some((k, v)) = line.split_once('=') {
            match k {
                "LoadState" => u.load_state = v.into(),
                "UnitFileState" => u.unit_file_state = v.into(),
                "ActiveState" => u.active_state = v.into(),
                "SubState" => u.sub_state = v.into(),
                "PrivateTmp" => u.private_tmp = v == "yes",
                _ => {}
            }
        }
    }
    u
}

pub fn unit_verdict(u: &UnitState) -> Verdict {
    if u.load_state != "loaded" {
        return Verdict::Warn("hark.service is not installed (run scripts/install.sh)".into());
    }
    let enabled = u.unit_file_state == "enabled";
    let active = u.active_state == "active";
    match (enabled, active) {
        (true, true) => Verdict::Ok(format!(
            "hark.service is enabled and active ({})",
            u.sub_state
        )),
        (false, true) => Verdict::Fail(format!(
            "hark.service is active but not enabled ({}); run systemctl --user enable hark",
            u.unit_file_state
        )),
        _ => Verdict::Fail(format!(
            "hark.service is {} ({}), unit file {}; see journalctl --user -u hark",
            u.active_state, u.sub_state, u.unit_file_state
        )),
    }
}

pub fn hash_verdict(what: &str, file: &Path, expected: &str, actual: &str) -> Verdict {
    if actual == expected {
        Verdict::Ok(format!("{what} {} sha256 matches the pin", file.display()))
    } else {
        Verdict::Fail(format!(
            "{what} {}: sha256 {actual} does not match the pinned {expected}; re-download it with hark models pull",
            file.display()
        ))
    }
}

fn run(cmd: &str, args: &[&str]) -> Result<String, String> {
    let out = Command::new(cmd)
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| format!("cannot run {cmd}: {e}"))?;
    if !out.status.success() {
        return Err(format!("{cmd} exited with {}", out.status));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[cfg(target_os = "macos")]
const UNIT_LABEL: &str = "launchd agent";
#[cfg(target_os = "macos")]
const FFMPEG_HINT: &str = "brew install ffmpeg";
#[cfg(windows)]
const UNIT_LABEL: &str = "scheduled task";
#[cfg(windows)]
const FFMPEG_HINT: &str = "winget install Gyan.FFmpeg";
#[cfg(not(any(target_os = "macos", windows)))]
const UNIT_LABEL: &str = "systemd unit";
#[cfg(not(any(target_os = "macos", windows)))]
const FFMPEG_HINT: &str =
    "use your package manager, for example apt install ffmpeg or dnf install ffmpeg";

fn check_ffmpeg() -> Verdict {
    match run("ffmpeg", &["-hide_banner", "-version"]) {
        Ok(out) => Verdict::Ok(out.lines().next().unwrap_or("ffmpeg").to_string()),
        Err(e) => Verdict::Fail(format!("{e}; install ffmpeg ({FFMPEG_HINT})")),
    }
}

/// True when `nvidia-smi -L` lists an NVIDIA GPU; always false off Linux.
#[cfg(not(target_os = "linux"))]
pub fn nvidia_card_present() -> bool {
    false
}

/// True when `nvidia-smi -L` lists an NVIDIA GPU.
#[cfg(target_os = "linux")]
pub fn nvidia_card_present() -> bool {
    run("nvidia-smi", &["-L"])
        .map(|o| o.lines().any(|l| l.starts_with("GPU ")))
        .unwrap_or(false)
}

fn check_device(cfg: &Config) -> Verdict {
    if !cfg!(any(feature = "vulkan", feature = "metal")) && cfg.engine.device != Device::Cpu {
        return Verdict::Warn(
            "this binary was built without the vulkan or metal feature; it runs on CPU".into(),
        );
    }
    device_verdict(cfg.engine.device, &enumerate_gpus(), nvidia_card_present())
}

/// Warns when a path root sits under /tmp or /var/tmp while the service has PrivateTmp: the service sees
/// its own empty /tmp, so jobs for files there would be refused or fail.
pub fn path_roots_verdict(roots: &[String], unit: &UnitState) -> Verdict {
    let under_tmp: Vec<&String> = roots
        .iter()
        .filter(|r| {
            let p = expand_tilde(r);
            let p = std::fs::canonicalize(&p).unwrap_or(p);
            p.starts_with("/tmp") || p.starts_with("/var/tmp")
        })
        .collect();
    if under_tmp.is_empty() || unit.load_state != "loaded" || !unit.private_tmp {
        return Verdict::Ok(format!("{} path root(s)", roots.len()));
    }
    Verdict::Warn(format!(
        "jobs.path_roots entry {} is under /tmp or /var/tmp, but hark.service runs with PrivateTmp=yes and cannot see those files; use a root outside /tmp (for example under your home)",
        under_tmp[0]
    ))
}

#[cfg(not(target_os = "linux"))]
fn unit_state() -> Option<UnitState> {
    None
}

#[cfg(target_os = "linux")]
fn unit_state() -> Option<UnitState> {
    let out = run(
        "systemctl",
        &[
            "--user",
            "show",
            "hark.service",
            "--property=LoadState,UnitFileState,ActiveState,SubState,PrivateTmp",
        ],
    )
    .ok()?;
    Some(parse_unit_show(&out))
}

#[cfg(target_os = "macos")]
fn check_unit() -> Verdict {
    // SAFETY: getuid has no preconditions and cannot fail.
    let uid = unsafe { libc::getuid() };
    let target = format!("gui/{uid}/com.matpb.hark");
    match run("launchctl", &["print", &target]) {
        Ok(_) => Verdict::Ok(format!("launchd agent {target} is loaded")),
        Err(e) => Verdict::Warn(format!(
            "launchd agent com.matpb.hark is not loaded ({e}); run scripts/install-macos.sh"
        )),
    }
}

#[cfg(windows)]
fn check_unit() -> Verdict {
    match run("schtasks", &["/Query", "/TN", "hark"]) {
        Ok(_) => Verdict::Ok("Scheduled Task hark is registered".into()),
        Err(e) => Verdict::Warn(format!(
            "Scheduled Task hark is not registered ({e}); run scripts/install-windows.ps1"
        )),
    }
}

#[cfg(target_os = "linux")]
fn check_unit() -> Verdict {
    match run(
        "systemctl",
        &[
            "--user",
            "show",
            "hark.service",
            "--property=LoadState,UnitFileState,ActiveState,SubState,PrivateTmp",
        ],
    ) {
        Ok(out) => unit_verdict(&parse_unit_show(&out)),
        Err(e) => Verdict::Warn(format!("cannot query the systemd user manager: {e}")),
    }
}

fn probe_port(addr: std::net::SocketAddr) -> PortState {
    if std::net::TcpListener::bind(addr).is_ok() {
        return PortState::Free;
    }
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(3))
        .build();
    let dial = if addr.ip().is_unspecified() {
        let ip: std::net::IpAddr = if addr.is_ipv4() {
            std::net::Ipv4Addr::LOCALHOST.into()
        } else {
            std::net::Ipv6Addr::LOCALHOST.into()
        };
        std::net::SocketAddr::new(ip, addr.port())
    } else {
        addr
    };
    match agent.get(&format!("http://{dial}/health")).call() {
        Ok(r) => match r.into_string() {
            Ok(body) => match serde_json::from_str::<serde_json::Value>(&body) {
                Ok(v) if v.get("version").is_some() && v.get("status").is_some() => {
                    PortState::Hark {
                        status: v["status"].as_str().unwrap_or("?").to_string(),
                        version: v["version"].as_str().unwrap_or("?").to_string(),
                    }
                }
                _ => PortState::Other("the /health body is not hark's".into()),
            },
            Err(e) => PortState::Other(format!("unreadable /health body: {e}")),
        },
        Err(e) => PortState::Other(format!("/health failed: {e}")),
    }
}

fn sha256_file(path: &Path, label: &str) -> std::io::Result<String> {
    let mut f = std::fs::File::open(path)?;
    let total = f.metadata()?.len();
    let show = total > 64 << 20 && std::io::stderr().is_terminal();
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        done += n as u64;
        if show {
            eprint!("\r  hashing {label}: {}%", done * 100 / total.max(1));
        }
    }
    if show {
        eprint!("\r{:width$}\r", "", width = 40 + label.len());
        let _ = std::io::stderr().flush();
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn check_model_file(what: &str, entry: &CatalogEntry, path: &Path) -> Verdict {
    match sha256_file(path, entry.file) {
        Ok(actual) => hash_verdict(what, path, entry.sha256, &actual),
        Err(e) => Verdict::Fail(format!("{what} {}: cannot read: {e}", path.display())),
    }
}

fn check_models(cfg: &Config, registry: &Registry, all: bool) -> Vec<(String, Verdict)> {
    let mut out = Vec::new();
    match registry.resolve(&cfg.engine.default_model) {
        Err(e) => out.push((
            "default model".into(),
            Verdict::Fail(format!("engine.default_model: {e}")),
        )),
        Ok(entry) => match registry.locate(entry) {
            None => out.push((
                "default model".into(),
                Verdict::Fail(format!(
                    "{} ({}) is not installed; run hark models pull {}",
                    entry.id, entry.file, entry.id
                )),
            )),
            Some(p) => out.push((
                "default model".into(),
                check_model_file(entry.id, entry, &p),
            )),
        },
    }
    if cfg.vad.enabled {
        let name = cfg.vad.model.as_str();
        let pin = CATALOG.iter().find(|e| e.file == name);
        match (registry.locate_file(name), pin) {
            (None, _) => out.push((
                "vad model".into(),
                Verdict::Warn(format!(
                    "{name} is not installed, silence trimming is off; run hark models pull <vad model id>"
                )),
            )),
            (Some(p), Some(entry)) => {
                out.push(("vad model".into(), check_model_file("vad", entry, &p)))
            }
            (Some(p), None) => out.push((
                "vad model".into(),
                Verdict::Warn(format!("{} has no catalog pin to check", p.display())),
            )),
        }
    }
    if all {
        for entry in CATALOG {
            let Some(p) = registry.locate(entry) else {
                continue;
            };
            if cfg.vad.enabled && entry.file == cfg.vad.model {
                continue;
            }
            if registry.resolve(&cfg.engine.default_model).ok() == Some(entry) {
                continue;
            }
            out.push((
                format!("model {}", entry.id),
                check_model_file(entry.id, entry, &p),
            ));
        }
    }
    out
}

fn report(name: &str, v: &Verdict, failed: &mut bool) {
    if matches!(v, Verdict::Fail(_)) {
        *failed = true;
    }
    println!("{}  {name}: {}", v.label(), v.message());
}

/// Runs every check and prints the report; returns the process exit code.
pub fn run_doctor(config: Option<PathBuf>, all: bool) -> i32 {
    let path = config.unwrap_or_else(default_config_path);
    let mut failed = false;
    report("ffmpeg", &check_ffmpeg(), &mut failed);

    let cfg = match Config::load(&path) {
        Ok(c) => {
            let msg = if path.exists() {
                format!("{} parses and validates", path.display())
            } else {
                format!("{} not present, using defaults", path.display())
            };
            report("config", &Verdict::Ok(msg), &mut failed);
            c
        }
        Err(e) => {
            report("config", &Verdict::Fail(e.to_string()), &mut failed);
            Config::default()
        }
    };

    report("vulkan", &check_device(&cfg), &mut failed);

    let registry = Registry::from_config(&cfg);
    for (name, v) in check_models(&cfg, &registry, all) {
        report(&name, &v, &mut failed);
    }

    if !cfg.server.api_key_file.is_empty() {
        let p = expand_tilde(&cfg.server.api_key_file);
        let v = match crate::auth::load_key(&cfg.server.api_key_file) {
            Ok(_) => permission_verdict(None, &p),
            Err(e) => permission_verdict(Some(e), &p),
        };
        report("api_key_file", &v, &mut failed);
    }

    match cfg.listen_addr() {
        Ok(addr) => report(
            "port",
            &port_verdict(&addr.to_string(), &probe_port(addr)),
            &mut failed,
        ),
        Err(e) => report("port", &Verdict::Fail(e.to_string()), &mut failed),
    }

    report(UNIT_LABEL, &check_unit(), &mut failed);
    if let Some(unit) = unit_state() {
        report(
            "jobs.path_roots",
            &path_roots_verdict(&cfg.jobs.path_roots, &unit),
            &mut failed,
        );
    }
    if failed { 1 } else { 0 }
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
    fn auto_with_both_picks_discrete_and_is_ok() {
        let g = [gpu(0, "Intel", false), gpu(1, "NVIDIA", true)];
        assert!(matches!(
            device_verdict(Device::Auto, &g, true),
            Verdict::Ok(m) if m.contains("gpu:1") && m.contains("discrete")
        ));
    }

    #[test]
    fn explicit_igpu_with_a_discrete_present_fails() {
        let g = [gpu(0, "Intel", false), gpu(1, "NVIDIA", true)];
        assert!(matches!(
            device_verdict(Device::Gpu(0), &g, true),
            Verdict::Fail(m) if m.contains("integrated")
        ));
    }

    #[test]
    fn only_igpu_with_nvidia_smi_card_warns_with_the_libegl_fix() {
        let g = [gpu(0, "Intel", false)];
        assert!(matches!(
            device_verdict(Device::Auto, &g, true),
            Verdict::Warn(m) if m.contains("libEGL")
        ));
        assert!(matches!(
            device_verdict(Device::Auto, &g, false),
            Verdict::Ok(_)
        ));
    }

    #[test]
    fn no_devices_and_cpu_and_bad_index() {
        assert!(matches!(
            device_verdict(Device::Auto, &[], true),
            Verdict::Warn(m) if m.contains("libEGL")
        ));
        assert!(matches!(
            device_verdict(Device::Auto, &[], false),
            Verdict::Warn(_)
        ));
        assert!(matches!(
            device_verdict(Device::Cpu, &[], false),
            Verdict::Ok(_)
        ));
        assert!(matches!(
            device_verdict(Device::Gpu(3), &[gpu(0, "A", true)], false),
            Verdict::Fail(_)
        ));
    }

    #[test]
    fn port_verdicts() {
        let a = "127.0.0.1:8797";
        assert!(matches!(port_verdict(a, &PortState::Free), Verdict::Ok(_)));
        let ok = PortState::Hark {
            status: "ok".into(),
            version: "0.1.0".into(),
        };
        assert!(matches!(port_verdict(a, &ok), Verdict::Ok(m) if m.contains("0.1.0")));
        let deg = PortState::Hark {
            status: "degraded".into(),
            version: "0.1.0".into(),
        };
        assert!(matches!(port_verdict(a, &deg), Verdict::Warn(_)));
        assert!(matches!(
            port_verdict(a, &PortState::Other("nginx".into())),
            Verdict::Fail(m) if m.contains("nginx")
        ));
    }

    #[test]
    fn port_probe_sees_free_and_foreign_listeners() {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        assert!(matches!(probe_port(addr), PortState::Other(_)));
        drop(l);
        assert_eq!(probe_port(addr), PortState::Free);
    }

    #[cfg(unix)]
    #[test]
    fn permission_verdicts_wrap_the_key_file_rules() {
        let p = Path::new("/k");
        assert!(matches!(permission_verdict(None, p), Verdict::Ok(_)));
        let problem = crate::auth::key_file_problem(p, true, 1, 1, 0o644);
        assert!(matches!(
            permission_verdict(problem, p),
            Verdict::Fail(m) if m.contains("0644")
        ));
    }

    #[test]
    fn unit_verdicts() {
        let show =
            "LoadState=loaded\nUnitFileState=enabled\nActiveState=active\nSubState=running\n";
        let u = parse_unit_show(show);
        assert_eq!(u.sub_state, "running");
        assert!(matches!(unit_verdict(&u), Verdict::Ok(_)));
        let missing =
            parse_unit_show("LoadState=not-found\nUnitFileState=\nActiveState=inactive\n");
        assert!(matches!(unit_verdict(&missing), Verdict::Warn(_)));
        let off = UnitState {
            load_state: "loaded".into(),
            unit_file_state: "disabled".into(),
            active_state: "inactive".into(),
            sub_state: "dead".into(),
            private_tmp: false,
        };
        assert!(matches!(unit_verdict(&off), Verdict::Fail(_)));
        let not_enabled = UnitState {
            unit_file_state: "disabled".into(),
            active_state: "active".into(),
            ..off.clone()
        };
        assert!(matches!(unit_verdict(&not_enabled), Verdict::Fail(m) if m.contains("enable")));
        let failed = UnitState {
            unit_file_state: "enabled".into(),
            active_state: "failed".into(),
            sub_state: "failed".into(),
            ..off
        };
        assert!(matches!(unit_verdict(&failed), Verdict::Fail(_)));
    }

    #[test]
    fn hash_verdicts_and_file_hashing() {
        let p = std::env::temp_dir().join(format!("hark-doctor-{}.bin", std::process::id()));
        std::fs::write(&p, b"abc").unwrap();
        let digest = sha256_file(&p, "t").unwrap();
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert!(matches!(
            hash_verdict("m", &p, &digest, &digest),
            Verdict::Ok(_)
        ));
        assert!(matches!(
            hash_verdict("m", &p, &"0".repeat(64), &digest),
            Verdict::Fail(_)
        ));
        std::fs::remove_file(p).ok();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn path_roots_under_tmp_warn_only_when_the_unit_has_private_tmp() {
        let unit = |private_tmp: bool, load: &str| UnitState {
            load_state: load.into(),
            private_tmp,
            ..UnitState::default()
        };
        let roots = |r: &[&str]| r.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let v = path_roots_verdict(&roots(&["~", "/tmp/media"]), &unit(true, "loaded"));
        assert!(
            matches!(&v, Verdict::Warn(m) if m.contains("/tmp/media") && m.contains("PrivateTmp")),
            "{v:?}"
        );
        let v = path_roots_verdict(&roots(&["/var/tmp"]), &unit(true, "loaded"));
        assert!(matches!(v, Verdict::Warn(_)));
        let v = path_roots_verdict(&roots(&["/tmp/media"]), &unit(false, "loaded"));
        assert!(matches!(v, Verdict::Ok(_)), "no PrivateTmp, no problem");
        let v = path_roots_verdict(&roots(&["/tmp/media"]), &unit(true, "not-found"));
        assert!(matches!(v, Verdict::Ok(_)), "no unit installed");
        let v = path_roots_verdict(&roots(&["~", "/srv/audio", "/tmpx"]), &unit(true, "loaded"));
        assert!(matches!(v, Verdict::Ok(_)));
        let parsed = parse_unit_show("LoadState=loaded\nPrivateTmp=yes\n");
        assert!(parsed.private_tmp);
        assert!(!parse_unit_show("PrivateTmp=no\n").private_tmp);
    }

    #[test]
    fn unit_show_ignores_unknown_lines() {
        let u = parse_unit_show("Foo=bar\nnoequals\nLoadState=loaded\n");
        assert_eq!(u.load_state, "loaded");
    }
}
