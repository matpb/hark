//! Vocabulary profiles: terms from config, term files and Wspr's config, held behind an
//! atomically swapped snapshot and reloaded on SIGHUP or when a referenced file changes.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

use crate::config::{Config, VocabularyProfile, expand_tilde, valid_profile_name};

pub const DEFAULT_PROFILE: &str = "default";
pub const NO_PROFILE: &str = "none";
/// Wspr's `config.json`, inside Wspr's own folder.
pub fn default_wspr_config() -> String {
    format!("{}/config.json", crate::config::wspr_dir())
}
const WSPR_POINTER: &str = "/transcription/custom_vocabulary";
const MAX_FILE_BYTES: u64 = 1 << 20;
pub const POLL_INTERVAL: Duration = Duration::from_secs(2);

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum VocabularyError {
    #[error("`{0}` is not a valid vocabulary profile name (expected [a-z0-9_-]{{1,64}} or none)")]
    InvalidName(String),
    #[error("vocabulary profile `{0}` does not exist")]
    Unknown(String),
}

pub const MAX_KEYWORDS: usize = 100;
pub const MAX_KEYWORD_CHARS: usize = 50;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KeywordError {
    #[error("a request may carry at most {MAX_KEYWORDS} keywords")]
    TooMany,
    #[error("keywords[{0}] is empty")]
    Empty(usize),
    #[error("keywords[{0}] must not contain '<', '>', line breaks or other control characters")]
    Forbidden(usize),
    #[error("keywords[{0}] is longer than {MAX_KEYWORD_CHARS} characters")]
    TooLong(usize),
}

/// Trimmed request keywords in order, later duplicates ignoring case dropped (first spelling kept).
pub fn request_keywords(raw: &[String]) -> Result<Vec<String>, KeywordError> {
    if raw.len() > MAX_KEYWORDS {
        return Err(KeywordError::TooMany);
    }
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(raw.len());
    for (i, item) in raw.iter().enumerate() {
        let term = item.trim();
        if term.is_empty() {
            return Err(KeywordError::Empty(i));
        }
        if term
            .chars()
            .any(|c| matches!(c, '<' | '>' | '\u{2028}' | '\u{2029}') || c.is_control())
        {
            return Err(KeywordError::Forbidden(i));
        }
        if term.chars().count() > MAX_KEYWORD_CHARS {
            return Err(KeywordError::TooLong(i));
        }
        if seen.insert(term.to_lowercase()) {
            out.push(term.to_string());
        }
    }
    Ok(out)
}

type Stamp = Option<(SystemTime, u64)>;

fn stamp(path: &Path) -> Stamp {
    std::fs::metadata(path)
        .ok()
        .map(|m| (m.modified().unwrap_or(SystemTime::UNIX_EPOCH), m.len()))
}

/// One immutable generation of every profile's merged terms, plus the files it was built from.
#[derive(Debug, Default)]
pub struct Snapshot {
    profiles: BTreeMap<String, Vec<String>>,
    watched: Vec<(PathBuf, Stamp)>,
}

impl Snapshot {
    pub fn terms(&self, name: &str) -> Option<&[String]> {
        self.profiles.get(name).map(Vec::as_slice)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.profiles.keys().map(String::as_str)
    }
}

/// Lines of a term file: one term per line, `#` comment lines and blank lines skipped.
pub fn parse_term_file(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

fn read_term_file(path: &Path) -> Vec<String> {
    match std::fs::metadata(path) {
        Ok(m) if m.len() > MAX_FILE_BYTES => {
            tracing::warn!(file = %path.display(), "vocabulary file is over 1 MiB; ignored");
            return Vec::new();
        }
        Ok(_) => {}
        Err(e) => {
            tracing::warn!(file = %path.display(), error = %e, "vocabulary file unreadable; ignored");
            return Vec::new();
        }
    }
    match std::fs::read_to_string(path) {
        Ok(text) => parse_term_file(&text),
        Err(e) => {
            tracing::warn!(file = %path.display(), error = %e, "vocabulary file unreadable; ignored");
            Vec::new()
        }
    }
}

/// Terms from Wspr's `/transcription/custom_vocabulary`. A missing file is normal (debug log);
/// malformed JSON or a missing pointer yields no terms and a warning; non-string entries are skipped.
pub fn read_wspr_terms(path: &Path) -> Vec<String> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::debug!(file = %path.display(), "no Wspr config; skipping its vocabulary");
            return Vec::new();
        }
        Err(e) => {
            tracing::warn!(file = %path.display(), error = %e, "Wspr config unreadable; ignored");
            return Vec::new();
        }
    };
    let json: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(file = %path.display(), error = %e, "Wspr config is not valid JSON; ignored");
            return Vec::new();
        }
    };
    let Some(list) = json.pointer(WSPR_POINTER).and_then(|v| v.as_array()) else {
        tracing::warn!(file = %path.display(), pointer = WSPR_POINTER, "Wspr config has no vocabulary list; ignored");
        return Vec::new();
    };
    let terms: Vec<String> = list
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    if terms.len() != list.len() {
        tracing::warn!(file = %path.display(), pointer = WSPR_POINTER, skipped = list.len() - terms.len(), "Wspr vocabulary has non-string entries; skipped");
    }
    terms
}

/// Config terms, then each file in order, then the Wspr list; trimmed, deduplicated, first wins.
fn merge_terms(groups: impl IntoIterator<Item = Vec<String>>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for t in groups.into_iter().flatten() {
        let t = t.trim();
        if !t.is_empty() && !t.contains('\0') && seen.insert(t.to_string()) {
            out.push(t.to_string());
        }
    }
    out
}

fn build_snapshot(tables: &BTreeMap<String, VocabularyProfile>) -> Snapshot {
    let mut tables = tables.clone();
    tables
        .entry(DEFAULT_PROFILE.to_string())
        .or_insert_with(|| VocabularyProfile {
            terms: Vec::new(),
            files: Vec::new(),
            wspr_config: default_wspr_config(),
        });
    let mut snap = Snapshot::default();
    let mut watched: Vec<PathBuf> = Vec::new();
    for (name, p) in &tables {
        let mut groups = vec![p.terms.clone()];
        for f in &p.files {
            let path = expand_tilde(f);
            groups.push(read_term_file(&path));
            watched.push(path);
        }
        if !p.wspr_config.is_empty() {
            let path = expand_tilde(&p.wspr_config);
            groups.push(read_wspr_terms(&path));
            watched.push(path);
        }
        let terms = merge_terms(groups);
        tracing::info!(profile = %name, terms = terms.len(), "vocabulary profile loaded");
        snap.profiles.insert(name.clone(), terms);
    }
    watched.sort();
    watched.dedup();
    snap.watched = watched
        .into_iter()
        .map(|p| {
            let s = stamp(&p);
            (p, s)
        })
        .collect();
    snap
}

pub struct VocabularyStore {
    tables: RwLock<BTreeMap<String, VocabularyProfile>>,
    current: RwLock<Arc<Snapshot>>,
}

impl VocabularyStore {
    pub fn new(tables: &BTreeMap<String, VocabularyProfile>) -> VocabularyStore {
        VocabularyStore {
            tables: RwLock::new(tables.clone()),
            current: RwLock::new(Arc::new(build_snapshot(tables))),
        }
    }

    /// The current generation; a held snapshot never changes under the caller.
    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.current
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    fn swap(&self, snap: Snapshot) {
        *self.current.write().unwrap_or_else(|p| p.into_inner()) = Arc::new(snap);
    }

    /// `None` is `default`, `none` is no vocabulary, anything else must be a known profile.
    pub fn resolve(&self, name: Option<&str>) -> Result<Vec<String>, VocabularyError> {
        let name = name.map(str::trim).filter(|n| !n.is_empty());
        let name = name.unwrap_or(DEFAULT_PROFILE);
        if name == NO_PROFILE {
            return Ok(Vec::new());
        }
        if !valid_profile_name(name) {
            return Err(VocabularyError::InvalidName(name.to_string()));
        }
        self.snapshot()
            .terms(name)
            .map(<[String]>::to_vec)
            .ok_or_else(|| VocabularyError::Unknown(name.to_string()))
    }

    pub fn replace_tables(&self, tables: &BTreeMap<String, VocabularyProfile>) {
        *self.tables.write().unwrap_or_else(|p| p.into_inner()) = tables.clone();
        self.swap(build_snapshot(tables));
    }

    /// Rebuilds when any referenced file changed in mtime or size; true when it did.
    pub fn refresh_files(&self) -> bool {
        let snap = self.snapshot();
        if snap.watched.iter().all(|(p, s)| stamp(p) == *s) {
            return false;
        }
        let tables = self
            .tables
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .clone();
        self.swap(build_snapshot(&tables));
        tracing::info!("vocabulary files changed; profiles reloaded");
        true
    }

    /// SIGHUP path: re-read only `[vocabulary.*]`. A bad file keeps the previous profiles.
    pub fn reload_from_config_file(&self, path: &Path, running: &Config) -> Result<(), String> {
        let new = Config::load(path).map_err(|e| {
            tracing::warn!(error = %e, "config reload failed; keeping the previous vocabulary");
            e.to_string()
        })?;
        let strip = |c: &Config| Config {
            vocabulary: BTreeMap::new(),
            ..c.clone()
        };
        if strip(&new) != strip(running) {
            tracing::info!("config keys outside [vocabulary.*] changed; they need a restart");
        }
        self.replace_tables(&new.vocabulary);
        Ok(())
    }

    /// Starts the SIGHUP and file-change reloader; the handler is installed before this returns.
    pub fn spawn_watcher(
        self: &Arc<Self>,
        config_path: PathBuf,
        running: Arc<Config>,
    ) -> std::io::Result<tokio::task::JoinHandle<()>> {
        #[cfg(unix)]
        let mut hup = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())?;
        #[cfg(not(unix))]
        let mut hup = ();
        let store = self.clone();
        Ok(tokio::spawn(async move {
            let mut tick = tokio::time::interval_at(
                tokio::time::Instant::from_std(Instant::now() + POLL_INTERVAL),
                POLL_INTERVAL,
            );
            loop {
                let s = store.clone();
                tokio::select! {
                    _ = hup_recv(&mut hup) => {
                        tracing::info!("SIGHUP: reloading vocabulary profiles");
                        let (path, running) = (config_path.clone(), running.clone());
                        let _ = tokio::task::spawn_blocking(move || {
                            let _ = s.reload_from_config_file(&path, &running);
                        })
                        .await;
                    }
                    _ = tick.tick() => {
                        let _ = tokio::task::spawn_blocking(move || s.refresh_files()).await;
                    }
                }
            }
        }))
    }
}

#[cfg(unix)]
async fn hup_recv(hup: &mut tokio::signal::unix::Signal) {
    hup.recv().await;
}

/// No SIGHUP off unix; the file poll still reloads.
#[cfg(not(unix))]
async fn hup_recv(_: &mut ()) {
    std::future::pending::<()>().await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("hark-vocab-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn profile(terms: &[&str]) -> VocabularyProfile {
        VocabularyProfile {
            terms: terms.iter().map(|s| s.to_string()).collect(),
            files: Vec::new(),
            wspr_config: String::new(),
        }
    }

    fn tables(entries: &[(&str, VocabularyProfile)]) -> BTreeMap<String, VocabularyProfile> {
        entries
            .iter()
            .map(|(n, p)| (n.to_string(), p.clone()))
            .collect()
    }

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn default_none_named_and_unknown_profiles_resolve() {
        let store = VocabularyStore::new(&tables(&[
            ("default", profile(&["Kubernetes", "Grafana"])),
            ("legal", profile(&["estoppel"])),
        ]));
        assert_eq!(store.resolve(None).unwrap(), s(&["Kubernetes", "Grafana"]));
        assert_eq!(
            store.resolve(Some("")).unwrap(),
            s(&["Kubernetes", "Grafana"])
        );
        assert_eq!(
            store.resolve(Some("default")).unwrap(),
            s(&["Kubernetes", "Grafana"])
        );
        assert_eq!(store.resolve(Some("legal")).unwrap(), s(&["estoppel"]));
        assert!(store.resolve(Some("none")).unwrap().is_empty());
        assert_eq!(
            store.resolve(Some("nope")),
            Err(VocabularyError::Unknown("nope".into()))
        );
        assert!(matches!(
            store.resolve(Some("Bad Name")),
            Err(VocabularyError::InvalidName(_))
        ));
        assert!(matches!(
            store.resolve(Some("../etc")),
            Err(VocabularyError::InvalidName(_))
        ));
    }

    #[test]
    fn builtin_default_exists_without_config_and_uses_the_wspr_path() {
        let store = VocabularyStore::new(&BTreeMap::new());
        let snap = store.snapshot();
        assert_eq!(snap.names().collect::<Vec<_>>(), vec!["default"]);
        assert!(
            snap.watched
                .iter()
                .any(|(p, _)| *p == expand_tilde(&default_wspr_config()))
        );
    }

    #[test]
    fn merge_order_is_config_then_files_then_wspr_with_first_occurrence_kept() {
        let d = dir("merge");
        std::fs::write(
            d.join("a.txt"),
            "# comment\n\n  Beta  \nAlpha\n\t\n# Gamma\nDelta\n",
        )
        .unwrap();
        std::fs::write(d.join("b.txt"), "Delta\nEpsilon\n").unwrap();
        std::fs::write(
            d.join("wspr.json"),
            r#"{"transcription":{"custom_vocabulary":["Epsilon","Zeta","Alpha"," Eta "]}}"#,
        )
        .unwrap();
        let p = VocabularyProfile {
            terms: s(&["Alpha", "Omega", "Alpha"]),
            files: vec![
                d.join("a.txt").display().to_string(),
                d.join("b.txt").display().to_string(),
            ],
            wspr_config: d.join("wspr.json").display().to_string(),
        };
        let store = VocabularyStore::new(&tables(&[("default", p)]));
        assert_eq!(
            store.resolve(None).unwrap(),
            s(&["Alpha", "Omega", "Beta", "Delta", "Epsilon", "Zeta", "Eta"])
        );
    }

    #[test]
    fn keywords_are_trimmed_in_order_and_deduped_ignoring_case() {
        assert_eq!(
            request_keywords(&s(&["  Ghidra ", "ghidra", "Zstandard"])),
            Ok(s(&["Ghidra", "Zstandard"]))
        );
        assert_eq!(request_keywords(&[]), Ok(Vec::new()));
        assert_eq!(
            request_keywords(&s(&["Ünï", "ÜNÏ", "b"])),
            Ok(s(&["Ünï", "b"]))
        );
        assert_eq!(
            request_keywords(&s(&["two words", "TWO WORDS"])),
            Ok(s(&["two words"]))
        );
    }

    #[test]
    fn keywords_reject_angle_brackets_and_control_characters() {
        assert_eq!(
            request_keywords(&s(&["a<b"])),
            Err(KeywordError::Forbidden(0))
        );
        assert_eq!(
            request_keywords(&s(&["a>b"])),
            Err(KeywordError::Forbidden(0))
        );
        assert_eq!(
            request_keywords(&s(&["ok", "x\ny"])),
            Err(KeywordError::Forbidden(1))
        );
        for bad in [
            "x\ry",
            "x\ty",
            "x\0y",
            "x\u{7f}y",
            "x\u{85}y",
            "x\u{2028}y",
            "x\u{2029}y",
        ] {
            assert_eq!(
                request_keywords(&s(&["ok", bad])),
                Err(KeywordError::Forbidden(1)),
                "{bad:?}"
            );
        }
        assert_eq!(
            request_keywords(&s(&["\nok\t"])),
            Ok(s(&["ok"])),
            "edges are trimmed first"
        );
    }

    #[test]
    fn keywords_reject_empty_items_by_their_received_index() {
        assert_eq!(
            request_keywords(&s(&["ok", "  "])),
            Err(KeywordError::Empty(1))
        );
        assert_eq!(request_keywords(&s(&[""])), Err(KeywordError::Empty(0)));
        assert_eq!(
            request_keywords(&s(&["a", "A", "", "b"])),
            Err(KeywordError::Empty(2)),
            "the index counts dropped duplicates"
        );
    }

    #[test]
    fn keywords_length_is_counted_in_characters_and_capped_at_50() {
        assert_eq!(
            request_keywords(&s(&[&"x".repeat(50)])),
            Ok(s(&[&"x".repeat(50)]))
        );
        assert_eq!(
            request_keywords(&s(&[&"x".repeat(51)])),
            Err(KeywordError::TooLong(0))
        );
        assert!(
            request_keywords(&s(&[&"é".repeat(50)])).is_ok(),
            "50 chars, 100 bytes"
        );
        assert_eq!(
            request_keywords(&s(&["ok", &"é".repeat(51)])),
            Err(KeywordError::TooLong(1))
        );
        assert!(request_keywords(&s(&[&format!("  {}  ", "x".repeat(50))])).is_ok());
    }

    #[test]
    fn keywords_count_is_capped_at_100_before_any_item_is_checked() {
        let hundred: Vec<String> = (0..100).map(|i| format!("term{i}")).collect();
        assert_eq!(request_keywords(&hundred).unwrap().len(), 100);
        let mut many = hundred.clone();
        many.push("term100".into());
        assert_eq!(request_keywords(&many), Err(KeywordError::TooMany));
        many[0] = "<".into();
        assert_eq!(request_keywords(&many), Err(KeywordError::TooMany));
        let same = vec!["a".to_string(); 101];
        assert_eq!(request_keywords(&same), Err(KeywordError::TooMany));
    }

    #[test]
    fn keyword_errors_read_as_the_api_messages() {
        assert_eq!(
            KeywordError::TooMany.to_string(),
            "a request may carry at most 100 keywords"
        );
        assert_eq!(KeywordError::Empty(3).to_string(), "keywords[3] is empty");
        assert_eq!(
            KeywordError::Forbidden(2).to_string(),
            "keywords[2] must not contain '<', '>', line breaks or other control characters"
        );
        assert_eq!(
            KeywordError::TooLong(1).to_string(),
            "keywords[1] is longer than 50 characters"
        );
    }

    #[test]
    fn term_file_parsing_skips_comments_and_blanks() {
        assert_eq!(
            parse_term_file("# c\n\nKubernetes\n  Grafana \n#x\n"),
            s(&["Kubernetes", "Grafana"])
        );
        assert!(parse_term_file("").is_empty());
    }

    #[test]
    fn tilde_in_files_is_expanded() {
        let home = dirs::home_dir().unwrap();
        let p = VocabularyProfile {
            files: vec!["~/does-not-exist-hark-test.txt".into()],
            ..Default::default()
        };
        let store = VocabularyStore::new(&tables(&[("default", p)]));
        let watched = &store.snapshot().watched;
        assert_eq!(watched[0].0, home.join("does-not-exist-hark-test.txt"));
    }

    #[test]
    fn wspr_import_never_errors() {
        let d = dir("wspr");
        assert!(read_wspr_terms(&d.join("missing.json")).is_empty());
        let write = |name: &str, body: &str| {
            let p = d.join(name);
            std::fs::write(&p, body).unwrap();
            p
        };
        assert!(read_wspr_terms(&write("bad.json", "{not json")).is_empty());
        assert!(read_wspr_terms(&write("empty.json", "")).is_empty());
        assert!(read_wspr_terms(&write("nopointer.json", r#"{"transcription":{}}"#)).is_empty());
        assert!(
            read_wspr_terms(&write(
                "notlist.json",
                r#"{"transcription":{"custom_vocabulary":"x"}}"#
            ))
            .is_empty()
        );
        assert_eq!(
            read_wspr_terms(&write(
                "mixed.json",
                r#"{"transcription":{"custom_vocabulary":["a",3,"b"]}}"#
            )),
            s(&["a", "b"])
        );
        assert_eq!(
            read_wspr_terms(&write(
                "ok.json",
                r#"{"transcription":{"custom_vocabulary":["a","b"]}}"#
            )),
            s(&["a", "b"])
        );
        assert!(read_wspr_terms(&d).is_empty(), "a directory is not a file");
    }

    #[test]
    fn broken_wspr_config_leaves_the_other_terms_in_place() {
        let d = dir("wsprmix");
        std::fs::write(d.join("w.json"), "{oops").unwrap();
        let p = VocabularyProfile {
            terms: s(&["Wspr"]),
            wspr_config: d.join("w.json").display().to_string(),
            ..Default::default()
        };
        let store = VocabularyStore::new(&tables(&[("default", p)]));
        assert_eq!(store.resolve(None).unwrap(), s(&["Wspr"]));
    }

    #[test]
    fn file_change_swaps_profiles_and_old_snapshots_stay_intact() {
        let d = dir("filechange");
        let f = d.join("terms.txt");
        std::fs::write(&f, "One\n").unwrap();
        let p = VocabularyProfile {
            files: vec![f.display().to_string()],
            ..Default::default()
        };
        let store = VocabularyStore::new(&tables(&[("default", p)]));
        let before = store.snapshot();
        assert!(!store.refresh_files(), "nothing changed yet");
        std::fs::write(&f, "One\nTwo\nThree\n").unwrap();
        assert!(store.refresh_files());
        assert_eq!(store.resolve(None).unwrap(), s(&["One", "Two", "Three"]));
        assert_eq!(before.terms("default").unwrap(), ["One"]);
        assert!(!store.refresh_files(), "stable after the reload");
        std::fs::remove_file(&f).unwrap();
        assert!(store.refresh_files());
        assert!(store.resolve(None).unwrap().is_empty());
        std::fs::write(&f, "Back\n").unwrap();
        assert!(store.refresh_files());
        assert_eq!(store.resolve(None).unwrap(), s(&["Back"]));
    }

    #[test]
    fn readers_never_see_a_half_swapped_state() {
        let a = tables(&[
            ("default", profile(&["a1", "a2"])),
            ("x", profile(&["a1", "a2"])),
        ]);
        let b = tables(&[
            ("default", profile(&["b1", "b2"])),
            ("x", profile(&["b1", "b2"])),
        ]);
        let store = Arc::new(VocabularyStore::new(&a));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ready = Arc::new(std::sync::Barrier::new(5));
        let readers: Vec<_> = (0..4)
            .map(|_| {
                let (store, stop, ready) = (store.clone(), stop.clone(), ready.clone());
                std::thread::spawn(move || {
                    let mut n = 0u64;
                    ready.wait();
                    while n == 0 || !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        let snap = store.snapshot();
                        assert_eq!(snap.terms("default"), snap.terms("x"), "torn snapshot");
                        n += 1;
                    }
                    n
                })
            })
            .collect();
        ready.wait();
        for i in 0..500 {
            store.replace_tables(if i % 2 == 0 { &b } else { &a });
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for r in readers {
            assert!(r.join().unwrap() > 0);
        }
    }

    #[test]
    fn config_reload_swaps_vocabulary_and_failure_keeps_the_previous_one() {
        let d = dir("reload");
        let cfg_path = d.join("config.toml");
        std::fs::write(&cfg_path, "[vocabulary.default]\nterms = [\"Old\"]\n").unwrap();
        let running = Config::load(&cfg_path).unwrap();
        let store = VocabularyStore::new(&running.vocabulary);
        assert_eq!(store.resolve(None).unwrap(), s(&["Old"]));

        std::fs::write(
            &cfg_path,
            "[vocabulary.default]\nterms = [\"New\"]\n[vocabulary.extra]\nterms = [\"E\"]\n[server]\nmax_upload_mb = 7\n",
        )
        .unwrap();
        store.reload_from_config_file(&cfg_path, &running).unwrap();
        assert_eq!(store.resolve(None).unwrap(), s(&["New"]));
        assert_eq!(store.resolve(Some("extra")).unwrap(), s(&["E"]));

        std::fs::write(&cfg_path, "[vocabulary.default\nbroken").unwrap();
        assert!(store.reload_from_config_file(&cfg_path, &running).is_err());
        assert_eq!(store.resolve(None).unwrap(), s(&["New"]));
        std::fs::write(&cfg_path, "[vocabulary.\"Bad Name\"]\nterms = []\n").unwrap();
        assert!(store.reload_from_config_file(&cfg_path, &running).is_err());
        assert_eq!(store.resolve(Some("extra")).unwrap(), s(&["E"]));
    }

    #[tokio::test]
    async fn sighup_reloads_the_config_and_polling_reloads_changed_files() {
        let d = dir("sighup");
        let cfg_path = d.join("config.toml");
        let terms_file = d.join("t.txt");
        std::fs::write(&terms_file, "FileOne\n").unwrap();
        std::fs::write(
            &cfg_path,
            format!(
                "[vocabulary.default]\nterms = [\"Before\"]\nfiles = [\"{}\"]\n",
                terms_file.display()
            ),
        )
        .unwrap();
        let running = Arc::new(Config::load(&cfg_path).unwrap());
        let store = Arc::new(VocabularyStore::new(&running.vocabulary));
        let _task = store.spawn_watcher(cfg_path.clone(), running).unwrap();
        assert_eq!(store.resolve(None).unwrap(), s(&["Before", "FileOne"]));

        std::fs::write(&terms_file, "FileOne\nFileTwo\n").unwrap();
        let wait_for = |want: Vec<String>| {
            let store = store.clone();
            async move {
                for _ in 0..80 {
                    if store.resolve(None).unwrap() == want {
                        return true;
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                false
            }
        };
        assert!(
            wait_for(s(&["Before", "FileOne", "FileTwo"])).await,
            "file poll did not reload"
        );

        std::fs::write(&cfg_path, "[vocabulary.default]\nterms = [\"After\"]\n").unwrap();
        let pid = std::process::id().to_string();
        let st = std::process::Command::new("kill")
            .args(["-HUP", &pid])
            .status()
            .unwrap();
        assert!(st.success());
        assert!(wait_for(s(&["After"])).await, "SIGHUP did not reload");
    }
}
