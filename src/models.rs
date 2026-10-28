use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use crate::config::{Config, expand_tilde};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Transcription,
    Vad,
}

#[derive(Debug, PartialEq, Eq)]
pub struct CatalogEntry {
    pub id: &'static str,
    pub file: &'static str,
    pub sha256: &'static str,
    /// Exact file size in bytes, from the Hugging Face LFS metadata.
    pub size: u64,
    pub kind: ModelKind,
}

pub const HF_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
pub const HF_VAD_BASE: &str = "https://huggingface.co/ggml-org/whisper-vad/resolve/main";

impl CatalogEntry {
    pub fn url(&self) -> String {
        let base = match self.kind {
            ModelKind::Transcription => HF_BASE,
            ModelKind::Vad => HF_VAD_BASE,
        };
        format!("{base}/{}", self.file)
    }
}

const fn tr(id: &'static str, file: &'static str, sha256: &'static str, size: u64) -> CatalogEntry {
    CatalogEntry {
        id,
        file,
        sha256,
        size,
        kind: ModelKind::Transcription,
    }
}

pub static CATALOG: &[CatalogEntry] = &[
    tr(
        "tiny",
        "ggml-tiny.bin",
        "be07e048e1e599ad46341c8d2a135645097a538221678b7acdd1b1919c6e1b21",
        77691713,
    ),
    tr(
        "tiny-q5_1",
        "ggml-tiny-q5_1.bin",
        "818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7",
        32152673,
    ),
    tr(
        "tiny-q8_0",
        "ggml-tiny-q8_0.bin",
        "c2085835d3f50733e2ff6e4b41ae8a2b8d8110461e18821b09a15c40c42d1cca",
        43537433,
    ),
    tr(
        "tiny.en",
        "ggml-tiny.en.bin",
        "921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f",
        77704715,
    ),
    tr(
        "tiny.en-q5_1",
        "ggml-tiny.en-q5_1.bin",
        "c77c5766f1cef09b6b7d47f21b546cbddd4157886b3b5d6d4f709e91e66c7c2b",
        32166155,
    ),
    tr(
        "tiny.en-q8_0",
        "ggml-tiny.en-q8_0.bin",
        "5bc2b3860aa151a4c6e7bb095e1fcce7cf12c7b020ca08dcec0c6d018bb7dd94",
        43550795,
    ),
    tr(
        "base",
        "ggml-base.bin",
        "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
        147951465,
    ),
    tr(
        "base-q5_1",
        "ggml-base-q5_1.bin",
        "422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898",
        59707625,
    ),
    tr(
        "base-q8_0",
        "ggml-base-q8_0.bin",
        "c577b9a86e7e048a0b7eada054f4dd79a56bbfa911fbdacf900ac5b567cbb7d9",
        81768585,
    ),
    tr(
        "base.en",
        "ggml-base.en.bin",
        "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002",
        147964211,
    ),
    tr(
        "base.en-q5_1",
        "ggml-base.en-q5_1.bin",
        "4baf70dd0d7c4247ba2b81fafd9c01005ac77c2f9ef064e00dcf195d0e2fdd2f",
        59721011,
    ),
    tr(
        "base.en-q8_0",
        "ggml-base.en-q8_0.bin",
        "a4d4a0768075e13cfd7e19df3ae2dbc4a68d37d36a7dad45e8410c9a34f8c87e",
        81781811,
    ),
    tr(
        "small",
        "ggml-small.bin",
        "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
        487601967,
    ),
    tr(
        "small-q5_1",
        "ggml-small-q5_1.bin",
        "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
        190085487,
    ),
    tr(
        "small-q8_0",
        "ggml-small-q8_0.bin",
        "49c8fb02b65e6049d5fa6c04f81f53b867b5ec9540406812c643f177317f779f",
        264464607,
    ),
    tr(
        "small.en",
        "ggml-small.en.bin",
        "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d",
        487614201,
    ),
    tr(
        "small.en-q5_1",
        "ggml-small.en-q5_1.bin",
        "bfdff4894dcb76bbf647d56263ea2a96645423f1669176f4844a1bf8e478ad30",
        190098681,
    ),
    tr(
        "small.en-q8_0",
        "ggml-small.en-q8_0.bin",
        "67a179f608ea6114bd3fdb9060e762b588a3fb3bd00c4387971be4d177958067",
        264477561,
    ),
    tr(
        "medium",
        "ggml-medium.bin",
        "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208",
        1533763059,
    ),
    tr(
        "medium-q5_0",
        "ggml-medium-q5_0.bin",
        "19fea4b380c3a618ec4723c3eef2eb785ffba0d0538cf43f8f235e7b3b34220f",
        539212467,
    ),
    tr(
        "medium-q8_0",
        "ggml-medium-q8_0.bin",
        "42a1ffcbe4167d224232443396968db4d02d4e8e87e213d3ee2e03095dea6502",
        823369779,
    ),
    tr(
        "medium.en",
        "ggml-medium.en.bin",
        "cc37e93478338ec7700281a7ac30a10128929eb8f427dda2e865faa8f6da4356",
        1533774781,
    ),
    tr(
        "medium.en-q5_0",
        "ggml-medium.en-q5_0.bin",
        "76733e26ad8fe1c7a5bf7531a9d41917b2adc0f20f2e4f5531688a8c6cd88eb0",
        539225533,
    ),
    tr(
        "medium.en-q8_0",
        "ggml-medium.en-q8_0.bin",
        "43fa2cd084de5a04399a896a9a7a786064e221365c01700cea4666005218f11c",
        823382461,
    ),
    tr(
        "large-v3",
        "ggml-large-v3.bin",
        "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2",
        3095033483,
    ),
    tr(
        "large-v3-q5_0",
        "ggml-large-v3-q5_0.bin",
        "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1",
        1081140203,
    ),
    tr(
        "large-v3-turbo",
        "ggml-large-v3-turbo.bin",
        "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
        1624555275,
    ),
    tr(
        "large-v3-turbo-q5_0",
        "ggml-large-v3-turbo-q5_0.bin",
        "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
        574041195,
    ),
    tr(
        "large-v3-turbo-q8_0",
        "ggml-large-v3-turbo-q8_0.bin",
        "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1",
        874188075,
    ),
    CatalogEntry {
        id: "silero-v5.1.2",
        file: "ggml-silero-v5.1.2.bin",
        sha256: "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf",
        size: 885_098,
        kind: ModelKind::Vad,
    },
];

pub fn catalog_entry(id: &str) -> Option<&'static CatalogEntry> {
    CATALOG.iter().find(|e| e.id == id)
}

/// The unquantized model a catalog id derives from: `large-v3-turbo-q5_0` gives `large-v3-turbo`.
pub fn base_model_id(id: &str) -> &str {
    ["-q5_0", "-q5_1", "-q8_0"]
        .iter()
        .find_map(|suffix| id.strip_suffix(suffix))
        .unwrap_or(id)
}

#[derive(Debug, PartialEq, Eq)]
pub enum DeleteOutcome {
    Removed,
    NotInstalled,
    /// Present only in a read-only extra directory; nothing was touched.
    ExtraDirOnly(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelError {
    #[error("The model `{0}` does not exist")]
    Unknown(String),
    #[error("The model `{0}` is not installed")]
    NotInstalled(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    pub created: u64,
}

#[derive(Debug, Clone)]
pub struct Registry {
    dirs: Vec<PathBuf>,
    aliases: BTreeMap<String, String>,
}

pub const OPENAI_ALIAS: &str = "whisper-1";

impl Registry {
    pub fn new(dirs: Vec<PathBuf>, aliases: BTreeMap<String, String>) -> Registry {
        Registry { dirs, aliases }
    }

    pub fn from_config(cfg: &Config) -> Registry {
        let mut dirs = vec![expand_tilde(&cfg.models.dir)];
        dirs.extend(cfg.models.extra_dirs.iter().map(|d| expand_tilde(d)));
        let mut aliases =
            BTreeMap::from([(OPENAI_ALIAS.to_string(), cfg.engine.default_model.clone())]);
        aliases.extend(cfg.aliases.clone());
        Registry::new(dirs, aliases)
    }

    pub fn aliases(&self) -> &BTreeMap<String, String> {
        &self.aliases
    }

    pub fn models_dir(&self) -> &std::path::Path {
        self.dirs.first().map_or(std::path::Path::new("."), |d| d)
    }

    /// Catalog ids only (no aliases); includes the VAD model. Used by pull and delete.
    pub fn lookup_catalog(&self, id: &str) -> Result<&'static CatalogEntry, ModelError> {
        catalog_entry(id).ok_or_else(|| ModelError::Unknown(id.to_string()))
    }

    /// Deletes the model file from `models.dir` only; extra directories are never touched.
    pub fn delete_from_models_dir(&self, entry: &CatalogEntry) -> std::io::Result<DeleteOutcome> {
        let own = self.models_dir().join(entry.file);
        if own.is_file() {
            std::fs::remove_file(&own)?;
            return Ok(DeleteOutcome::Removed);
        }
        Ok(match self.locate(entry) {
            Some(p) => DeleteOutcome::ExtraDirOnly(p),
            None => DeleteOutcome::NotInstalled,
        })
    }

    /// Alias table first, then catalog id. A model id is never used as a path.
    pub fn resolve(&self, id: &str) -> Result<&'static CatalogEntry, ModelError> {
        let target = self.aliases.get(id).map(String::as_str).unwrap_or(id);
        match catalog_entry(target) {
            Some(e) if e.kind == ModelKind::Transcription => Ok(e),
            _ => Err(ModelError::Unknown(id.to_string())),
        }
    }

    /// Alias table first, then any catalog id (the VAD model included); what a pull accepts.
    pub fn resolve_pullable(&self, id: &str) -> Result<&'static CatalogEntry, ModelError> {
        let target = self.aliases.get(id).map(String::as_str).unwrap_or(id);
        catalog_entry(target).ok_or_else(|| ModelError::Unknown(id.to_string()))
    }

    pub fn locate(&self, entry: &CatalogEntry) -> Option<PathBuf> {
        self.dirs
            .iter()
            .map(|d| d.join(entry.file))
            .find(|p| p.is_file())
    }

    /// Finds a plain file name (no separators) in the model directories; an absolute path must exist.
    pub fn locate_file(&self, name: &str) -> Option<PathBuf> {
        let p = std::path::Path::new(name);
        if p.is_absolute() {
            return p.is_file().then(|| p.to_path_buf());
        }
        if name.is_empty() || name.contains('/') || name.contains("..") {
            return None;
        }
        self.dirs.iter().map(|d| d.join(name)).find(|p| p.is_file())
    }

    /// Resolves and requires the file to be present.
    pub fn resolve_installed(
        &self,
        id: &str,
    ) -> Result<(&'static CatalogEntry, PathBuf), ModelError> {
        let entry = self.resolve(id)?;
        let path = self
            .locate(entry)
            .ok_or_else(|| ModelError::NotInstalled(id.to_string()))?;
        Ok((entry, path))
    }

    fn created(path: &std::path::Path) -> u64 {
        std::fs::metadata(path)
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    pub fn model_info(&self, id: &str) -> Result<ModelInfo, ModelError> {
        let (_, path) = self.resolve_installed(id)?;
        Ok(ModelInfo {
            id: id.to_string(),
            created: Self::created(&path),
        })
    }

    /// Installed transcription models, then aliases whose target is installed.
    pub fn list(&self) -> Vec<ModelInfo> {
        let mut out: Vec<ModelInfo> = CATALOG
            .iter()
            .filter(|e| e.kind == ModelKind::Transcription)
            .filter_map(|e| self.locate(e).map(|p| (e, p)))
            .map(|(e, p)| ModelInfo {
                id: e.id.to_string(),
                created: Self::created(&p),
            })
            .collect();
        for alias in self.aliases.keys() {
            if let Ok(info) = self.model_info(alias)
                && !out.iter().any(|m| m.id == info.id)
            {
                out.push(info);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_with(files: &[&str], aliases: &[(&str, &str)]) -> (Registry, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "hark-models-test-{}-{}",
            std::process::id(),
            files.join("_")
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for f in files {
            std::fs::write(dir.join(f), b"x").unwrap();
        }
        let aliases = aliases
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        (Registry::new(vec![dir.clone()], aliases), dir)
    }

    #[test]
    fn catalog_hashes_are_sha256_hex() {
        for e in CATALOG {
            assert_eq!(e.sha256.len(), 64, "{}", e.id);
            assert!(e.sha256.chars().all(|c| c.is_ascii_hexdigit()), "{}", e.id);
        }
        assert_eq!(
            catalog_entry("large-v3-turbo").unwrap().url(),
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
        );
    }

    #[test]
    fn catalog_ids_files_and_sizes_are_consistent() {
        let mut ids = std::collections::BTreeSet::new();
        for e in CATALOG {
            assert!(ids.insert(e.id), "duplicate id {}", e.id);
            assert_eq!(e.file, format!("ggml-{}.bin", e.id), "{}", e.id);
            assert!(e.size > 100_000, "{}", e.id);
            assert!(
                !e.id.contains(['/', '\\', '\0']) && !e.id.contains(".."),
                "{}",
                e.id
            );
        }
        for want in [
            "tiny",
            "tiny.en",
            "large-v3-turbo-q5_0",
            "large-v3-turbo-q8_0",
            "medium.en-q5_0",
            "small-q5_1",
        ] {
            assert!(catalog_entry(want).is_some(), "{want}");
        }
        assert_eq!(
            catalog_entry("silero-v5.1.2").unwrap().url(),
            "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin"
        );
    }

    #[test]
    fn quantized_ids_map_to_their_base_model() {
        assert_eq!(base_model_id("large-v3-turbo-q5_0"), "large-v3-turbo");
        assert_eq!(base_model_id("medium.en-q8_0"), "medium.en");
        assert_eq!(base_model_id("tiny-q5_1"), "tiny");
        assert_eq!(base_model_id("large-v3"), "large-v3");
        for e in CATALOG {
            assert!(
                e.kind == ModelKind::Vad || catalog_entry(base_model_id(e.id)).is_some(),
                "{} has no base model in the catalog",
                e.id
            );
        }
    }

    #[test]
    fn delete_touches_models_dir_only() {
        let own = std::env::temp_dir().join(format!("hark-del-own-{}", std::process::id()));
        let extra = std::env::temp_dir().join(format!("hark-del-extra-{}", std::process::id()));
        std::fs::create_dir_all(&own).unwrap();
        std::fs::create_dir_all(&extra).unwrap();
        std::fs::write(own.join("ggml-tiny.bin"), b"x").unwrap();
        std::fs::write(extra.join("ggml-base.bin"), b"x").unwrap();
        std::fs::write(extra.join("ggml-small.bin"), b"x").unwrap();
        std::fs::write(own.join("ggml-small.bin"), b"x").unwrap();
        let r = Registry::new(vec![own.clone(), extra.clone()], BTreeMap::new());
        let tiny = catalog_entry("tiny").unwrap();
        assert_eq!(
            r.delete_from_models_dir(tiny).unwrap(),
            DeleteOutcome::Removed
        );
        assert!(!own.join("ggml-tiny.bin").exists());
        assert_eq!(
            r.delete_from_models_dir(tiny).unwrap(),
            DeleteOutcome::NotInstalled
        );
        let base = catalog_entry("base").unwrap();
        assert_eq!(
            r.delete_from_models_dir(base).unwrap(),
            DeleteOutcome::ExtraDirOnly(extra.join("ggml-base.bin"))
        );
        assert!(extra.join("ggml-base.bin").exists(), "extra dir untouched");
        let small = catalog_entry("small").unwrap();
        assert_eq!(
            r.delete_from_models_dir(small).unwrap(),
            DeleteOutcome::Removed
        );
        assert!(extra.join("ggml-small.bin").exists(), "extra copy kept");
        assert!(
            r.lookup_catalog("whisper-1").is_err(),
            "no aliases for admin ops"
        );
        assert!(r.lookup_catalog("../x").is_err());
        std::fs::remove_dir_all(own).ok();
        std::fs::remove_dir_all(extra).ok();
    }

    #[test]
    fn alias_then_catalog() {
        let (r, _) = registry_with(&[], &[("whisper-1", "large-v3-turbo"), ("fast", "base")]);
        assert_eq!(r.resolve("whisper-1").unwrap().id, "large-v3-turbo");
        assert_eq!(r.resolve("fast").unwrap().id, "base");
        assert_eq!(
            r.resolve_pullable("whisper-1").unwrap().id,
            "large-v3-turbo"
        );
        assert_eq!(
            r.resolve_pullable("silero-v5.1.2").unwrap().file,
            "ggml-silero-v5.1.2.bin"
        );
        assert!(r.resolve_pullable("../base").is_err());
        assert_eq!(r.resolve("small").unwrap().id, "small");
    }

    #[test]
    fn unknown_and_pathish_ids_rejected() {
        let (r, _) = registry_with(&[], &[("evil", "../../etc/passwd")]);
        for id in [
            "nope",
            "",
            "../ggml-base.bin",
            "/etc/passwd",
            "base/../base",
            "ggml-base.bin",
            "BASE",
            "base\0",
            "evil",
        ] {
            assert!(
                matches!(r.resolve(id), Err(ModelError::Unknown(_))),
                "{id:?} resolved"
            );
        }
    }

    #[test]
    fn vad_model_is_not_a_transcription_model() {
        let (r, _) = registry_with(&[], &[]);
        assert!(r.resolve("silero-v5.1.2").is_err());
    }

    #[test]
    fn installed_detection_and_listing() {
        let (r, dir) = registry_with(
            &["ggml-base.bin", "ggml-silero-v5.1.2.bin"],
            &[("whisper-1", "base"), ("gone", "large-v3")],
        );
        assert!(r.resolve_installed("base").is_ok());
        assert_eq!(
            r.resolve_installed("small").unwrap_err(),
            ModelError::NotInstalled("small".into())
        );
        let ids: Vec<_> = r.list().into_iter().map(|m| m.id).collect();
        assert_eq!(ids, vec!["base", "whisper-1"]);
        let base = r.model_info("base").unwrap();
        assert_eq!(r.model_info("whisper-1").unwrap().created, base.created);
        assert!(base.created > 0);
        std::fs::remove_dir_all(dir).ok();
    }
}
