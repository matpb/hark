//! Optional bearer key (`server.api_key_file`): file checks, constant-time comparison.

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::config::expand_tilde;

/// SHA-256 of the key, so the comparison never depends on the length or content of either side.
#[derive(Clone)]
pub struct ApiKey {
    digest: [u8; 32],
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ApiKey(..)")
    }
}

fn digest(s: &str) -> [u8; 32] {
    Sha256::digest(s.as_bytes()).into()
}

/// Compares every byte regardless of where the first difference sits.
fn ct_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

impl ApiKey {
    pub fn new(key: &str) -> ApiKey {
        ApiKey {
            digest: digest(key),
        }
    }

    /// A key nothing can match; used when the key file cannot be read, so the server fails closed.
    pub fn unusable() -> ApiKey {
        ApiKey { digest: [0; 32] }
    }

    pub fn matches(&self, presented: &str) -> bool {
        ct_eq(&self.digest, &digest(presented))
    }
}

/// The token of an `Authorization: Bearer <token>` header value.
pub fn bearer_token(header: &str) -> Option<&str> {
    let (scheme, token) = header.split_once(' ')?;
    (scheme.eq_ignore_ascii_case("bearer") && !token.trim().is_empty()).then(|| token.trim())
}

/// Verdict on a key file's metadata: a regular file, owned by the running user, with no group or other access.
#[cfg(unix)]
pub fn key_file_problem(
    path: &Path,
    is_file: bool,
    owner_uid: u32,
    my_uid: u32,
    mode: u32,
) -> Option<String> {
    if !is_file {
        return Some(format!(
            "server.api_key_file {} is not a regular file",
            path.display()
        ));
    }
    if owner_uid != my_uid {
        return Some(format!(
            "server.api_key_file {} is owned by uid {owner_uid}, not by the user running hark (uid {my_uid})",
            path.display()
        ));
    }
    let mode = mode & 0o777;
    if mode & 0o177 != 0 {
        return Some(format!(
            "server.api_key_file {} has mode {mode:04o}; it must be 0600 (chmod 600 {})",
            path.display(),
            path.display()
        ));
    }
    None
}

/// Opens the key file once without following a final symlink, checks the open file's type, owner
/// and mode, and returns it for reading; nothing is looked up by path twice.
#[cfg(unix)]
fn open_checked(path: &Path) -> Result<std::fs::File, String> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|e| {
            if e.raw_os_error() == Some(libc::ELOOP) {
                format!(
                    "server.api_key_file {} is a symbolic link; point it at the key file itself",
                    path.display()
                )
            } else {
                format!("server.api_key_file {}: {e}", path.display())
            }
        })?;
    let meta = file
        .metadata()
        .map_err(|e| format!("server.api_key_file {}: {e}", path.display()))?;
    // SAFETY: geteuid has no preconditions.
    let uid = unsafe { libc::geteuid() };
    match key_file_problem(
        path,
        meta.is_file(),
        meta.uid(),
        uid,
        meta.permissions().mode(),
    ) {
        Some(e) => Err(e),
        None => Ok(file),
    }
}

/// Windows: refuses a symlink and a non-regular file; owner and mode checks are skipped (ACLs apply).
#[cfg(windows)]
fn open_checked(path: &Path) -> Result<std::fs::File, String> {
    let link = std::fs::symlink_metadata(path)
        .map_err(|e| format!("server.api_key_file {}: {e}", path.display()))?;
    if link.file_type().is_symlink() {
        return Err(format!(
            "server.api_key_file {} is a symbolic link; point it at the key file itself",
            path.display()
        ));
    }
    let file = std::fs::File::open(path)
        .map_err(|e| format!("server.api_key_file {}: {e}", path.display()))?;
    let is_file = file.metadata().is_ok_and(|m| m.is_file());
    if !is_file {
        return Err(format!(
            "server.api_key_file {} is not a regular file",
            path.display()
        ));
    }
    Ok(file)
}

/// A regular file, owned by the running user, with no group or other access (0600 or 0400).
pub fn check_key_file(path: &Path) -> Result<(), String> {
    open_checked(path).map(drop)
}

/// `Ok(None)` when no key file is configured.
pub fn load_key(api_key_file: &str) -> Result<Option<ApiKey>, String> {
    if api_key_file.is_empty() {
        return Ok(None);
    }
    let path = expand_tilde(api_key_file);
    let file = open_checked(&path)?;
    let mut text = String::new();
    file.take(64 * 1024)
        .read_to_string(&mut text)
        .map_err(|e| format!("server.api_key_file {}: {e}", path.display()))?;
    let key = text.trim();
    if key.is_empty() {
        return Err(format!("server.api_key_file {} is empty", path.display()));
    }
    Ok(Some(ApiKey::new(key)))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn file_with_mode(tag: &str, mode: u32, content: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("hark-key-{}-{tag}", std::process::id()));
        std::fs::write(&p, content).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
        p
    }

    #[test]
    fn matching_is_exact_and_length_blind() {
        let k = ApiKey::new("s3cret-key");
        assert!(k.matches("s3cret-key"));
        for bad in ["", "s3cret-ke", "s3cret-key ", "S3CRET-KEY", "s3cret-keyx"] {
            assert!(!k.matches(bad), "{bad:?}");
        }
        assert!(!ApiKey::unusable().matches(""));
        assert!(!ApiKey::unusable().matches("x"));
    }

    #[test]
    fn bearer_header_parsing() {
        assert_eq!(bearer_token("Bearer abc"), Some("abc"));
        assert_eq!(bearer_token("bearer abc "), Some("abc"));
        assert_eq!(bearer_token("Basic abc"), None);
        assert_eq!(bearer_token("Bearer"), None);
        assert_eq!(bearer_token("Bearer  "), None);
        assert_eq!(bearer_token(""), None);
    }

    #[test]
    fn key_file_must_be_0600_or_stricter() {
        let ok = file_with_mode("ok", 0o600, "abc\n");
        assert!(check_key_file(&ok).is_ok());
        let strict = file_with_mode("ro", 0o400, "abc\n");
        assert!(check_key_file(&strict).is_ok());
        for mode in [0o644, 0o640, 0o604, 0o660, 0o700, 0o777] {
            let p = file_with_mode(&format!("m{mode:o}"), mode, "abc");
            let e = check_key_file(&p).unwrap_err();
            assert!(e.contains("0600"), "{mode:o}: {e}");
            std::fs::remove_file(p).ok();
        }
        std::fs::remove_file(ok).ok();
        std::fs::remove_file(strict).ok();
    }

    #[test]
    fn a_symlinked_key_file_is_refused_even_when_the_target_is_fine() {
        let target = file_with_mode("link-target", 0o600, "abc\n");
        let link = std::env::temp_dir().join(format!("hark-key-{}-link", std::process::id()));
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let e = check_key_file(&link).unwrap_err();
        assert!(e.contains("symbolic link"), "{e}");
        let e = load_key(link.to_str().unwrap()).unwrap_err();
        assert!(e.contains("symbolic link"), "{e}");
        assert!(load_key(target.to_str().unwrap()).unwrap().is_some());
        std::fs::remove_file(link).ok();
        std::fs::remove_file(target).ok();
    }

    #[test]
    fn the_key_is_read_from_the_checked_descriptor() {
        let p = file_with_mode("fd", 0o600, "first-key\n");
        let file = open_checked(&p).unwrap();
        std::fs::remove_file(&p).unwrap();
        std::fs::write(&p, "swapped-key\n").unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o644)).unwrap();
        let mut text = String::new();
        std::io::Read::read_to_string(&mut &file, &mut text).unwrap();
        assert_eq!(text.trim(), "first-key");
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn key_file_verdicts_are_pure() {
        let p = Path::new("/k");
        assert!(key_file_problem(p, true, 1000, 1000, 0o100600).is_none());
        assert!(key_file_problem(p, true, 1000, 1000, 0o400).is_none());
        assert!(
            key_file_problem(p, false, 1000, 1000, 0o600)
                .unwrap()
                .contains("regular")
        );
        assert!(
            key_file_problem(p, true, 0, 1000, 0o600)
                .unwrap()
                .contains("owned by")
        );
        assert!(
            key_file_problem(p, true, 1000, 1000, 0o640)
                .unwrap()
                .contains("0640")
        );
    }

    #[test]
    fn key_file_must_be_a_regular_existing_nonempty_file() {
        assert!(check_key_file(Path::new("/nonexistent/hark.key")).is_err());
        assert!(check_key_file(&std::env::temp_dir()).is_err());
        let empty = file_with_mode("empty", 0o600, "  \n");
        let e = load_key(empty.to_str().unwrap()).unwrap_err();
        assert!(e.contains("empty"), "{e}");
        std::fs::remove_file(empty).ok();
        assert!(load_key("").unwrap().is_none());
        let good = file_with_mode("good", 0o600, "  my key \n");
        let key = load_key(good.to_str().unwrap()).unwrap().unwrap();
        assert!(key.matches("my key"));
        std::fs::remove_file(good).ok();
    }
}
