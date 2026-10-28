//! OS seams: private file modes, advisory locks, directory fsync, free space, heap trim.

use std::fs::{DirBuilder, File, OpenOptions};
use std::io;
use std::path::Path;

/// Owner-only (0600) on unix; the default elsewhere.
pub fn private_file(opts: &mut OpenOptions) -> &mut OpenOptions {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts
}

/// A directory builder that creates owner-only (0700) directories on unix.
pub fn private_dir_builder() -> DirBuilder {
    #[allow(unused_mut)]
    let mut b = DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        b.mode(0o700);
    }
    b
}

/// Takes an exclusive non-blocking lock on `file`; `Ok(false)` when another holder has it.
pub fn try_lock_exclusive(file: &File) -> io::Result<bool> {
    match file.try_lock() {
        Ok(()) => Ok(true),
        Err(std::fs::TryLockError::WouldBlock) => Ok(false),
        Err(std::fs::TryLockError::Error(e)) => Err(e),
    }
}

/// Flushes a directory entry change to disk; Windows cannot open a directory as a file, so it is a no-op there.
pub fn sync_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        File::open(dir)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        Ok(())
    }
}

/// Free bytes for unprivileged writes on the filesystem holding `path`.
#[cfg(unix)]
pub fn free_bytes(path: &Path) -> io::Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::other("path contains NUL"))?;
    // SAFETY: statvfs fills the zeroed struct for a valid NUL-terminated path.
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((st.f_bavail as u64).saturating_mul(st.f_frsize as u64))
}

#[cfg(windows)]
pub fn free_bytes(path: &Path) -> io::Result<u64> {
    fs4::available_space(path)
}

/// Sets glibc's arena cap; a no-op off glibc.
pub fn limit_malloc_arenas() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: mallopt runs before any other thread exists.
    unsafe {
        libc::mallopt(libc::M_ARENA_MAX, 2)
    };
}

/// Returns freed glibc heap to the OS; a no-op off glibc.
pub fn trim_heap() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: malloc_trim is thread-safe and takes no pointers.
    unsafe {
        libc::malloc_trim(0)
    };
}

/// Fills `buf` from the OS RNG: `/dev/urandom` on unix, `getrandom` on Windows.
pub fn fill_random(buf: &mut [u8]) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::io::Read;
        File::open("/dev/urandom")?.read_exact(buf)
    }
    #[cfg(windows)]
    {
        getrandom::getrandom(buf).map_err(|e| io::Error::other(e.to_string()))
    }
}
