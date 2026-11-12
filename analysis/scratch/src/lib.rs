//! A scratch directory for a test, removed when the test is done with it.
//!
//! A test that stages files under the system's temporary directory and
//! leaves them there fills the disk of a machine that keeps running: each
//! test process has its own id in the name, so no later run clears it.
//! `Scratch` owns such a directory and removes it when dropped, which a
//! panicking test does too as it unwinds. A directory the tests of a
//! binary share is kept until the process ends (`until_exit`).

use std::ffi::c_int;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// A directory that is removed, with all it holds, when this is dropped.
/// It reads as the path it owns.
#[derive(Debug)]
pub struct Scratch(PathBuf);

impl Scratch {
    /// Takes over `dir`: what is there now is removed, and whatever the
    /// test puts there is removed when this is dropped. The directory is
    /// not made, so a test may make it or leave that to what it runs.
    pub fn at(dir: impl Into<PathBuf>) -> Scratch {
        let dir = dir.into();
        remove(&dir);
        Scratch(dir)
    }

    /// The directory.
    pub fn path(&self) -> &Path {
        &self.0
    }

    /// Keeps the directory until the process ends, and removes it then:
    /// for a fixture the tests of one binary share from a `static`, whose
    /// value is never dropped. Returns the path.
    pub fn until_exit(self) -> PathBuf {
        let dir = self.0.clone();
        std::mem::forget(self);
        let mut kept = KEPT.lock().unwrap_or_else(|p| p.into_inner());
        if kept.is_empty() {
            // SAFETY: `atexit` is the C library's, takes a function that
            // stays valid for the life of the process, and may be called
            // from any thread.
            let registered = unsafe { atexit(remove_kept) };
            assert_eq!(registered, 0, "atexit refused the cleanup of {}", dir.display());
        }
        kept.push(dir.clone());
        dir
    }
}

/// The directories `until_exit` keeps.
static KEPT: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

unsafe extern "C" {
    fn atexit(callback: extern "C" fn()) -> c_int;
}

/// Removes what `until_exit` kept. It runs as the process exits, where a
/// thread may have died holding the lock, so it takes what it can and
/// never panics.
extern "C" fn remove_kept() {
    let kept = match KEPT.try_lock() {
        Ok(mut kept) => std::mem::take(&mut *kept),
        Err(std::sync::TryLockError::Poisoned(p)) => std::mem::take(&mut *p.into_inner()),
        Err(std::sync::TryLockError::WouldBlock) => return,
    };
    for dir in kept {
        remove(&dir);
    }
}

/// Removes the directory at `path` with all it holds, or the file a test
/// wrote there instead. A failure is left alone: there may be nothing
/// there, and on Windows a file still open elsewhere cannot be removed.
fn remove(path: &Path) {
    if std::fs::remove_dir_all(path).is_err() {
        let _ = std::fs::remove_file(path);
    }
}

impl Deref for Scratch {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for Scratch {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl From<&Scratch> for PathBuf {
    /// The path alone: the directory stays the `Scratch`'s to remove.
    fn from(scratch: &Scratch) -> PathBuf {
        scratch.0.clone()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        remove(&self.0);
    }
}
