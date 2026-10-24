//! What the cache holds of an artifact, read without fetching or writing.

use std::fs;
use std::path::Path;

use super::{Zoo, verify};
use crate::error::{Check, ZooError};
use crate::index::{Artifact, File};

/// An artifact's files in the cache, as [`Zoo::cache_status`] finds them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheStatus {
    /// Every file is there and verifies by size, SHA-256 and SHA-512.
    Cached,
    /// This file, the first in the index's order, is not there.
    Missing { file: String },
    /// This file is there and does not verify; [`Zoo::fetch`] would fetch
    /// it again.
    Corrupt { file: String, reason: ZooError },
}

/// One cached file: verified, absent, or there with other bytes.
pub(super) enum Local {
    Verified,
    Missing,
    Mismatch(ZooError),
}

/// The file at `local`, checked against `file`. A symlink, or anything
/// but a plain file, is an error: the zoo writes only plain files.
pub(super) fn local_file(artifact: &Artifact, file: &File, local: &Path) -> Result<Local, ZooError> {
    let io = |message: String| ZooError::Io { path: local.to_path_buf(), message };
    match fs::symlink_metadata(local) {
        Ok(m) if m.file_type().is_symlink() => Err(io("is a symlink; the zoo writes only plain files".into())),
        Ok(m) if !m.is_file() => Err(io("is not a plain file".into())),
        Ok(m) if m.len() != file.size => Ok(Local::Mismatch(ZooError::Integrity {
            id: artifact.id.clone(),
            path: file.path.to_string(),
            check: Check::Size,
            expected: file.size.to_string(),
            actual: m.len().to_string(),
        })),
        Ok(_) => {
            // Another process may remove the file between the stat and
            // the read; that is a missing file, not an error.
            let bytes = match fs::read(local) {
                Ok(bytes) => bytes,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Local::Missing),
                Err(e) => return Err(io(e.to_string())),
            };
            Ok(verify(artifact, file, &bytes).map_or_else(Local::Mismatch, |()| Local::Verified))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Local::Missing),
        Err(e) => Err(io(e.to_string())),
    }
}

/// Whether `<cache>/<id>` and the directories of `path` under it are all
/// there, each a plain directory. A symlink or a non-directory on the way
/// is an error, as in `make_dirs`; nothing is made.
fn dirs_exist(cache: &Path, id: &str, path: &str) -> Result<bool, ZooError> {
    let mut dir = cache.to_path_buf();
    if !dir.is_dir() {
        return Ok(false);
    }
    let mut parts: Vec<&str> = path.split('/').collect();
    parts.pop();
    for part in std::iter::once(id).chain(parts) {
        dir.push(part);
        match fs::symlink_metadata(&dir) {
            Ok(m) if m.file_type().is_symlink() => {
                return Err(ZooError::Io {
                    path: dir,
                    message: "is a symlink; the zoo makes only plain directories in its cache".into(),
                });
            }
            Ok(m) if m.is_dir() => {}
            Ok(_) => return Err(ZooError::Io { path: dir, message: "is not a directory".into() }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(ZooError::Io { path: dir, message: e.to_string() }),
        }
    }
    Ok(true)
}

impl Zoo<'_> {
    /// Whether the artifact `id` is in the cache, every file read and
    /// verified, with no fetch and no write. No licence is checked, and no
    /// path is handed out: [`Zoo::fetch`] hands out the files once the
    /// user may have them. A refused or unknown id is an error, as for
    /// [`Zoo::fetch`].
    pub fn cache_status(&self, id: &str) -> Result<CacheStatus, ZooError> {
        let a = self.index.get(id)?;
        for f in &a.files {
            let file = || f.path.to_string();
            if !dirs_exist(&self.cache, a.id.as_str(), f.path.as_str())? {
                return Ok(CacheStatus::Missing { file: file() });
            }
            match local_file(a, f, &self.cache.join(a.id.as_str()).join(f.path.as_str()))? {
                Local::Verified => {}
                Local::Missing => return Ok(CacheStatus::Missing { file: file() }),
                Local::Mismatch(reason) => return Ok(CacheStatus::Corrupt { file: file(), reason }),
            }
        }
        Ok(CacheStatus::Cached)
    }
}
