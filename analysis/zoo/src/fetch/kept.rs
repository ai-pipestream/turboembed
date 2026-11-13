//! The cache as the place an import keeps what it reads: a file an
//! importer downloaded is written where a fetch of the entry looks for
//! it, so the fetch that follows, or the same import run again, does not
//! download it a second time.

use std::fs;
use std::path::{Path, PathBuf};

use super::status::dirs_exist;
use super::{denied_again, make_dirs, write_atomic};
use crate::digest::Sha256;
use crate::error::ZooError;

/// One artifact's directory in a cache, `<cache>/<id>`, read and written
/// under the rules a fetch keeps: plain files in plain directories.
pub(crate) struct Kept<'a> {
    cache: &'a Path,
    id: &'a str,
}

impl<'a> Kept<'a> {
    pub(crate) fn new(cache: &'a Path, id: &'a str) -> Kept<'a> {
        Kept { cache, id }
    }

    fn local(&self, path: &str) -> PathBuf {
        self.cache.join(self.id).join(path)
    }

    /// The bytes of `path` when the cache holds it as a plain file of
    /// this size and SHA-256. A file that is absent or holds other bytes
    /// is none: the importer reads it from its source. A symlink, or
    /// anything but a plain file, is an error.
    pub(crate) fn held(&self, path: &str, size: u64, sha256: &Sha256) -> Result<Option<Vec<u8>>, ZooError> {
        if !dirs_exist(self.cache, self.id, path)? {
            return Ok(None);
        }
        let local = self.local(path);
        let io = |message: String| ZooError::Io { path: local.clone(), message };
        match denied_again(|| fs::symlink_metadata(&local)) {
            Ok(m) if m.file_type().is_symlink() => Err(io("is a symlink; the zoo writes only plain files".into())),
            Ok(m) if !m.is_file() => Err(io("is not a plain file".into())),
            Ok(m) if m.len() != size => Ok(None),
            Ok(_) => match denied_again(|| fs::read(&local)) {
                Ok(bytes) => Ok(Some(bytes).filter(|b| Sha256::of(b) == *sha256)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(io(e.to_string())),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(io(e.to_string())),
        }
    }

    /// Writes `bytes` as `path`, whole or not at all.
    pub(crate) fn keep(&self, path: &str, bytes: &[u8]) -> Result<(), ZooError> {
        make_dirs(self.cache, self.id, path)?;
        write_atomic(&self.local(path), bytes)
    }
}
