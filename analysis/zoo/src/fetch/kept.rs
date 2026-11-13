//! The cache as the place an import keeps what it reads: a file an
//! importer downloaded is written where a fetch of the entry looks for
//! it, so the fetch that follows, or the same import run again, does not
//! download it a second time.

use std::fs;
use std::path::{Path, PathBuf};

use super::status::dirs_exist;
use super::{denied_again, make_dirs, put_in_place, write_partial};
use crate::digest::Sha256;
use crate::error::ZooError;

/// One artifact's directory in a cache, `<cache>/<id>`, read and written
/// under the rules a fetch keeps: plain files in plain directories.
///
/// What an import keeps is staged: each file is written whole under a
/// temporary name beside its place, and [`Kept::accept`] puts them all in
/// place once the entry is accepted. An import that fails drops the
/// `Kept`, which removes what it staged and every directory it left
/// empty, so the cache is as the import found it.
pub(crate) struct Kept<'a> {
    cache: &'a Path,
    id: &'a str,
    /// Each staged file: its temporary name and its place.
    staged: Vec<(PathBuf, PathBuf)>,
}

impl<'a> Kept<'a> {
    pub(crate) fn new(cache: &'a Path, id: &'a str) -> Kept<'a> {
        Kept { cache, id, staged: Vec::new() }
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

    /// Stages `bytes` as `path`: written whole beside its place, and put
    /// there by [`Kept::accept`].
    pub(crate) fn keep(&mut self, path: &str, bytes: &[u8]) -> Result<(), ZooError> {
        let local = self.local(path);
        // Recorded first, so a directory made for a file that was never
        // written is removed with the rest.
        self.staged.push((PathBuf::new(), local.clone()));
        make_dirs(self.cache, self.id, path)?;
        let tmp = write_partial(&local, bytes, false)?;
        if let Some(last) = self.staged.last_mut() {
            last.0 = tmp;
        }
        Ok(())
    }

    /// Puts every staged file in its place, each whole. When one cannot
    /// be put there the rest are removed as a failed import's are.
    pub(crate) fn accept(mut self) -> Result<(), ZooError> {
        while let Some((tmp, local)) = self.staged.first() {
            put_in_place(tmp, local)?;
            self.staged.remove(0);
        }
        Ok(())
    }
}

impl Drop for Kept<'_> {
    /// Removes what was staged and not accepted, then each directory
    /// below the cache that this leaves empty. A directory that holds
    /// anything else stays: `remove_dir` takes only an empty one.
    fn drop(&mut self) {
        for (tmp, local) in &self.staged {
            if !tmp.as_os_str().is_empty() {
                let _ = fs::remove_file(tmp);
            }
            let mut dir = local.parent();
            while let Some(d) = dir.filter(|d| d.starts_with(self.cache) && *d != self.cache) {
                if fs::remove_dir(d).is_err() {
                    break;
                }
                dir = d.parent();
            }
        }
    }
}
