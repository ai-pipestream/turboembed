//! Places a model loads from besides its address: a content-addressed
//! store on disk, and a list of sources tried in order (#1613).
//!
//! A content-addressed store keeps each file, unpacked, at
//! `<root>/blobs/sha256/<hex>`: the layout a zoo server's store, a
//! library's local copy and an S3 bucket share, so any one can seed
//! another. A key always names the bytes the index pins.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::transport::Transport;
use super::{denied_again, unpack, verify, write_atomic};
use crate::digest::Sha256;
use crate::error::ZooError;
use crate::index::{Artifact, File, Url};

/// A content-addressed store on disk. As a transport it serves a file by
/// its SHA-256 alone and fetches nothing by address.
#[derive(Debug, Clone)]
pub struct Blobs {
    root: PathBuf,
}

impl Blobs {
    /// The store under `root`; nothing is read or made until it is used.
    pub fn new(root: impl Into<PathBuf>) -> Blobs {
        Blobs { root: root.into() }
    }

    /// Where the file with SHA-256 `sha256` is kept.
    pub fn path(&self, sha256: &Sha256) -> PathBuf {
        self.root.join("blobs").join("sha256").join(sha256.to_string())
    }

    /// Keeps `bytes` under their own SHA-256, written whole through a
    /// temporary name, and returns that SHA-256. A file already kept there
    /// is written again unless it holds those bytes. The store's
    /// directories are made one at a time, and a symlink or anything but a
    /// directory on the way is refused, so nothing is written outside the
    /// store; only `root` itself may be a symlink.
    pub fn put(&self, bytes: &[u8]) -> Result<Sha256, ZooError> {
        let sha256 = Sha256::of(bytes);
        let dir = self.dir(true).map_err(|(path, e)| ZooError::Io { path, message: e.to_string() })?;
        let dest = dir.join(sha256.to_string());
        let kept = read_plain(&dest, bytes.len() as u64).is_ok_and(|held| held == bytes);
        if !kept {
            write_atomic(&dest, bytes)?;
        }
        Ok(sha256)
    }

    /// The bytes kept for `file`, at most one more than its size, so the
    /// zoo's size check fails on a longer one. A symlink, or anything but
    /// a plain file or directory, on the way or as the file, is refused.
    fn read(&self, file: &File) -> Result<Vec<u8>, String> {
        let dir = self.dir(false).map_err(|(p, e)| format!("{}: {e}", p.display()))?;
        let p = dir.join(file.sha256.to_string());
        read_plain(&p, file.size).map_err(|e| format!("{}: {e}", p.display()))
    }

    /// `<root>/blobs/sha256`, each directory below `root` checked as a
    /// plain directory, and with `make`, made when absent.
    fn dir(&self, make: bool) -> Result<PathBuf, (PathBuf, std::io::Error)> {
        let mut dir = self.root.clone();
        if make {
            fs::create_dir_all(&dir).map_err(|e| (dir.clone(), e))?;
        }
        for part in ["blobs", "sha256"] {
            dir.push(part);
            match denied_again(|| fs::symlink_metadata(&dir)) {
                Ok(m) if m.file_type().is_symlink() => {
                    return Err((dir, std::io::Error::other("is a symlink; the store holds only plain directories")));
                }
                Ok(m) if m.is_dir() => {}
                Ok(_) => return Err((dir, std::io::Error::other("is not a directory"))),
                Err(e) if make && e.kind() == std::io::ErrorKind::NotFound => match fs::create_dir(&dir) {
                    Ok(()) => {}
                    // Made by another writer since: checked again.
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        let m = fs::symlink_metadata(&dir).map_err(|e| (dir.clone(), e))?;
                        if !m.is_dir() || m.file_type().is_symlink() {
                            return Err((dir, std::io::Error::other("is not a plain directory")));
                        }
                    }
                    Err(e) => return Err((dir, e)),
                },
                Err(e) => return Err((dir, e)),
            }
        }
        Ok(dir)
    }
}

fn read_plain(p: &Path, size: u64) -> std::io::Result<Vec<u8>> {
    let m = denied_again(|| fs::symlink_metadata(p))?;
    if !m.is_file() {
        return Err(std::io::Error::other("not a plain file"));
    }
    let mut out = Vec::new();
    fs::File::open(p)?.take(size.saturating_add(1)).read_to_end(&mut out)?;
    Ok(out)
}

impl Transport for Blobs {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        Err(format!("{url}: a content-addressed store serves a file by its SHA-256, not its address"))
    }

    fn get_file(&self, _artifact: &Artifact, file: &File) -> Result<Vec<u8>, String> {
        self.read(file)
    }
}

/// A local directory transport that serves files from `<root>/<id>/<path>`
/// (or `<root>/<path>` when `<root>` is the artifact directory itself),
/// checking that every component is a plain file.
#[derive(Debug, Clone)]
pub struct Directory {
    root: PathBuf,
}

impl Directory {
    /// The directory under `root`; nothing is read until a file is requested.
    pub fn new(root: impl Into<PathBuf>) -> Directory {
        Directory { root: root.into() }
    }

    /// The root directory this transport reads from.
    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl Transport for Directory {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        Err(format!("{url}: a directory transport serves a file by its artifact path, not its URL"))
    }

    fn get_file(&self, artifact: &Artifact, file: &File) -> Result<Vec<u8>, String> {
        let by_id = self.root.join(artifact.id.as_str()).join(file.path.as_str());
        let p = if by_id.exists() { by_id } else { self.root.join(file.path.as_str()) };
        read_plain(&p, file.size).map_err(|e| format!("{}: {e}", p.display()))
    }
}

/// A store a [`Sources`] list fills as it reads ([`Sources::then_keep`]).
pub trait Keep {
    /// Keeps `bytes` under their own SHA-256.
    fn keep(&self, bytes: &[u8]) -> Result<(), String>;
}

impl<K: Keep + ?Sized> Keep for std::sync::Arc<K> {
    fn keep(&self, bytes: &[u8]) -> Result<(), String> {
        (**self).keep(bytes)
    }
}

impl Keep for Blobs {
    fn keep(&self, bytes: &[u8]) -> Result<(), String> {
        self.put(bytes).map(|_| ()).map_err(|e| e.to_string())
    }
}

/// Sources tried in order for each file: the first whose bytes verify as
/// the file is kept, so a source that lacks the file, fails, or serves
/// other bytes passes the request to the next. When none has it, the error
/// names what each one said.
pub struct Sources {
    sources: Vec<(String, Box<dyn Transport + Send + Sync>)>,
    /// The store [`Sources::then_keep`] added, and its place in the list.
    keep: Option<(usize, Box<dyn Keep + Send + Sync>)>,
}

impl Sources {
    /// No source yet; [`Sources::then`] adds them in the order they are
    /// tried.
    pub fn new() -> Sources {
        Sources { sources: Vec::new(), keep: None }
    }

    /// `source`, named `name` in errors, tried after those added before.
    pub fn then(mut self, name: impl Into<String>, source: impl Transport + Send + Sync + 'static) -> Sources {
        self.sources.push((name.into(), Box::new(source)));
        self
    }

    /// `store`, tried here in order like any source, which also keeps
    /// every file a later source serves once it verifies: a library's own
    /// copy or a server's bucket ([`Blobs`], `S3Bucket`), in the one
    /// layout they share, filled as it reads.
    /// A file that cannot be kept (a read-only store, a full disk) is
    /// still handed over. Only one store keeps; a second call replaces the
    /// first as the keeper and leaves it in the list as a plain source. The
    /// keeper is passed over for an artifact whose class may not be
    /// redistributed ([`crate::LicenceClass::may_redistribute`]).
    pub fn then_keep<S>(mut self, name: impl Into<String>, store: S) -> Sources
    where
        S: Transport + Keep + Clone + Send + Sync + 'static,
    {
        self.keep = Some((self.sources.len(), Box::new(store.clone())));
        self.then(name, store)
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

impl Default for Sources {
    fn default() -> Sources {
        Sources::new()
    }
}

impl Transport for Sources {
    /// By address alone the bytes cannot be checked here, so the first
    /// source that answers wins; the zoo itself asks by file.
    fn get(&self, url: &Url, size: u64) -> Result<Vec<u8>, String> {
        let mut failed = Vec::new();
        for (name, s) in &self.sources {
            match s.get(url, size) {
                Ok(bytes) => return Ok(bytes),
                Err(e) => failed.push(format!("{name}: {e}")),
            }
        }
        Err(none_had(failed))
    }

    /// The keeping store neither serves nor keeps an artifact whose class
    /// may not be redistributed (`LocalOnly`, `Refused`): such a file comes
    /// only from the sources after it, and no copy is made of it.
    fn get_file(&self, artifact: &Artifact, file: &File) -> Result<Vec<u8>, String> {
        let keep = self.keep.as_ref().filter(|_| artifact.class.may_redistribute());
        let mut failed = Vec::new();
        for (at, (name, s)) in self.sources.iter().enumerate() {
            if keep.is_none() && self.keep.as_ref().is_some_and(|(k, _)| *k == at) {
                continue;
            }
            let bytes = match s.get_file(artifact, file) {
                Ok(b) => b,
                Err(e) => {
                    failed.push(format!("{name}: {e}"));
                    continue;
                }
            };
            match unpack(artifact, file, bytes).and_then(|b| verify(artifact, file, &b).map(|()| b)) {
                Ok(b) => {
                    if let Some((_, store)) = keep.filter(|(k, _)| *k != at) {
                        // Kept when it can be; the file is handed over either way.
                        let _ = store.keep(&b);
                    }
                    return Ok(b);
                }
                Err(e) => failed.push(format!("{name}: {e}")),
            }
        }
        Err(none_had(failed))
    }
}

fn none_had(failed: Vec<String>) -> String {
    if failed.is_empty() { "no source is configured".into() } else { failed.join("; ") }
}
