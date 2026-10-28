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
    /// temporary name, and returns that SHA-256. Bytes already kept are
    /// not written again.
    pub fn put(&self, bytes: &[u8]) -> Result<Sha256, ZooError> {
        let sha256 = Sha256::of(bytes);
        let dest = self.path(&sha256);
        let kept = fs::symlink_metadata(&dest).is_ok_and(|m| m.is_file() && m.len() == bytes.len() as u64);
        if !kept {
            write_atomic(&dest, bytes)?;
        }
        Ok(sha256)
    }

    /// The bytes kept for `file`, at most one more than its size, so the
    /// zoo's size check fails on a longer one. A symlink, or anything but
    /// a plain file, is refused.
    fn read(&self, file: &File) -> Result<Vec<u8>, String> {
        let p = self.path(&file.sha256);
        read_plain(&p, file.size).map_err(|e| format!("{}: {e}", p.display()))
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

/// Sources tried in order for each file: the first whose bytes verify as
/// the file is kept, so a source that lacks the file, fails, or serves
/// other bytes passes the request to the next. When none has it, the error
/// names what each one said.
pub struct Sources {
    sources: Vec<(String, Box<dyn Transport + Send + Sync>)>,
}

impl Sources {
    /// No source yet; [`Sources::then`] adds them in the order they are
    /// tried.
    pub fn new() -> Sources {
        Sources { sources: Vec::new() }
    }

    /// `source`, named `name` in errors, tried after those added before.
    pub fn then(mut self, name: impl Into<String>, source: impl Transport + Send + Sync + 'static) -> Sources {
        self.sources.push((name.into(), Box::new(source)));
        self
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

    fn get_file(&self, artifact: &Artifact, file: &File) -> Result<Vec<u8>, String> {
        let mut failed = Vec::new();
        for (name, s) in &self.sources {
            let bytes = match s.get_file(artifact, file) {
                Ok(b) => b,
                Err(e) => {
                    failed.push(format!("{name}: {e}"));
                    continue;
                }
            };
            match unpack(artifact, file, bytes).and_then(|b| verify(artifact, file, &b).map(|()| b)) {
                Ok(b) => return Ok(b),
                Err(e) => failed.push(format!("{name}: {e}")),
            }
        }
        Err(none_had(failed))
    }
}

fn none_had(failed: Vec<String>) -> String {
    if failed.is_empty() { "no source is configured".into() } else { failed.join("; ") }
}
