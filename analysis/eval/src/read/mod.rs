//! The readers of the sets' own file formats, one per format, and what
//! they share: reading a set's files while hashing them, and parsing JSON
//! and JSON Lines with errors that name the file and line.
//!
//! A reader takes the set's directory and returns a [`LoadedSet`]; it
//! knows nothing of fetching or licences, which are the
//! [`sets`](crate::sets) module's.
//!
//! [`LoadedSet`]: crate::set::LoadedSet

pub mod beir;
pub mod conteb;
pub mod gutenqa;
pub mod hicbench;
pub mod legalbench_rag;
pub mod offsets;
pub mod pyrandom;
pub mod qasper;
pub mod tar;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use turbo_goldens::hex::to_hex;
use turbo_goldens::sha::{Sha256, sha256};

use crate::Result;

/// A set's directory, and the SHA-256 of each file read from it.
pub struct SetDir {
    root: PathBuf,
    digests: BTreeMap<String, [u8; 32]>,
}

impl SetDir {
    pub fn new(root: &Path) -> SetDir {
        SetDir { root: root.to_path_buf(), digests: BTreeMap::new() }
    }

    /// The bytes of `rel`, a `/` separated path under the directory.
    pub fn bytes(&mut self, rel: &str) -> Result<Vec<u8>> {
        let path = self.root.join(rel);
        let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        self.digests.insert(rel.to_owned(), sha256(&bytes));
        Ok(bytes)
    }

    /// The text of `rel`, which must be UTF-8.
    pub fn text(&mut self, rel: &str) -> Result<String> {
        let bytes = self.bytes(rel)?;
        utf8(rel, bytes)
    }

    /// The names of the files directly under `rel` with extension `ext`,
    /// in byte order.
    pub fn list(&self, rel: &str, ext: &str) -> Result<Vec<String>> {
        let path = self.root.join(rel);
        let mut names = Vec::new();
        for e in fs::read_dir(&path).map_err(|e| format!("{}: {e}", path.display()))? {
            let e = e.map_err(|e| format!("{}: {e}", path.display()))?;
            let name = e.file_name().into_string().map_err(|n| format!("{}: {n:?} is not UTF-8", path.display()))?;
            if name.strip_suffix(ext).is_some_and(|stem| stem.ends_with('.')) && e.path().is_file() {
                names.push(name);
            }
        }
        names.sort();
        Ok(names)
    }

    /// The set's digest, as [`LoadedSet::sha256`] defines it.
    ///
    /// [`LoadedSet::sha256`]: crate::set::LoadedSet::sha256
    pub fn sha256(&self) -> String {
        let mut h = Sha256::new();
        for (path, digest) in &self.digests {
            h.update(to_hex(digest).as_bytes());
            h.update(b"  ");
            h.update(path.as_bytes());
            h.update(b"\n");
        }
        to_hex(&h.finish())
    }
}

/// `bytes` as UTF-8; the error names the file and the line of the first
/// bad byte.
pub fn utf8(file: &str, bytes: Vec<u8>) -> Result<String> {
    String::from_utf8(bytes).map_err(|e| {
        let at = e.utf8_error().valid_up_to();
        let line = 1 + e.as_bytes()[..at].iter().filter(|&&b| b == b'\n').count();
        format!("{file}:{line}: not UTF-8")
    })
}

/// One JSON document. Errors carry the file, line and column.
pub fn json<T: DeserializeOwned>(file: &str, text: &str) -> Result<T> {
    serde_json::from_str(text).map_err(|e| format!("{file}:{}:{}: {e}", e.line(), e.column()))
}

/// One JSON value per line, each with its 1 based line number. Blank
/// lines are skipped.
pub fn json_lines<T: DeserializeOwned>(file: &str, text: &str) -> Result<Vec<(usize, T)>> {
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str(line).map_err(|e| format!("{file}:{}: {e}", i + 1))?;
        rows.push((i + 1, row));
    }
    Ok(rows)
}

/// The separator sets published as chunks are joined with: their chunks
/// do not say what lay between them, so a document is its chunks in
/// order with one line break between each two.
pub const CHUNK_SEPARATOR: &str = "\n";

/// `chunks` joined with [`CHUNK_SEPARATOR`], and each chunk's byte range
/// in the result.
pub fn join_chunks<'a>(chunks: impl IntoIterator<Item = &'a str>) -> (String, Vec<(usize, usize)>) {
    let mut text = String::new();
    let mut ranges = Vec::new();
    for (i, c) in chunks.into_iter().enumerate() {
        if i > 0 {
            text.push_str(CHUNK_SEPARATOR);
        }
        let start = text.len();
        text.push_str(c);
        ranges.push((start, text.len()));
    }
    (text, ranges)
}
