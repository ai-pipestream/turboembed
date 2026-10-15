//! The evaluation sets of decision 10: where each is published, under
//! which licence, the unit its span judgements are in, and which reader
//! reads it. One file per set; adding a set is a file here and a line in
//! [`ALL`].
//!
//! A set lives in `<cache>/<name>/`, under a cache root the user names
//! (see [`fetch`](crate::fetch)). [`Set::load`] refuses a set with no
//! licence recorded, and a file whose SHA-256 is recorded and differs.

pub mod arguana;
pub mod conteb_covid_qa;
pub mod conteb_squad;
pub mod gutenqa;
pub mod hicbench;
pub mod legalbench_rag_mini;
pub mod nfcorpus;
pub mod qasper;
pub mod scifact;

use std::path::{Path, PathBuf};

use turbo_goldens::hex::to_hex;
use turbo_goldens::sha::sha256;

use crate::Result;
use crate::set::LoadedSet;

/// Every set, in the order a full run takes them.
pub const ALL: [&Set; 9] = [
    &scifact::SET,
    &nfcorpus::SET,
    &arguana::SET,
    &legalbench_rag_mini::SET,
    &conteb_covid_qa::SET,
    &conteb_squad::SET,
    &gutenqa::SET,
    &hicbench::SET,
    &qasper::SET,
];

/// The set called `name`.
pub fn find(name: &str) -> Result<&'static Set> {
    ALL.into_iter().find(|s| s.name == name).ok_or_else(|| {
        let names: Vec<&str> = ALL.iter().map(|s| s.name).collect();
        format!("no set {name}; the sets are {}", names.join(", "))
    })
}

pub struct Set {
    /// The name records carry, and the set's directory under the cache.
    pub name: &'static str,
    /// The licence the data is published under, and where it is stated.
    /// Empty when none could be found; such a set is not loaded.
    pub license: &'static str,
    pub source: Source,
    /// The unit of the set's own span judgements; `None` for a set with
    /// document judgements only.
    pub span_unit: Option<SpanUnit>,
    /// The reader, given the set's name and directory.
    pub read: fn(&str, &Path) -> Result<LoadedSet>,
}

/// Where a set's files come from.
pub enum Source {
    /// Files at published URLs, fetched into the set's directory.
    Files(&'static [RemoteFile]),
    /// Published only as a shared folder that cannot be fetched file by
    /// file; the user downloads it from `url` and unpacks it into the
    /// set's directory as `layout` says.
    Manual { url: &'static str, layout: &'static str },
    /// No published copy was found; `note` says what is known.
    Unpublished { note: &'static str },
}

pub struct RemoteFile {
    /// Where the file goes under the set's directory, `/` separated.
    pub path: &'static str,
    pub url: &'static str,
    pub sha256: Digest,
}

pub enum Digest {
    /// 64 lower case hex.
    Recorded(&'static str),
    /// The file has not been downloaded and checked yet, so there is no
    /// digest to pin. Fetching it fails and prints the digest of what was
    /// downloaded, to be checked and recorded here.
    NotRecorded,
}

/// The unit a set's own span judgements are given in. Every reader
/// converts them to UTF-8 byte offsets of the document as loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanUnit {
    /// `[start, end)` in Unicode code points (Python `str` indices).
    CodePoints,
    /// The judged text is quoted, without offsets, and located in the
    /// document by the reader.
    QuotedText,
    /// The judged unit is one of the set's own chunks, placed where the
    /// reader joined it into the document.
    Chunks,
}

impl Set {
    /// The set's directory under `cache`.
    pub fn dir(&self, cache: &Path) -> PathBuf {
        cache.join(self.name)
    }

    /// Refuses a set with no licence recorded.
    pub fn check_license(&self) -> Result<()> {
        if self.license.trim().is_empty() {
            return Err(format!("set {} has no licence recorded, so it is not used", self.name));
        }
        Ok(())
    }

    /// Checks the licence and every recorded digest, then reads the set
    /// from `cache`.
    pub fn load(&self, cache: &Path) -> Result<LoadedSet> {
        self.check_license()?;
        let dir = self.dir(cache);
        if !dir.is_dir() {
            return Err(format!("set {}: {} does not exist; fetch the set first", self.name, dir.display()));
        }
        self.verify(&dir)?;
        let set = (self.read)(self.name, &dir)?;
        if set.spans.is_some() != self.span_unit.is_some() {
            return Err(format!("set {}: the reader's span judgements disagree with its span unit", self.name));
        }
        Ok(set)
    }

    /// Checks each file with a recorded digest in `dir`.
    pub fn verify(&self, dir: &Path) -> Result<()> {
        let Source::Files(files) = self.source else { return Ok(()) };
        for f in files {
            if let Digest::Recorded(want) = f.sha256 {
                let path = dir.join(f.path);
                let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                let have = to_hex(&sha256(&bytes));
                if have != want {
                    return Err(format!("{}: SHA-256 is {have}, set {} records {want}", path.display(), self.name));
                }
            }
        }
        Ok(())
    }
}
