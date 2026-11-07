//! The detectors a layer's mentions name: a stable string id and a
//! version each, in a per-layer table that every mention indexes. Custom
//! extractors and learned models fit the same table, which an enum could
//! not cover.

use std::fmt;

use turbo_analysis::Fnv1a;

/// An entry of [`PiiDetectors`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PiiDetectorRef(u16);

impl PiiDetectorRef {
    pub const fn new(index: u16) -> PiiDetectorRef {
        PiiDetectorRef(index)
    }

    pub const fn index(self) -> u16 {
        self.0
    }
}

/// Why a detector entry was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DetectorError {
    /// The id is empty.
    EmptyId,
    /// The id or version is longer than [`PiiDetectors::MAX_BYTES`] bytes.
    TooLong(usize),
    /// The table would exceed [`PiiDetectors::MAX`] entries.
    Full,
    /// A reference indexes past the end of the table.
    Unknown(u16),
}

impl fmt::Display for DetectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DetectorError::EmptyId => f.write_str("a pii detector id must not be empty"),
            DetectorError::TooLong(n) => {
                write!(f, "a pii detector id or version is {n} bytes (max {})", PiiDetectors::MAX_BYTES)
            }
            DetectorError::Full => write!(f, "more than {} pii detectors", PiiDetectors::MAX),
            DetectorError::Unknown(i) => write!(f, "pii detector {i} is not in the table"),
        }
    }
}

impl std::error::Error for DetectorError {}

/// The detectors of one layer, in first-seen order. Equal id and version
/// pairs share one entry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PiiDetectors {
    ids: Vec<Box<str>>,
    versions: Vec<Box<str>>,
}

impl PiiDetectors {
    /// The most entries a table holds.
    pub const MAX: usize = 1024;
    /// The longest id or version, in bytes.
    pub const MAX_BYTES: usize = 128;

    pub const fn new() -> PiiDetectors {
        PiiDetectors { ids: Vec::new(), versions: Vec::new() }
    }

    /// The entry for `id` at `version`, added when new.
    pub fn add(&mut self, id: &str, version: &str) -> Result<PiiDetectorRef, DetectorError> {
        if id.is_empty() {
            return Err(DetectorError::EmptyId);
        }
        if id.len() > Self::MAX_BYTES {
            return Err(DetectorError::TooLong(id.len()));
        }
        if version.len() > Self::MAX_BYTES {
            return Err(DetectorError::TooLong(version.len()));
        }
        if let Some(i) = self.find(id, version) {
            return Ok(i);
        }
        if self.ids.len() >= Self::MAX {
            return Err(DetectorError::Full);
        }
        self.ids.push(id.into());
        self.versions.push(version.into());
        Ok(PiiDetectorRef((self.ids.len() - 1) as u16))
    }

    /// The entry for `id` at `version`, when present.
    pub fn find(&self, id: &str, version: &str) -> Option<PiiDetectorRef> {
        self.ids
            .iter()
            .zip(&self.versions)
            .position(|(i, v)| &**i == id && &**v == version)
            .map(|i| PiiDetectorRef(i as u16))
    }

    /// The id and version of `r`.
    pub fn get(&self, r: PiiDetectorRef) -> Option<(&str, &str)> {
        let i = r.0 as usize;
        Some((self.ids.get(i)?, self.versions.get(i)?))
    }

    pub fn check(&self, r: PiiDetectorRef) -> Result<(), DetectorError> {
        if (r.0 as usize) < self.ids.len() { Ok(()) } else { Err(DetectorError::Unknown(r.0)) }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Every entry, in table order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&str, &str)> {
        self.ids.iter().zip(&self.versions).map(|(i, v)| (&**i, &**v))
    }

    pub fn fold_hash(&self, h: &mut Fnv1a) {
        h.usize(self.ids.len());
        for (id, version) in self.iter() {
            h.str(id).str(version);
        }
    }
}
