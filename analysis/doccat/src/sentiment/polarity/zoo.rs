//! Loading a published [`PolarityVectorHead`] from the model zoo (`#1732`).
//!
//! [`PolarityVectorHead::from_zoo`] asks a [`Zoo`] for an artifact by id,
//! checks that it is a `KIND_HEAD` entry of one `FORMAT_HEAD` file, and
//! parses its `TPVH` bytes via [`PolarityVectorHead::from_bytes`]. The
//! licence check and the size, SHA-256 and SHA-512 checks are the zoo's.

use std::fmt;
use std::path::PathBuf;

use turbo_zoo::{Acceptances, Format, Kind, UserId, Zoo, ZooError};

use super::head::PolarityVectorHead;

/// MASC 3.0.0 MPQA 3-label polarity head trained over `answerdotai/ModernBERT-base` (768-dim).
pub const MASC_MODERNBERT_POLARITY_HEAD: &str = "turbo-en-masc-3.0.0-modernbert-base-polarity-head-1";

/// Why a polarity head could not be fetched or loaded from the zoo.
#[derive(Debug)]
pub enum PolarityZooError {
    /// The zoo refused the id or could not hand out its file.
    Zoo(ZooError),
    /// The artifact is not one `KIND_HEAD` `FORMAT_HEAD` file.
    NotAHead { id: String, kind: Kind, format: Format, files: usize },
    /// The verified file could not be read.
    Read { path: PathBuf, error: std::io::Error },
    /// The file bytes are not a valid [`PolarityVectorHead`].
    Head(String),
}

impl fmt::Display for PolarityZooError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolarityZooError::Zoo(e) => write!(f, "{e}"),
            PolarityZooError::NotAHead { id, kind, format, files } => {
                write!(f, "{id} is {kind} {format} with {files} files, not one KIND_HEAD FORMAT_HEAD file")
            }
            PolarityZooError::Read { path, error } => write!(f, "reading {}: {error}", path.display()),
            PolarityZooError::Head(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PolarityZooError {}

impl From<ZooError> for PolarityZooError {
    fn from(e: ZooError) -> Self {
        PolarityZooError::Zoo(e)
    }
}

impl PolarityVectorHead {
    /// Fetches and opens the head `id` from `zoo`, for `user`, reading the
    /// acceptances through `load` only when the entry needs some
    /// ([`Zoo::fetch_with`]).
    pub fn from_zoo(
        zoo: &Zoo<'_>,
        id: &str,
        user: &UserId,
        load: impl FnOnce() -> Result<Acceptances, ZooError>,
    ) -> Result<PolarityVectorHead, PolarityZooError> {
        let a = zoo.index().get(id)?;
        if a.kind != Kind::Head || a.format != Format::Head || a.files.len() != 1 {
            return Err(PolarityZooError::NotAHead {
                id: id.to_owned(),
                kind: a.kind,
                format: a.format,
                files: a.files.len(),
            });
        }
        let fetched = zoo.fetch_with(id, user, load)?;
        let path = fetched.single().expect("one file, checked above").to_path_buf();
        let bytes = std::fs::read(&path).map_err(|error| PolarityZooError::Read { path, error })?;
        PolarityVectorHead::from_bytes(&bytes).map_err(PolarityZooError::Head)
    }
}
