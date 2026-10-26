//! Loading a published [`EmotionVectorHead`] from the model zoo (`#926`).
//!
//! [`EmotionVectorHead::from_zoo`] asks a [`Zoo`] for an artifact by id,
//! checks that it is a `KIND_HEAD` entry of one `FORMAT_HEAD` file, and
//! parses its `TEVH` bytes via [`EmotionVectorHead::from_bytes`]. The
//! licence check and the size, SHA-256 and SHA-512 checks are the zoo's.

use std::fmt;
use std::path::PathBuf;

use turbo_zoo::{Acceptances, Format, Kind, UserId, Zoo, ZooError};

use super::{EmotionHeadError, EmotionVectorHead};

/// GoEmotions 28-label head trained over `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
pub const GOEMOTIONS_MINILM_HEAD: &str = "turbo-en-goemotions-minilm-l6-v2-head-1";

/// GoEmotions 28-label head trained over `BAAI/bge-small-en-v1.5` (384-dim).
pub const GOEMOTIONS_BGE_SMALL_HEAD: &str = "turbo-en-goemotions-bge-small-en-v1.5-head-1";

/// Why a head could not be fetched or loaded from the zoo.
#[derive(Debug)]
pub enum EmotionZooError {
    /// The zoo refused the id or could not hand out its file.
    Zoo(ZooError),
    /// The artifact is not one `KIND_HEAD` `FORMAT_HEAD` file.
    NotAHead { id: String, kind: Kind, format: Format, files: usize },
    /// The verified file could not be read.
    Read { path: PathBuf, error: std::io::Error },
    /// The file bytes are not a valid [`EmotionVectorHead`].
    Head(EmotionHeadError),
}

impl fmt::Display for EmotionZooError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmotionZooError::Zoo(e) => write!(f, "{e}"),
            EmotionZooError::NotAHead { id, kind, format, files } => {
                write!(f, "{id} is {kind} {format} with {files} files, not one KIND_HEAD FORMAT_HEAD file")
            }
            EmotionZooError::Read { path, error } => write!(f, "reading {}: {error}", path.display()),
            EmotionZooError::Head(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for EmotionZooError {}

impl From<ZooError> for EmotionZooError {
    fn from(e: ZooError) -> Self {
        EmotionZooError::Zoo(e)
    }
}

impl From<EmotionHeadError> for EmotionZooError {
    fn from(e: EmotionHeadError) -> Self {
        EmotionZooError::Head(e)
    }
}

impl EmotionVectorHead {
    /// Fetches and opens the head `id` from `zoo`, for `user`, reading the
    /// acceptances through `load` only when the entry needs some
    /// ([`Zoo::fetch_with`]).
    pub fn from_zoo(
        zoo: &Zoo<'_>,
        id: &str,
        user: &UserId,
        load: impl FnOnce() -> Result<Acceptances, ZooError>,
    ) -> Result<EmotionVectorHead, EmotionZooError> {
        let a = zoo.index().get(id)?;
        if a.kind != Kind::Head || a.format != Format::Head || a.files.len() != 1 {
            return Err(EmotionZooError::NotAHead {
                id: id.to_owned(),
                kind: a.kind,
                format: a.format,
                files: a.files.len(),
            });
        }
        let fetched = zoo.fetch_with(id, user, load)?;
        let path = fetched.single().expect("one file, checked above").to_path_buf();
        let bytes = std::fs::read(&path).map_err(|error| EmotionZooError::Read { path, error })?;
        Ok(EmotionVectorHead::from_bytes(&bytes)?)
    }
}
