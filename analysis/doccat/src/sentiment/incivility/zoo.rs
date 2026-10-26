//! Loading a published [`IncivilityVectorHead`] from the model zoo (`#888`).
//!
//! [`IncivilityVectorHead::from_zoo`] asks a [`Zoo`] for an artifact by id,
//! checks that it is a `KIND_HEAD` entry of one `FORMAT_HEAD` file, and
//! parses its `TIVH` bytes via [`IncivilityVectorHead::from_bytes`]. The
//! licence check and the size, SHA-256 and SHA-512 checks are the zoo's.

use std::fmt;
use std::path::PathBuf;

use turbo_zoo::{Acceptances, Format, Kind, UserId, Zoo, ZooError};

use super::{IncivilityHeadError, IncivilityVectorHead};

/// Civil Comments 6-label incivility head trained over `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
pub const CIVIL_COMMENTS_MINILM_HEAD: &str = "turbo-en-civil-comments-minilm-l6-v2-head-1";

/// Civil Comments 6-label incivility head with parent-comment context matrix trained over `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
pub const CIVIL_COMMENTS_MINILM_CONTEXT_HEAD: &str = "turbo-en-civil-comments-minilm-l6-v2-context-head-1";

/// Civil Comments 6-label incivility head trained over `BAAI/bge-small-en-v1.5` (384-dim).
pub const CIVIL_COMMENTS_BGE_SMALL_HEAD: &str = "turbo-en-civil-comments-bge-small-en-v1.5-head-1";

/// Civil Comments 6-label incivility head with parent-comment context matrix trained over `BAAI/bge-small-en-v1.5` (384-dim).
pub const CIVIL_COMMENTS_BGE_SMALL_CONTEXT_HEAD: &str = "turbo-en-civil-comments-bge-small-en-v1.5-context-head-1";

/// Why an incivility head could not be fetched or loaded from the zoo.
#[derive(Debug)]
pub enum IncivilityZooError {
    /// The zoo refused the id or could not hand out its file.
    Zoo(ZooError),
    /// The artifact is not one `KIND_HEAD` `FORMAT_HEAD` file.
    NotAHead { id: String, kind: Kind, format: Format, files: usize },
    /// The verified file could not be read.
    Read { path: PathBuf, error: std::io::Error },
    /// The file bytes are not a valid [`IncivilityVectorHead`].
    Head(IncivilityHeadError),
}

impl fmt::Display for IncivilityZooError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IncivilityZooError::Zoo(e) => write!(f, "{e}"),
            IncivilityZooError::NotAHead { id, kind, format, files } => {
                write!(f, "{id} is {kind} {format} with {files} files, not one KIND_HEAD FORMAT_HEAD file")
            }
            IncivilityZooError::Read { path, error } => write!(f, "reading {}: {error}", path.display()),
            IncivilityZooError::Head(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for IncivilityZooError {}

impl From<ZooError> for IncivilityZooError {
    fn from(e: ZooError) -> Self {
        IncivilityZooError::Zoo(e)
    }
}

impl From<IncivilityHeadError> for IncivilityZooError {
    fn from(e: IncivilityHeadError) -> Self {
        IncivilityZooError::Head(e)
    }
}

impl IncivilityVectorHead {
    /// Fetches and opens the head `id` from `zoo`, for `user`, reading the
    /// acceptances through `load` only when the entry needs some
    /// ([`Zoo::fetch_with`]).
    pub fn from_zoo(
        zoo: &Zoo<'_>,
        id: &str,
        user: &UserId,
        load: impl FnOnce() -> Result<Acceptances, ZooError>,
    ) -> Result<IncivilityVectorHead, IncivilityZooError> {
        let a = zoo.index().get(id)?;
        if a.kind != Kind::Head || a.format != Format::Head || a.files.len() != 1 {
            return Err(IncivilityZooError::NotAHead {
                id: id.to_owned(),
                kind: a.kind,
                format: a.format,
                files: a.files.len(),
            });
        }
        let fetched = zoo.fetch_with(id, user, load)?;
        let path = fetched.single().expect("one file, checked above").to_path_buf();
        let bytes = std::fs::read(&path).map_err(|error| IncivilityZooError::Read { path, error })?;
        Ok(IncivilityVectorHead::from_bytes(&bytes)?)
    }
}
