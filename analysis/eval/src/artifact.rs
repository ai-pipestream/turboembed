//! Storage for evaluation artifacts. Encoding and document/cache lifecycles
//! belong to their callers; this interface exposes no provider protocol.

use std::fmt;
use std::io::{self, Read, Write};

pub mod encrypted;
mod temporary;
pub use temporary::TemporaryFileStore;

/// A logical evaluation artifact name, never a path or provider URL.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArtifactKey(String);

impl ArtifactKey {
    pub fn new(name: impl Into<String>) -> Result<Self, StorageError> {
        let name = name.into();
        if name.is_empty()
            || name.len() > 128
            || name.starts_with('.')
            || name.ends_with('.')
            || !name.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
        {
            return Err(StorageError::InvalidKey);
        }
        Ok(Self(name))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ArtifactKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Typed failures at the storage boundary. `Published` explicitly means
/// the key became visible, but a subsequent durability operation failed.
#[derive(Debug)]
pub enum StorageError {
    InvalidKey,
    Security(encrypted::Error),
    NotFound(ArtifactKey),
    AlreadyExists(ArtifactKey),
    Io { operation: &'static str, source: io::Error },
    Published { key: ArtifactKey, source: io::Error },
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Security(error) => error.fmt(f),
            Self::InvalidKey => {
                f.write_str("artifact key needs 1..=128 ASCII letters, digits, '-', '_' or interior '.'")
            }
            Self::NotFound(k) => write!(f, "artifact {k}: not found"),
            Self::AlreadyExists(k) => write!(f, "artifact {k}: a completed artifact is never replaced"),
            Self::Io { operation, source } => write!(f, "artifact {operation}: {source}"),
            Self::Published { key, source } => {
                write!(f, "artifact {key} was published, but directory sync failed: {source}")
            }
        }
    }
}

impl std::error::Error for StorageError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } | Self::Published { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Publication succeeded. An unsuccessful staging cleanup is observable
/// here without claiming the committed artifact was not published.
#[derive(Debug)]
pub struct Publication {
    pub key: ArtifactKey,
    pub cleanup_error: Option<io::Error>,
}

/// An exclusively owned unpublished write. Dropping or aborting it must
/// never change a committed artifact. Finalize must refuse replacement.
pub trait PendingArtifact: Write {
    fn finalize(self: Box<Self>) -> Result<Publication, StorageError>;
    fn abort(self: Box<Self>) -> Result<(), StorageError>;
}

/// Evaluation-artifact storage with streaming byte access. The cut codec
/// is separate. Only these operations are currently required/supported;
/// this is not a generic cache, document exporter or format converter.
pub trait EvaluationArtifactStore {
    fn begin(&self, key: &ArtifactKey) -> Result<Box<dyn PendingArtifact>, StorageError>;
    fn open(&self, key: &ArtifactKey) -> Result<Box<dyn Read>, StorageError>;
    fn list(&self) -> Result<Vec<ArtifactKey>, StorageError>;
    fn delete(&self, key: &ArtifactKey) -> Result<(), StorageError>;
}
