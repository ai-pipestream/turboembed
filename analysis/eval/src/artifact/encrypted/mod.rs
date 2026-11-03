//! Client-side encryption for explicitly authorized evaluation artifacts.

use std::{fmt, io};

mod codec;
use codec::{Decoder, Encoder};
mod verified;
use verified::VerifiedReader;
mod policy;
mod store;
pub use policy::*;
pub use store::{EncryptedStore, SecurityContext};
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct KeyId(pub [u8; 16]);

/// An opaque principal and artifact binding, authenticated but never stored.
#[derive(Clone, Copy)]
pub(super) struct Binding(pub [u8; 32]);

/// Secret material deliberately has no Debug, Display or serialization API.
pub struct KeyMaterial([u8; 32]);
impl KeyMaterial {
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
}

/// A real deployment supplies this capability from its key service.
/// No key is persisted by this interface and failures never select plaintext.
pub trait KeyProvider: Send + Sync {
    fn active(&self) -> Result<(KeyId, KeyMaterial), Error>;
    fn resolve(&self, id: KeyId) -> Result<KeyMaterial, Error>;
}

#[derive(Debug)]
pub enum Error {
    UnavailableKey,
    RevokedKey,
    Denied,
    Retained,
    AuditUnavailable,
    InvalidArtifact,
    Entropy,
    Poisoned,
    Io,
}
impl fmt::Display for Error {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str(match self {
            Self::RevokedKey => "artifact key revoked",
            Self::Denied => "artifact access denied",
            Self::Retained => "artifact retention prevents deletion",
            Self::AuditUnavailable => "artifact audit unavailable",
            Self::UnavailableKey => "artifact key unavailable",
            Self::InvalidArtifact => "artifact authentication or completeness failed",
            Self::Entropy => "artifact entropy unavailable",
            Self::Poisoned => "artifact write previously failed",
            Self::Io => "artifact I/O failed",
        })
    }
}
impl std::error::Error for Error {}
impl From<io::Error> for Error {
    fn from(_: io::Error) -> Self {
        Self::Io
    }
}

/// Generate a random opaque identifier. Keep any document-name mapping in the
/// application's protected metadata store, never in the provider filename.
pub fn new_artifact_key() -> Result<super::ArtifactKey, Error> {
    use ring::rand::{SecureRandom, SystemRandom};
    let mut id = [0; 16];
    SystemRandom::new().fill(&mut id).map_err(|_| Error::Entropy)?;
    let name: String = id.iter().map(|b| format!("{b:02x}")).collect();
    super::ArtifactKey::new(name).map_err(|_| Error::InvalidArtifact)
}
