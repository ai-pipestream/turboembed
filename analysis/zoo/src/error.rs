//! What can go wrong in the zoo.

use std::fmt;
use std::path::PathBuf;

use crate::index::ArtifactId;
use crate::licence::{Licence, LicenceClass, Requirement};

/// Which check a file failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Check {
    Size,
    Sha256,
    Sha512,
    /// A compressed file's bytes on the wire, before unpacking.
    DownloadSize,
    DownloadSha256,
    /// The compressed bytes do not unpack to the file's size.
    Unpack,
}

impl fmt::Display for Check {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Check::Size => "size",
            Check::Sha256 => "SHA-256",
            Check::Sha512 => "SHA-512",
            Check::DownloadSize => "download size",
            Check::DownloadSha256 => "download SHA-256",
            Check::Unpack => "unpacked file",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZooError {
    /// The index is malformed or inconsistent.
    Index(String),
    /// The acceptance file is malformed or inconsistent.
    Acceptances(String),
    /// No artifact of the index has this id.
    NotFound(String),
    /// The artifact is under terms the zoo refuses: it is not listed and
    /// never fetched.
    Refused { id: ArtifactId, licences: Vec<Licence> },
    /// The artifact's class needs the user's acceptance of these licences,
    /// and the user has not accepted them.
    NotAccepted { id: ArtifactId, class: LicenceClass, user: String, missing: Vec<Requirement> },
    /// A fetched or cached file is not the one the index describes.
    Integrity { id: ArtifactId, path: String, check: Check, expected: String, actual: String },
    /// The transport could not fetch an address.
    Transport { url: String, message: String },
    /// Reading or writing a local file failed.
    Io { path: PathBuf, message: String },
}

impl fmt::Display for ZooError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZooError::Index(m) => write!(f, "zoo index: {m}"),
            ZooError::Acceptances(m) => write!(f, "acceptance file: {m}"),
            ZooError::NotFound(id) => write!(f, "{id}: the zoo has no artifact with this id"),
            ZooError::Refused { id, licences } => {
                write!(f, "{id}: refused, the zoo does not list or fetch artifacts under ")?;
                for (i, l) in licences.iter().enumerate() {
                    let sep = if i == 0 { "" } else { ", " };
                    match l.refusal() {
                        Some(r) => write!(f, "{sep}{l} ({r})")?,
                        None => write!(f, "{sep}{l}")?,
                    }
                }
                Ok(())
            }
            ZooError::NotAccepted { id, class, user, missing } => {
                write!(f, "{id}: {class} needs the licence accepted before it is fetched; {user} has not accepted ")?;
                for (i, r) in missing.iter().enumerate() {
                    let sep = if i == 0 { "" } else { ", " };
                    write!(f, "{sep}{r}")?;
                }
                Ok(())
            }
            ZooError::Integrity { id, path, check, expected, actual } => {
                write!(f, "{id}/{path}: {check} is {actual}, the index says {expected}")
            }
            ZooError::Transport { url, message } => write!(f, "GET {url}: {message}"),
            ZooError::Io { path, message } => write!(f, "{}: {message}", path.display()),
        }
    }
}

impl std::error::Error for ZooError {}
