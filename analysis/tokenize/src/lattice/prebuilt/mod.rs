//! Versioned prebuilt dictionaries with shared, checked runtime tables.
mod image;
mod read;
mod validate;
mod write;

use super::{DictionaryLimits, MecabDictionary};
pub use image::PrebuiltImage;
pub(crate) use image::{Table, Text};
use std::{fmt, path::Path};

const MAGIC: &[u8; 8] = b"TLATTICE";
const VERSION: u32 = 1;
const HEADER: usize = 512;
const SECTIONS: usize = 15;
const DIRECTORY: usize = 72;

/// A prebuilt dictionary could not be read or written.
#[derive(Debug)]
pub enum PrebuiltError {
    Io(std::io::Error),
    Invalid(&'static str),
}
impl fmt::Display for PrebuiltError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "prebuilt dictionary: {e}"),
            Self::Invalid(detail) => write!(f, "prebuilt dictionary: {detail}"),
        }
    }
}
impl std::error::Error for PrebuiltError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}
impl From<std::io::Error> for PrebuiltError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl MecabDictionary {
    /// Writes the runtime tables and source digest. Original source files are not embedded.
    pub fn write_prebuilt(&self, path: impl AsRef<Path>) -> Result<(), PrebuiltError> {
        std::fs::write(path, self.to_prebuilt()?)?;
        Ok(())
    }

    /// Reads an image file into shared aligned storage, checking all bounds.
    pub fn load_prebuilt(path: impl AsRef<Path>, limits: DictionaryLimits) -> Result<Self, PrebuiltError> {
        Self::from_prebuilt(PrebuiltImage::read(path)?, limits)
    }
}
