//! An OpenNLP model file: a zip holding `manifest.properties` and the
//! model's artifacts, such as `token.model` (the classifier) and
//! `abbreviations.dictionary`.
//!
//! OpenNLP's `BaseModel` requires the manifest to hold `OpenNLP-Version`,
//! `Component-Name` and `Language`, and refuses a model whose version's
//! major number is above 3. Opening a file here checks the same. Which
//! component a model is for, and what its other properties mean, is for
//! the component reading it to check.

use std::borrow::Cow;
use std::path::Path;

use crate::dictionary::{Dictionary, Entries};
use crate::error::{Error, Result};
use crate::model::MaxentModel;
use crate::properties::Properties;
use crate::zip::Zip;

/// The manifest entry's name.
pub const MANIFEST_ENTRY: &str = "manifest.properties";

/// An opened model file: its manifest, read, and its archive, from which
/// artifacts are read on request.
#[derive(Debug, Clone)]
pub struct ModelFile {
    manifest: Properties,
    archive: Zip<Vec<u8>>,
}

impl ModelFile {
    /// Reads a model file from disk and opens it.
    pub fn read(path: impl AsRef<Path>) -> Result<ModelFile> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|error| Error::Io { path: path.to_path_buf(), error })?;
        ModelFile::open(bytes)
    }

    /// Opens a model file held in memory.
    pub fn open(bytes: Vec<u8>) -> Result<ModelFile> {
        let archive = Zip::new(bytes)?;
        let manifest = archive.entry(MANIFEST_ENTRY)?;
        let manifest = Properties::load(&manifest)
            .map_err(|error| Error::Entry { entry: MANIFEST_ENTRY.into(), error: Box::new(error) })?;
        let version =
            manifest.get("OpenNLP-Version").ok_or(Error::Manifest { key: "OpenNLP-Version", message: "missing" })?;
        let major = major_version(version).ok_or(Error::Manifest {
            key: "OpenNLP-Version",
            message: "not a version of the form major.minor.patch",
        })?;
        if major > 3 {
            return Err(Error::Manifest {
                key: "OpenNLP-Version",
                message: "models above major version 3 are not supported",
            });
        }
        for key in ["Component-Name", "Language"] {
            if manifest.get(key).is_none() {
                return Err(Error::Manifest { key, message: "missing" });
            }
        }
        Ok(ModelFile { manifest, archive })
    }

    /// The manifest's properties.
    pub fn manifest(&self) -> &Properties {
        &self.manifest
    }

    /// The `Language` property.
    pub fn language(&self) -> &str {
        self.manifest.get("Language").expect("checked when the file was opened")
    }

    /// The `Component-Name` property, such as `TokenizerME` or
    /// `SentenceDetectorME`.
    pub fn component_name(&self) -> &str {
        self.manifest.get("Component-Name").expect("checked when the file was opened")
    }

    /// The `OpenNLP-Version` property: the version that trained the model.
    pub fn opennlp_version(&self) -> &str {
        self.manifest.get("OpenNLP-Version").expect("checked when the file was opened")
    }

    /// The zip archive, for listing its entries.
    pub fn archive(&self) -> &Zip<Vec<u8>> {
        &self.archive
    }

    /// An artifact's bytes.
    pub fn entry(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        self.archive.entry(name)
    }

    /// Reads the maxent or perceptron model in the named entry.
    pub fn maxent(&self, name: &str) -> Result<MaxentModel> {
        let bytes = self.entry(name)?;
        MaxentModel::read(&bytes).map_err(|error| Error::Entry { entry: name.into(), error: Box::new(error) })
    }

    /// Reads the dictionary in the named entry, or `None` when the archive
    /// has no such entry (the model was trained without one).
    pub fn dictionary(&self, name: &str) -> Result<Option<Dictionary>> {
        if !self.archive.contains(name) {
            return Ok(None);
        }
        let bytes = self.entry(name)?;
        Dictionary::read(&bytes).map(Some).map_err(|error| Error::Entry { entry: name.into(), error: Box::new(error) })
    }

    /// Reads the named dictionary entry with every entry's attributes, or
    /// `None` when the archive has no such entry.
    pub fn dictionary_entries(&self, name: &str) -> Result<Option<Entries>> {
        if !self.archive.contains(name) {
            return Ok(None);
        }
        let bytes = self.entry(name)?;
        Entries::read(&bytes).map(Some).map_err(|error| Error::Entry { entry: name.into(), error: Box::new(error) })
    }
}

/// The major number of a version string, read as OpenNLP's
/// `Version.parse` reads it: the text before the first dot, which must be
/// followed by a second dot, parsed as an integer.
fn major_version(version: &str) -> Option<u32> {
    let (major, rest) = version.split_once('.')?;
    let (minor, rest) = rest.split_once('.')?;
    let patch_end = rest.find(['.', '-']).unwrap_or(rest.len());
    let patch = &rest[..patch_end];
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(major) || !digits(minor) || !digits(patch) {
        return None;
    }
    major.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions() {
        assert_eq!(major_version("2.5.4"), Some(2));
        assert_eq!(major_version("1.5.3"), Some(1));
        assert_eq!(major_version("3.0.0-SNAPSHOT"), Some(3));
        assert_eq!(major_version("0.1.0-alpha2"), Some(0));
        assert_eq!(major_version("1.9.0.1"), Some(1));
        assert_eq!(major_version("2.5"), None);
        assert_eq!(major_version("x.1.2"), None);
    }
}
