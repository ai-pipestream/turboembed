//! Disabled classical model container.
//!
//! The public constructors return Error::Disabled before filesystem access or
//! archive parsing. The type remains until dependent ports are deleted.

use std::borrow::Cow;
use std::path::Path;
use turbo_sha256::{Digest, sha256};

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
    /// OpenNLP model loading is disabled by #1511. No file is opened.
    pub fn read(_path: impl AsRef<Path>) -> Result<ModelFile> {
        Err(Error::Disabled)
    }

    /// OpenNLP model loading is disabled by #1511. No archive is decoded.
    pub fn open(_bytes: Vec<u8>) -> Result<ModelFile> {
        Err(Error::Disabled)
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

    /// The model file's bytes, exactly as they were read: what a SHA-256
    /// of the file, recorded as a layer's provenance, is computed over.
    pub fn bytes(&self) -> &[u8] {
        self.archive.bytes()
    }

    /// The SHA-256 of the model file, the value `sha256sum` prints for it
    /// and the one a layer's provenance records.
    pub fn sha256(&self) -> Digest {
        sha256(self.bytes())
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
