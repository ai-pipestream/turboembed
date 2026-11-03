//! Model archive container (`manifest.properties` + classifier/dictionary entries)
//! produced by `turbo-train-components` (`ModelPackage`), with external legacy
//! `.bin` file loading disabled under `#1511`.

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
    /// External OpenNLP `.bin` model loading is disabled by `#1511`. No file is opened.
    pub fn read(_path: impl AsRef<Path>) -> Result<ModelFile> {
        Err(Error::Disabled)
    }

    /// External OpenNLP `.bin` model loading is disabled by `#1511`. No archive is decoded.
    /// Use [`ModelFile::from_trained_archive`] to open archives produced by `turbo-train-components`.
    pub fn open(_bytes: Vec<u8>) -> Result<ModelFile> {
        Err(Error::Disabled)
    }

    /// Opens a trained model archive produced by `turbo-train-components` (`ModelPackage::write`).
    pub fn from_trained_archive(bytes: Vec<u8>) -> Result<ModelFile> {
        let archive = Zip::new(bytes)?;
        let manifest = archive.entry(MANIFEST_ENTRY)?;
        let manifest = Properties::load(&manifest)
            .map_err(|error| Error::Entry { entry: MANIFEST_ENTRY.into(), error: Box::new(error) })?;
        if manifest.get("OpenNLP-Version").is_some() {
            return Err(Error::Manifest {
                key: "OpenNLP-Version",
                message: "legacy OpenNLP manifest is not supported",
            });
        }
        match manifest.get("Turbo-Format-Version") {
            Some("1") => {}
            Some(_) => {
                return Err(Error::Manifest {
                    key: "Turbo-Format-Version",
                    message: "unsupported native archive format version",
                });
            }
            None => {
                return Err(Error::Manifest {
                    key: "Turbo-Format-Version",
                    message: "missing native archive format version",
                });
            }
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

    /// The `Turbo-Format-Version` property of the native archive.
    pub fn format_version(&self) -> &str {
        self.manifest.get("Turbo-Format-Version").expect("checked when the file was opened")
    }

    /// Compatibility accessor returning [`Self::format_version`] for model provenance revision.
    pub fn opennlp_version(&self) -> &str {
        self.format_version()
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

    /// Reads the maxent, QN, perceptron, or NaiveBayes model in the named entry.
    pub fn maxent(&self, name: &str) -> Result<MaxentModel> {
        let bytes = self.entry(name)?;
        MaxentModel::from_trained_bytes(&bytes)
            .map_err(|error| Error::Entry { entry: name.into(), error: Box::new(error) })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zip::stored_archive;

    fn manifest_error(manifest: &str) -> String {
        let z = stored_archive(MANIFEST_ENTRY, manifest.as_bytes(), b"", b"");
        ModelFile::from_trained_archive(z).unwrap_err().to_string()
    }

    #[test]
    fn a_complete_manifest_opens() {
        let z = stored_archive(MANIFEST_ENTRY, b"Turbo-Format-Version=1\nComponent-Name=X\nLanguage=en\n", b"", b"");
        let file = ModelFile::from_trained_archive(z).unwrap();
        assert_eq!(file.format_version(), "1");
        assert_eq!(file.opennlp_version(), "1");
        assert_eq!(file.component_name(), "X");
        assert_eq!(file.language(), "en");
    }

    #[test]
    fn manifest_errors_are_rejected_before_model_entry_decoding() {
        let z = stored_archive("token.model", b"not-a-valid-model", b"", b"");
        assert_eq!(
            ModelFile::from_trained_archive(z).unwrap_err().to_string(),
            "the archive has no entry \"manifest.properties\""
        );
        assert_eq!(
            manifest_error("Component-Name=X\nLanguage=en\n"),
            "manifest property Turbo-Format-Version: missing native archive format version"
        );
        assert_eq!(
            manifest_error("Turbo-Format-Version=2\nComponent-Name=X\nLanguage=en\n"),
            "manifest property Turbo-Format-Version: unsupported native archive format version"
        );
        assert_eq!(
            manifest_error("OpenNLP-Version=1.0.0\nComponent-Name=X\nLanguage=en\n"),
            "manifest property OpenNLP-Version: legacy OpenNLP manifest is not supported"
        );
        assert_eq!(
            manifest_error("Turbo-Format-Version=1\nOpenNLP-Version=1.0.0\nComponent-Name=X\nLanguage=en\n"),
            "manifest property OpenNLP-Version: legacy OpenNLP manifest is not supported"
        );
        assert_eq!(
            manifest_error("Turbo-Format-Version=1\nLanguage=en\n"),
            "manifest property Component-Name: missing"
        );
        assert_eq!(manifest_error("Turbo-Format-Version=1\nComponent-Name=X\n"), "manifest property Language: missing");
    }
}
