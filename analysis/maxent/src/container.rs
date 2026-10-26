//! An OpenNLP model file: a zip holding `manifest.properties` and the
//! model's artifacts, such as `token.model` (the classifier) and
//! `abbreviations.dictionary`.
//!
//! OpenNLP's `BaseModel` requires the manifest to hold `OpenNLP-Version`,
//! `Component-Name` and `Language`, and refuses a model whose version
//! does not parse or whose major number is not between 0 and 3 (the fork
//! this crate follows accepts 0, which its preview releases use). Opening
//! a file here checks the same. Which
//! component a model is for, and what its other properties mean, is for
//! the component reading it to check.

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
        if !(0..=3).contains(&major) {
            return Err(Error::Manifest {
                key: "OpenNLP-Version",
                message: "only models of major version 0 to 3 are supported",
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

/// The major number of a version string, or `None` where OpenNLP's
/// `Version.parse` throws. The version ends at the third dot if there is
/// one, else at the first dash, else at the end of the string; the text
/// before the first dot, between the first and second dots, and from the
/// second dot to that end must each be an `Integer.parseInt` integer. A
/// first dash before the second dot, with no third dot, makes Java's
/// `substring` throw, which also fails the load.
fn major_version(version: &str) -> Option<i32> {
    let first_dot = version.find('.')?;
    let second_dot = first_dot + 1 + version[first_dot + 1..].find('.')?;
    let third_dot = version[second_dot + 1..].find('.').map(|i| second_dot + 1 + i);
    let end = match (third_dot, version.find('-')) {
        (Some(dot), _) => dot,
        (None, Some(dash)) => dash,
        (None, None) => version.len(),
    };
    if end < second_dot + 1 {
        return None;
    }
    let major = parse_int(&version[..first_dot])?;
    parse_int(&version[first_dot + 1..second_dot])?;
    parse_int(&version[second_dot + 1..end])?;
    Some(major)
}

/// Java's `Integer.parseInt(s)`: an optional `+` or `-`, then at least one
/// decimal digit, where a digit is any UTF-16 unit `Character.digit`
/// gives a value for in radix 10 (any `Nd` character in the BMP, such as
/// `٣` or `３`), with the result in `i32` range.
fn parse_int(s: &str) -> Option<i32> {
    let (negative, digits) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    let limit: i64 = if negative { 1 << 31 } else { (1 << 31) - 1 };
    let mut value: i64 = 0;
    for c in digits.chars() {
        value = value * 10 + i64::from(digit(c)?);
        if value > limit {
            return None;
        }
    }
    Some(if negative { -value } else { value } as i32)
}

/// `Character.digit(c, 10)` for a character in the BMP: its decimal value
/// if it is in category `Nd`. Unicode assigns `Nd` characters in runs of
/// ten contiguous code points from zero to nine, so the value is the
/// number of `Nd` code points directly before it, modulo ten. A character
/// outside the BMP is two surrogate units in Java, neither a digit.
fn digit(c: char) -> Option<u32> {
    let cp = u32::from(c);
    if c.is_ascii_digit() {
        return Some(cp - u32::from(b'0'));
    }
    if cp > 0xFFFF || !turbo_unicode::java::code_point::is_digit(cp) {
        return None;
    }
    let mut before = 0;
    while before < cp && turbo_unicode::java::code_point::is_digit(cp - before - 1) {
        before += 1;
    }
    Some(before % 10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zip::stored_archive;

    /// The error opening a model file whose only entry is a manifest with
    /// this text gives, as a string.
    fn manifest_error(manifest: &str) -> String {
        let z = stored_archive(MANIFEST_ENTRY, manifest.as_bytes(), b"", b"");
        ModelFile::open(z).unwrap_err().to_string()
    }

    #[test]
    fn a_complete_manifest_opens() {
        let z = stored_archive(MANIFEST_ENTRY, b"OpenNLP-Version=+3.0.0\nComponent-Name=X\nLanguage=en\n", b"", b"");
        let file = ModelFile::open(z).unwrap();
        assert_eq!(file.opennlp_version(), "+3.0.0");
        assert_eq!(file.component_name(), "X");
        assert_eq!(file.language(), "en");
    }

    #[test]
    fn manifest_errors() {
        let z = stored_archive("token.model", b"", b"", b"");
        assert_eq!(ModelFile::open(z).unwrap_err().to_string(), "the archive has no entry \"manifest.properties\"");
        assert_eq!(manifest_error("Component-Name=X\nLanguage=en\n"), "manifest property OpenNLP-Version: missing");
        assert_eq!(manifest_error("OpenNLP-Version=2.5.4\nLanguage=en\n"), "manifest property Component-Name: missing");
        assert_eq!(manifest_error("OpenNLP-Version=2.5.4\nComponent-Name=X\n"), "manifest property Language: missing");
        assert_eq!(
            manifest_error("OpenNLP-Version=2.5\nComponent-Name=X\nLanguage=en\n"),
            "manifest property OpenNLP-Version: not a version of the form major.minor.patch"
        );
        assert_eq!(
            manifest_error("OpenNLP-Version=1.2.3-rc.1\nComponent-Name=X\nLanguage=en\n"),
            "manifest property OpenNLP-Version: not a version of the form major.minor.patch"
        );
        assert_eq!(
            manifest_error("OpenNLP-Version=4.0.0\nComponent-Name=X\nLanguage=en\n"),
            "manifest property OpenNLP-Version: only models of major version 0 to 3 are supported"
        );
        assert_eq!(
            manifest_error("OpenNLP-Version=-1.0.0.0\nComponent-Name=X\nLanguage=en\n"),
            "manifest property OpenNLP-Version: only models of major version 0 to 3 are supported"
        );
    }

    #[test]
    fn versions() {
        assert_eq!(major_version("2.5.4"), Some(2));
        assert_eq!(major_version("1.5.3"), Some(1));
        assert_eq!(major_version("3.0.0-SNAPSHOT"), Some(3));
        assert_eq!(major_version("0.1.0-alpha2"), Some(0));
        assert_eq!(major_version("1.9.0.1"), Some(1));
        assert_eq!(major_version("1.9.0.x-y"), Some(1));
        assert_eq!(major_version("2.5"), None);
        assert_eq!(major_version("x.1.2"), None);
        assert_eq!(major_version("2.5."), None);
        assert_eq!(major_version("2..5"), None);
    }

    #[test]
    fn a_third_dot_ends_the_version_before_a_dash_does() {
        // The patch is "3-rc", which parseInt refuses, as Java does.
        assert_eq!(major_version("1.2.3-rc.1"), None);
        assert_eq!(major_version("1.2.3-rc1"), Some(1));
    }

    #[test]
    fn a_dash_before_the_second_dot_fails_as_substring_does() {
        assert_eq!(major_version("1-a.2.3"), None);
        assert_eq!(major_version("1.2-a.3"), None);
    }

    #[test]
    fn parse_int_follows_java() {
        assert_eq!(major_version("+3.0.0"), Some(3));
        assert_eq!(major_version("-1.0.0.0"), Some(-1));
        assert_eq!(major_version("+.0.0"), None);
        assert_eq!(major_version("-.0.0"), None);
        assert_eq!(major_version("++1.0.0"), None);
        assert_eq!(parse_int("2147483647"), Some(i32::MAX));
        assert_eq!(parse_int("-2147483648"), Some(i32::MIN));
        assert_eq!(parse_int("2147483648"), None);
        assert_eq!(parse_int("-2147483649"), None);
        assert_eq!(parse_int("00000000000000000000001"), Some(1));
        // Arabic-Indic, Devanagari and fullwidth digits.
        assert_eq!(parse_int("\u{663}"), Some(3));
        assert_eq!(parse_int("\u{967}\u{960}"), None);
        assert_eq!(parse_int("\u{967}\u{966}"), Some(10));
        assert_eq!(parse_int("\u{FF12}"), Some(2));
        // Outside the BMP: two surrogates in Java, not digits.
        assert_eq!(parse_int("\u{1D7D9}"), None);
        // Nl and No are not decimal digits.
        assert_eq!(parse_int("\u{2163}"), None);
        assert_eq!(parse_int("\u{B2}"), None);
    }
}
