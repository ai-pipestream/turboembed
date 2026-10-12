//! A TokenizerME model: OpenNLP's `TokenizerModel` and the parts of
//! `TokenizerFactory` a loaded model uses.

use std::fmt;
use std::path::Path;

use turbo_maxent::{Dictionary, MaxentModel, ModelFile};

use super::alnum::{AlphaNumeric, DEFAULT_PATTERN, PatternError};
use crate::sha256::{Digest, sha256};

/// The `Component-Name` of a tokenizer model.
pub const COMPONENT_NAME: &str = "TokenizerME";
/// The entry holding the classifier.
pub const MODEL_ENTRY: &str = "token.model";
/// The entry holding the optional abbreviation dictionary.
pub const ABBREVIATIONS_ENTRY: &str = "abbreviations.dictionary";

/// Why a tokenizer model could not be loaded.
#[derive(Debug)]
pub enum ModelError {
    /// The file, the classifier or the dictionary could not be read.
    Maxent(turbo_maxent::Error),
    /// The manifest names another component.
    Component(String),
    /// `useAlphaNumericOptimization`, which OpenNLP requires, is missing.
    MissingOptimizationFlag,
    /// The classifier's outcomes are not `T` and `F`.
    Outcomes(Vec<String>),
    /// The alphanumeric optimization is on and its pattern cannot be read.
    Pattern(PatternError),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Maxent(e) => write!(f, "{e}"),
            ModelError::Component(name) => write!(f, "the model is for {name:?}, not {COMPONENT_NAME}"),
            ModelError::MissingOptimizationFlag => {
                f.write_str("manifest property useAlphaNumericOptimization: missing, and it is mandatory")
            }
            ModelError::Outcomes(found) => write!(f, "the classifier's outcomes are {found:?}, not T and F"),
            ModelError::Pattern(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::Maxent(e) => Some(e),
            ModelError::Pattern(e) => Some(e),
            _ => None,
        }
    }
}

impl From<turbo_maxent::Error> for ModelError {
    fn from(e: turbo_maxent::Error) -> ModelError {
        ModelError::Maxent(e)
    }
}

/// What each outcome of the classifier means for the tokenizer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Outcome {
    /// The outcome is `T`, a token boundary.
    pub split: bool,
    /// The index `getIndex` returns for the outcome's name: the first
    /// outcome of that name. The token's probability is read there.
    pub index: usize,
}

/// A loaded tokenizer model.
#[derive(Debug, Clone)]
pub struct TokenizerModel {
    language: String,
    opennlp_version: String,
    maxent: MaxentModel,
    outcomes: [Outcome; 2],
    abbreviations: Option<Dictionary>,
    pattern: String,
    alphanumeric: Option<AlphaNumeric>,
    sha256: Digest,
}

impl TokenizerModel {
    /// Reads a model file from disk.
    pub fn read(path: impl AsRef<Path>) -> Result<TokenizerModel, ModelError> {
        TokenizerModel::open(ModelFile::read(path)?)
    }

    /// Reads the tokenizer from an opened model file, checking what
    /// OpenNLP's `TokenizerModel` and `TokenizerFactory` check: the
    /// component is `TokenizerME`, `useAlphaNumericOptimization` is
    /// present, and the classifier has exactly the outcomes `T` and `F`.
    ///
    /// The alphanumeric optimization is on when the manifest's
    /// `useAlphaNumericOptimization` is `true` in any case, as
    /// `Boolean.valueOf` reads it. Its pattern is the manifest's
    /// `alphaNumericPattern`, or else [`DEFAULT_PATTERN`] whatever the
    /// model's language: `TokenizerFactory.getAlphaNumericPattern` asks
    /// `lang.Factory` for the language's pattern with the factory's own
    /// language field, which a factory made for a loaded model never sets,
    /// so `Factory` answers with its default. The pattern is read only
    /// when the optimization is on, the only case in which it is used.
    pub fn open(file: ModelFile) -> Result<TokenizerModel, ModelError> {
        if file.component_name() != COMPONENT_NAME {
            return Err(ModelError::Component(file.component_name().to_string()));
        }
        let manifest = file.manifest();
        let optimize = manifest
            .get("useAlphaNumericOptimization")
            .ok_or(ModelError::MissingOptimizationFlag)?
            .eq_ignore_ascii_case("true");
        let pattern = manifest.get("alphaNumericPattern").unwrap_or(DEFAULT_PATTERN).to_string();
        let alphanumeric =
            if optimize { Some(AlphaNumeric::parse(&pattern).map_err(ModelError::Pattern)?) } else { None };
        let maxent = file.maxent(MODEL_ENTRY)?;
        let outcomes = outcomes(&maxent)?;
        let abbreviations = file.dictionary(ABBREVIATIONS_ENTRY)?;
        Ok(TokenizerModel {
            language: file.language().to_string(),
            opennlp_version: file.opennlp_version().to_string(),
            maxent,
            outcomes,
            abbreviations,
            pattern,
            alphanumeric,
            sha256: sha256(file.bytes()),
        })
    }

    /// The same model with `abbreviations` as its abbreviation dictionary,
    /// as if the model file held it: it is both the dictionary the
    /// tokenizer keeps known abbreviations whole by and the one the
    /// `pabb` feature tests. The model's SHA-256 still names the file it
    /// was read from.
    pub fn with_abbreviations(mut self, abbreviations: Option<Dictionary>) -> TokenizerModel {
        self.abbreviations = abbreviations;
        self
    }

    /// The same model with the alphanumeric optimization on, with
    /// `pattern`, or off for `None`, as a `TokenizerFactory` built with
    /// those settings would have it.
    pub fn with_alphanumeric_optimization(mut self, pattern: Option<&str>) -> Result<TokenizerModel, PatternError> {
        match pattern {
            Some(p) => {
                self.alphanumeric = Some(AlphaNumeric::parse(p)?);
                self.pattern = p.to_string();
            }
            None => self.alphanumeric = None,
        }
        Ok(self)
    }

    /// The manifest's `Language`.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// The manifest's `OpenNLP-Version`: the version that trained it.
    pub fn opennlp_version(&self) -> &str {
        &self.opennlp_version
    }

    /// The classifier.
    pub fn maxent(&self) -> &MaxentModel {
        &self.maxent
    }

    /// The abbreviation dictionary, when the model has one.
    pub fn abbreviations(&self) -> Option<&Dictionary> {
        self.abbreviations.as_ref()
    }

    /// The alphanumeric pattern: the manifest's, or the default.
    pub fn alphanumeric_pattern(&self) -> &str {
        &self.pattern
    }

    /// The alphanumeric check, when the optimization is on.
    pub fn alphanumeric(&self) -> Option<&AlphaNumeric> {
        self.alphanumeric.as_ref()
    }

    /// The SHA-256 of the model file.
    pub fn sha256(&self) -> Digest {
        self.sha256
    }

    pub(crate) fn outcome(&self, i: usize) -> Outcome {
        self.outcomes[i]
    }
}

/// `ModelUtil.validateOutcomes(model, "T", "F")`: exactly two outcomes,
/// each `T` or `F`.
fn outcomes(maxent: &MaxentModel) -> Result<[Outcome; 2], ModelError> {
    let names: Vec<&str> = maxent.outcomes().collect();
    if names.len() != 2 || names.iter().any(|&n| n != "T" && n != "F") {
        return Err(ModelError::Outcomes(names.iter().map(|n| n.to_string()).collect()));
    }
    let outcome = |i: usize| Outcome {
        split: names[i] == "T",
        index: maxent.outcome_index(names[i]).expect("the outcome's own name"),
    };
    Ok([outcome(0), outcome(1)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models")
    }

    #[test]
    fn reads_the_english_model() {
        let m = TokenizerModel::read(models().join("opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin")).unwrap();
        assert_eq!(m.language(), "en");
        assert_eq!(m.opennlp_version(), "2.5.4");
        assert_eq!(m.alphanumeric_pattern(), "^[A-Za-z0-9]+$");
        assert!(m.alphanumeric().is_none());
        assert!(m.abbreviations().is_none());
        assert_eq!(m.sha256().to_hex(), "95139111197ba9f1165fc12be8d4cbac05ea3912f995d167efa1a0e04e3e649a");
        assert_eq!(m.maxent().outcomes().collect::<Vec<_>>(), ["F", "T"]);
        assert_eq!(m.outcome(0), Outcome { split: false, index: 0 });
        assert_eq!(m.outcome(1), Outcome { split: true, index: 1 });
    }

    #[test]
    fn reads_the_abbreviation_dictionary() {
        let m = TokenizerModel::read(models().join("tokenizer-abbreviations.bin")).unwrap();
        let d = m.abbreviations().unwrap();
        assert_eq!(d.len(), 3);
        assert!(d.contains("dr."));
        let m = m.with_abbreviations(None);
        assert!(m.abbreviations().is_none());
    }

    /// A zip of stored entries, enough for `Zip` to read.
    fn stored_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut z = Vec::new();
        let mut dir = Vec::new();
        for &(name, data) in entries {
            let local = z.len() as u32;
            let crc = turbo_maxent::crc32(data);
            z.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
            z.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            for v in [crc, data.len() as u32, data.len() as u32] {
                z.extend_from_slice(&v.to_le_bytes());
            }
            z.extend_from_slice(&(name.len() as u16).to_le_bytes());
            z.extend_from_slice(&[0, 0]);
            z.extend_from_slice(name.as_bytes());
            z.extend_from_slice(data);
            dir.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
            dir.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            for v in [crc, data.len() as u32, data.len() as u32] {
                dir.extend_from_slice(&v.to_le_bytes());
            }
            dir.extend_from_slice(&(name.len() as u16).to_le_bytes());
            dir.extend_from_slice(&[0; 12]);
            dir.extend_from_slice(&local.to_le_bytes());
            dir.extend_from_slice(name.as_bytes());
        }
        let at = z.len() as u32;
        z.extend_from_slice(&dir);
        z.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
        z.extend_from_slice(&[0, 0, 0, 0]);
        let n = entries.len() as u16;
        z.extend_from_slice(&n.to_le_bytes());
        z.extend_from_slice(&n.to_le_bytes());
        z.extend_from_slice(&(dir.len() as u32).to_le_bytes());
        z.extend_from_slice(&at.to_le_bytes());
        z.extend_from_slice(&[0, 0]);
        z
    }

    /// The English classifier under a manifest of `extra` lines.
    fn with_manifest(extra: &str, classifier: &[u8]) -> Result<TokenizerModel, ModelError> {
        let manifest = format!("OpenNLP-Version=2.5.4\nLanguage=de\n{extra}");
        let bytes = stored_zip(&[("manifest.properties", manifest.as_bytes()), (MODEL_ENTRY, classifier)]);
        let model = TokenizerModel::open(ModelFile::open(bytes.clone()).unwrap());
        if let Ok(m) = &model {
            assert_eq!(m.sha256(), sha256(&bytes));
        }
        model
    }

    fn english_classifier() -> Vec<u8> {
        let file = ModelFile::read(models().join("opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin")).unwrap();
        file.entry(MODEL_ENTRY).unwrap().into_owned()
    }

    #[test]
    fn manifest_checks() {
        let c = english_classifier();
        let e = with_manifest("Component-Name=SentenceDetectorME\nuseAlphaNumericOptimization=false\n", &c);
        assert_eq!(e.unwrap_err().to_string(), "the model is for \"SentenceDetectorME\", not TokenizerME");
        let e = with_manifest("Component-Name=TokenizerME\n", &c).unwrap_err();
        assert!(matches!(e, ModelError::MissingOptimizationFlag));
        let gis = include_bytes!("../../../maxent/tests/fixtures/models/gis.model");
        let e = with_manifest("Component-Name=TokenizerME\nuseAlphaNumericOptimization=false\n", gis).unwrap_err();
        assert_eq!(e.to_string(), "the classifier's outcomes are [\"yes\", \"no\", \"maybe\"], not T and F");
    }

    #[test]
    fn the_optimization_flag_and_its_pattern() {
        let c = english_classifier();
        // On, in any case, with no pattern: the default, not the German one.
        let m = with_manifest("Component-Name=TokenizerME\nuseAlphaNumericOptimization=TRUE\n", &c).unwrap();
        assert_eq!(m.alphanumeric_pattern(), DEFAULT_PATTERN);
        assert!(m.alphanumeric().unwrap().test("abc"));
        assert!(!m.alphanumeric().unwrap().test("\u{e4}"));
        // Anything but "true" is off.
        let m = with_manifest("Component-Name=TokenizerME\nuseAlphaNumericOptimization=yes\n", &c).unwrap();
        assert!(m.alphanumeric().is_none());
        // A pattern the check cannot read is refused when it would be used...
        let manifest = "Component-Name=TokenizerME\nuseAlphaNumericOptimization=true\nalphaNumericPattern=^\\\\w+$\n";
        let e = with_manifest(manifest, &c).unwrap_err();
        assert_eq!(e.to_string(), "alphanumeric pattern \"^\\w+$\": not of the form ^[characters]+$");
        // ...and kept as text when it would not.
        let manifest = "Component-Name=TokenizerME\nuseAlphaNumericOptimization=false\nalphaNumericPattern=^\\\\w+$\n";
        assert_eq!(with_manifest(manifest, &c).unwrap().alphanumeric_pattern(), "^\\w+$");
    }
}
