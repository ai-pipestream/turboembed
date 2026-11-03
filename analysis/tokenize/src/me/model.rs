//! A TokenizerME model: OpenNLP's `TokenizerModel` and the parts of
//! `TokenizerFactory` a loaded model uses.

use std::fmt;
use std::path::Path;

use turbo_maxent::{Dictionary, JavaString, MaxentModel, ModelFile};

use super::alnum::{AlphaNumeric, DEFAULT_PATTERN, PatternError};
use super::context::AffixLimits;
use turbo_maxent::Digest;

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
    affixes: AffixLimits,
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

    /// Opens a trained `TokenizerME` archive produced by `turbo-train-components`
    /// (`TrainedTokenizerModel::package`).
    pub fn from_trained_archive(bytes: Vec<u8>) -> Result<TokenizerModel, ModelError> {
        TokenizerModel::open(ModelFile::from_trained_archive(bytes)?)
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
            affixes: AffixLimits::of(&maxent),
            maxent,
            outcomes,
            abbreviations,
            pattern,
            alphanumeric,
            sha256: file.sha256(),
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

    /// The length of the classifier's longest `p=` and `s=` predicates.
    pub fn affix_limits(&self) -> AffixLimits {
        self.affixes
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
    let names: Vec<&JavaString> = maxent.outcomes().collect();
    if names.len() != 2 || names.iter().any(|&n| *n != "T" && *n != "F") {
        return Err(ModelError::Outcomes(names.iter().map(|n| n.to_string_lossy()).collect()));
    }
    let outcome = |i: usize| Outcome {
        split: *names[i] == "T",
        index: maxent.outcome_index(names[i]).expect("the outcome's own name"),
    };
    Ok([outcome(0), outcome(1)])
}
