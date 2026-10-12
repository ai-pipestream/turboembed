//! A sentence detector model: OpenNLP's `SentenceModel` with the settings
//! `SentenceDetectorFactory` reads from it.

use turbo_analysis::ModelRef;
use turbo_maxent::{Dictionary, MaxentModel, ModelFile};

use crate::abbreviation::AbbreviationIndex;
use crate::eos::EndOfSentence;
use crate::error::ModelError;

/// The classifier's entry in the model archive.
pub const MODEL_ENTRY: &str = "sent.model";
/// The abbreviation dictionary's entry, when the model has one.
pub const ABBREVIATIONS_ENTRY: &str = "abbreviations.dictionary";
/// The one factory class whose behaviour this port reproduces.
pub const FACTORY: &str = "opennlp.tools.sentdetect.SentenceDetectorFactory";

/// An opened `SentenceDetectorME` model.
///
/// The end characters are the manifest's `eosCharacters`. When the
/// manifest has none (or an empty value), they are `. ! ?` whatever the
/// language: `SentenceDetectorFactory` looks the language default up with
/// a language code it never sets on a factory loaded from a file, so the
/// Portuguese and Thai defaults of `lang/Factory` never apply to a loaded
/// model. For the same reason a Thai model is read with the default
/// context generator, as the JVM reads it. Every model OpenNLP's factory
/// writes has the property.
#[derive(Debug)]
pub struct SentenceModel {
    language: String,
    use_token_end: bool,
    eos: EndOfSentence,
    maxent: MaxentModel,
    split: usize,
    abbreviations: Option<Dictionary>,
    index: Option<AbbreviationIndex>,
    model_ref: ModelRef,
}

impl SentenceModel {
    /// Reads the model from an opened model file, checking what OpenNLP
    /// checks: the component, the factory, a `useTokenEnd` property and the
    /// outcomes `s` and `n`.
    pub fn open(file: ModelFile) -> Result<SentenceModel, ModelError> {
        if file.component_name() != "SentenceDetectorME" {
            return Err(ModelError::Component(file.component_name().into()));
        }
        let manifest = file.manifest();
        if let Some(factory) = manifest.get("factory")
            && factory != FACTORY
        {
            return Err(ModelError::Factory(factory.into()));
        }
        // Boolean.valueOf: true for "true" in any case, false otherwise.
        let use_token_end =
            manifest.get("useTokenEnd").ok_or(ModelError::MissingUseTokenEnd)?.eq_ignore_ascii_case("true");
        let eos = match manifest.get("eosCharacters") {
            Some(chars) if !chars.is_empty() => EndOfSentence::from_chars(chars).map_err(ModelError::EosCharacters)?,
            _ => EndOfSentence::default(),
        };
        let maxent = file.maxent(MODEL_ENTRY)?;
        let split = maxent.outcome_index("s");
        let (Some(split), Some(_), 2) = (split, maxent.outcome_index("n"), maxent.num_outcomes()) else {
            return Err(ModelError::Outcomes(maxent.outcomes().map(String::from).collect()));
        };
        let abbreviations = file.dictionary(ABBREVIATIONS_ENTRY)?;
        let index = abbreviations.as_ref().map(AbbreviationIndex::new);
        let model_ref = ModelRef {
            id: format!("opennlp:SentenceDetectorME:{}", file.language()),
            revision: file.opennlp_version().into(),
            sha256: file.sha256().to_hex(),
        };
        Ok(SentenceModel {
            language: file.language().into(),
            use_token_end,
            eos,
            maxent,
            split,
            abbreviations,
            index,
            model_ref,
        })
    }

    /// The manifest's `Language`.
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Whether a sentence starts after the whitespace that follows its
    /// end character (`useTokenEnd`), or right after the end character.
    pub fn use_token_end(&self) -> bool {
        self.use_token_end
    }

    /// The characters that can end a sentence.
    pub fn eos(&self) -> &EndOfSentence {
        &self.eos
    }

    /// The classifier.
    pub fn maxent(&self) -> &MaxentModel {
        &self.maxent
    }

    /// The index of outcome `s` (a split); the other outcome is `n`.
    pub fn split_outcome(&self) -> usize {
        self.split
    }

    /// The abbreviation dictionary, when the model has one.
    pub fn abbreviations(&self) -> Option<&Dictionary> {
        self.abbreviations.as_ref()
    }

    pub(crate) fn abbreviation_index(&self) -> Option<&AbbreviationIndex> {
        self.index.as_ref()
    }

    /// What layers record as their model: `opennlp:SentenceDetectorME:`
    /// and the language, the OpenNLP version that trained it, and the
    /// SHA-256 of the model file.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }
}
