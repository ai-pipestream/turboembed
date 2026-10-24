//! Document categorizer model (`DoccatModel`), ported from
//! `opennlp.tools.doccat.DoccatModel`.

use std::sync::Arc;

use turbo_analysis::{ModelRef, Tag, TagSet};
use turbo_maxent::{MaxentModel, ModelFile};

use crate::error::DoccatError;
use crate::features::{
    BAG_OF_WORDS_CLASS, BagOfWordsFeatureGenerator, BuiltinFeatureGenerator, DoccatContextGenerator, EMOJI_CLASS,
    EmojiFeatureGenerator, NGRAM_CLASS, NGramFeatureGenerator,
};
use turbo_sequence::PatternMode;

/// `DoccatModel.COMPONENT_NAME`.
pub const COMPONENT_NAME: &str = "DocumentCategorizerME";
/// `DoccatModel.DOCCAT_MODEL_ENTRY_NAME`.
pub const MODEL_ENTRY: &str = "doccat.model";
/// Canonical class name of `DoccatFactory`.
pub const FACTORY_NAME: &str = "opennlp.tools.doccat.DoccatFactory";
/// Manifest property storing the comma-separated feature generator class names.
pub const FEATURE_GENERATORS_KEY: &str = "doccat.featureGenerators";

/// An opened or trained `DoccatModel` holding the `doccat.model` classifier,
/// the category `TagSet` in model outcome order, and the configured feature
/// generators from `DoccatFactory`.
#[derive(Clone, Debug)]
pub struct DoccatModel {
    language: String,
    maxent: Arc<MaxentModel>,
    categories: Arc<TagSet>,
    context_generator: DoccatContextGenerator,
    model_ref: ModelRef,
}

impl DoccatModel {
    /// Opens a `DoccatModel` from an OpenNLP `.bin` `ModelFile`.
    ///
    /// Validates `Component-Name == "DocumentCategorizerME"`, checks `factory`
    /// if present in the manifest, parses `doccat.featureGenerators` (or
    /// defaults to `BagOfWordsFeatureGenerator`), and loads `doccat.model`.
    pub fn open(file: ModelFile) -> Result<Self, DoccatError> {
        if file.component_name() != COMPONENT_NAME {
            return Err(DoccatError::Component(file.component_name().into()));
        }
        if let Some(factory) = file.manifest().get("factory")
            && factory != FACTORY_NAME
        {
            return Err(DoccatError::Factory(factory.into()));
        }
        let builtins = match file.manifest().get(FEATURE_GENERATORS_KEY) {
            Some(prop) => parse_feature_generators(prop)?,
            None => vec![BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::new())],
        };
        let maxent = file.maxent(MODEL_ENTRY)?;
        if let Err(outcome) = maxent.outcome_texts() {
            return Err(DoccatError::OutcomeNotText(outcome.clone()));
        }
        let labels: Vec<&str> = (0..maxent.num_outcomes())
            .map(|i| maxent.outcome_str(i).expect("outcomes were verified to be valid UTF-8"))
            .collect();
        let categories = TagSet::new("opennlp:doccat", &labels);
        let language = file.language().to_owned();
        let model_ref = ModelRef {
            id: format!("opennlp:{COMPONENT_NAME}:{language}"),
            revision: file.opennlp_version().into(),
            sha256: file.sha256().to_hex(),
        };
        Ok(Self {
            language,
            maxent: Arc::new(maxent),
            categories,
            context_generator: DoccatContextGenerator::from_builtins(builtins),
            model_ref,
        })
    }

    /// Constructs an in-memory `DoccatModel` from a `MaxentModel` and a
    /// `DoccatContextGenerator`.
    pub fn from_maxent(
        language: impl Into<String>,
        maxent: MaxentModel,
        context_generator: DoccatContextGenerator,
    ) -> Result<Self, DoccatError> {
        if let Err(outcome) = maxent.outcome_texts() {
            return Err(DoccatError::OutcomeNotText(outcome.clone()));
        }
        let labels: Vec<&str> = (0..maxent.num_outcomes())
            .map(|i| maxent.outcome_str(i).expect("outcomes were verified to be valid UTF-8"))
            .collect();
        let categories = TagSet::new("opennlp:doccat", &labels);
        let language = language.into();
        let model_ref = ModelRef {
            id: format!("opennlp:{COMPONENT_NAME}:{language}"),
            revision: "3.0.0".into(),
            sha256: String::new(),
        };
        Ok(Self { language, maxent: Arc::new(maxent), categories, context_generator, model_ref })
    }

    /// Replaces the `DoccatContextGenerator` (for example, when attaching
    /// custom `NGramFeatureGenerator` window sizes or `BagOfWordsFeatureGenerator`
    /// all-letter settings after loading a `.bin` archive whose manifest stores
    /// only class names).
    pub fn with_context_generator(mut self, context_generator: DoccatContextGenerator) -> Self {
        self.context_generator = context_generator;
        self
    }

    /// Returns the model's language code (`Language-Code` in `manifest.properties`).
    pub fn language(&self) -> &str {
        &self.language
    }

    /// Returns the underlying `MaxentModel` (`doccat.model`).
    pub fn maxent(&self) -> &MaxentModel {
        &self.maxent
    }

    /// Returns the closed `TagSet` of category names in outcome order.
    pub fn categories(&self) -> &Arc<TagSet> {
        &self.categories
    }

    /// Returns the number of categories in the model (`getNumberOfCategories`).
    pub fn num_categories(&self) -> usize {
        self.categories.len()
    }

    /// Returns the category name at `index` (`getCategory`).
    pub fn category(&self, index: usize) -> Option<&str> {
        if index < self.categories.len() { Some(self.categories.label(Tag(index as u16))) } else { None }
    }

    /// Returns the outcome index of `category` (`getIndex`).
    pub fn index(&self, category: &str) -> Option<usize> {
        self.categories.tag(category).map(|t| t.0 as usize)
    }

    /// Returns the configured [`DoccatContextGenerator`].
    pub fn context_generator(&self) -> &DoccatContextGenerator {
        &self.context_generator
    }

    /// Returns the built-in feature generators configured in `DoccatFactory`.
    pub fn feature_generators(&self) -> &[BuiltinFeatureGenerator] {
        self.context_generator.builtins()
    }

    /// Returns the [`ModelRef`] recorded in layer provenance.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }

    /// Computes a deterministic FNV-1a configuration hash over the model's
    /// language, category labels, and feature generator parameters.
    pub(crate) fn config_hash_value(&self) -> u64 {
        let mut hash = context_generator_config_hash(&self.context_generator);
        feed_bytes(&mut hash, self.language.as_bytes());
        feed_bytes(&mut hash, &[0]);
        for i in 0..self.categories.len() {
            feed_bytes(&mut hash, self.categories.label(Tag(i as u16)).as_bytes());
            feed_bytes(&mut hash, &[0]);
        }
        hash
    }
}

pub(crate) fn feed_bytes(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash = (*hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
}

/// Deterministic FNV-1a hash over a [`DoccatContextGenerator`]'s built-in and
/// custom feature generators.
pub(crate) fn context_generator_config_hash(cg: &DoccatContextGenerator) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for g in cg.builtins() {
        match g {
            BuiltinFeatureGenerator::BagOfWords(g) => {
                feed_bytes(
                    &mut hash,
                    &[
                        0,
                        u8::from(g.uses_only_all_letter_tokens()),
                        match g.pattern_mode() {
                            PatternMode::Utf16Units => 0,
                            PatternMode::CodePoints => 1,
                        },
                    ],
                );
            }
            BuiltinFeatureGenerator::NGram(g) => {
                feed_bytes(&mut hash, &[1]);
                feed_bytes(&mut hash, &(g.min_gram() as u64).to_le_bytes());
                feed_bytes(&mut hash, &(g.max_gram() as u64).to_le_bytes());
            }
            BuiltinFeatureGenerator::Emoji(_) => {
                feed_bytes(&mut hash, &[2]);
            }
        }
    }
    if !cg.custom().is_empty() {
        feed_bytes(&mut hash, &[3]);
        feed_bytes(&mut hash, &(cg.custom().len() as u64).to_le_bytes());
        for g in cg.custom() {
            let dbg = format!("{g:?}");
            feed_bytes(&mut hash, dbg.as_bytes());
            feed_bytes(&mut hash, &[0]);
        }
    }
    hash
}

/// Encodes a [`BuiltinFeatureGenerator`] as its canonical Java class name when
/// using default settings, or with its non-default parameters appended so
/// archives round-trip `NGramFeatureGenerator` bounds and
/// `BagOfWordsFeatureGenerator` token filters losslessly.
pub(crate) fn encode_single_generator(g: &BuiltinFeatureGenerator) -> String {
    match g {
        BuiltinFeatureGenerator::BagOfWords(bow) => {
            if !bow.uses_only_all_letter_tokens() && bow.pattern_mode() == PatternMode::Utf16Units {
                BAG_OF_WORDS_CLASS.to_owned()
            } else {
                let mode = match bow.pattern_mode() {
                    PatternMode::Utf16Units => "utf16",
                    PatternMode::CodePoints => "codepoints",
                };
                format!("{BAG_OF_WORDS_CLASS}(letters={}:mode={mode})", bow.uses_only_all_letter_tokens())
            }
        }
        BuiltinFeatureGenerator::NGram(ng) => {
            if ng.min_gram() == 2 && ng.max_gram() == 2 {
                NGRAM_CLASS.to_owned()
            } else {
                format!("{NGRAM_CLASS}({}:{})", ng.min_gram(), ng.max_gram())
            }
        }
        BuiltinFeatureGenerator::Emoji(_) => EMOJI_CLASS.to_owned(),
    }
}

/// Encodes all feature generators of `cg` as a comma-separated specification,
/// returning an error if `cg` contains unserializable custom generators or no
/// built-in generators.
pub(crate) fn encode_feature_generators(cg: &DoccatContextGenerator) -> Result<String, String> {
    if !cg.custom().is_empty() {
        return Err("custom feature generators cannot be serialized in a model archive".into());
    }
    if cg.builtins().is_empty() {
        return Err("context_generator has no built-in feature generators to serialize".into());
    }
    let specs: Vec<String> = cg.builtins().iter().map(encode_single_generator).collect();
    Ok(specs.join(","))
}

/// Parses one feature generator class name or parameterized specification.
pub(crate) fn parse_single_generator(class_name: &str) -> Result<BuiltinFeatureGenerator, DoccatError> {
    let trimmed = class_name.trim();
    match trimmed {
        BAG_OF_WORDS_CLASS => Ok(BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::new())),
        NGRAM_CLASS => Ok(BuiltinFeatureGenerator::NGram(NGramFeatureGenerator::default())),
        EMOJI_CLASS => Ok(BuiltinFeatureGenerator::Emoji(EmojiFeatureGenerator::new())),
        other => {
            if let Some(inner) = other
                .strip_prefix(BAG_OF_WORDS_CLASS)
                .and_then(|s| s.strip_prefix('('))
                .and_then(|s| s.strip_suffix(')'))
            {
                let (letters_kv, mode_kv) =
                    inner.split_once(':').ok_or_else(|| DoccatError::UnknownFeatureGenerator(other.to_owned()))?;
                let letters = match letters_kv.strip_prefix("letters=") {
                    Some("true") => true,
                    Some("false") => false,
                    _ => return Err(DoccatError::UnknownFeatureGenerator(other.to_owned())),
                };
                let mode = match mode_kv.strip_prefix("mode=") {
                    Some("utf16") => PatternMode::Utf16Units,
                    Some("codepoints") => PatternMode::CodePoints,
                    _ => return Err(DoccatError::UnknownFeatureGenerator(other.to_owned())),
                };
                return Ok(BuiltinFeatureGenerator::BagOfWords(
                    BagOfWordsFeatureGenerator::with_only_all_letter_tokens(letters).with_pattern_mode(mode),
                ));
            }
            if let Some(inner) =
                other.strip_prefix(NGRAM_CLASS).and_then(|s| s.strip_prefix('(')).and_then(|s| s.strip_suffix(')'))
            {
                let (min_s, max_s) =
                    inner.split_once(':').ok_or_else(|| DoccatError::UnknownFeatureGenerator(other.to_owned()))?;
                let min_gram: usize =
                    min_s.parse().map_err(|_| DoccatError::UnknownFeatureGenerator(other.to_owned()))?;
                let max_gram: usize =
                    max_s.parse().map_err(|_| DoccatError::UnknownFeatureGenerator(other.to_owned()))?;
                return Ok(BuiltinFeatureGenerator::NGram(NGramFeatureGenerator::new(min_gram, max_gram)?));
            }
            Err(DoccatError::UnknownFeatureGenerator(other.to_owned()))
        }
    }
}

/// Parses the comma-separated `doccat.featureGenerators` manifest property,
/// matching `DoccatFactory.getFeatureGenerators()`.
pub fn parse_feature_generators(prop: &str) -> Result<Vec<BuiltinFeatureGenerator>, DoccatError> {
    let mut out = Vec::new();
    for part in prop.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            out.push(BuiltinFeatureGenerator::from_class_name(trimmed)?);
        }
    }
    if out.is_empty() {
        out.push(BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::new()));
    }
    Ok(out)
}
