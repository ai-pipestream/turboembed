//! Feature-generator factory (`DoccatFactory`) and descriptor serialization for
//! document categorization models.

use std::sync::Arc;

use turbo_analysis::Fnv1a;
use turbo_sequence::PatternMode;

use crate::error::DoccatError;
use crate::features::{
    BAG_OF_WORDS_CLASS, BagOfWordsFeatureGenerator, BuiltinFeatureGenerator, DoccatContextGenerator,
    DoccatFeatureGenerator, EMOJI_CLASS, EmojiFeatureGenerator, NGRAM_CLASS, NGramFeatureGenerator,
};

/// Configures the [`DoccatContextGenerator`] feature extractors used by
/// [`MultiLabelDoccatModel`](crate::multilabel::MultiLabelDoccatModel) and
/// sparse feature selection.
#[derive(Clone, Debug)]
pub struct DoccatFactory {
    context_generator: DoccatContextGenerator,
}

impl Default for DoccatFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl DoccatFactory {
    /// Creates a `DoccatFactory` with the default feature generator
    /// ([`BagOfWordsFeatureGenerator::new`]).
    pub fn new() -> Self {
        Self { context_generator: DoccatContextGenerator::default() }
    }

    /// Creates a `DoccatFactory` from built-in feature generators.
    pub fn with_feature_generators(generators: Vec<BuiltinFeatureGenerator>) -> Self {
        Self { context_generator: DoccatContextGenerator::from_builtins(generators) }
    }

    /// Creates a `DoccatFactory` from user-supplied [`DoccatFeatureGenerator`]
    /// trait objects.
    pub fn with_custom_generators(generators: Vec<Arc<dyn DoccatFeatureGenerator>>) -> Self {
        Self { context_generator: DoccatContextGenerator::from_custom(generators) }
    }

    /// Creates a `DoccatFactory` from a pre-configured [`DoccatContextGenerator`].
    pub fn from_context_generator(context_generator: DoccatContextGenerator) -> Self {
        Self { context_generator }
    }

    /// Returns a reference to the underlying [`DoccatContextGenerator`].
    pub fn context_generator(&self) -> &DoccatContextGenerator {
        &self.context_generator
    }

    /// Returns the built-in feature generators configured on this factory.
    pub fn feature_generators(&self) -> &[BuiltinFeatureGenerator] {
        self.context_generator.builtins()
    }
}

pub(crate) fn feed_bytes(hash: &mut u64, bytes: &[u8]) {
    *hash = Fnv1a::with_seed(*hash).bytes(bytes).finish();
}

pub(crate) fn context_generator_config_hash(cg: &DoccatContextGenerator) -> u64 {
    let mut hash = Fnv1a::new().finish();
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

/// Encodes a [`BuiltinFeatureGenerator`] as its canonical descriptor string.
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

/// Parses a comma-separated feature generator specification into built-in
/// generators, defaulting to [`BagOfWordsFeatureGenerator::new`] when empty.
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
