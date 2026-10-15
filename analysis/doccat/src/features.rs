//! OpenNLP's document categorizer feature generators (`FeatureGenerator`,
//! `DoccatFeatureGenerator`, `BagOfWordsFeatureGenerator`,
//! `NGramFeatureGenerator`, `EmojiFeatureGenerator`, and
//! `DoccatContextGenerator`).

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use turbo_sequence::{PatternMode, StringPattern};

use crate::error::DoccatError;

/// Canonical Java class name of `BagOfWordsFeatureGenerator`.
pub const BAG_OF_WORDS_CLASS: &str = "opennlp.tools.doccat.BagOfWordsFeatureGenerator";
/// Canonical Java class name of `NGramFeatureGenerator`.
pub const NGRAM_CLASS: &str = "opennlp.tools.doccat.NGramFeatureGenerator";
/// Canonical Java class name of `EmojiFeatureGenerator`.
pub const EMOJI_CLASS: &str = "opennlp.tools.doccat.EmojiFeatureGenerator";

/// A `FeatureGenerator` of `opennlp.tools.doccat` as exported in an SVM
/// document categorizer sidecar.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FeatureGenerator {
    /// `BagOfWordsFeatureGenerator()`: `bow=` and the token, for every
    /// token.
    BagOfWords,
    /// `NGramFeatureGenerator(min, max)`: from each token, `ng=` and then
    /// `:` and a token for each of the next `min` to `max` tokens, one
    /// feature per length.
    NGram { min: u32, max: u32 },
}

/// The range `NGramFeatureGenerator` refuses: a bound below 1 or a minimum
/// above the maximum.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NGramRangeError {
    pub min: i64,
    pub max: i64,
}

impl fmt::Display for NGramRangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.min > 0 && self.max > 0 {
            write!(f, "n-gram range {}..={}: the minimum is above the maximum", self.min, self.max)
        } else {
            write!(f, "n-gram range {}..={}: both bounds must be at least 1", self.min, self.max)
        }
    }
}

impl std::error::Error for NGramRangeError {}

impl FeatureGenerator {
    /// `new NGramFeatureGenerator(min, max)`, with its checks.
    pub fn ngram(min: i64, max: i64) -> Result<FeatureGenerator, NGramRangeError> {
        if min > 0 && max > 0 && min <= max && max <= i64::from(u32::MAX) {
            Ok(FeatureGenerator::NGram { min: min as u32, max: max as u32 })
        } else {
            Err(NGramRangeError { min, max })
        }
    }

    /// Calls `emit` with each feature of `tokens`, built in `buf`.
    pub fn generate(&self, tokens: &[&str], buf: &mut String, emit: &mut impl FnMut(&str)) {
        match *self {
            FeatureGenerator::BagOfWords => {
                for t in tokens {
                    buf.clear();
                    buf.push_str("bow=");
                    buf.push_str(t);
                    emit(buf);
                }
            }
            FeatureGenerator::NGram { min, max } => {
                let (min, max) = (min as usize, max as usize);
                if tokens.len() < min {
                    return;
                }
                for i in 0..=tokens.len() - min {
                    buf.clear();
                    buf.push_str("ng=");
                    for (y, t) in tokens[i..].iter().take(max).enumerate() {
                        buf.push(':');
                        buf.push_str(t);
                        if y + 1 >= min {
                            emit(buf);
                        }
                    }
                }
            }
        }
    }
}

/// Every feature of every generator, in turn.
pub fn for_each_feature(
    generators: &[FeatureGenerator],
    tokens: &[&str],
    buf: &mut String,
    mut emit: impl FnMut(&str),
) {
    for g in generators {
        g.generate(tokens, buf, &mut emit);
    }
}

/// Trait for extracting document-level context features from tokenized text,
/// ported from `opennlp.tools.doccat.FeatureGenerator`.
pub trait DoccatFeatureGenerator: Send + Sync + fmt::Debug {
    /// Appends features extracted from `text` and `extra_information` to `out`.
    fn extract_features(&self, text: &[&str], extra_information: &BTreeMap<String, String>, out: &mut Vec<String>);

    /// Convenience helper that returns the extracted features as a new `Vec<String>`.
    fn collect_features(&self, text: &[&str], extra_information: &BTreeMap<String, String>) -> Vec<String> {
        let mut out = Vec::new();
        self.extract_features(text, extra_information, &mut out);
        out
    }
}

impl DoccatFeatureGenerator for FeatureGenerator {
    fn extract_features(&self, text: &[&str], _extra_information: &BTreeMap<String, String>, out: &mut Vec<String>) {
        let mut buf = String::new();
        self.generate(text, &mut buf, &mut |f| out.push(f.to_owned()));
    }
}

/// Generates `bow=<word>` features for each token in a document, ported from
/// `opennlp.tools.doccat.BagOfWordsFeatureGenerator`.
///
/// When `use_only_all_letter_tokens` is `true`, only tokens for which
/// `StringPattern::recognize(word, pattern_mode).is_all_letter()` is `true`
/// emit a `bow=` feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BagOfWordsFeatureGenerator {
    use_only_all_letter_tokens: bool,
    pattern_mode: PatternMode,
}

impl Default for BagOfWordsFeatureGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl BagOfWordsFeatureGenerator {
    /// Creates a `BagOfWordsFeatureGenerator` that includes all tokens
    /// (`use_only_all_letter_tokens = false`).
    pub const fn new() -> Self {
        Self { use_only_all_letter_tokens: false, pattern_mode: PatternMode::CodePoints }
    }

    /// Creates a `BagOfWordsFeatureGenerator` configured with
    /// `use_only_all_letter_tokens`.
    pub const fn with_only_all_letter_tokens(use_only_all_letter_tokens: bool) -> Self {
        Self { use_only_all_letter_tokens, pattern_mode: PatternMode::CodePoints }
    }

    /// Creates a `BagOfWordsFeatureGenerator` that emits features only for
    /// all-letter tokens (`use_only_all_letter_tokens = true`).
    pub const fn only_all_letter_tokens() -> Self {
        Self::with_only_all_letter_tokens(true)
    }

    /// Sets the `PatternMode` (`CodePoints` or `Utf16Units`) used when
    /// checking `is_all_letter()`.
    pub const fn with_pattern_mode(mut self, mode: PatternMode) -> Self {
        self.pattern_mode = mode;
        self
    }

    /// Returns whether only all-letter tokens are included.
    pub const fn uses_only_all_letter_tokens(&self) -> bool {
        self.use_only_all_letter_tokens
    }

    /// Returns the `PatternMode` used when checking all-letter tokens.
    pub const fn pattern_mode(&self) -> PatternMode {
        self.pattern_mode
    }
}

impl DoccatFeatureGenerator for BagOfWordsFeatureGenerator {
    fn extract_features(&self, text: &[&str], _extra_information: &BTreeMap<String, String>, out: &mut Vec<String>) {
        out.reserve(text.len());
        for &word in text {
            if self.use_only_all_letter_tokens {
                let pattern = StringPattern::recognize(word, self.pattern_mode);
                if pattern.is_all_letter() {
                    let mut feat = String::with_capacity(4 + word.len());
                    feat.push_str("bow=");
                    feat.push_str(word);
                    out.push(feat);
                }
            } else {
                let mut feat = String::with_capacity(4 + word.len());
                feat.push_str("bow=");
                feat.push_str(word);
                out.push(feat);
            }
        }
    }
}

/// Generates token n-gram features (`ng=:<t1>:...:<tk>`) for window lengths in
/// `min_gram..=max_gram`, ported from `opennlp.tools.doccat.NGramFeatureGenerator`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NGramFeatureGenerator {
    min_gram: usize,
    max_gram: usize,
}

impl Default for NGramFeatureGenerator {
    /// Default bigram generator `(2, 2)`, matching `new NGramFeatureGenerator()`.
    fn default() -> Self {
        Self { min_gram: 2, max_gram: 2 }
    }
}

impl NGramFeatureGenerator {
    /// Creates an `NGramFeatureGenerator` for n-grams of length
    /// `min_gram..=max_gram`.
    ///
    /// Returns [`DoccatError::InvalidNGramRange`] if `min_gram == 0` or
    /// `min_gram > max_gram` (matching OpenNLP's `InvalidFormatException`).
    pub fn new(min_gram: usize, max_gram: usize) -> Result<Self, DoccatError> {
        if min_gram > 0 && min_gram <= max_gram {
            Ok(Self { min_gram, max_gram })
        } else {
            Err(DoccatError::InvalidNGramRange { min_gram, max_gram })
        }
    }

    /// Returns the minimum n-gram length (`>= 1`).
    pub const fn min_gram(&self) -> usize {
        self.min_gram
    }

    /// Returns the maximum n-gram length (`>= min_gram`).
    pub const fn max_gram(&self) -> usize {
        self.max_gram
    }
}

impl DoccatFeatureGenerator for NGramFeatureGenerator {
    fn extract_features(&self, text: &[&str], _extra_information: &BTreeMap<String, String>, out: &mut Vec<String>) {
        if text.len() < self.min_gram {
            return;
        }
        let last_start = text.len() - self.min_gram;
        let mut sb = String::new();
        for i in 0..=last_start {
            sb.clear();
            sb.push_str("ng=");
            let limit = self.max_gram.min(text.len() - i);
            for j in 0..limit {
                sb.push(':');
                sb.push_str(text[i + j]);
                let gram_count = j + 1;
                if gram_count >= self.min_gram {
                    out.push(sb.clone());
                }
            }
        }
    }
}

/// Generates coarse emoji sentiment, entity type, category, and ISO region
/// features (`emojiSentiment=`, `emojiType=`, `emojiCategory=`, `emojiRegion=`)
/// for emoji tokens, ported from `opennlp.tools.doccat.EmojiFeatureGenerator`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct EmojiFeatureGenerator;

impl EmojiFeatureGenerator {
    /// Creates an `EmojiFeatureGenerator` over OpenNLP's bundled and derived
    /// emoji annotations.
    pub const fn new() -> Self {
        Self
    }
}

impl DoccatFeatureGenerator for EmojiFeatureGenerator {
    fn extract_features(&self, text: &[&str], _extra_information: &BTreeMap<String, String>, out: &mut Vec<String>) {
        for &token in text {
            turbo_normalize::emoji::features(token, out);
        }
    }
}

/// One of the three built-in OpenNLP `doccat` feature generators that can be
/// named in a model manifest's `doccat.featureGenerators` property.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinFeatureGenerator {
    /// `opennlp.tools.doccat.BagOfWordsFeatureGenerator`.
    BagOfWords(BagOfWordsFeatureGenerator),
    /// `opennlp.tools.doccat.NGramFeatureGenerator`.
    NGram(NGramFeatureGenerator),
    /// `opennlp.tools.doccat.EmojiFeatureGenerator`.
    Emoji(EmojiFeatureGenerator),
}

impl BuiltinFeatureGenerator {
    /// Returns the canonical Java class name written to `doccat.featureGenerators`.
    pub const fn class_name(&self) -> &'static str {
        match self {
            BuiltinFeatureGenerator::BagOfWords(_) => BAG_OF_WORDS_CLASS,
            BuiltinFeatureGenerator::NGram(_) => NGRAM_CLASS,
            BuiltinFeatureGenerator::Emoji(_) => EMOJI_CLASS,
        }
    }

    /// Instantiates a built-in feature generator with its default constructor
    /// from its canonical Java class name, matching `ExtensionLoader.instantiateExtension`
    /// in `DoccatFactory.getFeatureGenerators()`.
    pub fn from_class_name(class_name: &str) -> Result<Self, DoccatError> {
        match class_name.trim() {
            BAG_OF_WORDS_CLASS => Ok(BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::new())),
            NGRAM_CLASS => Ok(BuiltinFeatureGenerator::NGram(NGramFeatureGenerator::default())),
            EMOJI_CLASS => Ok(BuiltinFeatureGenerator::Emoji(EmojiFeatureGenerator::new())),
            other => Err(DoccatError::UnknownFeatureGenerator(other.to_owned())),
        }
    }
}

impl DoccatFeatureGenerator for BuiltinFeatureGenerator {
    fn extract_features(&self, text: &[&str], extra_information: &BTreeMap<String, String>, out: &mut Vec<String>) {
        match self {
            BuiltinFeatureGenerator::BagOfWords(g) => g.extract_features(text, extra_information, out),
            BuiltinFeatureGenerator::NGram(g) => g.extract_features(text, extra_information, out),
            BuiltinFeatureGenerator::Emoji(g) => g.extract_features(text, extra_information, out),
        }
    }
}

/// Context generator that chains one or more [`DoccatFeatureGenerator`]s in order,
/// ported from `opennlp.tools.doccat.DocumentCategorizerContextGenerator`.
#[derive(Clone, Debug)]
pub struct DoccatContextGenerator {
    builtins: Vec<BuiltinFeatureGenerator>,
    custom: Vec<Arc<dyn DoccatFeatureGenerator>>,
}

impl Default for DoccatContextGenerator {
    /// Default context generator with a single `BagOfWordsFeatureGenerator`.
    fn default() -> Self {
        Self {
            builtins: vec![BuiltinFeatureGenerator::BagOfWords(BagOfWordsFeatureGenerator::new())],
            custom: Vec::new(),
        }
    }
}

impl DoccatContextGenerator {
    /// Creates a `DoccatContextGenerator` from a list of built-in generators.
    pub fn from_builtins(builtins: Vec<BuiltinFeatureGenerator>) -> Self {
        Self { builtins, custom: Vec::new() }
    }

    /// Creates a `DoccatContextGenerator` from custom [`DoccatFeatureGenerator`]s.
    pub fn from_custom(custom: Vec<Arc<dyn DoccatFeatureGenerator>>) -> Self {
        Self { builtins: Vec::new(), custom }
    }

    /// Appends a custom [`DoccatFeatureGenerator`] to run after the built-in generators.
    pub fn with_custom(mut self, generator: Arc<dyn DoccatFeatureGenerator>) -> Self {
        self.custom.push(generator);
        self
    }

    /// Returns the built-in generators configured on this context generator.
    pub fn builtins(&self) -> &[BuiltinFeatureGenerator] {
        &self.builtins
    }

    /// Returns the custom generators configured on this context generator.
    pub fn custom(&self) -> &[Arc<dyn DoccatFeatureGenerator>] {
        &self.custom
    }

    /// Extracts all context features for `text` and `extra_information` into `out`.
    pub fn extract_features(&self, text: &[&str], extra_information: &BTreeMap<String, String>, out: &mut Vec<String>) {
        out.clear();
        for g in &self.builtins {
            g.extract_features(text, extra_information, out);
        }
        for g in &self.custom {
            g.extract_features(text, extra_information, out);
        }
    }

    /// Returns all context features for `text` and `extra_information`
    /// (`DocumentCategorizerContextGenerator.getContext`).
    pub fn context(&self, text: &[&str], extra_information: &BTreeMap<String, String>) -> Vec<String> {
        let mut out = Vec::new();
        self.extract_features(text, extra_information, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features(generators: &[FeatureGenerator], tokens: &[&str]) -> Vec<String> {
        let mut out = Vec::new();
        for_each_feature(generators, tokens, &mut String::new(), |f| out.push(f.to_string()));
        out
    }

    const TOKENS: [&str; 7] = ["a", "b", "c", "d", "e", "f", "g"];

    // BagOfWordsFeatureGeneratorTest: testEmpty, testUseAllTokens.
    #[test]
    fn bag_of_words_prefixes_every_token() {
        let bow = [FeatureGenerator::BagOfWords];
        assert!(features(&bow, &[]).is_empty());
        assert_eq!(
            features(&bow, &["it", "is", "12.345", "feet", "long"]),
            ["bow=it", "bow=is", "bow=12.345", "bow=feet", "bow=long"]
        );
    }

    // NGramFeatureGeneratorTest: testEmpty and the five ranges.
    #[test]
    fn ngrams_join_tokens_after_a_colon_each() {
        let ngram = |min, max| [FeatureGenerator::ngram(min, max).unwrap()];
        assert!(features(&ngram(2, 2), &[]).is_empty());
        assert_eq!(features(&ngram(1, 1), &TOKENS), ["ng=:a", "ng=:b", "ng=:c", "ng=:d", "ng=:e", "ng=:f", "ng=:g"]);
        assert_eq!(features(&ngram(2, 2), &TOKENS), ["ng=:a:b", "ng=:b:c", "ng=:c:d", "ng=:d:e", "ng=:e:f", "ng=:f:g"]);
        assert_eq!(features(&ngram(3, 3), &TOKENS), ["ng=:a:b:c", "ng=:b:c:d", "ng=:c:d:e", "ng=:d:e:f", "ng=:e:f:g"]);
        assert_eq!(
            features(&ngram(1, 2), &TOKENS),
            [
                "ng=:a", "ng=:a:b", "ng=:b", "ng=:b:c", "ng=:c", "ng=:c:d", "ng=:d", "ng=:d:e", "ng=:e", "ng=:e:f",
                "ng=:f", "ng=:f:g", "ng=:g"
            ]
        );
        assert_eq!(
            features(&ngram(1, 3), &TOKENS),
            [
                "ng=:a",
                "ng=:a:b",
                "ng=:a:b:c",
                "ng=:b",
                "ng=:b:c",
                "ng=:b:c:d",
                "ng=:c",
                "ng=:c:d",
                "ng=:c:d:e",
                "ng=:d",
                "ng=:d:e",
                "ng=:d:e:f",
                "ng=:e",
                "ng=:e:f",
                "ng=:e:f:g",
                "ng=:f",
                "ng=:f:g",
                "ng=:g"
            ]
        );
        assert!(features(&ngram(2, 3), &["x"]).is_empty());
        assert_eq!(features(&ngram(2, 3), &["x", "y"]), ["ng=:x:y"]);
    }

    #[test]
    fn generators_run_in_turn() {
        let g = [FeatureGenerator::BagOfWords, FeatureGenerator::ngram(2, 2).unwrap()];
        assert_eq!(features(&g, &["x", "y"]), ["bow=x", "bow=y", "ng=:x:y"]);
    }

    // NGramFeatureGeneratorTest: testInvalidGramSize1 and 2.
    #[test]
    fn an_ngram_range_needs_positive_ordered_bounds() {
        assert_eq!(FeatureGenerator::ngram(0, 2), Err(NGramRangeError { min: 0, max: 2 }));
        assert_eq!(FeatureGenerator::ngram(2, -1), Err(NGramRangeError { min: 2, max: -1 }));
        assert_eq!(FeatureGenerator::ngram(3, 2), Err(NGramRangeError { min: 3, max: 2 }));
        assert!(FeatureGenerator::ngram(3, 2).unwrap_err().to_string().contains("above the maximum"));
        assert!(FeatureGenerator::ngram(0, 2).unwrap_err().to_string().contains("at least 1"));
    }
}
