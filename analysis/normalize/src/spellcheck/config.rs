//! The engine's configuration, with OpenNLP's `SymSpellConfig` defaults.

use std::fmt;

use super::distance::Metric;

/// The largest count the engine keeps: OpenNLP's counts are Java `long`s
/// that saturate at `Long.MAX_VALUE`, and the compound scoring computes
/// with them, so counts here saturate at the same value.
pub const MAX_COUNT: u64 = i64::MAX as u64;

/// `corpus_word_count` value that makes the engine derive the corpus size
/// from the sum of the counts added.
pub const DERIVE_CORPUS_WORD_COUNT: u64 = 0;

/// How the deletes index is built and how candidates are verified. It is
/// fixed for the life of an engine.
///
/// - `max_dictionary_edit_distance` (default 2): the largest edit distance
///   the index answers; a lookup may not ask for more.
/// - `prefix_length` (default 7): only the first this many code points of
///   a word produce deletes, which keeps the index small for long words.
/// - `count_threshold` (default 1): a word is indexed once its count
///   reaches this.
/// - `metric` (default optimal string alignment): verifies candidates.
/// - `corpus_word_count` (default derived): the corpus size N of the
///   compound scoring; 0 derives it from the counts added.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymSpellConfig {
    max_dictionary_edit_distance: usize,
    prefix_length: usize,
    count_threshold: u64,
    metric: Metric,
    corpus_word_count: u64,
}

impl Default for SymSpellConfig {
    fn default() -> SymSpellConfig {
        SymSpellConfig {
            max_dictionary_edit_distance: 2,
            prefix_length: 7,
            count_threshold: 1,
            metric: Metric::DamerauOsa,
            corpus_word_count: DERIVE_CORPUS_WORD_COUNT,
        }
    }
}

impl SymSpellConfig {
    /// A builder that starts from the defaults.
    pub fn builder() -> SymSpellConfigBuilder {
        SymSpellConfigBuilder { config: SymSpellConfig::default() }
    }

    pub fn max_dictionary_edit_distance(&self) -> usize {
        self.max_dictionary_edit_distance
    }

    pub fn prefix_length(&self) -> usize {
        self.prefix_length
    }

    pub fn count_threshold(&self) -> u64 {
        self.count_threshold
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The fixed corpus size, or [`DERIVE_CORPUS_WORD_COUNT`].
    pub fn corpus_word_count(&self) -> u64 {
        self.corpus_word_count
    }
}

/// Sets the fields of a [`SymSpellConfig`]; `build` checks them together.
#[derive(Copy, Clone, Debug)]
pub struct SymSpellConfigBuilder {
    config: SymSpellConfig,
}

impl SymSpellConfigBuilder {
    pub fn max_dictionary_edit_distance(mut self, value: usize) -> Self {
        self.config.max_dictionary_edit_distance = value;
        self
    }

    pub fn prefix_length(mut self, value: usize) -> Self {
        self.config.prefix_length = value;
        self
    }

    pub fn count_threshold(mut self, value: u64) -> Self {
        self.config.count_threshold = value;
        self
    }

    pub fn metric(mut self, value: Metric) -> Self {
        self.config.metric = value;
        self
    }

    pub fn corpus_word_count(mut self, value: u64) -> Self {
        self.config.corpus_word_count = value;
        self
    }

    /// The configuration, if its fields are consistent: a prefix length
    /// of at least 1 and greater than the maximum edit distance, a count
    /// threshold of at least 1, and counts no larger than [`MAX_COUNT`].
    pub fn build(self) -> Result<SymSpellConfig, ConfigError> {
        let c = self.config;
        if c.prefix_length < 1 {
            return Err(ConfigError::PrefixLengthZero);
        }
        if c.count_threshold < 1 {
            return Err(ConfigError::CountThresholdZero);
        }
        if c.prefix_length <= c.max_dictionary_edit_distance {
            return Err(ConfigError::PrefixNotLongerThanDistance {
                prefix_length: c.prefix_length,
                max_dictionary_edit_distance: c.max_dictionary_edit_distance,
            });
        }
        if c.count_threshold > MAX_COUNT || c.corpus_word_count > MAX_COUNT {
            return Err(ConfigError::CountTooLarge);
        }
        Ok(c)
    }
}

/// Why a configuration was refused.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    PrefixLengthZero,
    CountThresholdZero,
    PrefixNotLongerThanDistance { prefix_length: usize, max_dictionary_edit_distance: usize },
    CountTooLarge,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::PrefixLengthZero => write!(f, "prefixLength must be >= 1"),
            ConfigError::CountThresholdZero => write!(f, "countThreshold must be >= 1"),
            ConfigError::PrefixNotLongerThanDistance { prefix_length, max_dictionary_edit_distance } => write!(
                f,
                "prefixLength ({prefix_length}) must be > maxDictionaryEditDistance ({max_dictionary_edit_distance})"
            ),
            ConfigError::CountTooLarge => write!(f, "a count must not exceed {MAX_COUNT}"),
        }
    }
}

impl std::error::Error for ConfigError {}
