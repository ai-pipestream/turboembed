//! Deterministic lexicon and negation-scope sentiment scorer (#320).
//!
//! Ports the rule-based sentiment model of Hutto and Gilbert (2014),
//! *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social
//! Media Text* (ICWSM-14), onto the layered analysis API:
//!
//! - [`VaderLexicon`] loads the MIT-licensed VADER lexicon (`vader_lexicon.txt`)
//!   or a caller-supplied TSV table (`token\tvalence`).
//! - [`LexiconSentiment::score_text`] runs in VADER-compatible raw-text mode
//!   and reproduces Python `vaderSentiment`'s `SentimentIntensityAnalyzer`
//!   scores (`neg`, `neu`, `pos`, `compound`) across its reference suite.
//! - [`LexiconSentiment::score_sentence`] and the [`Annotator`] implementation
//!   (`turbo-doccat:lexicon-sentiment`) read `opennlp:sentences` and
//!   `opennlp:tokens`, plus optional `opennlp:dependencies` and `turbo:emoji`,
//!   and write `turbo:opinions` ([`Opinions`], with exact token evidence spans)
//!   and sentence-level polarity in `turbo:categories<opennlp:sentences>`
//!   ([`Categories<Sentences>`]).
//! - Emoji scoring uses OpenNLP's own Apache-2.0 `emoji-annotations.txt`
//!   ([`turbo_normalize::emoji::opennlp_annotation`]) rather than any external
//!   CC BY-SA emoji ranking table.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use turbo_analysis::{Annotator, Bound, Error, Ix, IxRange, Key, Layer, ModelRef, Refs, Span, Tag, TagSet, View};
use turbo_analysis_layers::{Categories, Dependencies, Opinion, Opinions, Polarity, Sentences, Tokens};
use turbo_maxent::sha256;
use turbo_normalize::emoji::opennlp_annotation;
use turbo_normalize_layers::Emoji;

/// Empirically derived mean sentiment intensity rating increase for booster words.
pub const B_INCR: f64 = 0.293;

/// Empirically derived mean sentiment intensity rating decrease for dampener words.
pub const B_DECR: f64 = -0.293;

/// Empirically derived mean sentiment intensity rating increase for ALLCAPS emphasis.
pub const C_INCR: f64 = 0.733;

/// Negation multiplier applied to negated valence scores.
pub const N_SCALAR: f64 = -0.74;

/// Normalization constant (`alpha`) for computing the compound score in `[-1.0, 1.0]`.
pub const DEFAULT_ALPHA: f64 = 15.0;

/// Standard VADER compound threshold (`0.05`) for classifying positive (`>= 0.05`),
/// negative (`<= -0.05`), and neutral (`(-0.05, 0.05)`) sentences.
pub const COMPOUND_THRESHOLD: f64 = 0.05;

/// Named instance of `Categories<Sentences>` written by [`LexiconSentiment::bound`].
pub const SENTENCE_SENTIMENT: Key<Categories<Sentences>> = Key::named("sentiment");

mod annotate;
mod dependencies;
mod helpers;
mod rules;
mod scoring;
mod tables;
#[cfg(test)]
mod tests;
mod valence;
mod vocabulary;

use dependencies::*;
pub use helpers::{
    allcap_differential, is_all_caps, is_negation_word, normalize_score, scalar_inc_dec, strip_punc_if_word,
};
use helpers::{booster_value, round_half_to_even, special_case_value};
use rules::*;
use tables::*;
pub use vocabulary::VaderLexicon;

/// How negation scope is determined when scoring tokenized sentences.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum NegationMode {
    /// Use the `opennlp:dependencies` layer when present on the document,
    /// falling back to VADER's three-token preceding window when absent.
    #[default]
    Auto,
    /// Always use VADER's three-token preceding window, even if a dependency
    /// layer is present.
    Window,
    /// Use dependency edges when provided, falling back to VADER's three-token
    /// window only when no dependency layer is supplied.
    Dependency,
}

impl NegationMode {
    pub const fn model_id(self) -> &'static str {
        match self {
            NegationMode::Auto => "vader:lexicon:auto",
            NegationMode::Window => "vader:lexicon:window",
            NegationMode::Dependency => "vader:lexicon:dependency",
        }
    }
}

/// VADER polarity and compound scores for a text or sentence.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct VaderScore {
    /// Negative proportion (`0.0..=1.0`).
    pub neg: f64,
    /// Neutral proportion (`0.0..=1.0`).
    pub neu: f64,
    /// Positive proportion (`0.0..=1.0`).
    pub pos: f64,
    /// Normalized, weighted composite score (`-1.0..=1.0`).
    pub compound: f64,
}

impl VaderScore {
    /// `neg` rounded to 3 decimal places as `vaderSentiment` reports it.
    pub fn neg_rounded(self) -> f64 {
        round_half_to_even(self.neg, 3)
    }

    /// `neu` rounded to 3 decimal places as `vaderSentiment` reports it.
    pub fn neu_rounded(self) -> f64 {
        round_half_to_even(self.neu, 3)
    }

    /// `pos` rounded to 3 decimal places as `vaderSentiment` reports it.
    pub fn pos_rounded(self) -> f64 {
        round_half_to_even(self.pos, 3)
    }

    /// `compound` rounded to 4 decimal places as `vaderSentiment` reports it.
    pub fn compound_rounded(self) -> f64 {
        round_half_to_even(self.compound, 4)
    }

    /// Classifies `compound` using VADER's standard `+/- 0.05` threshold.
    pub fn polarity(self) -> Polarity {
        if self.compound >= COMPOUND_THRESHOLD {
            Polarity::Positive
        } else if self.compound <= -COMPOUND_THRESHOLD {
            Polarity::Negative
        } else {
            Polarity::Neutral
        }
    }
}

/// Detailed sentiment result for one tokenized sentence.
#[derive(Clone, Debug, PartialEq)]
pub struct SentenceSentiment {
    /// Sentence-level `neg`, `neu`, `pos`, and `compound` scores.
    pub score: VaderScore,
    /// Opinion evidence spans (`IxRange<Tokens>`) that contributed non-zero
    /// sentiment in this sentence, in token order.
    pub opinions: Vec<Opinion>,
    /// Whether dependency-based negation scope was used for this sentence.
    pub used_dependencies: bool,
}

/// Deterministic VADER-style lexicon and negation sentiment scorer (#320).
#[derive(Clone, Debug)]
pub struct LexiconSentiment {
    lexicon: Arc<VaderLexicon>,
    negation_mode: NegationMode,
    use_opennlp_emoji: bool,
    alpha: f64,
    categories: Arc<TagSet>,
    model_ref: ModelRef,
}

impl Default for LexiconSentiment {
    fn default() -> Self {
        Self::new()
    }
}

impl LexiconSentiment {
    /// Creates a `LexiconSentiment` scorer with the bundled VADER lexicon,
    /// [`NegationMode::Auto`], and OpenNLP emoji sentiment annotations enabled.
    pub fn new() -> LexiconSentiment {
        Self::with_lexicon(VaderLexicon::bundled())
    }

    /// Creates a `LexiconSentiment` scorer with a custom [`VaderLexicon`].
    pub fn with_lexicon(lexicon: Arc<VaderLexicon>) -> LexiconSentiment {
        let negation_mode = NegationMode::Auto;
        let model_ref =
            ModelRef { id: negation_mode.model_id().into(), revision: "1".into(), sha256: lexicon.sha256_hex().into() };
        let categories = TagSet::new("sentiment:polarity", &["positive", "negative", "neutral"]);
        LexiconSentiment {
            lexicon,
            negation_mode,
            use_opennlp_emoji: true,
            alpha: DEFAULT_ALPHA,
            categories,
            model_ref,
        }
    }

    /// Configures how negation scope is resolved ([`NegationMode::Auto`],
    /// [`NegationMode::Window`], or [`NegationMode::Dependency`]).
    pub fn with_negation_mode(mut self, mode: NegationMode) -> LexiconSentiment {
        self.negation_mode = mode;
        self.model_ref =
            ModelRef { id: mode.model_id().into(), revision: "1".into(), sha256: self.lexicon.sha256_hex().into() };
        self
    }

    /// Enables or disables fallback to OpenNLP's `emoji-annotations.txt`
    /// coarse sentiment scores (`-2..=2`) for Unicode emoji tokens.
    pub fn with_opennlp_emoji(mut self, enabled: bool) -> LexiconSentiment {
        self.use_opennlp_emoji = enabled;
        self
    }

    /// Returns the negation mode.
    pub fn negation_mode(&self) -> NegationMode {
        self.negation_mode
    }

    /// Returns the underlying lexicon.
    pub fn lexicon(&self) -> &VaderLexicon {
        &self.lexicon
    }

    /// Returns the `TagSet` (`["positive", "negative", "neutral"]`) used for
    /// `Categories<Sentences>`.
    pub fn categories_tagset(&self) -> &Arc<TagSet> {
        &self.categories
    }

    /// Binds the second output (`Categories<Sentences>`) to the `"sentiment"`
    /// instance ([`SENTENCE_SENTIMENT`]).
    pub fn bound(self) -> Bound<LexiconSentiment> {
        Bound::new(self).write(1, SENTENCE_SENTIMENT.name)
    }
}
