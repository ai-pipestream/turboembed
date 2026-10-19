//! N-gram language modeling with Stupid Backoff
//! (`opennlp.tools.languagemodel.LanguageModel` and
//! `opennlp.tools.languagemodel.NGramLanguageModel`).
//!
//! Probabilities are accumulated in log space with `StrictMath.log` and
//! exponentiated with `StrictMath.exp` from [`turbo_jvm::strict_math`] so that
//! floating-point scores match the JVM bit for bit.

use std::ops::{Deref, DerefMut};

use turbo_jvm::strict_math::{exp, log};

use crate::error::{Error, Result};
use crate::model::NGramModel;
use crate::utils::NGramUtils;

/// Default maximum n-gram order (`3`, trigram) used by [`NGramLanguageModel::default`].
pub const DEFAULT_N: usize = 3;

/// Stupid Backoff discount factor (`0.4`) applied at each backoff step.
const STUPID_BACKOFF_ALPHA: f64 = 0.4;

/// A language model that scores token sequences and predicts the most probable
/// continuation n-gram (`opennlp.tools.languagemodel.LanguageModel`).
pub trait LanguageModel {
    /// Calculates the probability of `tokens` in the vocabulary of this model,
    /// returning a value in `[0.0, 1.0]`.
    fn calculate_probability<S: AsRef<str>>(&self, tokens: &[S]) -> f64;

    /// Predicts the most probable next token sequence from the model's n-grams
    /// given the preceding `tokens`, or `None` if the model is empty.
    fn predict_next_tokens<S: AsRef<str>>(&self, tokens: &[S]) -> Option<Vec<String>>;
}

/// An n-gram language model using Stupid Backoff
/// (`opennlp.tools.languagemodel.NGramLanguageModel`).
///
/// Extends [`NGramModel`] with order `n` (`n >= 1`). Calling [`add`](Self::add)
/// inserts all contiguous token sub-grams of lengths `1..=n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NGramLanguageModel {
    n: usize,
    model: NGramModel,
}

impl Default for NGramLanguageModel {
    fn default() -> Self {
        Self { n: DEFAULT_N, model: NGramModel::new() }
    }
}

impl NGramLanguageModel {
    /// Creates an empty [`NGramLanguageModel`] with order `n`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidOrder`] if `n == 0`.
    pub fn new(n: usize) -> Result<Self> {
        if n == 0 {
            return Err(Error::InvalidOrder(n));
        }
        Ok(Self { n, model: NGramModel::new() })
    }

    /// Reads an [`NGramLanguageModel`] with default order [`DEFAULT_N`] (`3`)
    /// from OpenNLP dictionary XML bytes.
    pub fn from_xml(xml: &[u8]) -> Result<Self> {
        Self::from_xml_with_order(xml, DEFAULT_N)
    }

    /// Reads an [`NGramLanguageModel`] with order `n` from OpenNLP dictionary
    /// XML bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidOrder`] if `n == 0`, or an XML error if `xml` is
    /// malformed.
    pub fn from_xml_with_order(xml: &[u8], n: usize) -> Result<Self> {
        if n == 0 {
            return Err(Error::InvalidOrder(n));
        }
        let model = NGramModel::read(xml)?;
        Ok(Self { n, model })
    }

    /// Returns the n-gram order `n` of this language model.
    pub fn order(&self) -> usize {
        self.n
    }

    /// Returns a reference to the underlying [`NGramModel`].
    pub fn model(&self) -> &NGramModel {
        &self.model
    }

    /// Returns a mutable reference to the underlying [`NGramModel`].
    pub fn model_mut(&mut self) -> &mut NGramModel {
        &mut self.model
    }

    /// Adds all contiguous token n-grams of lengths `1..=self.order()` from
    /// `tokens` (`NGramLanguageModel.add(String...)`).
    pub fn add<S: AsRef<str>>(&mut self, tokens: &[S]) -> Result<()> {
        self.model.add_range(tokens, 1, self.n)
    }

    /// Computes the Stupid Backoff score for a single `ngram` slice, matching
    /// `NGramLanguageModel.stupidBackoff(StringList)`.
    fn stupid_backoff<S: AsRef<str>>(&self, ngram: &[S]) -> f64 {
        let count = self.model.get_count(ngram);
        if ngram.len() > 1 {
            let n_minus_one = NGramUtils::get_n_minus_one_token_first(ngram);
            let mut n_minus_one_count = self.model.get_count(n_minus_one) as f64;
            if n_minus_one_count <= 0.0 {
                n_minus_one_count = self.model.size() as f64; // prevent Infinity or negative denominator
            }
            if count > 0 {
                count as f64 / n_minus_one_count
            } else {
                let next_ngram = NGramUtils::get_n_minus_one_token_last(ngram);
                STUPID_BACKOFF_ALPHA * self.stupid_backoff(next_ngram)
            }
        } else if count > 0 {
            count as f64 / self.model.size() as f64
        } else {
            0.0
        }
    }
}

impl Deref for NGramLanguageModel {
    type Target = NGramModel;

    fn deref(&self) -> &Self::Target {
        &self.model
    }
}

impl DerefMut for NGramLanguageModel {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.model
    }
}

impl LanguageModel for NGramLanguageModel {
    fn calculate_probability<S: AsRef<str>>(&self, tokens: &[S]) -> f64 {
        if self.model.size() > 0 {
            let mut score = 0.0_f64;
            for ngram in NGramUtils::get_ngrams(tokens, self.n) {
                let score_for_ngram = self.stupid_backoff(ngram);
                score += log(score_for_ngram);
                if score.is_nan() {
                    score = 0.0;
                    break;
                }
            }
            exp(score)
        } else {
            0.0
        }
    }

    fn predict_next_tokens<S: AsRef<str>>(&self, tokens: &[S]) -> Option<Vec<String>> {
        let mut max_prob = f64::NEG_INFINITY;
        let mut best: Option<Vec<String>> = None;
        let mut sequence: Vec<&str> = Vec::new();

        for ngram in self.model.iter() {
            sequence.clear();
            sequence.extend(tokens.iter().map(AsRef::as_ref));
            sequence.extend(ngram.iter().map(String::as_str));
            let v = self.calculate_probability(&sequence);
            if v > max_prob {
                max_prob = v;
                best = Some(ngram.to_vec());
            }
        }

        best
    }
}
