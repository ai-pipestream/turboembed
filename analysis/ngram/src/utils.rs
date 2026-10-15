//! N-gram generation and probability estimation utilities
//! (`opennlp.tools.ngram.NGramGenerator` and `opennlp.tools.ngram.NGramUtils`).

use std::collections::HashSet;
use std::hash::Hash;

use crate::error::{Error, Result};

/// Generates string n-grams joined by a separator from token lists or character
/// slices, porting `opennlp.tools.ngram.NGramGenerator`.
pub struct NGramGenerator;

impl NGramGenerator {
    /// Generates all contiguous `n`-grams from `input`, joining tokens in each
    /// window with `separator`.
    ///
    /// Returns an empty vector when `n == 0` or `input.len() < n`.
    pub fn generate<S: AsRef<str>>(input: &[S], n: usize, separator: &str) -> Vec<String> {
        if n == 0 || input.len() < n {
            return Vec::new();
        }
        let mut out_grams = Vec::with_capacity(input.len() - n + 1);
        for window in input.windows(n) {
            let mut sb = String::new();
            for (j, token) in window.iter().enumerate() {
                if j > 0 {
                    sb.push_str(separator);
                }
                sb.push_str(token.as_ref());
            }
            out_grams.push(sb);
        }
        out_grams
    }

    /// Generates all contiguous `n`-grams from character slice `input`, joining
    /// characters in each window with `separator`.
    ///
    /// Returns an empty vector when `n == 0` or `input.len() < n`.
    pub fn generate_chars(input: &[char], n: usize, separator: &str) -> Vec<String> {
        if n == 0 || input.len() < n {
            return Vec::new();
        }
        let mut out_grams = Vec::with_capacity(input.len() - n + 1);
        for window in input.windows(n) {
            let mut sb = String::new();
            for (j, &ch) in window.iter().enumerate() {
                if j > 0 {
                    sb.push_str(separator);
                }
                sb.push(ch);
            }
            out_grams.push(sb);
        }
        out_grams
    }
}

/// N-gram slicing and probability estimation helpers, porting
/// `opennlp.tools.ngram.NGramUtils`.
pub struct NGramUtils;

impl NGramUtils {
    /// Calculates the Laplace-smoothed probability of `ngram` over `set` with
    /// smoothing constant `k` (`NGramUtils.calculateLaplaceSmoothingProbability`):
    ///
    /// `(count(ngram, set) + k) / (count(prefix(ngram), set) + k * 1.0)`
    pub fn calculate_laplace_smoothing_probability<T, S>(ngram: &[T], set: &[S], k: f64) -> f64
    where
        T: PartialEq,
        S: AsRef<[T]>,
    {
        (Self::count(ngram, set) + k) / (Self::count(Self::get_n_minus_one_token_first(ngram), set) + k * 1.0)
    }

    /// Calculates the Maximum Likelihood (ML) unigram probability of `word` in
    /// `set` (`NGramUtils.calculateUnigramMLProbability`).
    pub fn calculate_unigram_ml_probability<T, S>(word: &T, set: &[S]) -> f64
    where
        T: PartialEq,
        S: AsRef<[T]>,
    {
        let voc_size: f64 = set.iter().map(|s| s.as_ref().len() as f64).sum();
        Self::count(std::slice::from_ref(word), set) / voc_size
    }

    /// Calculates the Maximum Likelihood (ML) bigram probability `P(x1 | x0)` in
    /// `set` (`NGramUtils.calculateBigramMLProbability`).
    pub fn calculate_bigram_ml_probability<T, S>(x0: &T, x1: &T, set: &[S]) -> f64
    where
        T: PartialEq + Clone,
        S: AsRef<[T]>,
    {
        Self::calculate_ngram_ml_probability(&[x0.clone(), x1.clone()], set)
    }

    /// Calculates the Maximum Likelihood (ML) trigram probability
    /// `P(x2 | x0, x1)` in `set` (`NGramUtils.calculateTrigramMLProbability`).
    pub fn calculate_trigram_ml_probability<T, S>(x0: &T, x1: &T, x2: &T, set: &[S]) -> f64
    where
        T: PartialEq + Clone,
        S: AsRef<[T]>,
    {
        Self::calculate_ngram_ml_probability(&[x0.clone(), x1.clone(), x2.clone()], set)
    }

    /// Calculates the Maximum Likelihood (ML) probability of `ngram` given its
    /// `(n - 1)`-token prefix in `set` (`NGramUtils.calculateNgramMLProbability`).
    pub fn calculate_ngram_ml_probability<T, S>(ngram: &[T], set: &[S]) -> f64
    where
        T: PartialEq,
        S: AsRef<[T]>,
    {
        let ngram_minus_one = Self::get_n_minus_one_token_first(ngram);
        Self::count(ngram, set) / Self::count(ngram_minus_one, set)
    }

    /// Calculates the prior-smoothed bigram probability of `(x0, x1)` with
    /// smoothing weight `k` (`NGramUtils.calculateBigramPriorSmoothingProbability`).
    pub fn calculate_bigram_prior_smoothing_probability<T, S>(x0: &T, x1: &T, set: &[S], k: f64) -> f64
    where
        T: PartialEq + Clone,
        S: AsRef<[T]>,
    {
        let bigram = [x0.clone(), x1.clone()];
        (Self::count(&bigram, set) + k * Self::calculate_unigram_ml_probability(x1, set))
            / (Self::count(std::slice::from_ref(x0), set) + k * set.len() as f64)
    }

    /// Calculates the trigram linear interpolation probability of `(x0, x1, x2)`
    /// with weights `(lambda1, lambda2, lambda3)`
    /// (`NGramUtils.calculateTrigramLinearInterpolationProbability`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInterpolationWeights`] if any weight is `<= 0.0`
    /// or `|(lambda1 + lambda2 + lambda3) - 1.0| > 0.0001`.
    pub fn calculate_trigram_linear_interpolation_probability<T, S>(
        x0: &T,
        x1: &T,
        x2: &T,
        set: &[S],
        lambda1: f64,
        lambda2: f64,
        lambda3: f64,
    ) -> Result<f64>
    where
        T: PartialEq + Clone,
        S: AsRef<[T]>,
    {
        if lambda1 <= 0.0 || lambda2 <= 0.0 || lambda3 <= 0.0 || (lambda1 + lambda2 + lambda3 - 1.0).abs() > 0.0001 {
            return Err(Error::InvalidInterpolationWeights { lambda1, lambda2, lambda3 });
        }
        Ok(lambda1 * Self::calculate_trigram_ml_probability(x0, x1, x2, set)
            + lambda2 * Self::calculate_bigram_ml_probability(x1, x2, set)
            + lambda3 * Self::calculate_unigram_ml_probability(x2, set))
    }

    /// Calculates the missing probability mass for `ngram` under `discount` over
    /// `set` (`NGramUtils.calculateMissingNgramProbabilityMass`).
    pub fn calculate_missing_ngram_probability_mass<T, S>(ngram: &[T], discount: f64, set: &[S]) -> f64
    where
        T: PartialEq + Eq + Hash + Clone,
        S: AsRef<[T]>,
    {
        let mut missing_mass = 0.0;
        let count_word = Self::count(ngram, set);
        let mut seen = HashSet::new();
        let mut extended = Vec::with_capacity(ngram.len() + 1);
        for sentence in set {
            for word in sentence.as_ref() {
                if seen.insert(word) {
                    extended.clear();
                    extended.extend_from_slice(ngram);
                    extended.push(word.clone());
                    missing_mass += (Self::count(&extended, set) - discount) / count_word;
                }
            }
        }
        1.0 - missing_mass
    }

    /// Returns the `(n - 1)`-token prefix of `ngram` (`ngram[..ngram.len() - 1]`),
    /// matching `NGramUtils.getNMinusOneTokenFirst`.
    ///
    /// Returns an empty slice if `ngram` is empty.
    pub fn get_n_minus_one_token_first<T>(ngram: &[T]) -> &[T] {
        if ngram.is_empty() { &[] } else { &ngram[..ngram.len() - 1] }
    }

    /// Returns the `(n - 1)`-token suffix of `ngram` (`ngram[1..]`),
    /// matching `NGramUtils.getNMinusOneTokenLast`.
    ///
    /// Returns an empty slice if `ngram` is empty.
    pub fn get_n_minus_one_token_last<T>(ngram: &[T]) -> &[T] {
        if ngram.is_empty() { &[] } else { &ngram[1..] }
    }

    /// Returns all contiguous slices of length `size` from `sequence`
    /// (`NGramUtils.getNGrams`).
    ///
    /// As in `NGramUtils.getNGrams(T[] sequence, int size)`, when `size == 0`
    /// or `size >= sequence.len()`, a single slice covering the entire `sequence`
    /// is returned.
    pub fn get_ngrams<T>(sequence: &[T], size: usize) -> Vec<&[T]> {
        if size == 0 || size >= sequence.len() {
            vec![sequence]
        } else {
            let count = sequence.len() - size + 1;
            let mut ngrams = Vec::with_capacity(count);
            for i in 0..count {
                ngrams.push(&sequence[i..i + size]);
            }
            ngrams
        }
    }

    /// Counts how many sentences in `sentences` contain `ngram` starting at the
    /// first occurrence of `ngram[0]` in each sentence, matching
    /// `NGramUtils.count(StringList, Iterable<StringList>)`.
    pub fn count<T, S>(ngram: &[T], sentences: &[S]) -> f64
    where
        T: PartialEq,
        S: AsRef<[T]>,
    {
        if ngram.is_empty() {
            return 0.0;
        }
        let mut count = 0.0;
        for sentence in sentences {
            let s = sentence.as_ref();
            if let Some(idx0) = s.iter().position(|t| t == &ngram[0])
                && s.len() >= idx0 + ngram.len()
                && s[idx0 + 1..idx0 + ngram.len()] == ngram[1..]
            {
                count += 1.0;
            }
        }
        count
    }
}
