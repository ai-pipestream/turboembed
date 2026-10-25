//! Character n-gram frequency table (`opennlp.tools.ngram.NGramCharModel`).
//!
//! [`NGramCharModel`] counts character n-grams after lowercasing each code point
//! with Java's `Character.toLowerCase(int)` (`StringUtil.toLowerCase` and
//! `StringUtil.toLowerCaseCharBuffer`).

use std::collections::HashMap;
use std::fmt;

use turbo_unicode::java::{
    lowercase_code_points as java_lowercase, lowercase_code_points_changes as needs_java_lowercase,
};

use crate::error::{Error, Result};
use crate::model::{decode_utf16_slice, lowercase_utf16_units};

/// A counted collection of character n-grams, porting
/// `opennlp.tools.ngram.NGramCharModel`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NGramCharModel {
    ngrams: HashMap<String, i32>,
}

impl NGramCharModel {
    /// Creates an empty [`NGramCharModel`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the count of `ngram`, or `0` if it is not present.
    ///
    /// As in `NGramCharModel.getCount(CharSequence)`, the probe is matched
    /// directly without additional lowercasing.
    pub fn get_count(&self, ngram: &str) -> i32 {
        self.ngrams.get(ngram).copied().unwrap_or(0)
    }

    /// Sets the count of an existing `ngram` to `count`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoSuchNGram`] if `ngram` is not in the model
    /// (matching OpenNLP's `NoSuchElementException`).
    pub fn set_count(&mut self, ngram: &str, count: i32) -> Result<()> {
        let Some(slot) = self.ngrams.get_mut(ngram) else {
            return Err(Error::NoSuchNGram);
        };
        *slot = count;
        Ok(())
    }

    /// Lowercases `ngram` code point by code point (`StringUtil.toLowerCase`)
    /// and increments its count by `1` (inserting it with count `1` if absent).
    pub fn add(&mut self, ngram: &str) {
        if !needs_java_lowercase(ngram) {
            // Already lowercase: probe with the borrowed key and allocate only
            // when the n-gram is new.
            if let Some(count) = self.ngrams.get_mut(ngram) {
                *count = count.wrapping_add(1);
            } else {
                self.ngrams.insert(ngram.to_owned(), 1);
            }
            return;
        }
        let lowered = java_lowercase(ngram);
        if let Some(count) = self.ngrams.get_mut(&lowered) {
            *count = count.wrapping_add(1);
        } else {
            self.ngrams.insert(lowered, 1);
        }
    }

    /// Extracts all UTF-16 character substrings of lengths `min_length..=max_length`
    /// from `chars` after lowercasing via `StringUtil.toLowerCaseCharBuffer`, and
    /// increments the count of each substring.
    pub fn add_chars(&mut self, chars: &str, min_length: usize, max_length: usize) -> Result<()> {
        if min_length < 1 || max_length < 1 || min_length > max_length {
            return Err(Error::InvalidLengthRange { min_length, max_length });
        }
        let units = lowercase_utf16_units(chars);
        let mut gram_buf = String::new();
        for length_index in min_length..=max_length {
            if length_index > units.len() {
                break;
            }
            for text_index in 0..=units.len() - length_index {
                decode_utf16_slice(&units[text_index..text_index + length_index], &mut gram_buf);
                self.add(&gram_buf);
            }
        }
        Ok(())
    }

    /// Removes `ngram` from the model if present.
    ///
    /// As in `NGramCharModel.remove(CharSequence)`, the key is matched directly
    /// without lowercasing.
    pub fn remove(&mut self, ngram: &str) -> bool {
        self.ngrams.remove(ngram).is_some()
    }

    /// Returns `true` if the model contains `ngram` after lowercasing it with
    /// `StringUtil.toLowerCase` (`NGramCharModel.contains`).
    pub fn contains(&self, ngram: &str) -> bool {
        if !needs_java_lowercase(ngram) {
            return self.ngrams.contains_key(ngram);
        }
        let lowered = java_lowercase(ngram);
        self.ngrams.contains_key(&lowered)
    }

    /// Returns the number of distinct character n-grams in the model
    /// (`NGramCharModel.size()`).
    pub fn size(&self) -> usize {
        self.ngrams.len()
    }

    /// Returns the number of distinct character n-grams in the model.
    pub fn len(&self) -> usize {
        self.ngrams.len()
    }

    /// Returns `true` if the model contains no n-grams.
    pub fn is_empty(&self) -> bool {
        self.ngrams.is_empty()
    }

    /// Returns the sum of counts across all character n-grams in the model
    /// (`NGramCharModel.numberOfGrams()`).
    pub fn number_of_grams(&self) -> i32 {
        self.ngrams.values().copied().fold(0_i32, i32::wrapping_add)
    }

    /// Removes every n-gram whose count is strictly less than `cutoff_under` or
    /// strictly greater than `cutoff_over`.
    pub fn cutoff(&mut self, cutoff_under: i32, cutoff_over: i32) {
        if cutoff_under > 0 || cutoff_over < i32::MAX {
            self.ngrams.retain(|_, count| *count >= cutoff_under && *count <= cutoff_over);
        }
    }

    /// Iterates over the distinct character n-grams in the model.
    pub fn iter(&self) -> impl Iterator<Item = &str> {
        self.ngrams.keys().map(String::as_str)
    }

    /// Iterates over `(ngram, count)` pairs in the model.
    pub fn iter_with_counts(&self) -> impl Iterator<Item = (&str, i32)> {
        self.ngrams.iter().map(|(k, v)| (k.as_str(), *v))
    }
}

impl fmt::Display for NGramCharModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Size: {}", self.size())
    }
}
