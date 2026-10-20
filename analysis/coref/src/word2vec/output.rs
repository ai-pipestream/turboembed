//! Trained vectors: a [`WordVectors`] provider and its text exports.

use std::fmt::Write;

use super::vocab::Vocabulary;
use crate::WordVectors;

/// The vectors of a training run, one per vocabulary word.
#[derive(Clone, Debug)]
pub struct Word2VecVectors {
    dimension: usize,
    vocab: Vocabulary,
    data: Vec<f32>,
    config_hash: u64,
}

impl Word2VecVectors {
    pub(super) fn new(dimension: usize, vocab: Vocabulary, data: Vec<f32>, config_hash: u64) -> Word2VecVectors {
        debug_assert_eq!(data.len(), dimension * vocab.len());
        Word2VecVectors { dimension, vocab, data, config_hash }
    }

    /// The vocabulary, in row order.
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocab
    }

    /// The word2vec text format: a `count dimension` header, then one
    /// line per word of the word and its values, most frequent first.
    /// `Err` when a word holds whitespace, which the format cannot carry.
    pub fn to_word2vec_text(&self) -> Result<String, String> {
        let mut out = format!("{} {}\n", self.vocab.len(), self.dimension);
        self.write_rows(&mut out)?;
        Ok(out)
    }

    /// The GloVe text format: the word2vec rows without the header.
    pub fn to_glove_text(&self) -> Result<String, String> {
        let mut out = String::new();
        self.write_rows(&mut out)?;
        Ok(out)
    }

    fn write_rows(&self, out: &mut String) -> Result<(), String> {
        for (i, word) in self.vocab.words().iter().enumerate() {
            if word.is_empty() || word.chars().any(char::is_whitespace) {
                return Err(format!("word {word:?} cannot be written in a text vector format"));
            }
            out.push_str(word);
            for v in &self.data[i * self.dimension..(i + 1) * self.dimension] {
                // Debug prints the shortest text that reads back the same.
                write!(out, " {v:?}").expect("writing to a String");
            }
            out.push('\n');
        }
        Ok(())
    }
}

impl WordVectors for Word2VecVectors {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn vector(&self, word: &str) -> Option<&[f32]> {
        let i = self.vocab.id(word)? as usize;
        Some(&self.data[i * self.dimension..(i + 1) * self.dimension])
    }

    /// The parameters (seed included), the vocabulary and every token
    /// of the corpus in order: everything a run depends on.
    fn config_hash(&self) -> u64 {
        self.config_hash
    }
}
