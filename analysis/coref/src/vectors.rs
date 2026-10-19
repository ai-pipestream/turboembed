//! Vector providers for the ranker: [`WordVectors`] for head similarity
//! features and [`TokenVectors`] for contextual span features.

use std::collections::HashMap;
use std::hash::Hasher;

use crate::hash::Fnv;

/// A vector per word for the features that compare mention heads by
/// distributional similarity (`opennlp.tools.coref.WordVectors`).
pub trait WordVectors: Send + Sync {
    /// The length of every vector, positive.
    fn dimension(&self) -> usize;

    /// The vector of a lowercased word: [`dimension`](Self::dimension)
    /// finite values, or `None` for a word without one.
    fn vector(&self, word: &str) -> Option<&[f32]>;

    /// A hash, stable across runs, of everything that decides the
    /// vectors (their source and settings), recorded in the annotator's
    /// configuration hash.
    fn config_hash(&self) -> u64;
}

/// A contextual vector per token, computed a sentence at a time
/// (`opennlp.tools.coref.TokenVectors`), for the span features of Lee
/// et al. (EMNLP 2017).
pub trait TokenVectors: Send + Sync {
    /// The length of every vector, positive.
    fn dimension(&self) -> usize;

    /// One vector of [`dimension`](Self::dimension) finite values per
    /// token of one sentence.
    fn vectors(&self, sentence: &[&str]) -> Vec<Vec<f32>>;

    /// A hash, stable across runs, of everything that decides the
    /// vectors (the encoder's model and settings), recorded in the
    /// annotator's configuration hash.
    fn config_hash(&self) -> u64;
}

/// Word vectors held in a map.
#[derive(Clone, Debug, Default)]
pub struct MapWordVectors {
    dimension: usize,
    vectors: HashMap<String, Vec<f32>>,
}

impl MapWordVectors {
    /// An empty table of `dimension`-long vectors.
    pub fn new(dimension: usize) -> MapWordVectors {
        MapWordVectors { dimension, vectors: HashMap::new() }
    }

    /// Adds or replaces the vector of a lowercased word; `Err` when its
    /// length is not the dimension or a value is not finite.
    pub fn insert(&mut self, word: impl Into<String>, vector: Vec<f32>) -> Result<(), String> {
        let word = word.into();
        if vector.len() != self.dimension {
            return Err(format!("vector for {word} has dimension {}, expected {}", vector.len(), self.dimension));
        }
        if let Some(d) = vector.iter().position(|v| !v.is_finite()) {
            return Err(format!("vector for {word} has a non-finite value at dimension {d}"));
        }
        self.vectors.insert(word, vector);
        Ok(())
    }
}

impl WordVectors for MapWordVectors {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn vector(&self, word: &str) -> Option<&[f32]> {
        self.vectors.get(word).map(Vec::as_slice)
    }

    /// The dimension and the entries, combined by a sum so the map's
    /// order does not matter.
    fn config_hash(&self) -> u64 {
        let mut sum = 0u64;
        for (word, vector) in &self.vectors {
            let mut h = Fnv::default();
            h.str(word);
            vector.iter().for_each(|v| h.write_u32(v.to_bits()));
            sum = sum.wrapping_add(h.finish());
        }
        let mut h = Fnv::default();
        h.write_u64(self.dimension as u64);
        h.write_u64(sum);
        h.finish()
    }
}

/// Runs the encoder over each sentence's tokens and checks its answer:
/// one vector per token, each of the encoder's dimension, all finite.
pub(crate) fn encode(
    encoder: &dyn TokenVectors,
    forms: &[&str],
    sentence_of_token: &[usize],
) -> Result<Vec<Vec<f32>>, String> {
    let dimension = encoder.dimension();
    let mut all = Vec::with_capacity(forms.len());
    let mut start = 0;
    while start < forms.len() {
        let mut end = start + 1;
        while end < forms.len() && sentence_of_token[end] == sentence_of_token[start] {
            end += 1;
        }
        let vectors = encoder.vectors(&forms[start..end]);
        if vectors.len() != end - start {
            return Err(format!("token encoder returned {} vectors for {} tokens", vectors.len(), end - start));
        }
        for (t, v) in vectors.iter().enumerate() {
            if v.is_empty() {
                return Err(format!("token encoder returned an empty vector for token {}", start + t));
            }
            if v.len() != dimension {
                return Err(format!(
                    "token encoder returned dimension {} for token {}, expected {dimension}",
                    v.len(),
                    start + t
                ));
            }
            if let Some(d) = v.iter().position(|x| !x.is_finite()) {
                return Err(format!(
                    "token encoder returned a non-finite value for token {} at dimension {d}",
                    start + t
                ));
            }
        }
        all.extend(vectors);
        start = end;
    }
    Ok(all)
}
