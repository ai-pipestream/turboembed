//! The vocabulary, frequent-word subsampling and the noise distribution.

use std::collections::HashMap;
use std::hash::Hasher;

use crate::hash::Fnv;

/// The words that get vectors, most frequent first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vocabulary {
    words: Vec<String>,
    counts: Vec<u64>,
    index: HashMap<String, u32>,
    total: u64,
}

impl Vocabulary {
    /// The words occurring at least `min_count` times, by count
    /// descending then byte order, cut to `max_vocab`.
    pub(super) fn build(corpus: &[Vec<String>], min_count: u64, max_vocab: Option<usize>) -> Vocabulary {
        let mut counts: HashMap<&str, u64> = HashMap::new();
        for word in corpus.iter().flatten() {
            *counts.entry(word.as_str()).or_insert(0) += 1;
        }
        let mut entries: Vec<(&str, u64)> = counts.into_iter().filter(|&(_, c)| c >= min_count).collect();
        entries.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        if let Some(max) = max_vocab {
            entries.truncate(max);
        }
        let words: Vec<String> = entries.iter().map(|&(w, _)| w.to_owned()).collect();
        let counts: Vec<u64> = entries.iter().map(|&(_, c)| c).collect();
        let index = words.iter().enumerate().map(|(i, w)| (w.clone(), i as u32)).collect();
        let total = counts.iter().sum();
        Vocabulary { words, counts, index, total }
    }

    pub fn len(&self) -> usize {
        self.words.len()
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// The words, most frequent first, ties in byte order.
    pub fn words(&self) -> &[String] {
        &self.words
    }

    /// The occurrence count of each word, in [`words`](Self::words) order.
    pub fn counts(&self) -> &[u64] {
        &self.counts
    }

    /// The occurrences of all vocabulary words.
    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn id(&self, word: &str) -> Option<u32> {
        self.index.get(word).copied()
    }

    /// The probability of keeping an occurrence of word `id` under
    /// threshold `t`: `min(1, sqrt(t / f(w)))` for the word's corpus
    /// frequency `f(w)`, so it is dropped with `1 - sqrt(t / f(w))`;
    /// 1 when `t` is 0.
    pub fn keep_probability(&self, id: u32, t: f64) -> f64 {
        if t == 0.0 {
            return 1.0;
        }
        let f = self.counts[id as usize] as f64 / self.total as f64;
        (t / f).sqrt().min(1.0)
    }

    pub(super) fn hash_into(&self, h: &mut Fnv) {
        h.write_u64(self.words.len() as u64);
        for (w, &c) in self.words.iter().zip(&self.counts) {
            h.str(w);
            h.write_u64(c);
        }
    }
}

/// The noise distribution: each word's count raised to 0.75, as
/// cumulative probabilities searched with a uniform draw.
pub(super) struct Noise {
    cumulative: Vec<f64>,
}

impl Noise {
    pub fn new(vocab: &Vocabulary) -> Noise {
        let weights: Vec<f64> = vocab.counts.iter().map(|&c| (c as f64).powf(0.75)).collect();
        let sum: f64 = weights.iter().sum();
        let mut running = 0.0;
        let cumulative = weights
            .iter()
            .map(|w| {
                running += w / sum;
                running
            })
            .collect();
        Noise { cumulative }
    }

    /// The word whose cumulative interval holds `u` in `[0, 1)`.
    pub fn sample(&self, u: f64) -> u32 {
        let i = self.cumulative.partition_point(|&c| c <= u);
        i.min(self.cumulative.len() - 1) as u32
    }

    #[cfg(test)]
    fn probability(&self, id: usize) -> f64 {
        self.cumulative[id] - if id == 0 { 0.0 } else { self.cumulative[id - 1] }
    }
}

#[cfg(test)]
mod tests {
    use super::{Noise, Vocabulary};

    fn corpus(text: &[&str]) -> Vec<Vec<String>> {
        text.iter().map(|s| s.split(' ').map(str::to_owned).collect()).collect()
    }

    #[test]
    fn orders_by_count_then_bytes_and_applies_min_count_and_max_vocab() {
        let c = corpus(&["b a c a", "d b a e e", "z"]);
        let v = Vocabulary::build(&c, 1, None);
        assert_eq!(v.words(), ["a", "b", "e", "c", "d", "z"]);
        assert_eq!(v.counts(), [3, 2, 2, 1, 1, 1]);
        assert_eq!(v.total(), 10);
        assert_eq!(v.id("e"), Some(2));
        let v = Vocabulary::build(&c, 2, None);
        assert_eq!(v.words(), ["a", "b", "e"]);
        assert_eq!(v.id("c"), None);
        let v = Vocabulary::build(&c, 1, Some(2));
        assert_eq!(v.words(), ["a", "b"]);
        assert!(Vocabulary::build(&c, 4, None).is_empty());
    }

    #[test]
    fn subsampling_drops_frequent_words_only() {
        let c = corpus(&["a a a a a a a a b c"]);
        let v = Vocabulary::build(&c, 1, None);
        // f(a) = 0.8: keep sqrt(0.1 / 0.8); f(b) = 0.1: keep 1.
        assert!((v.keep_probability(0, 0.1) - (0.125f64).sqrt()).abs() < 1e-12);
        assert_eq!(v.keep_probability(1, 0.1), 1.0);
        assert_eq!(v.keep_probability(0, 0.0), 1.0);
    }

    #[test]
    fn noise_follows_counts_to_the_three_quarters() {
        let c = corpus(&["a a a a a a a a a a a a a a a a b"]);
        let v = Vocabulary::build(&c, 1, None);
        let noise = Noise::new(&v);
        // 16^0.75 = 8 against 1.
        assert!((noise.probability(0) - 8.0 / 9.0).abs() < 1e-12);
        assert!((noise.probability(1) - 1.0 / 9.0).abs() < 1e-12);
        assert_eq!(noise.sample(0.0), 0);
        assert_eq!(noise.sample(0.88), 0);
        assert_eq!(noise.sample(0.9), 1);
        assert_eq!(noise.sample(0.999_999_999), 1);
    }
}
