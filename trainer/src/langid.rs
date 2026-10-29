//! Deterministic softmax trainer for hashed n-gram language identification.

use std::collections::BTreeMap;

use crate::real::Real;

/// A training example: text and its language index.
#[derive(Clone, Debug)]
pub struct Example<'a> {
    pub text: &'a str,
    pub language: usize,
}

/// Configuration for the language identification trainer.
#[derive(Clone, Debug)]
pub struct Config {
    /// Range of character n-gram sizes.
    pub ngram_range: (u8, u8),
    /// Number of hash buckets.
    pub buckets: u32,
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate.
    pub learning_rate: f32,
}

impl Default for Config {
    fn default() -> Config {
        Config { ngram_range: (1, 3), buckets: 1 << 16, epochs: 5, learning_rate: 0.1 }
    }
}

/// Extracts hashed Unicode scalar n-grams in deterministic bucket order.
pub fn features(text: &str, ngram_range: (u8, u8), buckets: u32) -> BTreeMap<u32, f32> {
    let chars: Vec<char> = text.chars().collect();
    let mut features = BTreeMap::new();
    for n in ngram_range.0..=ngram_range.1 {
        for window in chars.windows(n as usize) {
            let mut hash = 2166136261u32;
            for &c in window {
                let mut encoded = [0u8; 4];
                for &b in c.encode_utf8(&mut encoded).as_bytes() {
                    hash ^= b as u32;
                    hash = hash.wrapping_mul(16777619);
                }
            }
            *features.entry(hash % buckets).or_insert(0.0) += 1.0;
        }
    }
    features
}

/// Trains a language identification model.
///
/// Returns weights `[languages, buckets]` and biases `[languages]`.
pub fn train<R: Real>(examples: &[Example], languages: usize, config: &Config) -> (Vec<R>, Vec<R>) {
    let buckets = config.buckets as usize;
    let mut weights = vec![R::ZERO; languages * buckets];
    let mut bias = vec![R::ZERO; languages];
    let lr = R::from_f64(config.learning_rate as f64);

    for _ in 0..config.epochs {
        for example in examples {
            let feats = features(example.text, config.ngram_range, config.buckets);
            let mut scores = bias.clone();
            for (&bucket, &count) in &feats {
                let count = R::from_f64(count as f64);
                for (i, score) in scores.iter_mut().enumerate() {
                    *score += weights[i * buckets + bucket as usize] * count;
                }
            }
            let max = scores.iter().fold(R::from_f64(f64::NEG_INFINITY), |a, &b| if b > a { b } else { a });
            let mut sum = R::ZERO;
            for score in scores.iter_mut() {
                *score = (*score - max).exp();
                sum += *score;
            }
            for score in scores.iter_mut() {
                *score = *score / sum;
            }
            for (i, score) in scores.iter_mut().enumerate() {
                let target = if i == example.language { R::ONE } else { R::ZERO };
                let grad = *score - target;
                bias[i] -= lr * grad;
                for (&bucket, &count) in &feats {
                    weights[i * buckets + bucket as usize] -= lr * grad * R::from_f64(count as f64);
                }
            }
        }
    }
    (weights, bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn predict(text: &str, weights: &[f32], bias: &[f32], config: &Config) -> usize {
        let features = features(text, config.ngram_range, config.buckets);
        let mut scores = bias.to_vec();
        for (bucket, count) in features {
            for (i, score) in scores.iter_mut().enumerate() {
                *score += weights[i * config.buckets as usize + bucket as usize] * count;
            }
        }
        scores.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap().0
    }

    #[test]
    fn learns_a_simple_distinction_deterministically() {
        let examples = [
            Example { text: "hello world", language: 0 },
            Example { text: "hallo welt", language: 1 },
            Example { text: "hello there", language: 0 },
            Example { text: "hallo dort", language: 1 },
        ];
        let config = Config { buckets: 256, epochs: 20, ..Config::default() };
        let (weights, bias) = train::<f32>(&examples, 2, &config);
        assert_eq!(predict("hello", &weights, &bias, &config), 0);
        assert_eq!(predict("hallo", &weights, &bias, &config), 1);
        assert_eq!(train::<f32>(&examples, 2, &config), (weights, bias));
    }

    #[test]
    fn features_count_unicode_scalars() {
        let features = features("éé", (1, 1), 16);
        assert_eq!(features.len(), 1);
        assert_eq!(features.values().copied().collect::<Vec<_>>(), [2.0]);
    }
}
