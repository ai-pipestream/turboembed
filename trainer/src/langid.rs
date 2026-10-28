//! Deterministic softmax trainer for hashed n-gram language identification.

use std::collections::HashMap;

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

/// Hashes a byte string into a bucket.
fn hash(bytes: &[u8], buckets: u32) -> u32 {
    let mut h: u32 = 2166136261;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h % buckets
}

/// Extracts hashed character n-gram features from `text`.
pub fn features(text: &str, ngram_range: (u8, u8), buckets: u32) -> HashMap<u32, f32> {
    let bytes = text.as_bytes();
    let mut features = HashMap::new();
    let (min_n, max_n) = ngram_range;
    for n in min_n..=max_n {
        let n = n as usize;
        if bytes.len() < n {
            continue;
        }
        for window in bytes.windows(n) {
            let bucket = hash(window, buckets);
            *features.entry(bucket).or_insert(0.0) += 1.0;
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

    #[test]
    fn learns_a_simple_distinction() {
        let examples = [
            Example { text: "hello world", language: 0 },
            Example { text: "hallo welt", language: 1 },
            Example { text: "hello there", language: 0 },
            Example { text: "hallo dort", language: 1 },
        ];
        let config = Config { epochs: 20, ..Config::default() };
        let (weights, bias) = train::<f32>(&examples, 2, &config);
        assert_eq!(weights.len(), 2 * config.buckets as usize);
        assert_eq!(bias.len(), 2);
    }

    #[test]
    fn features_are_deterministic() {
        let a = features("hello", (1, 3), 16);
        let b = features("hello", (1, 3), 16);
        assert_eq!(a, b);
    }
}
