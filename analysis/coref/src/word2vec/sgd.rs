//! Stochastic gradient descent with negative sampling, after the
//! reference `word2vec.c`, on one thread.

use super::rng::Rng;
use super::vocab::{Noise, Vocabulary};
use super::{MAX_MATRIX_BYTES, Word2VecArch, Word2VecError, Word2VecParameters};

/// Beyond this magnitude the logistic is taken as 0 or 1.
const MAX_EXP: f32 = 6.0;

fn sigmoid(x: f32) -> f32 {
    if x > MAX_EXP {
        1.0
    } else if x < -MAX_EXP {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

fn dot(u: &[f32], v: &[f32]) -> f32 {
    u.iter().zip(v).map(|(a, b)| a * b).sum()
}

struct State<'a> {
    params: &'a Word2VecParameters,
    noise: Noise,
    rng: Rng,
    dim: usize,
    /// Input (word) vectors, row per vocabulary word: the result.
    syn0: Vec<f32>,
    /// Output (context) vectors for negative sampling.
    syn1: Vec<f32>,
    alpha: f32,
}

impl State<'_> {
    /// One prediction of `target` from `input`: the target as the
    /// positive and `negative` noise words, skipping a noise draw equal
    /// to the target. Adds the gradient for `input` to `grad`.
    fn predict(&mut self, input: &[f32], target: u32, grad: &mut [f32]) {
        let dim = self.dim;
        for d in 0..=self.params.negative {
            let (word, label) = if d == 0 {
                (target, 1.0)
            } else {
                let w = self.noise.sample(self.rng.unit());
                if w == target {
                    continue;
                }
                (w, 0.0)
            };
            let row = &mut self.syn1[word as usize * dim..(word as usize + 1) * dim];
            let g = (label - sigmoid(dot(input, row))) * self.alpha;
            for k in 0..dim {
                grad[k] += g * row[k];
                row[k] += g * input[k];
            }
        }
    }

    fn skip_gram(&mut self, sentence: &[u32], pos: usize, reach: usize, grad: &mut [f32], input: &mut [f32]) {
        let dim = self.dim;
        let target = sentence[pos];
        for c in context(sentence.len(), pos, reach) {
            let ctx = sentence[c] as usize * dim;
            input.copy_from_slice(&self.syn0[ctx..ctx + dim]);
            grad.fill(0.0);
            self.predict(input, target, grad);
            self.syn0[ctx..ctx + dim].iter_mut().zip(grad.iter()).for_each(|(v, g)| *v += g);
        }
    }

    fn cbow(&mut self, sentence: &[u32], pos: usize, reach: usize, grad: &mut [f32], input: &mut [f32]) {
        let dim = self.dim;
        input.fill(0.0);
        let mut n = 0usize;
        for c in context(sentence.len(), pos, reach) {
            let ctx = sentence[c] as usize * dim;
            input.iter_mut().zip(&self.syn0[ctx..ctx + dim]).for_each(|(m, v)| *m += v);
            n += 1;
        }
        if n == 0 {
            return;
        }
        input.iter_mut().for_each(|m| *m /= n as f32);
        grad.fill(0.0);
        self.predict(input, sentence[pos], grad);
        for c in context(sentence.len(), pos, reach) {
            let ctx = sentence[c] as usize * dim;
            self.syn0[ctx..ctx + dim].iter_mut().zip(grad.iter()).for_each(|(v, g)| *v += g);
        }
    }
}

/// The positions within `reach` of `pos`, other than `pos`.
fn context(len: usize, pos: usize, reach: usize) -> impl Iterator<Item = usize> {
    (pos.saturating_sub(reach)..(pos + reach + 1).min(len)).filter(move |&c| c != pos)
}

/// The element count of a `words` by `dim` matrix, `Err` when it or
/// its size in bytes overflows or passes [`MAX_MATRIX_BYTES`].
fn matrix_len(words: usize, dim: usize) -> Result<usize, Word2VecError> {
    words
        .checked_mul(dim)
        .filter(|&n| n.checked_mul(size_of::<f32>()).is_some_and(|b| b as u64 <= MAX_MATRIX_BYTES))
        .ok_or(Word2VecError::MatrixTooLarge { words, dimension: dim })
}

/// Trains on sentences of vocabulary ids and returns the word vectors,
/// row by row in vocabulary order. Every weight is checked to be finite
/// after each epoch, one pass over both matrices.
pub(super) fn train(
    params: &Word2VecParameters,
    vocab: &Vocabulary,
    sentences: &[Vec<u32>],
) -> Result<Vec<f32>, Word2VecError> {
    let dim = params.dimension;
    let len = matrix_len(vocab.len(), dim)?;
    let mut rng = Rng::new(params.seed);
    let syn0 = (0..len).map(|_| (rng.unit() as f32 - 0.5) / dim as f32).collect();
    let mut s =
        State { params, noise: Noise::new(vocab), rng, dim, syn0, syn1: vec![0.0; len], alpha: params.learning_rate };
    let keep: Vec<f64> = (0..vocab.len() as u32).map(|id| vocab.keep_probability(id, params.subsample)).collect();
    let per_epoch: u64 = sentences.iter().map(|s| s.len() as u64).sum();
    let total = per_epoch as f64 * params.epochs as f64;
    let (mut grad, mut input) = (vec![0.0f32; dim], vec![0.0f32; dim]);
    let mut kept = Vec::new();
    let mut seen = 0u64;
    for epoch in 1..=params.epochs {
        for sentence in sentences {
            kept.clear();
            for &id in sentence {
                if keep[id as usize] >= 1.0 || s.rng.unit() < keep[id as usize] {
                    kept.push(id);
                }
            }
            for pos in 0..kept.len() {
                let progress = seen as f64 / (total + 1.0);
                s.alpha = ((params.learning_rate as f64 * (1.0 - progress)) as f32).max(params.min_learning_rate);
                let reach = params.window - s.rng.below(params.window);
                match params.arch {
                    Word2VecArch::SkipGram => s.skip_gram(&kept, pos, reach, &mut grad, &mut input),
                    Word2VecArch::Cbow => s.cbow(&kept, pos, reach, &mut grad, &mut input),
                }
                seen += 1;
            }
            // Dropped occurrences still count toward the decay.
            seen += (sentence.len() - kept.len()) as u64;
        }
        if !s.syn0.iter().chain(&s.syn1).all(|v| v.is_finite()) {
            return Err(Word2VecError::Diverged { epoch });
        }
    }
    Ok(s.syn0)
}

#[cfg(test)]
mod tests {
    use super::{MAX_MATRIX_BYTES, Word2VecError, context, matrix_len, sigmoid};

    #[test]
    fn matrix_sizes_are_checked_before_allocation() {
        assert_eq!(matrix_len(3, 4), Ok(12));
        let at_limit = (MAX_MATRIX_BYTES / 4) as usize;
        assert_eq!(matrix_len(at_limit, 1), Ok(at_limit));
        let over = Word2VecError::MatrixTooLarge { words: at_limit + 1, dimension: 1 };
        assert_eq!(matrix_len(at_limit + 1, 1), Err(over));
        let overflow = Word2VecError::MatrixTooLarge { words: 2, dimension: usize::MAX };
        assert_eq!(matrix_len(2, usize::MAX), Err(overflow));
        let bytes = Word2VecError::MatrixTooLarge { words: 1, dimension: usize::MAX / 2 };
        assert_eq!(matrix_len(1, usize::MAX / 2), Err(bytes));
    }

    #[test]
    fn context_stays_inside_the_sentence_and_skips_the_center() {
        assert_eq!(context(5, 0, 2).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(context(5, 2, 1).collect::<Vec<_>>(), [1, 3]);
        assert_eq!(context(5, 4, 9).collect::<Vec<_>>(), [0, 1, 2, 3]);
        assert_eq!(context(1, 0, 3).count(), 0);
    }

    #[test]
    fn sigmoid_saturates_beyond_max_exp() {
        assert_eq!(sigmoid(7.0), 1.0);
        assert_eq!(sigmoid(-7.0), 0.0);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-7);
    }
}
