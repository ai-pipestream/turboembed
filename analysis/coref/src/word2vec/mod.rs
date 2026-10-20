//! Static word vectors trained from a tokenized corpus with word2vec
//! (Mikolov et al. 2013): skip-gram or continuous bag of words, both
//! with negative sampling.
//!
//! [`Word2VecTrainer::train`] builds a [`Vocabulary`] (words seen at
//! least `min_count` times, most frequent first, ties in byte order,
//! optionally cut to `max_vocab`), drops frequent words with
//! probability `1 - sqrt(t / f(w))`, draws noise words from the unigram
//! distribution raised to the power 0.75, and decays the learning rate
//! linearly to a floor. One thread and a seeded generator make a run
//! reproducible: the same parameters and corpus give the same vectors.
//! The result, [`Word2VecVectors`], is a [`WordVectors`] provider for
//! [`CorefAnnotator`](crate::CorefAnnotator)'s similarity features.

use std::fmt;
use std::hash::Hasher;

use crate::WordVectors;
use crate::hash::Fnv;
use crate::input::lowercase;

mod output;
mod rng;
mod sgd;
mod text;
mod vocab;

pub use output::Word2VecVectors;
pub use text::read_text;
pub use vocab::Vocabulary;

/// The largest vector length [`Word2VecParameters::validate`] accepts.
pub const MAX_DIMENSION: usize = 1 << 16;

/// The largest size in bytes of each of the two training matrices
/// (vocabulary words by dimension `f32`s), checked before either is
/// allocated.
pub const MAX_MATRIX_BYTES: u64 = 1 << 34;

/// The training objective.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Word2VecArch {
    /// Each word predicts each word of its context.
    SkipGram,
    /// The mean of the context vectors predicts the word.
    Cbow,
}

/// Training settings; [`Default`] gives word2vec's usual values.
#[derive(Clone, Debug, PartialEq)]
pub struct Word2VecParameters {
    pub arch: Word2VecArch,
    /// The vector length, positive and at most [`MAX_DIMENSION`].
    pub dimension: usize,
    /// The largest context reach on each side, positive; each position
    /// draws its reach uniformly from `1..=window`.
    pub window: usize,
    /// Noise words per prediction, positive.
    pub negative: usize,
    /// Passes over the corpus, positive.
    pub epochs: usize,
    /// The fewest occurrences a word needs to get a vector.
    pub min_count: u64,
    /// Keeps only this many of the most frequent words; positive.
    pub max_vocab: Option<usize>,
    /// The starting learning rate, finite and positive.
    pub learning_rate: f32,
    /// The floor of the decayed rate, finite, at least 0 and at most
    /// `learning_rate`.
    pub min_learning_rate: f32,
    /// The subsampling threshold `t`, finite and at least 0; 0 keeps
    /// every occurrence.
    pub subsample: f64,
    /// Lowercases tokens as `String.toLowerCase(Locale.ROOT)` does, the
    /// form [`WordVectors::vector`] is asked for.
    pub lowercase: bool,
    /// Seeds the initial vectors, subsampling, window and noise draws.
    pub seed: u64,
}

impl Default for Word2VecParameters {
    fn default() -> Word2VecParameters {
        Word2VecParameters {
            arch: Word2VecArch::SkipGram,
            dimension: 100,
            window: 5,
            negative: 5,
            epochs: 5,
            min_count: 5,
            max_vocab: None,
            learning_rate: 0.025,
            min_learning_rate: 0.025 * 1e-4,
            subsample: 1e-3,
            lowercase: true,
            seed: 1,
        }
    }
}

/// Why parameters or a corpus cannot be trained on.
#[derive(Clone, Debug, PartialEq)]
pub enum Word2VecError {
    ZeroDimension,
    /// Above [`MAX_DIMENSION`].
    DimensionTooLarge(usize),
    ZeroWindow,
    ZeroNegative,
    ZeroEpochs,
    ZeroMaxVocab,
    /// Not finite or not positive.
    LearningRate(f32),
    /// Not finite, negative, or above the learning rate.
    MinLearningRate(f32),
    /// Not finite or negative.
    Subsample(f64),
    /// The corpus has no tokens.
    EmptyCorpus,
    /// No word occurs `min_count` times.
    EmptyVocabulary {
        min_count: u64,
    },
    /// A training matrix of this many words by this dimension would
    /// pass [`MAX_MATRIX_BYTES`].
    MatrixTooLarge {
        words: usize,
        dimension: usize,
    },
    /// A weight stopped being finite during this epoch (from 1),
    /// typically from a learning rate too large for the data.
    Diverged {
        epoch: usize,
    },
}

impl fmt::Display for Word2VecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Word2VecError::ZeroDimension => write!(f, "word2vec dimension must be positive"),
            Word2VecError::DimensionTooLarge(d) => {
                write!(f, "word2vec dimension {d} is above the maximum {MAX_DIMENSION}")
            }
            Word2VecError::ZeroWindow => write!(f, "word2vec window must be positive"),
            Word2VecError::ZeroNegative => write!(f, "word2vec negative samples must be positive"),
            Word2VecError::ZeroEpochs => write!(f, "word2vec epochs must be positive"),
            Word2VecError::ZeroMaxVocab => write!(f, "word2vec max_vocab must be positive"),
            Word2VecError::LearningRate(v) => write!(f, "word2vec learning rate {v} must be finite and positive"),
            Word2VecError::MinLearningRate(v) => {
                write!(f, "word2vec minimum learning rate {v} must be finite, non-negative and at most the rate")
            }
            Word2VecError::Subsample(v) => {
                write!(f, "word2vec subsample threshold {v} must be finite and non-negative")
            }
            Word2VecError::EmptyCorpus => write!(f, "word2vec corpus has no tokens"),
            Word2VecError::EmptyVocabulary { min_count } => {
                write!(f, "word2vec vocabulary is empty: no word occurs {min_count} times")
            }
            Word2VecError::MatrixTooLarge { words, dimension } => {
                write!(f, "word2vec matrix of {words} words by {dimension} passes {MAX_MATRIX_BYTES} bytes")
            }
            Word2VecError::Diverged { epoch } => {
                write!(f, "word2vec training diverged in epoch {epoch}: a weight is not finite")
            }
        }
    }
}

impl std::error::Error for Word2VecError {}

impl Word2VecParameters {
    /// `Err` naming the first setting out of range.
    pub fn validate(&self) -> Result<(), Word2VecError> {
        if self.dimension == 0 {
            return Err(Word2VecError::ZeroDimension);
        }
        if self.dimension > MAX_DIMENSION {
            return Err(Word2VecError::DimensionTooLarge(self.dimension));
        }
        if self.window == 0 {
            return Err(Word2VecError::ZeroWindow);
        }
        if self.negative == 0 {
            return Err(Word2VecError::ZeroNegative);
        }
        if self.epochs == 0 {
            return Err(Word2VecError::ZeroEpochs);
        }
        if self.max_vocab == Some(0) {
            return Err(Word2VecError::ZeroMaxVocab);
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(Word2VecError::LearningRate(self.learning_rate));
        }
        let floor = self.min_learning_rate;
        if !floor.is_finite() || floor < 0.0 || floor > self.learning_rate {
            return Err(Word2VecError::MinLearningRate(floor));
        }
        if !self.subsample.is_finite() || self.subsample < 0.0 {
            return Err(Word2VecError::Subsample(self.subsample));
        }
        Ok(())
    }

    fn hash_into(&self, h: &mut Fnv) {
        h.str("turbo-coref word2vec 1");
        h.write_u8(match self.arch {
            Word2VecArch::SkipGram => 0,
            Word2VecArch::Cbow => 1,
        });
        for v in [self.dimension, self.window, self.negative, self.epochs, self.max_vocab.unwrap_or(0)] {
            h.write_u64(v as u64);
        }
        h.write_u64(self.min_count);
        h.write_u32(self.learning_rate.to_bits());
        h.write_u32(self.min_learning_rate.to_bits());
        h.write_u64(self.subsample.to_bits());
        h.write_u8(self.lowercase as u8);
        h.write_u64(self.seed);
    }
}

/// Trains [`Word2VecVectors`] with validated parameters.
#[derive(Clone, Debug)]
pub struct Word2VecTrainer {
    params: Word2VecParameters,
}

impl Word2VecTrainer {
    /// `Err` when a parameter is out of range.
    pub fn new(params: Word2VecParameters) -> Result<Word2VecTrainer, Word2VecError> {
        params.validate()?;
        Ok(Word2VecTrainer { params })
    }

    pub fn parameters(&self) -> &Word2VecParameters {
        &self.params
    }

    /// The vocabulary training would use for a corpus of sentences.
    pub fn vocabulary<S, W>(&self, sentences: &[S]) -> Result<Vocabulary, Word2VecError>
    where
        S: AsRef<[W]>,
        W: AsRef<str>,
    {
        let corpus = self.normalize(sentences);
        self.build_vocabulary(&corpus)
    }

    /// Trains on a corpus of tokenized sentences; a context never
    /// crosses a sentence boundary. `Err` as well when the matrices
    /// would pass [`MAX_MATRIX_BYTES`] (checked before allocating) or a
    /// weight stops being finite (checked after every epoch).
    pub fn train<S, W>(&self, sentences: &[S]) -> Result<Word2VecVectors, Word2VecError>
    where
        S: AsRef<[W]>,
        W: AsRef<str>,
    {
        let corpus = self.normalize(sentences);
        let vocab = self.build_vocabulary(&corpus)?;
        let ids: Vec<Vec<u32>> = corpus.iter().map(|s| s.iter().filter_map(|w| vocab.id(w)).collect()).collect();
        let vectors = sgd::train(&self.params, &vocab, &ids)?;
        let mut h = Fnv::default();
        self.params.hash_into(&mut h);
        vocab.hash_into(&mut h);
        h.write_u64(corpus.len() as u64);
        for sentence in &corpus {
            h.write_u64(sentence.len() as u64);
            sentence.iter().for_each(|w| h.str(w));
        }
        Ok(Word2VecVectors::new(self.params.dimension, vocab, vectors, h.finish()))
    }

    fn normalize<S, W>(&self, sentences: &[S]) -> Vec<Vec<String>>
    where
        S: AsRef<[W]>,
        W: AsRef<str>,
    {
        let form = |w: &W| if self.params.lowercase { lowercase(w.as_ref()) } else { w.as_ref().to_owned() };
        sentences.iter().map(|s| s.as_ref().iter().map(form).collect()).collect()
    }

    fn build_vocabulary(&self, corpus: &[Vec<String>]) -> Result<Vocabulary, Word2VecError> {
        if corpus.iter().all(Vec::is_empty) {
            return Err(Word2VecError::EmptyCorpus);
        }
        let vocab = Vocabulary::build(corpus, self.params.min_count, self.params.max_vocab);
        if vocab.is_empty() {
            return Err(Word2VecError::EmptyVocabulary { min_count: self.params.min_count });
        }
        Ok(vocab)
    }
}

/// The cosine of two words' vectors, `None` when either has none or is
/// all zeros.
pub fn similarity(vectors: &dyn WordVectors, a: &str, b: &str) -> Option<f64> {
    let c = crate::features::cosine(vectors.vector(a)?, vectors.vector(b)?);
    (c >= -1.0).then_some(c)
}
