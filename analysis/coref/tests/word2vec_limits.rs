//! Word2vec size limits and numerical divergence: oversized matrices
//! and runaway updates are typed errors, never a panic, a huge
//! allocation or non-finite vectors.

use turbo_coref::{Word2VecArch, Word2VecError, Word2VecParameters, Word2VecTrainer};

fn run(p: Word2VecParameters, corpus: &[Vec<&str>]) -> Result<(), Word2VecError> {
    Word2VecTrainer::new(p)?.train(corpus).map(|_| ())
}

#[test]
fn huge_dimension_is_refused_before_any_allocation() {
    // Codex's boundary case on 7cb0c500: validate() passed, then
    // `vocab.len() * dimension` overflowed in sgd::train.
    let p = Word2VecParameters { dimension: usize::MAX, min_count: 1, ..Word2VecParameters::default() };
    let err = run(p, &[vec!["a", "b"]]).unwrap_err();
    assert!(matches!(err, Word2VecError::DimensionTooLarge(usize::MAX)), "{err:?}");
}

#[test]
fn matrix_past_the_allocation_limit_is_refused_after_the_vocabulary() {
    use turbo_coref::word2vec::{MAX_DIMENSION, MAX_MATRIX_BYTES};
    // One word more than a MAX_DIMENSION matrix can hold within the limit.
    let words = (MAX_MATRIX_BYTES / (MAX_DIMENSION * size_of::<f32>()) as u64) as usize + 1;
    let names: Vec<String> = (0..words).map(|i| format!("w{i}")).collect();
    let corpus = vec![names.iter().map(String::as_str).collect::<Vec<_>>()];
    let p = Word2VecParameters { dimension: MAX_DIMENSION, min_count: 1, ..Word2VecParameters::default() };
    let err = run(p, &corpus).unwrap_err();
    assert_eq!(err, Word2VecError::MatrixTooLarge { words, dimension: MAX_DIMENSION });
}

fn runaway(arch: Word2VecArch) -> Word2VecParameters {
    Word2VecParameters {
        arch,
        dimension: 4,
        epochs: 2,
        min_count: 1,
        subsample: 0.0,
        learning_rate: f32::MAX,
        min_learning_rate: f32::MAX,
        ..Word2VecParameters::default()
    }
}

fn runaway_corpus() -> Vec<Vec<&'static str>> {
    vec![vec!["a", "b", "c", "a", "c", "b"]; 4]
}

#[test]
fn skip_gram_divergence_is_an_error_not_nan_vectors() {
    // Codex's case on 7cb0c500: Ok with [NaN, NaN, NaN, NaN] for "a".
    let err = run(runaway(Word2VecArch::SkipGram), &runaway_corpus()).unwrap_err();
    assert!(matches!(err, Word2VecError::Diverged { .. }), "{err:?}");
}

#[test]
fn cbow_divergence_is_an_error_not_nan_vectors() {
    let err = run(runaway(Word2VecArch::Cbow), &runaway_corpus()).unwrap_err();
    assert!(matches!(err, Word2VecError::Diverged { .. }), "{err:?}");
}
