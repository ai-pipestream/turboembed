//! Boundary regressions for the feedforward POS trainer (#675): a sample whose tags do not line
//! up with its tokens must fail with a typed error, as the BiLSTM trainer does, never panic or
//! train on a truncated sentence.

use turbo_postag::neural::feedforward::{FeedforwardPosTrainer, FeedforwardSettings};
use turbo_postag::neural::{NeuralError, PosSample};

fn sample(sentence: &[&str], tags: &[&str]) -> PosSample {
    // The fields are public, so a sample can bypass `PosSample::new`'s length check.
    PosSample {
        sentence: sentence.iter().map(|s| (*s).to_owned()).collect(),
        tags: tags.iter().map(|s| (*s).to_owned()).collect(),
    }
}

fn settings() -> FeedforwardSettings {
    FeedforwardSettings::new(4, 4, 1, 2, 0.05, 0.0, 0.0, 1, 1, 7).unwrap()
}

#[test]
fn trainer_rejects_a_sample_with_fewer_tags_than_tokens() {
    let samples = [sample(&["the", "dog"], &["DT", "NN"]), sample(&["a", "cat", "sat"], &["DT", "NN"])];
    let err = FeedforwardPosTrainer::train(&samples, settings()).unwrap_err();
    assert!(matches!(err, NeuralError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn trainer_rejects_a_sample_with_more_tags_than_tokens() {
    let samples = [sample(&["the", "dog"], &["DT", "NN", "VB"])];
    let err = FeedforwardPosTrainer::train(&samples, settings()).unwrap_err();
    assert!(matches!(err, NeuralError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn trainer_still_trains_on_aligned_samples() {
    let samples = [sample(&["the", "dog"], &["DT", "NN"]), sample(&["a", "cat"], &["DT", "NN"])];
    let model = FeedforwardPosTrainer::train(&samples, settings()).unwrap();
    assert_eq!(model.tags(), ["DT", "NN"]);
}
