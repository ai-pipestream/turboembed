//! Parity with the neural part-of-speech taggers of the OpenNLP addons
//! (`neural-postag`) on a JVM: the goldens of `PostagNeuralGolden.java` in
//! `analysis/goldens-tool`, under `analysis/goldens/data/postag-neural/<run>`.
//!
//! Each run trained one model on the JVM from `inputs/tagged/wordtag.txt`
//! and wrote two goldens: `tagged/wordtag.golden` (the training sentences,
//! with the model file) and `sequence/sentences.golden` (a fixed token
//! list). Every golden of every run is read and compared:
//!
//! - The model file loads, writes back to the same bytes, and has the
//!   recorded magic, length, SHA-256 and tags.
//! - Feedforward runs: every context symbol, embedding row, vector row and
//!   emission of the left to right decoding, the tags, the probabilities
//!   and the sequence scores are exact.
//! - BiLSTM runs: the tags and the CRF Viterbi paths are exact. The
//!   emissions may differ in the last bits: the LSTM gate sigmoids call
//!   `Math.exp`, which HotSpot computes with an intrinsic routine that is
//!   not ported (see the sigmoid in `src/neural/lstm.rs`), and so do the
//!   CRF's normalizations (`Math.exp` and `Math.log`). The emissions, the
//!   CRF marginals (from the Rust emissions and from the JVM's), the tag
//!   probabilities and the sequence scores are within [`BILSTM_TOLERANCE`],
//!   and the test reports how many are bit for bit.
//! - Training: the Rust trainers are run with each run's settings on the
//!   same corpus. The feedforward model file is the JVM's, byte for byte,
//!   which also checks the `java.util.Random` stream the trainers draw
//!   from. A BiLSTM model trained in Rust scores the training sentences
//!   within [`BILSTM_TRAINED_TOLERANCE`] of the JVM's model.

mod check;
mod golden;
mod packed;
mod run;

use turbo_analysis::Annotator;
use turbo_postag::neural::{
    BilstmPosModel, BilstmPosTagger, BilstmPosTrainer, FeedforwardPosModel, FeedforwardPosTrainer,
};

use crate::check::*;
use crate::golden::*;
use crate::run::*;

/// Loads the JVM's model of `run`, checks that it writes back to the same
/// bytes, and compares both goldens.
fn check_inference(run: &Run, totals: &mut Totals) {
    let (name, bytes) = (&run.name, &run.bytes);
    let mut written = Vec::new();
    if run.is_bilstm() {
        let model = BilstmPosModel::load(bytes.as_slice()).unwrap_or_else(|e| panic!("{name}: {e}"));
        model.serialize(&mut written).unwrap();
        assert!(&written == bytes, "{name}: the model writes back to other bytes");
        let tagger = BilstmPosTagger::jvm_parity(model.clone());
        let model_ref = Annotator::model(&tagger).expect("a ModelRef");
        assert_eq!(model_ref.sha256, sha256_hex(bytes), "{name}: ModelRef");
        for (what, g) in run.goldens() {
            assert_eq!(strs(layer(g, "turbo:neural-tags"), "tag"), model.tags(), "{what}: tags");
            check_bilstm(&what, g, &model, &tagger, totals);
            packed::check_packed(&what, g, &model, totals);
            totals.goldens += 1;
        }
    } else {
        let model = FeedforwardPosModel::load(bytes.as_slice()).unwrap_or_else(|e| panic!("{name}: {e}"));
        model.serialize(&mut written).unwrap();
        assert!(&written == bytes, "{name}: the model writes back to other bytes");
        assert_eq!(model.model_ref().sha256, sha256_hex(bytes), "{name}: ModelRef");
        for (what, g) in run.goldens() {
            assert_eq!(strs(layer(g, "turbo:neural-tags"), "tag"), model.tags(), "{what}: tags");
            check_feedforward(&what, g, &model, totals);
            totals.goldens += 1;
        }
    }
    totals.runs += 1;
}

/// Trains a model in Rust with the run's settings and corpus and compares
/// it with the JVM's: the feedforward file byte for byte, the BiLSTM
/// emissions on the training sentences within [`BILSTM_TRAINED_TOLERANCE`].
/// Returns the count of compared emissions, empty for a feedforward run.
fn check_training(run: &Run) -> Count {
    let name = &run.name;
    let mut count = Count::default();
    if run.is_bilstm() {
        let settings = bilstm_settings(run.settings());
        let trained = if run.with_vectors() {
            BilstmPosTrainer::train_with_vectors(&run.samples, settings, Some(vectors), None, None)
        } else {
            BilstmPosTrainer::train(&run.samples, settings)
        }
        .unwrap_or_else(|e| panic!("{name}: training: {e}"));
        let tags = strs(layer(&run.training, "turbo:neural-tags"), "tag");
        assert_eq!(trained.tags(), tags, "{name}: trained tags");
        let emissions = table(&run.training, "turbo:neural-emissions", "score", &run.train_sentences, tags.len());
        for (k, (sentence, jvm_emissions)) in run.train_sentences.iter().zip(&emissions).enumerate() {
            for (index, (row, jvm_row)) in trained.score(sentence).unwrap().iter().zip(jvm_emissions).enumerate() {
                assert_eq!(row.len(), jvm_row.len(), "{name}: sentence {k} token {index}: scores");
                for (tag, (&score, &jvm)) in row.iter().zip(jvm_row).enumerate() {
                    count.compare(score, jvm, BILSTM_TRAINED_TOLERANCE, &|| {
                        format!("{name}: trained in Rust: sentence {k} token {index} tag {tag}: emission")
                    });
                }
            }
        }
    } else {
        let settings = feedforward_settings(run.settings());
        let trained = if run.with_vectors() {
            FeedforwardPosTrainer::train_with_vectors(&run.samples, settings, vectors)
        } else {
            FeedforwardPosTrainer::train(&run.samples, settings)
        }
        .unwrap_or_else(|e| panic!("{name}: training: {e}"));
        let mut trained_bytes = Vec::new();
        trained.serialize(&mut trained_bytes).unwrap();
        assert!(trained_bytes == run.bytes, "{name}: the model trained in Rust is not the JVM's file");
    }
    count
}

#[test]
fn every_neural_postag_golden_matches_the_jvm() {
    let mut totals = Totals::default();
    for t in each_run(&runs(), |run| {
        let mut totals = Totals::default();
        check_inference(run, &mut totals);
        totals
    }) {
        totals += t;
    }
    assert_eq!(totals.goldens, 16, "goldens");
    eprintln!(
        "{} runs, {} goldens, {} sentences: {} tags and {} feedforward context slots equal",
        totals.runs, totals.goldens, totals.sentences, totals.tags, totals.contexts
    );
    eprintln!("feedforward emissions: {}", totals.ff_emissions);
    eprintln!("BiLSTM emissions: {}", totals.bilstm_emissions);
    eprintln!("CRF marginals of the JVM's emissions: {}", totals.marginals_of_jvm_emissions);
    eprintln!("CRF marginals: {}", totals.marginals);
    eprintln!("tag probabilities: {}", totals.probabilities);
    eprintln!("sequence scores: {}", totals.sequence_scores);
    eprintln!("packed (default numerics): {} tags equal", totals.packed_tags);
    eprintln!("packed emissions: {}", totals.packed_emissions);
    eprintln!("packed tag probabilities: {}", totals.packed_probabilities);
    if let Some(device) = packed::test_device() {
        assert!(totals.device_tags > 0, "no golden ran on {device:?}");
        eprintln!("packed on {device:?}: {} tags equal", totals.device_tags);
        eprintln!("packed emissions on {device:?}: {}", totals.device_emissions);
        eprintln!("packed tag probabilities on {device:?}: {}", totals.device_probabilities);
        let (n, same) = (totals.fastest_tags, totals.fastest_equal);
        eprintln!("FASTEST on {device:?}: {same} of {n} tags equal the JVM's");
        assert!(n > 0 && same as f64 >= 0.999 * n as f64, "FASTEST on {device:?}: {same} of {n} tags equal");
    }
}

#[test]
fn trainers_reproduce_the_jvm_models() {
    let runs = runs();
    let mut trained = Count::default();
    for c in each_run(&runs, check_training) {
        trained += c;
    }
    let feedforward = runs.iter().filter(|r| !r.is_bilstm()).count();
    eprintln!("{feedforward} feedforward models trained in Rust are the JVM's files byte for byte");
    eprintln!("emissions of the BiLSTM models trained in Rust: {trained}");
}
