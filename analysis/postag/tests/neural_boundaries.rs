//! Boundary regressions for the CRF scorer and the BiLSTM-CRF model (#649):
//! dimension mismatches, empty tag sets, and out-of-range gold indices must
//! fail with typed errors, never panic, and `BilstmPosModel::new` must hold
//! the shape contract `load` enforces.

use turbo_postag::neural::NeuralError;
use turbo_postag::neural::bilstm_model::{BilstmPosModel, UNKNOWN, char_vocabulary_from_strs};
use turbo_postag::neural::crf::CrfScorer;
use turbo_postag::neural::lstm::LstmLayer;

// ---- CrfScorer dimensions ---------------------------------------------------

type CrfInputs = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>);

fn crf_inputs() -> CrfInputs {
    let emissions = vec![vec![0.4, -0.3], vec![-0.2, 0.5]];
    let transitions = vec![vec![0.1, -0.5], vec![0.4, 0.2]];
    let start = vec![0.2, -0.1];
    let end = vec![-0.15, 0.25];
    (emissions, transitions, start, end)
}

#[test]
fn viterbi_rejects_ragged_or_mismatched_dimensions() {
    let (emissions, transitions, start, end) = crf_inputs();

    // An emission row short of the tag count.
    let mut bad = emissions.clone();
    bad[1] = vec![0.1];
    assert!(matches!(CrfScorer::viterbi(&bad, &transitions, &start, &end), Err(NeuralError::InvalidArgument(_))));
    assert!(matches!(CrfScorer::marginals(&bad, &transitions, &start, &end), Err(NeuralError::InvalidArgument(_))));

    // A transition row count or width that is not the tag count.
    let bad_tr = vec![vec![0.1, -0.5]];
    assert!(matches!(CrfScorer::viterbi(&emissions, &bad_tr, &start, &end), Err(NeuralError::InvalidArgument(_))));
    let bad_tr = vec![vec![0.1, -0.5, 0.0], vec![0.4, 0.2, 0.0]];
    assert!(matches!(CrfScorer::viterbi(&emissions, &bad_tr, &start, &end), Err(NeuralError::InvalidArgument(_))));

    // Start and end of a length other than the tag count.
    assert!(matches!(CrfScorer::viterbi(&emissions, &transitions, &[0.0], &end), Err(NeuralError::InvalidArgument(_))));
    assert!(matches!(
        CrfScorer::viterbi(&emissions, &transitions, &start, &[0.0, 0.0, 0.0]),
        Err(NeuralError::InvalidArgument(_))
    ));

    // No tags at all, with tokens to score: an error, not a made-up path.
    let two_rows = vec![vec![0.1], vec![0.2]];
    assert!(matches!(CrfScorer::viterbi(&two_rows, &[], &[], &[]), Err(NeuralError::InvalidArgument(_))));
    assert!(matches!(CrfScorer::marginals(&two_rows, &[], &[], &[]), Err(NeuralError::InvalidArgument(_))));

    // Empty input is still an empty path, and honest inputs still decode.
    assert_eq!(CrfScorer::viterbi(&[], &transitions, &start, &end).unwrap(), Vec::<usize>::new());
    assert_eq!(CrfScorer::viterbi(&emissions, &transitions, &start, &end).unwrap().len(), 2);
}

#[test]
fn viterbi_rejects_non_finite_scores() {
    let (emissions, transitions, start, end) = crf_inputs();
    let mut bad = emissions.clone();
    bad[0][1] = f64::NAN;
    assert!(matches!(CrfScorer::viterbi(&bad, &transitions, &start, &end), Err(NeuralError::NonFiniteScore(_))));
    let mut bad_tr = transitions.clone();
    bad_tr[0][0] = f64::INFINITY;
    assert!(matches!(CrfScorer::viterbi(&emissions, &bad_tr, &start, &end), Err(NeuralError::NonFiniteScore(_))));
}

#[test]
fn loss_and_gradients_rejects_bad_gold() {
    let (emissions, transitions, start, end) = crf_inputs();
    let d_emissions = vec![vec![0.0; 2]; 2];
    let d_transitions = vec![vec![0.0; 2]; 2];
    let d_start = vec![0.0; 2];
    let d_end = vec![0.0; 2];
    let run = |gold: &[usize]| {
        CrfScorer::loss_and_gradients(
            &emissions,
            &transitions,
            &start,
            &end,
            gold,
            &mut d_emissions.clone(),
            &mut d_transitions.clone(),
            &mut d_start.clone(),
            &mut d_end.clone(),
        )
    };
    // A gold sequence short of the token count.
    assert!(matches!(run(&[1]), Err(NeuralError::InvalidArgument(_))));
    // A gold tag outside the tag set.
    assert!(matches!(run(&[0, 2]), Err(NeuralError::InvalidArgument(_))));
    // A valid gold sequence still scores.
    assert!(run(&[0, 1]).is_ok());
}

// ---- BilstmPosModel::new shape contract -------------------------------------

fn lstm(input: usize, hidden: usize) -> LstmLayer {
    let gates = 4 * hidden;
    LstmLayer::of_weights(
        input,
        hidden,
        vec![vec![0.0; input]; gates],
        vec![vec![0.0; hidden]; gates],
        vec![0.0; gates],
    )
    .unwrap()
}

struct Parts {
    word_entries: Vec<(String, usize)>,
    char_entries: Vec<(Vec<u8>, usize)>,
    tags: Vec<String>,
    word_embeddings: Vec<Vec<f64>>,
    char_embeddings: Vec<Vec<f64>>,
    char_forward: LstmLayer,
    char_backward: LstmLayer,
    word_forward: LstmLayer,
    word_backward: LstmLayer,
    output_weights: Vec<Vec<f64>>,
    output_bias: Vec<f64>,
}

fn parts() -> Parts {
    // word_dim 2, char dim 3, char hidden 2, word input 2 + 4 = 6, word hidden 2, tags 2.
    Parts {
        word_entries: vec![(UNKNOWN.to_owned(), 0)],
        char_entries: char_vocabulary_from_strs([(UNKNOWN, 0)]),
        tags: vec!["A".into(), "B".into()],
        word_embeddings: vec![vec![0.0; 2]],
        char_embeddings: vec![vec![0.0; 3]],
        char_forward: lstm(3, 2),
        char_backward: lstm(3, 2),
        word_forward: lstm(6, 2),
        word_backward: lstm(6, 2),
        output_weights: vec![vec![0.0; 4]; 2],
        output_bias: vec![0.0; 2],
    }
}

fn build(p: Parts) -> Result<BilstmPosModel, NeuralError> {
    BilstmPosModel::new(
        p.word_entries,
        p.char_entries,
        p.tags,
        p.word_embeddings,
        p.char_embeddings,
        p.char_forward,
        p.char_backward,
        p.word_forward,
        p.word_backward,
        None,
        None,
        p.output_weights,
        p.output_bias,
        5,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
}

#[test]
fn new_rejects_empty_or_ragged_embedding_tables() {
    let mut p = parts();
    p.word_embeddings = Vec::new();
    assert!(matches!(build(p), Err(NeuralError::InvalidArgument(_))), "empty word embeddings");

    let mut p = parts();
    p.char_embeddings = Vec::new();
    assert!(matches!(build(p), Err(NeuralError::InvalidArgument(_))), "empty char embeddings");

    let mut p = parts();
    p.word_embeddings = vec![vec![0.0; 2], vec![0.0; 3]];
    p.word_entries.push(("dog".into(), 1));
    assert!(matches!(build(p), Err(NeuralError::InvalidArgument(_))), "ragged word embeddings");

    let mut p = parts();
    p.word_embeddings[0][1] = f64::NAN;
    assert!(matches!(build(p), Err(NeuralError::InvalidArgument(_))), "non-finite weight");
}

#[test]
fn new_rejects_vocabulary_rows_past_the_tables() {
    let mut p = parts();
    p.word_entries.push(("dog".into(), 5));
    assert!(matches!(build(p), Err(NeuralError::InvalidArgument(_))), "word row 5 of 1");

    let mut p = parts();
    let d_key = char_vocabulary_from_strs([("d", 0)])[0].0.clone();
    p.char_entries.push((d_key, 2));
    assert!(matches!(build(p), Err(NeuralError::InvalidArgument(_))), "char row 2 of 1");
}

#[test]
fn new_rejects_output_and_crf_shape_mismatches() {
    let mut p = parts();
    p.output_weights = vec![vec![0.0; 3]; 2];
    assert!(matches!(build(p), Err(NeuralError::InvalidArgument(_))), "narrow output rows");

    let mut p = parts();
    p.output_bias = vec![0.0];
    assert!(matches!(build(p), Err(NeuralError::InvalidArgument(_))), "short output bias");

    let mut p = parts();
    p.tags = vec!["A".into(), "B".into(), "C".into()];
    assert!(matches!(build(p), Err(NeuralError::InvalidArgument(_))), "output rows != tags");
}

#[test]
fn new_rejects_encoder_width_mismatch() {
    let mut p = parts();
    p.word_forward = lstm(5, 2);
    assert!(matches!(build(p), Err(NeuralError::InvalidArgument(_))), "word forward input 5 != 6");

    let mut p = parts();
    p.char_backward = lstm(3, 3);
    assert!(matches!(build(p), Err(NeuralError::InvalidArgument(_))), "char hidden mismatch");

    let mut p = parts();
    p.word_backward = lstm(6, 3);
    assert!(matches!(build(p), Err(NeuralError::InvalidArgument(_))), "word hidden mismatch");
}

#[test]
fn new_accepts_and_scores_the_valid_model() {
    let model = build(parts()).unwrap();
    let scores = model.score(&["hello"]).unwrap();
    assert_eq!(scores.len(), 1);
    assert_eq!(scores[0].len(), 2);
}
