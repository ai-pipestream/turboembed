//! `BilstmPosModel::new` boundaries: an in-memory model must hold the shape, value, and
//! vocabulary contract `load` enforces on the wire format, including every LSTM layer's
//! own `W`, `U`, and `b` tensors.

use turbo_postag::neural::NeuralError;
use turbo_postag::neural::bilstm_model::{BilstmPosModel, UNKNOWN, char_vocabulary_from_strs};
use turbo_postag::neural::lstm::LstmLayer;

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
    word_forward2: Option<LstmLayer>,
    word_backward2: Option<LstmLayer>,
    output_weights: Vec<Vec<f64>>,
    output_bias: Vec<f64>,
    pretrained_entries: Option<Vec<(String, usize)>>,
    pretrained_vectors: Option<Vec<Vec<f32>>>,
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
        word_forward2: None,
        word_backward2: None,
        output_weights: vec![vec![0.0; 4]; 2],
        output_bias: vec![0.0; 2],
        pretrained_entries: None,
        pretrained_vectors: None,
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
        p.word_forward2,
        p.word_backward2,
        p.output_weights,
        p.output_bias,
        5,
        p.pretrained_entries,
        p.pretrained_vectors,
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

// ---- BilstmPosModel::new LSTM tensor contract --------------------------------

fn rejected(p: Parts, case: &str) {
    let result = build(p);
    assert!(matches!(result, Err(NeuralError::InvalidArgument(_))), "{case}: {:?}", result.err());
}

#[test]
fn new_rejects_ragged_lstm_weight_rows() {
    let mut p = parts();
    p.word_forward.w_mut()[0].pop();
    rejected(p, "word forward W row 0 one short");

    let mut p = parts();
    p.char_forward.w_mut()[3].push(0.0);
    rejected(p, "char forward W row 3 one long");

    let mut p = parts();
    p.char_backward.u_mut()[1].pop();
    rejected(p, "char backward U row 1 one short");

    let mut p = parts();
    p.word_backward.u_mut()[7].clear();
    rejected(p, "word backward U row 7 empty");

    // Every row narrowed by the same amount: consistent as a matrix, so only a check against
    // the layer's own input and hidden sizes catches it.
    let mut p = parts();
    p.word_forward.w_mut().iter_mut().for_each(|row| {
        row.pop();
    });
    rejected(p, "word forward W uniformly one column short of input_size");

    let mut p = parts();
    p.char_forward.u_mut().iter_mut().for_each(|row| row.push(0.0));
    rejected(p, "char forward U uniformly one column past hidden_size");
}

#[test]
fn new_rejects_non_finite_lstm_weights() {
    let mut p = parts();
    p.char_forward.w_mut()[0][0] = f64::NAN;
    rejected(p, "NaN in char forward W");

    let mut p = parts();
    p.char_forward.u_mut()[2][1] = f64::INFINITY;
    rejected(p, "Inf in char forward U");

    let mut p = parts();
    p.char_forward.b_mut()[5] = f64::NEG_INFINITY;
    rejected(p, "-Inf in char forward b");

    let mut p = parts();
    p.word_forward.w_mut()[4][5] = f64::INFINITY;
    rejected(p, "Inf in word forward W");

    let mut p = parts();
    p.word_forward.u_mut()[0][0] = f64::NAN;
    rejected(p, "NaN in word forward U");

    let mut p = parts();
    p.word_forward.b_mut()[0] = f64::NAN;
    rejected(p, "NaN in word forward b");

    let mut p = parts();
    p.word_backward.b_mut()[7] = f64::INFINITY;
    rejected(p, "Inf in word backward b");
}

/// The base parts with a second word layer pair: input 2 x first-layer hidden, hidden 2,
/// so the output width stays 4.
fn stacked_parts() -> Parts {
    let mut p = parts();
    p.word_forward2 = Some(lstm(4, 2));
    p.word_backward2 = Some(lstm(4, 2));
    p
}

#[test]
fn new_checks_the_second_word_layer_tensors() {
    assert!(build(stacked_parts()).is_ok(), "the well-formed stacked model");

    let mut p = stacked_parts();
    p.word_forward2.as_mut().unwrap().w_mut()[2].pop();
    rejected(p, "second forward W row 2 one short");

    let mut p = stacked_parts();
    p.word_backward2.as_mut().unwrap().u_mut()[0][1] = f64::NAN;
    rejected(p, "NaN in second backward U");

    let mut p = stacked_parts();
    p.word_backward2.as_mut().unwrap().b_mut()[3] = f64::INFINITY;
    rejected(p, "Inf in second backward b");
}

// ---- BilstmPosModel::new vocabulary contract shared with load ----------------

#[test]
fn new_rejects_duplicate_tags() {
    let mut p = parts();
    p.tags = vec!["A".into(), "A".into()];
    rejected(p, "duplicate tag A");
}

#[test]
fn new_rejects_duplicate_word_and_char_entries() {
    let mut p = parts();
    p.word_embeddings = vec![vec![0.0; 2]; 2];
    p.word_entries = vec![(UNKNOWN.to_owned(), 0), ("dog".into(), 1), ("dog".into(), 0)];
    rejected(p, "duplicate word entry dog");

    let mut p = parts();
    p.char_embeddings = vec![vec![0.0; 3]; 2];
    p.char_entries = char_vocabulary_from_strs([(UNKNOWN, 0), ("d", 1), ("d", 1)]);
    rejected(p, "duplicate char entry d");
}

#[test]
fn new_requires_unknown_at_row_zero() {
    let mut p = parts();
    p.word_entries = vec![("dog".into(), 0)];
    rejected(p, "no UNKNOWN word entry");

    let mut p = parts();
    p.word_embeddings = vec![vec![0.0; 2]; 2];
    p.word_entries = vec![("dog".into(), 0), (UNKNOWN.to_owned(), 1)];
    rejected(p, "UNKNOWN word entry at row 1");

    let mut p = parts();
    p.char_entries = char_vocabulary_from_strs([("d", 0)]);
    rejected(p, "no UNKNOWN char entry");

    let mut p = parts();
    p.char_embeddings = vec![vec![0.0; 3]; 2];
    p.char_entries = char_vocabulary_from_strs([("d", 0), (UNKNOWN, 1)]);
    rejected(p, "UNKNOWN char entry at row 1");
}

/// The base parts with one-column pretrained vectors, which widen the word LSTM input to 7.
fn pretrained_parts(entries: Vec<(String, usize)>) -> Parts {
    let mut p = parts();
    p.word_forward = lstm(7, 2);
    p.word_backward = lstm(7, 2);
    p.pretrained_entries = Some(entries);
    p.pretrained_vectors = Some(vec![vec![0.5]; 2]);
    p
}

#[test]
fn new_rejects_duplicate_pretrained_entries() {
    assert!(build(pretrained_parts(vec![("dog".into(), 0), ("cat".into(), 1)])).is_ok(), "unique pretrained entries");
    rejected(pretrained_parts(vec![("dog".into(), 0), ("dog".into(), 1)]), "duplicate pretrained entry dog");
}

#[test]
fn every_model_new_accepts_round_trips_through_load() {
    for (case, p) in [
        ("plain", parts()),
        ("stacked", stacked_parts()),
        ("pretrained", pretrained_parts(vec![("dog".into(), 0), ("cat".into(), 1)])),
    ] {
        let model = build(p).unwrap_or_else(|e| panic!("{case}: {e:?}"));
        let mut bytes = Vec::new();
        model.serialize(&mut bytes).unwrap();
        let loaded = BilstmPosModel::load(bytes.as_slice()).unwrap_or_else(|e| panic!("{case}: load refused {e:?}"));
        assert_eq!(loaded.score(&["hello"]).unwrap(), model.score(&["hello"]).unwrap(), "{case}");
    }
}
