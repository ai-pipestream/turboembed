//! A two-tag BiLSTM-CRF model small enough to write out in a test, for
//! the tests that need both numerics on one model.

use turbo_postag::neural::bilstm_model::{BilstmPosModel, UNKNOWN, char_vocabulary_from_strs};
use turbo_postag::neural::lstm::LstmLayer;

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

/// Two tags, every weight zero but a uniform CRF transition matrix of
/// `transition`: every emission is zero, so each token's marginal is 1/2.
pub fn model(transition: f64) -> BilstmPosModel {
    model_with(transition, 0.0, 0.0, 0.0)
}

/// [`model`] with the forward char LSTM's gate biases at `char_bias`, tag
/// A's output bias at `out_bias`, and the unknown character's embedding
/// starting with `char_embedding`.
pub fn model_with(transition: f64, char_bias: f64, out_bias: f64, char_embedding: f64) -> BilstmPosModel {
    let mut char_forward = lstm(3, 2);
    if char_bias != 0.0 {
        char_forward =
            LstmLayer::of_weights(3, 2, vec![vec![0.0; 3]; 8], vec![vec![0.0; 2]; 8], vec![char_bias; 8]).unwrap();
    }
    BilstmPosModel::new(
        vec![(UNKNOWN.to_owned(), 0)],
        char_vocabulary_from_strs([(UNKNOWN, 0)]),
        vec!["A".into(), "B".into()],
        vec![vec![0.0; 2]],
        vec![vec![char_embedding, 0.0, 0.0]],
        char_forward,
        lstm(3, 2),
        lstm(6, 2),
        lstm(6, 2),
        None,
        None,
        vec![vec![0.0; 4]; 2],
        vec![out_bias, 0.0],
        5,
        None,
        None,
        Some(vec![vec![transition; 2]; 2]),
        Some(vec![0.0; 2]),
        Some(vec![0.0; 2]),
        None,
        None,
    )
    .unwrap()
}
