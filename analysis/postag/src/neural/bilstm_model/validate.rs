//! The shape, value, and vocabulary contract every `BilstmPosModel` holds, whether `new`
//! builds it from in-memory tensors or `load` reads it from the `ONLP-BLPT` wire format.
//!
//! The vocabulary rules live in [`check_vocabulary`] and [`check_unique_tags`], which `load`
//! calls with its own `Format` error type, so the two entry points cannot drift apart.

use std::collections::HashSet;
use std::hash::Hash;

use crate::neural::NeuralError;
use crate::neural::lstm::{LstmLayer, MAX_HIDDEN_SIZE};

use super::UNKNOWN;

/// The `UNKNOWN` symbol as the modified UTF-8 bytes a character vocabulary key holds.
pub(super) const UNKNOWN_BYTES: &[u8] = b"*UNK*";

/// One vocabulary's rules: every key distinct, every row inside a table of `row_count` rows,
/// and, when `unknown` names a key, that key present and mapped to row `0`. Returns the broken
/// rule as text so `new` and `load` can each wrap it in their own error variant.
pub(super) fn check_vocabulary<K: Eq + Hash>(
    entries: &[(K, usize)],
    row_count: usize,
    name: &str,
    unknown: Option<&K>,
) -> Result<(), String> {
    let mut seen = HashSet::with_capacity(entries.len());
    let mut unknown_row = None;
    for (key, row) in entries {
        if !seen.insert(key) {
            return Err(format!("duplicate {name} entry at row {row}"));
        }
        if *row >= row_count {
            return Err(format!("{name} row {row} >= {row_count}"));
        }
        if unknown == Some(key) {
            unknown_row = Some(*row);
        }
    }
    if unknown.is_some() && unknown_row != Some(0) {
        return Err(format!("{name} must map {UNKNOWN} to row 0"));
    }
    Ok(())
}

/// The tag set rule: every tag distinct.
pub(super) fn check_unique_tags(tags: &[String]) -> Result<(), String> {
    let mut seen = HashSet::with_capacity(tags.len());
    match tags.iter().find(|tag| !seen.insert(tag.as_str())) {
        Some(tag) => Err(format!("duplicate tag {tag}")),
        None => Ok(()),
    }
}

/// One LSTM layer's tensors against its declared sizes: both sizes positive and the hidden size
/// at most `MAX_HIDDEN_SIZE`, `w` exactly `4 * hidden` rows of `input` columns, `u` exactly
/// `4 * hidden` rows of `hidden` columns, `b` exactly `4 * hidden` values, all finite. The
/// declared sizes alone are not enough: `w_mut` and `u_mut` hand out the rows, so a caller can
/// narrow, widen, or overwrite them after `LstmLayer::of_weights` checked the layer.
fn check_lstm_tensors(
    name: &str,
    input: usize,
    hidden: usize,
    w: &[Vec<f64>],
    u: &[Vec<f64>],
    b: &[f64],
) -> Result<(), NeuralError> {
    let bad = |detail: String| NeuralError::InvalidArgument(format!("{name} LSTM {detail}"));
    if input == 0 || hidden == 0 || hidden > MAX_HIDDEN_SIZE {
        return Err(bad(format!("input size {input} and hidden size {hidden} must be in range")));
    }
    let gates = 4 * hidden;
    for (tensor, rows, cols) in [("W", w, input), ("U", u, hidden)] {
        if rows.len() != gates {
            return Err(bad(format!("{tensor} holds {} of {gates} rows", rows.len())));
        }
        if let Some(r) = rows.iter().position(|row| row.len() != cols) {
            return Err(bad(format!("{tensor} row {r} holds {} of {cols} columns", rows[r].len())));
        }
        if let Some(v) = rows.iter().flatten().find(|v| !v.is_finite()) {
            return Err(bad(format!("{tensor} holds non-finite weight {v}")));
        }
    }
    if b.len() != gates {
        return Err(bad(format!("b holds {} of {gates} values", b.len())));
    }
    if let Some(v) = b.iter().find(|v| !v.is_finite()) {
        return Err(bad(format!("b holds non-finite weight {v}")));
    }
    Ok(())
}

fn check_lstm(name: &str, layer: &LstmLayer) -> Result<(), NeuralError> {
    check_lstm_tensors(name, layer.input_size(), layer.hidden_size(), layer.w(), layer.u(), layer.b())
}

/// Every value finite.
fn check_finite(name: &str, values: impl IntoIterator<Item = f64>) -> Result<(), NeuralError> {
    for v in values {
        if !v.is_finite() {
            return Err(NeuralError::InvalidArgument(format!("non-finite {name} weight {v}")));
        }
    }
    Ok(())
}

/// A matrix with at least one row and the same column count in every row; returns the width.
fn check_matrix(name: &str, m: &[Vec<f64>]) -> Result<usize, NeuralError> {
    let Some(first) = m.first() else {
        return Err(NeuralError::InvalidArgument(format!("{name} must not be empty")));
    };
    let cols = first.len();
    if let Some(r) = m.iter().position(|row| row.len() != cols) {
        return Err(NeuralError::InvalidArgument(format!("{name} row {r} holds {} of {cols} columns", m[r].len())));
    }
    check_finite(name, m.iter().flat_map(|row| row.iter().copied()))?;
    Ok(cols)
}

/// The shape and value contract `load` enforces on the wire format, held for models built in
/// memory so no scoring path can index past a table or read a non-finite weight.
#[allow(clippy::too_many_arguments)]
pub(super) fn validate_tensors(
    word_entries: &[(String, usize)],
    char_entries: &[(Vec<u8>, usize)],
    tags: &[String],
    word_embeddings: &[Vec<f64>],
    char_embeddings: &[Vec<f64>],
    char_forward: &LstmLayer,
    char_backward: &LstmLayer,
    word_forward: &LstmLayer,
    word_backward: &LstmLayer,
    word_forward2: &Option<LstmLayer>,
    word_backward2: &Option<LstmLayer>,
    output_weights: &[Vec<f64>],
    output_bias: &[f64],
    pretrained_entries: &[(String, usize)],
    pretrained_vectors: &Option<Vec<Vec<f32>>>,
    transition_weights: &Option<Vec<Vec<f64>>>,
    start_weights: &Option<Vec<f64>>,
    end_weights: &Option<Vec<f64>>,
    adapter_weights: &Option<Vec<Vec<f64>>>,
    adapter_bias: &Option<Vec<f64>>,
) -> Result<(), NeuralError> {
    let bad = |detail: String| NeuralError::InvalidArgument(detail);
    let word_dim = check_matrix("word embeddings", word_embeddings)?;
    let char_dim = check_matrix("character embeddings", char_embeddings)?;
    if word_dim == 0 || char_dim == 0 {
        return Err(bad("embedding tables must hold at least one column".into()));
    }
    check_vocabulary(word_entries, word_embeddings.len(), "words", Some(&UNKNOWN.to_owned())).map_err(bad)?;
    check_vocabulary(char_entries, char_embeddings.len(), "characters", Some(&UNKNOWN_BYTES.to_vec())).map_err(bad)?;
    check_unique_tags(tags).map_err(bad)?;
    for (name, layer) in [
        ("character forward", char_forward),
        ("character backward", char_backward),
        ("word forward", word_forward),
        ("word backward", word_backward),
    ] {
        check_lstm(name, layer)?;
    }
    for (name, layer) in [("second word forward", word_forward2), ("second word backward", word_backward2)] {
        if let Some(layer) = layer {
            check_lstm(name, layer)?;
        }
    }
    if char_forward.input_size() != char_dim || char_backward.input_size() != char_dim {
        return Err(bad("character LSTM input sizes must match the character embedding width".into()));
    }
    if char_backward.hidden_size() != char_forward.hidden_size() {
        return Err(bad("character LSTM hidden sizes differ".into()));
    }

    let pretrained_dim = match pretrained_vectors {
        None if !pretrained_entries.is_empty() => {
            return Err(bad("pretrained entries require pretrained vectors".into()));
        }
        None => 0,
        Some(vecs) => {
            let Some(first) = vecs.first() else {
                return Err(bad("pretrained vectors must not be empty".into()));
            };
            let width = first.len();
            if width == 0 {
                return Err(bad("pretrained vectors must hold at least one column".into()));
            }
            if let Some(r) = vecs.iter().position(|row| row.len() != width) {
                return Err(bad(format!("pretrained vector row {r} holds {} of {width} columns", vecs[r].len())));
            }
            if let Some(v) = vecs.iter().flatten().find(|v| !v.is_finite()) {
                return Err(bad(format!("non-finite pretrained vector value {v}")));
            }
            check_vocabulary(pretrained_entries, vecs.len(), "pretrained", None).map_err(bad)?;
            width
        }
    };

    let expected_input = word_dim + 2 * char_forward.hidden_size() + pretrained_dim;
    if word_forward.input_size() != expected_input {
        return Err(bad(format!("word forward input size {} != {expected_input}", word_forward.input_size())));
    }
    if word_backward.input_size() != expected_input {
        return Err(bad(format!("word backward input size {} != {expected_input}", word_backward.input_size())));
    }
    if word_backward.hidden_size() != word_forward.hidden_size() {
        return Err(bad("word LSTM hidden sizes differ".into()));
    }
    let hidden = match (word_forward2, word_backward2) {
        (Some(wf2), Some(wb2)) => {
            if wf2.input_size() != 2 * word_forward.hidden_size() {
                return Err(bad("second-layer forward input size must be twice the first-layer hidden size".into()));
            }
            if wb2.input_size() != 2 * word_forward.hidden_size() || wb2.hidden_size() != wf2.hidden_size() {
                return Err(bad("second-layer backward LSTM must mirror the forward one".into()));
            }
            wf2.hidden_size()
        }
        _ => word_forward.hidden_size(),
    };

    if output_weights.len() != tags.len() {
        return Err(bad(format!("output weights {} rows for {} tags", output_weights.len(), tags.len())));
    }
    let out_width = check_matrix("output weights", output_weights)?;
    if out_width != 2 * hidden {
        return Err(bad(format!("output weights width {out_width} != {}", 2 * hidden)));
    }
    if output_bias.len() != tags.len() {
        return Err(bad(format!("output bias {} for {} tags", output_bias.len(), tags.len())));
    }
    check_finite("output bias", output_bias.iter().copied())?;

    if let (Some(tr), Some(st), Some(en)) = (transition_weights, start_weights, end_weights) {
        if tr.len() != tags.len() {
            return Err(bad(format!("CRF transition weights {} rows for {} tags", tr.len(), tags.len())));
        }
        let tr_width = check_matrix("CRF transition weights", tr)?;
        if tr_width != tags.len() {
            return Err(bad(format!("CRF transition width {tr_width} for {} tags", tags.len())));
        }
        if st.len() != tags.len() || en.len() != tags.len() {
            return Err(bad("CRF start and end weights must match the tag count".into()));
        }
        check_finite("CRF start weights", st.iter().copied())?;
        check_finite("CRF end weights", en.iter().copied())?;
    }

    if let (Some(aw), Some(ab)) = (adapter_weights, adapter_bias) {
        if pretrained_dim == 0 {
            return Err(bad("an adapter requires pretrained vectors".into()));
        }
        if aw.len() != pretrained_dim {
            return Err(bad(format!("adapter weights {} rows for width {pretrained_dim}", aw.len())));
        }
        let aw_width = check_matrix("adapter weights", aw)?;
        if aw_width != pretrained_dim {
            return Err(bad(format!("adapter weights width {aw_width} != {pretrained_dim}")));
        }
        if ab.len() != pretrained_dim {
            return Err(bad(format!("adapter bias {} for width {pretrained_dim}", ab.len())));
        }
        check_finite("adapter bias", ab.iter().copied())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_lstm_tensors;
    use crate::neural::NeuralError;

    /// A gate bias vector of the wrong length cannot be built through the public `LstmLayer`
    /// API (`b_mut` hands out a fixed-length slice), so the tensor check is exercised directly.
    #[test]
    fn rejects_a_short_or_long_gate_bias() {
        let (input, hidden) = (3, 2);
        let w = vec![vec![0.0; input]; 4 * hidden];
        let u = vec![vec![0.0; hidden]; 4 * hidden];
        assert!(check_lstm_tensors("test", input, hidden, &w, &u, &[0.0; 8]).is_ok());
        for len in [0, 7, 9] {
            let b = vec![0.0; len];
            let result = check_lstm_tensors("test", input, hidden, &w, &u, &b);
            assert!(matches!(result, Err(NeuralError::InvalidArgument(_))), "b of {len}: {result:?}");
        }
    }

    #[test]
    fn rejects_short_row_counts_and_zero_sizes() {
        let w = vec![vec![0.0; 3]; 7];
        let u = vec![vec![0.0; 2]; 8];
        assert!(check_lstm_tensors("test", 3, 2, &w, &u, &[0.0; 8]).is_err(), "W one row short");
        let w = vec![vec![0.0; 3]; 8];
        let u = vec![vec![0.0; 2]; 9];
        assert!(check_lstm_tensors("test", 3, 2, &w, &u, &[0.0; 8]).is_err(), "U one row long");
        assert!(check_lstm_tensors("test", 0, 2, &[], &[], &[]).is_err(), "zero input size");
        assert!(check_lstm_tensors("test", 3, 0, &[], &[], &[]).is_err(), "zero hidden size");
    }
}
