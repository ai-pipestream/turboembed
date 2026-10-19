//! The shape and value contract `load` enforces, held for models built in memory.

use crate::neural::NeuralError;
use crate::neural::lstm::LstmLayer;

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
    if let Some((k, r)) = word_entries.iter().find(|(_, r)| *r >= word_embeddings.len()) {
        return Err(bad(format!("word {k:?} row {r} past {} embedding rows", word_embeddings.len())));
    }
    if let Some((_, r)) = char_entries.iter().find(|(_, r)| *r >= char_embeddings.len()) {
        return Err(bad(format!("character entry row {r} past {} embedding rows", char_embeddings.len())));
    }
    if char_forward.input_size() != char_dim || char_backward.input_size() != char_dim {
        return Err(bad("character LSTM input sizes must match the character embedding width".into()));
    }
    if char_backward.hidden_size() != char_forward.hidden_size() {
        return Err(bad("character LSTM hidden sizes differ".into()));
    }

    let pretrained_dim = match pretrained_vectors {
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
            if let Some((k, r)) = pretrained_entries.iter().find(|(_, r)| *r >= vecs.len()) {
                return Err(bad(format!("pretrained {k:?} row {r} past {} vector rows", vecs.len())));
            }
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
