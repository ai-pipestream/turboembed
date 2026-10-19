//! `ONLP-BLPT-1`..`5` binary reading and validation.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::neural::lstm::{LstmLayer, MAX_HIDDEN_SIZE};
use crate::neural::{DataReader, NeuralError, encode_str_modified_utf8, encode_unit_modified_utf8};

use super::{BilstmPosModel, MAGIC_V1, MAGIC_V2, MAGIC_V3, MAGIC_V4, MAGIC_V5, UNKNOWN};

const UNKNOWN_BYTES: &[u8] = b"*UNK*";
const INITIAL_CAPACITY: usize = 4096;
const MAX_DIMENSION: i32 = i32::MAX / 8;

fn read_positive_dim<R: Read>(data: &mut DataReader<R>, name: &str) -> Result<usize, NeuralError> {
    let value = data.read_i32()?;
    if !(1..=MAX_DIMENSION).contains(&value) {
        return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} = {value}")));
    }
    Ok(value as usize)
}

fn read_non_negative_count<R: Read>(data: &mut DataReader<R>, name: &str) -> Result<usize, NeuralError> {
    let value = data.read_i32()?;
    if !(0..=MAX_DIMENSION).contains(&value) {
        return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} = {value}")));
    }
    Ok(value as usize)
}

fn read_str_vocabulary<R: Read>(
    data: &mut DataReader<R>,
    name: &str,
    allow_empty: bool,
) -> Result<Vec<(String, usize)>, NeuralError> {
    let size = if allow_empty {
        read_non_negative_count(data, &format!("{name} vocabulary size"))?
    } else {
        read_positive_dim(data, &format!("{name} vocabulary size"))?
    };
    let mut entries = Vec::with_capacity(size.min(INITIAL_CAPACITY));
    let mut seen = HashSet::with_capacity(size.min(INITIAL_CAPACITY));
    for _ in 0..size {
        let key = data.read_utf()?;
        let row = data.read_i32()?;
        if row < 0 {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: negative {name} row {row}")));
        }
        if !seen.insert(key.clone()) {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: duplicate {name} entry {key}")));
        }
        entries.push((key, row as usize));
    }
    Ok(entries)
}

fn read_bytes_vocabulary<R: Read>(
    data: &mut DataReader<R>,
    name: &str,
    allow_empty: bool,
) -> Result<Vec<(Vec<u8>, usize)>, NeuralError> {
    let size = if allow_empty {
        read_non_negative_count(data, &format!("{name} vocabulary size"))?
    } else {
        read_positive_dim(data, &format!("{name} vocabulary size"))?
    };
    let mut entries = Vec::with_capacity(size.min(INITIAL_CAPACITY));
    let mut seen = HashSet::with_capacity(size.min(INITIAL_CAPACITY));
    for _ in 0..size {
        let key = data.read_utf_bytes()?;
        let row = data.read_i32()?;
        if row < 0 {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: negative {name} row {row}")));
        }
        if !seen.insert(key.clone()) {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: duplicate {name} entry")));
        }
        entries.push((key, row as usize));
    }
    Ok(entries)
}

fn validate_vocabulary_rows(entries: &[(String, usize)], row_count: usize, name: &str) -> Result<(), NeuralError> {
    let mut unk_row = None;
    for (k, r) in entries {
        if *r >= row_count {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} row {r} >= {row_count}")));
        }
        if k == UNKNOWN {
            unk_row = Some(*r);
        }
    }
    if unk_row != Some(0) {
        return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} must map {UNKNOWN} to row 0")));
    }
    Ok(())
}

fn validate_bytes_vocabulary_rows(
    entries: &[(Vec<u8>, usize)],
    row_count: usize,
    name: &str,
) -> Result<(), NeuralError> {
    let mut unk_row = None;
    for (k, r) in entries {
        if *r >= row_count {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} row {r} >= {row_count}")));
        }
        if k.as_slice() == UNKNOWN_BYTES {
            unk_row = Some(*r);
        }
    }
    if unk_row != Some(0) {
        return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} must map {UNKNOWN} to row 0")));
    }
    Ok(())
}

fn validate_pretrained_rows(entries: &[(String, usize)], row_count: usize) -> Result<(), NeuralError> {
    for (_, r) in entries {
        if *r >= row_count {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: pretrained row {r} >= {row_count}")));
        }
    }
    Ok(())
}

fn read_tags<R: Read>(data: &mut DataReader<R>) -> Result<Vec<String>, NeuralError> {
    let count = read_positive_dim(data, "tag count")?;
    let mut tags = Vec::with_capacity(count.min(INITIAL_CAPACITY));
    let mut seen = HashSet::with_capacity(count.min(INITIAL_CAPACITY));
    for _ in 0..count {
        let tag = data.read_utf()?;
        if !seen.insert(tag.clone()) {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: duplicate tag {tag}")));
        }
        tags.push(tag);
    }
    Ok(tags)
}

fn read_lstm<R: Read>(
    data: &mut DataReader<R>,
    name: &str,
    expected_input: Option<usize>,
    expected_hidden: Option<usize>,
) -> Result<LstmLayer, NeuralError> {
    let input_size = read_positive_dim(data, &format!("{name} inputSize"))?;
    let hidden_size = read_positive_dim(data, &format!("{name} hiddenSize"))?;
    if hidden_size > MAX_HIDDEN_SIZE {
        return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} hiddenSize = {hidden_size}")));
    }
    if let Some(exp_in) = expected_input
        && input_size != exp_in
    {
        return Err(NeuralError::Format(format!(
            "Invalid BiLSTM POS model: {name} inputSize {input_size} != {exp_in}"
        )));
    }
    if let Some(exp_hid) = expected_hidden
        && hidden_size != exp_hid
    {
        return Err(NeuralError::Format(format!(
            "Invalid BiLSTM POS model: {name} hiddenSize {hidden_size} != {exp_hid}"
        )));
    }
    let gates = 4 * hidden_size;
    let w = read_f64_matrix_exact(data, &format!("{name}.w"), gates, input_size)?;
    let u = read_f64_matrix_exact(data, &format!("{name}.u"), gates, hidden_size)?;
    let b = read_f64_vector_exact(data, &format!("{name}.b"), gates)?;
    LstmLayer::of_weights(input_size, hidden_size, w, u, b)
}

fn read_f64_matrix<R: Read>(data: &mut DataReader<R>, name: &str) -> Result<Vec<Vec<f64>>, NeuralError> {
    let rows = read_positive_dim(data, &format!("{name} rows"))?;
    let cols = read_positive_dim(data, &format!("{name} cols"))?;
    read_f64_rows(data, name, rows, cols)
}

fn read_f64_matrix_exact<R: Read>(
    data: &mut DataReader<R>,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<Vec<Vec<f64>>, NeuralError> {
    let rows = read_positive_dim(data, &format!("{name} rows"))?;
    let cols = read_positive_dim(data, &format!("{name} cols"))?;
    if rows != expected_rows || cols != expected_cols {
        return Err(NeuralError::Format(format!(
            "Invalid BiLSTM POS model: {name} shape {rows}x{cols} != {expected_rows}x{expected_cols}"
        )));
    }
    read_f64_rows(data, name, rows, cols)
}

fn read_f64_rows<R: Read>(
    data: &mut DataReader<R>,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<Vec<f64>>, NeuralError> {
    let mut matrix = Vec::with_capacity(rows.min(INITIAL_CAPACITY));
    for _ in 0..rows {
        matrix.push(read_f64_values(data, name, cols)?);
    }
    Ok(matrix)
}

fn read_f64_vector_exact<R: Read>(
    data: &mut DataReader<R>,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f64>, NeuralError> {
    let len = read_positive_dim(data, &format!("{name} length"))?;
    if len != expected_len {
        return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: {name} length {len} != {expected_len}")));
    }
    read_f64_values(data, name, len)
}

fn read_f64_values<R: Read>(data: &mut DataReader<R>, name: &str, len: usize) -> Result<Vec<f64>, NeuralError> {
    let mut out = Vec::with_capacity(len.min(INITIAL_CAPACITY));
    for _ in 0..len {
        let v = data.read_f64()?;
        if !v.is_finite() {
            return Err(NeuralError::Format(format!("Invalid BiLSTM POS model: non-finite {name} value {v}")));
        }
        out.push(v);
    }
    Ok(out)
}

fn read_f32_rows<R: Read>(data: &mut DataReader<R>, rows: usize, cols: usize) -> Result<Vec<Vec<f32>>, NeuralError> {
    let mut matrix = Vec::with_capacity(rows.min(INITIAL_CAPACITY));
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols.min(INITIAL_CAPACITY));
        for _ in 0..cols {
            let v = data.read_f32()?;
            if !v.is_finite() {
                return Err(NeuralError::Format(format!(
                    "Invalid BiLSTM POS model: non-finite pretrained vector value {v}"
                )));
            }
            row.push(v);
        }
        matrix.push(row);
    }
    Ok(matrix)
}

/// Helper to build a character vocabulary entry list from `&str` keys.
pub fn char_vocabulary_from_strs<I, S>(entries: I) -> Vec<(Vec<u8>, usize)>
where
    I: IntoIterator<Item = (S, usize)>,
    S: AsRef<str>,
{
    entries.into_iter().map(|(k, v)| (encode_str_modified_utf8(k.as_ref()), v)).collect()
}

/// Helper to build a character vocabulary entry from a single UTF-16 code unit.
pub fn char_key_from_unit(unit: u16) -> Vec<u8> {
    encode_unit_modified_utf8(unit)
}
impl BilstmPosModel {
    /// Reads and validates a `BilstmPosModel` from a stream in `ONLP-BLPT-1`..`5` format.
    pub fn load<R: Read>(reader: R) -> Result<Self, NeuralError> {
        let mut data = DataReader::new(reader);
        let magic = data.read_utf()?;
        let (stacked, crf, adapter) = match magic.as_str() {
            MAGIC_V1 => (false, false, false),
            MAGIC_V2 => (false, true, false),
            MAGIC_V3 => (true, false, false),
            MAGIC_V4 => (true, true, false),
            MAGIC_V5 => {
                let s = data.read_boolean()?;
                let c = data.read_boolean()?;
                (s, c, true)
            }
            _ => {
                return Err(NeuralError::Format(format!("Unrecognized BiLSTM POS model header: {magic}")));
            }
        };

        let word_entries = read_str_vocabulary(&mut data, "words", false)?;
        let char_entries = read_bytes_vocabulary(&mut data, "characters", false)?;
        let tags = read_tags(&mut data)?;

        let word_embeddings = read_f64_matrix(&mut data, "wordEmbeddings")?;
        let char_embeddings = read_f64_matrix(&mut data, "charEmbeddings")?;
        validate_vocabulary_rows(&word_entries, word_embeddings.len(), "words")?;
        validate_bytes_vocabulary_rows(&char_entries, char_embeddings.len(), "characters")?;

        let char_dim = char_embeddings[0].len();
        let word_dim = word_embeddings[0].len();
        let char_forward = read_lstm(&mut data, "charForward", Some(char_dim), None)?;
        let char_backward = read_lstm(&mut data, "charBackward", Some(char_dim), Some(char_forward.hidden_size()))?;
        let word_forward = read_lstm(&mut data, "wordForward", None, None)?;
        let word_backward =
            read_lstm(&mut data, "wordBackward", Some(word_forward.input_size()), Some(word_forward.hidden_size()))?;

        let mut encoder_width = 2 * word_forward.hidden_size();
        let (word_forward2, word_backward2) = if stacked {
            let wf2 = read_lstm(&mut data, "wordForward2", Some(encoder_width), None)?;
            let wb2 = read_lstm(&mut data, "wordBackward2", Some(encoder_width), Some(wf2.hidden_size()))?;
            encoder_width = 2 * wf2.hidden_size();
            (Some(wf2), Some(wb2))
        } else {
            (None, None)
        };

        let output_weights = read_f64_matrix_exact(&mut data, "outputWeights", tags.len(), encoder_width)?;
        let output_bias = read_f64_vector_exact(&mut data, "outputBias", tags.len())?;

        let (transition_weights, start_weights, end_weights) = if crf {
            let tr = read_f64_matrix_exact(&mut data, "transitionWeights", tags.len(), tags.len())?;
            let st = read_f64_vector_exact(&mut data, "startWeights", tags.len())?;
            let en = read_f64_vector_exact(&mut data, "endWeights", tags.len())?;
            (Some(tr), Some(st), Some(en))
        } else {
            (None, None, None)
        };

        let max_word_length = read_positive_dim(&mut data, "maxWordLength")?;
        let mut expected_word_input = word_dim + 2 * char_forward.hidden_size();

        let (pretrained_entries, pretrained_vectors) = if data.read_boolean()? {
            let p_entries = read_str_vocabulary(&mut data, "pretrained", true)?;
            let vec_dim = read_positive_dim(&mut data, "pretrained vector width")?;
            let row_count = read_positive_dim(&mut data, "pretrained vector rows")?;
            validate_pretrained_rows(&p_entries, row_count)?;
            let p_vecs = read_f32_rows(&mut data, row_count, vec_dim)?;
            expected_word_input += vec_dim;
            (Some(p_entries), Some(p_vecs))
        } else {
            if adapter {
                return Err(NeuralError::Format(
                    "Invalid BiLSTM POS model: adapter requires pretrained vectors".into(),
                ));
            }
            (None, None)
        };

        if word_forward.input_size() != expected_word_input {
            return Err(NeuralError::Format(format!(
                "Invalid BiLSTM POS model: wordForward inputSize {} != {}",
                word_forward.input_size(),
                expected_word_input
            )));
        }

        let (adapter_weights, adapter_bias) = if adapter {
            let p_vecs = pretrained_vectors.as_ref().unwrap();
            let vec_dim = p_vecs[0].len();
            let aw = read_f64_matrix_exact(&mut data, "adapterWeights", vec_dim, vec_dim)?;
            let ab = read_f64_vector_exact(&mut data, "adapterBias", vec_dim)?;
            (Some(aw), Some(ab))
        } else {
            (None, None)
        };

        Self::new(
            word_entries,
            char_entries,
            tags,
            word_embeddings,
            char_embeddings,
            char_forward,
            char_backward,
            word_forward,
            word_backward,
            word_forward2,
            word_backward2,
            output_weights,
            output_bias,
            max_word_length,
            pretrained_entries,
            pretrained_vectors,
            transition_weights,
            start_weights,
            end_weights,
            adapter_weights,
            adapter_bias,
        )
    }

    /// Reads and validates a `BilstmPosModel` from a file path.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, NeuralError> {
        let file = File::open(path)?;
        Self::load(BufReader::new(file))
    }
}
