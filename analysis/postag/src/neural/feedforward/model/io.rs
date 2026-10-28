//! The `ONLP-FFPT-1` / `ONLP-FFPT-2` binary format: writing and reading a
//! [`FeedforwardPosModel`] and checking every table it declares.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::neural::feedforward::context::{PRETRAINED_SLOTS, SLOTS, UNKNOWN};
use crate::neural::{DataReader, DataWriter, NeuralError};

use super::*;

impl FeedforwardPosModel {
    /// Serializes this model in `ONLP-FFPT-1` or `ONLP-FFPT-2` format to `writer`.
    pub fn serialize<W: Write>(&self, writer: W) -> Result<(), NeuralError> {
        let mut out = DataWriter::new(writer);
        out.write_utf(if self.pretrained_size > 0 { MAGIC_V2 } else { MAGIC_V1 })?;
        write_vocab(&mut out, &self.word_entries)?;
        write_vocab(&mut out, &self.suffix_entries)?;
        write_vocab(&mut out, &self.shape_entries)?;
        write_vocab(&mut out, &self.tag_feature_entries)?;
        out.write_i32(self.tags.len() as i32)?;
        for tag in &self.tags {
            out.write_utf(tag)?;
        }
        out.write_i32(self.embedding_size as i32)?;
        write_f32_matrix(&mut out, &self.embeddings)?;
        write_f32_matrix(&mut out, &self.hidden_weights)?;
        write_f32_vector(&mut out, &self.hidden_bias)?;
        write_f32_matrix(&mut out, &self.output_weights)?;
        write_f32_vector(&mut out, &self.output_bias)?;
        if self.pretrained_size > 0 {
            out.write_i32(self.pretrained_size as i32)?;
            write_vocab(&mut out, &self.pretrained_entries)?;
            write_f32_matrix(&mut out, &self.pretrained_vectors)?;
        }
        out.flush()?;
        Ok(())
    }

    /// Serializes this model to a file path.
    pub fn serialize_to_path(&self, path: impl AsRef<Path>) -> Result<(), NeuralError> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.serialize(&mut writer)?;
        writer.flush()?;
        Ok(())
    }

    /// Reads and validates a `FeedforwardPosModel` from a stream in `ONLP-FFPT-1` or `ONLP-FFPT-2` format.
    pub fn load<R: Read>(reader: R) -> Result<Self, NeuralError> {
        let mut inp = DataReader::new(reader);
        let magic = inp.read_utf()?;
        if magic != MAGIC_V1 && magic != MAGIC_V2 {
            return Err(NeuralError::Format(format!("not a FeedforwardPOSModel stream (magic '{magic}')")));
        }
        let word_entries = read_vocab(&mut inp, true)?;
        let suffix_entries = read_vocab(&mut inp, true)?;
        let shape_entries = read_vocab(&mut inp, true)?;
        let tag_feature_entries = read_vocab(&mut inp, true)?;

        let num_tags = read_count(&mut inp, "tag count", 1)?;
        let mut tags = Vec::with_capacity(num_tags.min(INITIAL_CAPACITY));
        let mut seen_tags = HashSet::with_capacity(num_tags.min(INITIAL_CAPACITY));
        for _ in 0..num_tags {
            let tag = inp.read_utf()?;
            if !seen_tags.insert(tag.clone()) {
                return Err(NeuralError::Format(format!("duplicate tag: {tag}")));
            }
            tags.push(tag);
        }

        let embedding_size = read_count(&mut inp, "embedding size", 1)?;
        let embeddings = read_f32_matrix(&mut inp, "embeddings", false)?;
        let hidden_weights = read_f32_matrix(&mut inp, "hidden weights", false)?;
        let hidden_bias = read_f32_vector(&mut inp, "hidden bias", 1)?;
        let output_weights = read_f32_matrix(&mut inp, "output weights", false)?;
        let output_bias = read_f32_vector(&mut inp, "output bias", 1)?;

        let (pretrained_size, pretrained_entries, pretrained_vectors) = if magic == MAGIC_V2 {
            let p_size = read_count(&mut inp, "pretrained vector size", 1)?;
            let p_entries = read_vocab(&mut inp, false)?;
            let p_vecs = read_f32_matrix(&mut inp, "pretrained vectors", true)?;
            validate_rows(&p_entries, p_vecs.len(), "pretrained")?;
            if !p_vecs.is_empty() && p_vecs[0].len() != p_size {
                return Err(NeuralError::Format(format!("pretrained vector width {} != {p_size}", p_vecs[0].len())));
            }
            (p_size, p_entries, p_vecs)
        } else {
            (0, Vec::new(), Vec::new())
        };

        validate_rows(&word_entries, embeddings.len(), "words")?;
        validate_rows(&suffix_entries, embeddings.len(), "suffixes")?;
        validate_rows(&shape_entries, embeddings.len(), "shapes")?;
        validate_rows(&tag_feature_entries, embeddings.len(), "tags")?;

        if embeddings[0].len() != embedding_size {
            return Err(NeuralError::Format(format!("embedding width {} != {embedding_size}", embeddings[0].len())));
        }
        let expected_input = SLOTS
            .checked_mul(embedding_size)
            .and_then(|a| PRETRAINED_SLOTS.checked_mul(pretrained_size).and_then(|b| a.checked_add(b)))
            .ok_or_else(|| NeuralError::Format("input width overflow".into()))?;
        if hidden_weights[0].len() != expected_input {
            return Err(NeuralError::Format(format!(
                "hidden input width {} != {expected_input}",
                hidden_weights[0].len()
            )));
        }
        if hidden_bias.len() != hidden_weights.len() {
            return Err(NeuralError::Format(format!(
                "hidden bias length {} != {}",
                hidden_bias.len(),
                hidden_weights.len()
            )));
        }
        if output_weights.len() != tags.len() {
            return Err(NeuralError::Format(format!("output row count {} != {}", output_weights.len(), tags.len())));
        }
        if output_weights[0].len() != hidden_weights.len() {
            return Err(NeuralError::Format(format!(
                "output column count {} != {}",
                output_weights[0].len(),
                hidden_weights.len()
            )));
        }
        if output_bias.len() != tags.len() {
            return Err(NeuralError::Format(format!("output bias length {} != {}", output_bias.len(), tags.len())));
        }

        Ok(Self::with_pretrained(
            word_entries,
            suffix_entries,
            shape_entries,
            tag_feature_entries,
            tags,
            embedding_size,
            embeddings,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
            pretrained_size,
            pretrained_entries,
            pretrained_vectors,
        ))
    }

    /// Reads and validates a `FeedforwardPosModel` from a file path.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, NeuralError> {
        let file = File::open(path)?;
        Self::load(BufReader::new(file))
    }
}

fn write_vocab<W: Write>(out: &mut DataWriter<W>, entries: &[(String, usize)]) -> Result<(), NeuralError> {
    out.write_i32(entries.len() as i32)?;
    for (k, v) in entries {
        out.write_utf(k)?;
        out.write_i32(*v as i32)?;
    }
    Ok(())
}

fn read_vocab<R: Read>(inp: &mut DataReader<R>, require_unknown: bool) -> Result<Vec<(String, usize)>, NeuralError> {
    let size = read_count(inp, "vocabulary size", usize::from(require_unknown))?;
    let mut entries = Vec::with_capacity(size.min(INITIAL_CAPACITY));
    let mut seen = HashSet::with_capacity(size.min(INITIAL_CAPACITY));
    let mut has_unk = false;
    for _ in 0..size {
        let key = inp.read_utf()?;
        let row = inp.read_i32()?;
        if row < 0 {
            return Err(NeuralError::Format(format!("negative row index: {row}")));
        }
        if !seen.insert(key.clone()) {
            return Err(NeuralError::Format(format!("duplicate vocabulary entry: {key}")));
        }
        if key == UNKNOWN {
            has_unk = true;
        }
        entries.push((key, row as usize));
    }
    if require_unknown && !has_unk {
        return Err(NeuralError::Format(format!("missing required vocabulary entry: {UNKNOWN}")));
    }
    Ok(entries)
}

fn validate_rows(entries: &[(String, usize)], row_count: usize, name: &str) -> Result<(), NeuralError> {
    for (k, r) in entries {
        if *r >= row_count {
            return Err(NeuralError::Format(format!("{name} row index for '{k}' out of bounds: {r} >= {row_count}")));
        }
    }
    Ok(())
}

fn write_f32_matrix<W: Write>(out: &mut DataWriter<W>, m: &[Vec<f32>]) -> Result<(), NeuralError> {
    let cols = m.first().map_or(0, Vec::len);
    out.write_i32(m.len() as i32)?;
    out.write_i32(cols as i32)?;
    for row in m {
        for &v in row {
            out.write_f32(v)?;
        }
    }
    Ok(())
}

fn read_f32_matrix<R: Read>(
    inp: &mut DataReader<R>,
    name: &str,
    allow_empty: bool,
) -> Result<Vec<Vec<f32>>, NeuralError> {
    let rows = read_count(inp, &format!("{name} rows"), usize::from(!allow_empty))?;
    let cols = read_count(inp, &format!("{name} columns"), 0)?;
    if rows == 0 {
        if cols != 0 {
            return Err(NeuralError::Format(format!("{name} columns must be 0 when rows is 0: {cols}")));
        }
        return Ok(Vec::new());
    }
    if cols < 1 {
        return Err(NeuralError::Format(format!("invalid {name} columns: {cols}")));
    }
    let mut m = Vec::with_capacity(rows.min(INITIAL_CAPACITY));
    for _ in 0..rows {
        m.push(read_f32_values(inp, name, cols)?);
    }
    Ok(m)
}

fn write_f32_vector<W: Write>(out: &mut DataWriter<W>, v: &[f32]) -> Result<(), NeuralError> {
    out.write_i32(v.len() as i32)?;
    for &x in v {
        out.write_f32(x)?;
    }
    Ok(())
}

fn read_f32_vector<R: Read>(inp: &mut DataReader<R>, name: &str, min: usize) -> Result<Vec<f32>, NeuralError> {
    let len = read_count(inp, &format!("{name} length"), min)?;
    read_f32_values(inp, name, len)
}

fn read_f32_values<R: Read>(inp: &mut DataReader<R>, name: &str, len: usize) -> Result<Vec<f32>, NeuralError> {
    let mut out = Vec::with_capacity(len.min(INITIAL_CAPACITY));
    for _ in 0..len {
        let v = inp.read_f32()?;
        if !v.is_finite() {
            return Err(NeuralError::Format(format!("non-finite value in {name}: {v}")));
        }
        out.push(v);
    }
    Ok(out)
}

fn read_count<R: Read>(inp: &mut DataReader<R>, name: &str, min: usize) -> Result<usize, NeuralError> {
    let val = inp.read_i32()?;
    if val < (min as i32) || val > MAX_LENGTH {
        return Err(NeuralError::Format(format!("invalid {name}: {val}")));
    }
    Ok(val as usize)
}
