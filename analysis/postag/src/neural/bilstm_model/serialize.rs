//! `ONLP-BLPT-1`..`5` binary writing.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::neural::lstm::LstmLayer;
use crate::neural::{DataWriter, NeuralError};

use super::{BilstmPosModel, MAGIC_V1, MAGIC_V2, MAGIC_V3, MAGIC_V4, MAGIC_V5};

fn write_str_vocabulary<W: Write>(data: &mut DataWriter<W>, entries: &[(String, usize)]) -> Result<(), NeuralError> {
    data.write_i32(entries.len() as i32)?;
    for (k, v) in entries {
        data.write_utf(k)?;
        data.write_i32(*v as i32)?;
    }
    Ok(())
}

fn write_bytes_vocabulary<W: Write>(data: &mut DataWriter<W>, entries: &[(Vec<u8>, usize)]) -> Result<(), NeuralError> {
    data.write_i32(entries.len() as i32)?;
    for (k, v) in entries {
        data.write_utf_bytes(k)?;
        data.write_i32(*v as i32)?;
    }
    Ok(())
}

fn write_f64_matrix<W: Write>(data: &mut DataWriter<W>, m: &[Vec<f64>]) -> Result<(), NeuralError> {
    let cols = m.first().map_or(0, Vec::len);
    // The header promises every row has `cols` columns; refuse a ragged
    // matrix rather than write a stream no reader can parse.
    if let Some(r) = m.iter().position(|row| row.len() != cols) {
        return Err(NeuralError::InvalidArgument(format!("matrix row {r} holds {} of {cols} columns", m[r].len())));
    }
    data.write_i32(m.len() as i32)?;
    data.write_i32(cols as i32)?;
    for row in m {
        for &v in row {
            data.write_f64(v)?;
        }
    }
    Ok(())
}

fn write_f64_vector<W: Write>(data: &mut DataWriter<W>, v: &[f64]) -> Result<(), NeuralError> {
    data.write_i32(v.len() as i32)?;
    for &x in v {
        data.write_f64(x)?;
    }
    Ok(())
}

fn write_lstm<W: Write>(data: &mut DataWriter<W>, layer: &LstmLayer) -> Result<(), NeuralError> {
    data.write_i32(layer.input_size() as i32)?;
    data.write_i32(layer.hidden_size() as i32)?;
    write_f64_matrix(data, layer.w())?;
    write_f64_matrix(data, layer.u())?;
    write_f64_vector(data, layer.b())
}

impl BilstmPosModel {
    /// Serializes this model in the `ONLP-BLPT-1`..`5` binary format to `writer`.
    pub fn serialize<W: Write>(&self, writer: W) -> Result<(), NeuralError> {
        let mut data = DataWriter::new(writer);
        let stacked = self.is_stacked();
        let crf = self.is_crf();
        if self.has_adapter() {
            data.write_utf(MAGIC_V5)?;
            data.write_boolean(stacked)?;
            data.write_boolean(crf)?;
        } else {
            let magic = match (stacked, crf) {
                (false, false) => MAGIC_V1,
                (false, true) => MAGIC_V2,
                (true, false) => MAGIC_V3,
                (true, true) => MAGIC_V4,
            };
            data.write_utf(magic)?;
        }
        write_str_vocabulary(&mut data, &self.word_entries)?;
        write_bytes_vocabulary(&mut data, &self.char_entries)?;
        data.write_i32(self.tags.len() as i32)?;
        for tag in &self.tags {
            data.write_utf(tag)?;
        }
        write_f64_matrix(&mut data, &self.word_embeddings)?;
        write_f64_matrix(&mut data, &self.char_embeddings)?;
        write_lstm(&mut data, &self.char_forward)?;
        write_lstm(&mut data, &self.char_backward)?;
        write_lstm(&mut data, &self.word_forward)?;
        write_lstm(&mut data, &self.word_backward)?;
        if let (Some(wf2), Some(wb2)) = (&self.word_forward2, &self.word_backward2) {
            write_lstm(&mut data, wf2)?;
            write_lstm(&mut data, wb2)?;
        }
        write_f64_matrix(&mut data, &self.output_weights)?;
        write_f64_vector(&mut data, &self.output_bias)?;
        if let (Some(tr), Some(st), Some(en)) = (&self.transition_weights, &self.start_weights, &self.end_weights) {
            write_f64_matrix(&mut data, tr)?;
            write_f64_vector(&mut data, st)?;
            write_f64_vector(&mut data, en)?;
        }
        data.write_i32(self.max_word_length as i32)?;
        if let Some(ref p_vecs) = self.pretrained_vectors {
            data.write_boolean(true)?;
            write_str_vocabulary(&mut data, &self.pretrained_entries)?;
            let vec_dim = self.pretrained_dim();
            data.write_i32(vec_dim as i32)?;
            data.write_i32(p_vecs.len() as i32)?;
            for row in p_vecs {
                for &v in row {
                    data.write_f32(v)?;
                }
            }
        } else {
            data.write_boolean(false)?;
        }
        if let (Some(aw), Some(ab)) = (&self.adapter_weights, &self.adapter_bias) {
            write_f64_matrix(&mut data, aw)?;
            write_f64_vector(&mut data, ab)?;
        }
        data.flush()?;
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
}
