//! Legacy model export retained until #1532; importing is disabled under #1511.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

use crate::neural::{DataWriter, NeuralError};

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

    /// Returns the retirement error without reading the supplied stream.
    pub fn load<R: Read>(_reader: R) -> Result<Self, NeuralError> {
        Err(NeuralError::Disabled)
    }

    /// Returns the retirement error without accessing the filesystem.
    pub fn load_from_path(_path: impl AsRef<Path>) -> Result<Self, NeuralError> {
        Err(NeuralError::Disabled)
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

fn write_f32_vector<W: Write>(out: &mut DataWriter<W>, v: &[f32]) -> Result<(), NeuralError> {
    out.write_i32(v.len() as i32)?;
    for &x in v {
        out.write_f32(x)?;
    }
    Ok(())
}
