//! `THEAD001` binary serialization and deserialization for [`super::ClassifierHead`].

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

use turbo_analysis::{ModelRef, Tag, TagSet};

use super::{ClassifierHead, DenseLayer, HEAD_MAGIC, HeadMode, HiddenActivation};
use crate::centroids::{Cursor, write_str};

impl ClassifierHead {
    /// Serializes this head to the plain typed `THEAD001` binary format.
    pub fn write_to<W: Write>(&self, mut w: W) -> io::Result<()> {
        w.write_all(HEAD_MAGIC)?;
        w.write_all(&[self.mode as u8, self.hidden_activation as u8, self.layers.len() as u8])?;
        write_str(&mut w, &self.model.id)?;
        write_str(&mut w, &self.model.revision)?;
        write_str(&mut w, &self.model.sha256)?;
        write_str(&mut w, &self.tags.name)?;
        w.write_all(&(self.tags.len() as u32).to_le_bytes())?;
        for i in 0..self.tags.len() {
            write_str(&mut w, self.tags.label(Tag(i as u16)))?;
        }
        for layer in &self.layers {
            w.write_all(&layer.in_dim.to_le_bytes())?;
            w.write_all(&layer.out_dim.to_le_bytes())?;
            for &weight in &layer.weights {
                w.write_all(&weight.to_le_bytes())?;
            }
            for &b in &layer.bias {
                w.write_all(&b.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Serializes this head into an in-memory byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_to(&mut buf).expect("writing to Vec<u8> cannot fail");
        buf
    }

    /// Writes this head to `path` in the `THEAD001` binary format.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        fs::write(path, self.to_bytes())
    }

    /// Loads a `ClassifierHead` from a byte slice in the `THEAD001` binary
    /// format.
    pub fn from_bytes(bytes: &[u8]) -> Result<ClassifierHead, String> {
        let mut cur = Cursor::new(bytes);
        let magic = cur.read_array::<8>()?;
        if &magic != HEAD_MAGIC {
            return Err("invalid ClassifierHead header (expected THEAD001)".into());
        }
        let mode = HeadMode::from_byte(cur.read_u8()?)?;
        let hidden_activation = HiddenActivation::from_byte(cur.read_u8()?)?;
        let n_layers = cur.read_u8()? as usize;
        if n_layers == 0 || n_layers > 2 {
            return Err(format!("THEAD001 supports 1 or 2 layers, got {n_layers}"));
        }
        let model = ModelRef { id: cur.read_string()?, revision: cur.read_string()?, sha256: cur.read_string()? };
        let tag_set_name = cur.read_string()?;
        let n_labels = cur.read_u32()? as usize;
        if n_labels == 0 || n_labels > u16::MAX as usize || n_labels > cur.remaining() / 4 {
            return Err("THEAD001 label count is invalid or truncated".into());
        }
        let mut labels = Vec::with_capacity(n_labels);
        for _ in 0..n_labels {
            labels.push(cur.read_string()?);
        }
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let tags = TagSet::new(&tag_set_name, &label_refs);
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let in_dim = cur.read_u32()?;
            let out_dim = cur.read_u32()?;
            let w_count = (out_dim as usize)
                .checked_mul(in_dim as usize)
                .ok_or_else(|| "THEAD001 layer dimensions overflow usize".to_string())?;
            let count = w_count.checked_add(out_dim as usize).ok_or("layer size overflow")?;
            if count > cur.remaining() / 8 {
                return Err("truncated classifier weights".into());
            }
            let mut weights = Vec::with_capacity(w_count);
            for _ in 0..w_count {
                weights.push(cur.read_f64()?);
            }
            let mut bias = Vec::with_capacity(out_dim as usize);
            for _ in 0..out_dim as usize {
                bias.push(cur.read_f64()?);
            }
            layers.push(DenseLayer::new(in_dim, out_dim, weights, bias)?);
        }
        cur.finish()?;
        match layers.len() {
            1 => {
                if hidden_activation != HiddenActivation::None {
                    return Err("1-layer THEAD001 head must have HiddenActivation::None".into());
                }
                let l = layers.remove(0);
                if l.out_dim as usize != tags.len() {
                    return Err(format!("THEAD001 output dim {} does not match {} labels", l.out_dim, tags.len()));
                }
                Self::linear(model, l.in_dim, tags, mode, l.weights, l.bias)
            }
            2 => {
                let l2 = layers.pop().unwrap();
                let l1 = layers.pop().unwrap();
                if l1.out_dim != l2.in_dim {
                    return Err(format!(
                        "THEAD001 layer 1 out_dim {} does not match layer 2 in_dim {}",
                        l1.out_dim, l2.in_dim
                    ));
                }
                if l2.out_dim as usize != tags.len() {
                    return Err(format!("THEAD001 output dim {} does not match {} labels", l2.out_dim, tags.len()));
                }
                Self::mlp(
                    model,
                    l1.in_dim,
                    l1.out_dim,
                    tags,
                    mode,
                    hidden_activation,
                    l1.weights,
                    l1.bias,
                    l2.weights,
                    l2.bias,
                )
            }
            _ => unreachable!(),
        }
    }

    /// Reads a `ClassifierHead` from any reader in the `THEAD001` format.
    pub fn read_from<R: Read>(mut r: R) -> Result<ClassifierHead, String> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf).map_err(|e| e.to_string())?;
        Self::from_bytes(&buf)
    }

    /// Loads a `ClassifierHead` from `path`.
    pub fn load(path: &Path) -> Result<ClassifierHead, String> {
        let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }
}
