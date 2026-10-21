//! A PyTorch checkpoint (`torch.save`'s zip form) read as data and written
//! out as safetensors. The archive's entries are stored (`zip`), and its
//! `data.pkl` is read by a stack machine that runs no code (`pickle`): it
//! knows only the opcodes and three globals a state dict of F32 tensors
//! uses. The legacy (pre-zip) format, other dtypes, non-contiguous tensors
//! and anything else are refused by name.

mod pickle;
mod zip;

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::Result;

pub use pickle::TensorRef;

/// One F32 tensor of a checkpoint, its values still in the archive.
pub struct Tensor<'a> {
    pub shape: Vec<usize>,
    /// Little-endian F32 values, row-major.
    pub data: &'a [u8],
}

/// Every tensor of the checkpoint, by name.
pub fn read(bytes: &[u8]) -> Result<BTreeMap<String, Tensor<'_>>> {
    if !bytes.starts_with(b"PK\x03\x04") {
        return Err("torch: not a zip archive; the legacy torch.save format is not read".into());
    }
    let entries = zip::entries(bytes)?;
    let pkl = entries
        .iter()
        .filter(|e| e.name.ends_with("/data.pkl") && e.name.matches('/').count() == 1)
        .collect::<Vec<_>>();
    let [pkl] = pkl.as_slice() else { return Err("torch: the archive holds no single data.pkl".into()) };
    let root = &pkl.name[..pkl.name.len() - "data.pkl".len()];
    let file = |name: &str| entries.iter().find(|e| e.name == format!("{root}{name}")).map(|e| e.data);
    if file("byteorder").is_some_and(|b| b != b"little") {
        return Err("torch: only little-endian checkpoints are read".into());
    }
    let mut out = BTreeMap::new();
    for (name, t) in pickle::state_dict(pkl.data)? {
        let storage = file(&format!("data/{}", t.key)).ok_or_else(|| format!("torch: {name}: no storage {}", t.key))?;
        if Some(storage.len() as u64) != t.numel.checked_mul(4) {
            return Err(format!("torch: {name}: storage {} is not {} F32 values", t.key, t.numel));
        }
        if t.size.len() != t.stride.len() {
            return Err(format!("torch: {name}: size and stride differ in rank"));
        }
        let mut expect = 1u64;
        for (&n, &s) in t.size.iter().zip(&t.stride).rev() {
            if n > 1 && s != expect {
                return Err(format!("torch: {name}: not contiguous"));
            }
            expect = expect.checked_mul(n).ok_or_else(|| format!("torch: {name}: size overflows"))?;
        }
        let end = t.offset.checked_add(expect).filter(|&e| e <= t.numel);
        let end = end.ok_or_else(|| format!("torch: {name}: lies outside its storage"))?;
        let data = &storage[t.offset as usize * 4..end as usize * 4];
        let shape = t.size.iter().map(|&n| n as usize).collect();
        if out.insert(name.clone(), Tensor { shape, data }).is_some() {
            return Err(format!("torch: {name} listed twice"));
        }
    }
    Ok(out)
}

/// The tensors `keep` selects, renamed by it, as canonical sorted F32
/// safetensors.
pub fn to_safetensors(tensors: &BTreeMap<String, Tensor<'_>>, keep: impl Fn(&str) -> Option<String>) -> Vec<u8> {
    let mut header = BTreeMap::<String, Value>::new();
    let mut picked: Vec<(String, &Tensor<'_>)> = tensors.iter().filter_map(|(k, t)| keep(k).map(|n| (n, t))).collect();
    picked.sort_by(|a, b| a.0.cmp(&b.0));
    let mut values = Vec::new();
    for (name, t) in picked {
        let start = values.len();
        values.extend_from_slice(t.data);
        header.insert(name, json!({"dtype": "F32", "shape": t.shape, "data_offsets": [start, values.len()]}));
    }
    let mut h = serde_json::to_vec(&header).expect("a JSON map of strings and numbers");
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(values);
    out
}

#[cfg(test)]
mod tests;
