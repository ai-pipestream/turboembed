//! Reading a safetensors file already in memory: an 8-byte little-endian
//! header length, a JSON header naming each tensor's dtype, shape and byte
//! range, then the data.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::status::{Result, invalid};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    I32,
    F16,
    Bf16,
    F32,
    Other,
}

impl Dtype {
    pub fn size(self) -> Option<usize> {
        match self {
            Dtype::I32 | Dtype::F32 => Some(4),
            Dtype::F16 | Dtype::Bf16 => Some(2),
            Dtype::Other => None,
        }
    }
}

pub struct Tensor<'a> {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data: &'a [u8],
}

impl Tensor<'_> {
    pub fn i32s(&self) -> Vec<i32> {
        self.data.chunks_exact(4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    pub fn f32s(&self) -> Vec<f32> {
        self.data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }
}

pub struct File<'a> {
    name: String,
    tensors: BTreeMap<String, Tensor<'a>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

impl<'a> File<'a> {
    /// `name` is the bundle path, for messages.
    pub fn parse(name: &str, bytes: &'a [u8]) -> Result<File<'a>> {
        let bad = |why: String| invalid(format!("{name}: {why}"));
        if bytes.len() < 8 {
            return Err(bad("shorter than a safetensors header".into()));
        }
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let body = &bytes[8..];
        if n > body.len() as u64 {
            return Err(bad(format!("header length {n} is past the end of the file")));
        }
        let (header, data) = body.split_at(n as usize);
        let raw: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(header).map_err(|e| bad(format!("header: {e}")))?;
        let mut tensors = BTreeMap::new();
        for (key, v) in raw {
            if key == "__metadata__" {
                continue;
            }
            let e: Entry = serde_json::from_value(v).map_err(|e| bad(format!("{key}: {e}")))?;
            let dtype = match e.dtype.as_str() {
                "I32" => Dtype::I32,
                "F16" => Dtype::F16,
                "BF16" => Dtype::Bf16,
                "F32" => Dtype::F32,
                _ => Dtype::Other,
            };
            let [begin, end] = e.data_offsets;
            if begin > end || end > data.len() as u64 {
                return Err(bad(format!("{key}: data_offsets [{begin}, {end}] are outside the data")));
            }
            if let Some(size) = dtype.size() {
                let count = e.shape.iter().try_fold(1u64, |a, &d| a.checked_mul(d));
                if count.and_then(|c| c.checked_mul(size as u64)) != Some(end - begin) {
                    return Err(bad(format!("{key}: shape {:?} does not match {} bytes", e.shape, end - begin)));
                }
            }
            let data = &data[begin as usize..end as usize];
            tensors.insert(key, Tensor { dtype, shape: e.shape, data });
        }
        Ok(File { name: name.to_owned(), tensors })
    }

    /// The tensor `key`, if the file has it.
    pub fn tensor(&self, key: &str) -> Option<&Tensor<'a>> {
        self.tensors.get(key)
    }

    /// The tensor `key`, which must have `dtype` and `ndim` dimensions.
    pub fn get(&self, key: &str, dtype: Dtype, ndim: usize) -> Result<&Tensor<'a>> {
        let t = self.tensors.get(key).ok_or_else(|| invalid(format!("{}: no tensor {key:?}", self.name)))?;
        if t.dtype != dtype || t.shape.len() != ndim {
            return Err(invalid(format!(
                "{}: {key} is not {dtype:?} with {ndim} dimensions (shape {:?})",
                self.name, t.shape
            )));
        }
        Ok(t)
    }
}
