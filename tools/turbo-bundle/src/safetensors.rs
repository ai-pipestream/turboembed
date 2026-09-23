//! Minimal safetensors reader: enough to extract one 2-D float tensor as a
//! little-endian f32 table. The format is an 8-byte little-endian header
//! length, a JSON header mapping tensor names to `{dtype, shape,
//! data_offsets}`, then the raw data.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use serde::Deserialize;

/// Tensor entry in the JSON header.
#[derive(Debug, Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: (u64, u64),
}

/// A parsed safetensors file held in memory.
pub struct SafeTensors {
    header: BTreeMap<String, Entry>,
    data: Vec<u8>,
}

impl SafeTensors {
    /// Read and parse a file.
    pub fn open(path: &Path) -> Result<Self, String> {
        let mut f = std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(|e| format!("read header length of {}: {e}", path.display()))?;
        let n = u64::from_le_bytes(len);
        if n > 100 * 1024 * 1024 {
            return Err(format!("safetensors header of {n} bytes is implausibly large"));
        }
        let mut header = vec![0u8; n as usize];
        f.read_exact(&mut header).map_err(|e| format!("read header of {}: {e}", path.display()))?;
        let mut parsed: BTreeMap<String, serde_json::Value> =
            serde_json::from_slice(&header).map_err(|e| format!("safetensors header JSON: {e}"))?;
        parsed.remove("__metadata__");
        let mut entries = BTreeMap::new();
        for (k, v) in parsed {
            let e: Entry = serde_json::from_value(v).map_err(|e| format!("tensor `{k}` header: {e}"))?;
            entries.insert(k, e);
        }
        let mut data = Vec::new();
        f.read_to_end(&mut data).map_err(|e| format!("read data of {}: {e}", path.display()))?;
        Ok(Self { header: entries, data })
    }

    /// Tensor names.
    pub fn names(&self) -> Vec<&str> {
        self.header.keys().map(String::as_str).collect()
    }

    /// Extract a 2-D tensor as f32 values (row-major) and its shape.
    /// Supports `F32`, `F16`, and `BF16`.
    pub fn table_f32(&self, name: &str) -> Result<(Vec<f32>, [u64; 2]), String> {
        let e = self
            .header
            .get(name)
            .ok_or_else(|| format!("tensor `{name}` not found; available: {}", self.names().join(", ")))?;
        if e.shape.len() != 2 {
            return Err(format!("tensor `{name}` has shape {:?}; expected 2-D", e.shape));
        }
        let (start, end) = e.data_offsets;
        let bytes = self
            .data
            .get(start as usize..end as usize)
            .ok_or_else(|| format!("tensor `{name}` offsets {start}..{end} exceed the data section"))?;
        let count = (e.shape[0] * e.shape[1]) as usize;
        let values: Vec<f32> = match e.dtype.as_str() {
            "F32" => {
                if bytes.len() != count * 4 {
                    return Err(format!("tensor `{name}` has {} bytes, expected {}", bytes.len(), count * 4));
                }
                bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            }
            "F16" => {
                if bytes.len() != count * 2 {
                    return Err(format!("tensor `{name}` has {} bytes, expected {}", bytes.len(), count * 2));
                }
                bytes.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
            }
            "BF16" => {
                if bytes.len() != count * 2 {
                    return Err(format!("tensor `{name}` has {} bytes, expected {}", bytes.len(), count * 2));
                }
                bytes.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()
            }
            other => return Err(format!("tensor `{name}` dtype {other} is not supported; use F32, F16, or BF16")),
        };
        Ok((values, [e.shape[0], e.shape[1]]))
    }
}

/// IEEE half to single.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            (sign << 31) | ((e as u32) << 23) | (f << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversion() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x3555) - 0.333_25).abs() < 1e-4);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!((f16_to_f32(0x0001) - 5.960_464_5e-8).abs() < 1e-12);
    }

    #[test]
    fn parses_a_small_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("m.safetensors");
        let header = serde_json::json!({
            "embeddings": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
            "__metadata__": {"x": "y"}
        })
        .to_string();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        let st = SafeTensors::open(&path).unwrap();
        let (vals, shape) = st.table_f32("embeddings").unwrap();
        assert_eq!(shape, [2, 3]);
        assert_eq!(vals, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(st.table_f32("nope").unwrap_err().contains("available: embeddings"));
    }
}
