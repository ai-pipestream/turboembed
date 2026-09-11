//! Tensor wire-format helpers for the OIP V2 raw byte representation.
//!
//! OIP transports large tensors as flat little-endian byte blobs in
//! `raw_input_contents` / `raw_output_contents`, paired positionally with the
//! `inputs` / `outputs` tensor metadata (name, dtype string, shape). These
//! helpers cover dtype parsing, element sizing, shape/byte-length validation,
//! and packing/unpacking of the common numeric dtypes.

use std::fmt;

/// OIP V2 tensor data types, matching the spec's dtype strings
/// (`"FP32"`, `"INT64"`, `"BYTES"`, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Int8,
    Int16,
    Int32,
    Int64,
    Fp16,
    Bf16,
    Fp32,
    Fp64,
    /// Variable-length byte strings; each element is length-prefixed with a
    /// little-endian u32 in the raw representation.
    Bytes,
}

impl DataType {
    /// Parse an OIP dtype string (case-sensitive, as produced by the spec).
    pub fn from_oip(s: &str) -> Result<Self, TensorError> {
        Ok(match s {
            "BOOL" => Self::Bool,
            "UINT8" => Self::Uint8,
            "UINT16" => Self::Uint16,
            "UINT32" => Self::Uint32,
            "UINT64" => Self::Uint64,
            "INT8" => Self::Int8,
            "INT16" => Self::Int16,
            "INT32" => Self::Int32,
            "INT64" => Self::Int64,
            "FP16" => Self::Fp16,
            "BF16" => Self::Bf16,
            "FP32" => Self::Fp32,
            "FP64" => Self::Fp64,
            "BYTES" => Self::Bytes,
            other => return Err(TensorError::UnknownDataType(other.to_string())),
        })
    }

    /// The OIP dtype string for this data type.
    pub fn as_oip(&self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::Uint8 => "UINT8",
            Self::Uint16 => "UINT16",
            Self::Uint32 => "UINT32",
            Self::Uint64 => "UINT64",
            Self::Int8 => "INT8",
            Self::Int16 => "INT16",
            Self::Int32 => "INT32",
            Self::Int64 => "INT64",
            Self::Fp16 => "FP16",
            Self::Bf16 => "BF16",
            Self::Fp32 => "FP32",
            Self::Fp64 => "FP64",
            Self::Bytes => "BYTES",
        }
    }

    /// Fixed element size in bytes, or `None` for variable-length `BYTES`.
    pub fn element_size(&self) -> Option<usize> {
        match self {
            Self::Bool | Self::Uint8 | Self::Int8 => Some(1),
            Self::Uint16 | Self::Int16 | Self::Fp16 | Self::Bf16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Fp32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Fp64 => Some(8),
            Self::Bytes => None,
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_oip())
    }
}

/// Errors produced by tensor wire-format helpers.
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("unknown OIP data type {0:?}")]
    UnknownDataType(String),
    #[error("shape {shape:?} contains a non-positive dimension; concrete tensors need fully specified shapes")]
    InvalidShape { shape: Vec<i64> },
    #[error("raw content length {actual} bytes does not match expected {expected} bytes for dtype {dtype} and shape {shape:?}")]
    LengthMismatch {
        dtype: DataType,
        shape: Vec<i64>,
        expected: usize,
        actual: usize,
    },
    #[error("raw content length {actual} is not a multiple of element size {element_size} for dtype {dtype}")]
    Misaligned {
        dtype: DataType,
        element_size: usize,
        actual: usize,
    },
    #[error("truncated BYTES tensor: {0}")]
    TruncatedBytes(String),
}

/// Number of elements described by a concrete shape.
///
/// Returns an error if any dimension is negative or zero-dimension products
/// overflow (`-1` variable dims are only legal in metadata, not payloads).
pub fn element_count(shape: &[i64]) -> Result<usize, TensorError> {
    let mut count: usize = 1;
    for &dim in shape {
        if dim < 0 {
            return Err(TensorError::InvalidShape {
                shape: shape.to_vec(),
            });
        }
        count = count
            .checked_mul(dim as usize)
            .ok_or(TensorError::InvalidShape {
                shape: shape.to_vec(),
            })?;
    }
    Ok(count)
}

/// Validate that a raw byte blob is the right length for `dtype` + `shape`.
///
/// For fixed-size dtypes this checks `element_count * element_size`. For
/// `BYTES` it walks the u32 length-prefixed elements and checks the count.
pub fn validate_raw(dtype: DataType, shape: &[i64], raw: &[u8]) -> Result<(), TensorError> {
    let count = element_count(shape)?;
    match dtype.element_size() {
        Some(size) => {
            let expected = count * size;
            if raw.len() != expected {
                return Err(TensorError::LengthMismatch {
                    dtype,
                    shape: shape.to_vec(),
                    expected,
                    actual: raw.len(),
                });
            }
            Ok(())
        }
        None => {
            let elements = unpack_bytes(raw)?;
            if elements.len() != count {
                return Err(TensorError::LengthMismatch {
                    dtype,
                    shape: shape.to_vec(),
                    expected: count,
                    actual: elements.len(),
                });
            }
            Ok(())
        }
    }
}

macro_rules! pack_unpack {
    ($pack:ident, $unpack:ident, $ty:ty, $dtype:expr) => {
        /// Pack a slice into the OIP raw little-endian byte representation.
        pub fn $pack(values: &[$ty]) -> Vec<u8> {
            let mut out = Vec::with_capacity(values.len() * std::mem::size_of::<$ty>());
            for v in values {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }

        /// Unpack an OIP raw little-endian byte blob into typed values.
        pub fn $unpack(raw: &[u8]) -> Result<Vec<$ty>, TensorError> {
            const SIZE: usize = std::mem::size_of::<$ty>();
            if raw.len() % SIZE != 0 {
                return Err(TensorError::Misaligned {
                    dtype: $dtype,
                    element_size: SIZE,
                    actual: raw.len(),
                });
            }
            Ok(raw
                .chunks_exact(SIZE)
                .map(|c| <$ty>::from_le_bytes(c.try_into().expect("chunks_exact")))
                .collect())
        }
    };
}

pack_unpack!(pack_fp32, unpack_fp32, f32, DataType::Fp32);
pack_unpack!(pack_fp64, unpack_fp64, f64, DataType::Fp64);
pack_unpack!(pack_i32, unpack_i32, i32, DataType::Int32);
pack_unpack!(pack_i64, unpack_i64, i64, DataType::Int64);
pack_unpack!(pack_u32, unpack_u32, u32, DataType::Uint32);
pack_unpack!(pack_u64, unpack_u64, u64, DataType::Uint64);

/// Pack variable-length `BYTES` elements: each element is prefixed with its
/// little-endian u32 length, per the OIP raw representation.
pub fn pack_bytes<T: AsRef<[u8]>>(elements: &[T]) -> Vec<u8> {
    let total: usize = elements.iter().map(|e| 4 + e.as_ref().len()).sum();
    let mut out = Vec::with_capacity(total);
    for e in elements {
        let e = e.as_ref();
        out.extend_from_slice(&(e.len() as u32).to_le_bytes());
        out.extend_from_slice(e);
    }
    out
}

/// Unpack a raw `BYTES` blob into its variable-length elements.
pub fn unpack_bytes(raw: &[u8]) -> Result<Vec<Vec<u8>>, TensorError> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < raw.len() {
        if cursor + 4 > raw.len() {
            return Err(TensorError::TruncatedBytes(format!(
                "length prefix at offset {cursor} runs past end ({} bytes total)",
                raw.len()
            )));
        }
        let len = u32::from_le_bytes(raw[cursor..cursor + 4].try_into().expect("4 bytes"))
            as usize;
        cursor += 4;
        if cursor + len > raw.len() {
            return Err(TensorError::TruncatedBytes(format!(
                "element of {len} bytes at offset {cursor} runs past end ({} bytes total)",
                raw.len()
            )));
        }
        out.push(raw[cursor..cursor + len].to_vec());
        cursor += len;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_roundtrip() {
        for s in [
            "BOOL", "UINT8", "UINT16", "UINT32", "UINT64", "INT8", "INT16", "INT32", "INT64",
            "FP16", "BF16", "FP32", "FP64", "BYTES",
        ] {
            assert_eq!(DataType::from_oip(s).unwrap().as_oip(), s);
        }
        assert!(DataType::from_oip("float32").is_err());
    }

    #[test]
    fn fp32_roundtrip_little_endian() {
        let values = [1.0f32, -2.5, 3.25e10, f32::MIN_POSITIVE];
        let raw = pack_fp32(&values);
        assert_eq!(raw.len(), 16);
        assert_eq!(&raw[..4], &1.0f32.to_le_bytes());
        assert_eq!(unpack_fp32(&raw).unwrap(), values);
    }

    #[test]
    fn misaligned_raw_rejected() {
        assert!(matches!(
            unpack_fp32(&[0u8; 6]),
            Err(TensorError::Misaligned { .. })
        ));
    }

    #[test]
    fn validate_raw_lengths() {
        let raw = pack_fp32(&[0.0; 6]);
        assert!(validate_raw(DataType::Fp32, &[2, 3], &raw).is_ok());
        assert!(matches!(
            validate_raw(DataType::Fp32, &[2, 4], &raw),
            Err(TensorError::LengthMismatch { .. })
        ));
        assert!(matches!(
            validate_raw(DataType::Fp32, &[-1, 3], &raw),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn bytes_roundtrip() {
        let elements: Vec<&[u8]> = vec![b"hello", b"", b"inference"];
        let raw = pack_bytes(&elements);
        let back = unpack_bytes(&raw).unwrap();
        assert_eq!(back, elements.iter().map(|e| e.to_vec()).collect::<Vec<_>>());
        assert!(validate_raw(DataType::Bytes, &[3], &raw).is_ok());
        assert!(validate_raw(DataType::Bytes, &[2], &raw).is_err());
    }

    #[test]
    fn truncated_bytes_rejected() {
        let mut raw = pack_bytes(&[b"hello".as_slice()]);
        raw.truncate(raw.len() - 1);
        assert!(matches!(
            unpack_bytes(&raw),
            Err(TensorError::TruncatedBytes(_))
        ));
    }
}
