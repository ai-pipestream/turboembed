use std::ffi::CString;

use crate::bundle::AlignedBytes;
use crate::manifest::{Artifact, StaticEmbedding, TensorRole, role_name};
use crate::safetensors;
use crate::status::{Result, invalid};
use crate::{
    TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_DTYPE_F64, TURBO_DTYPE_I8, TURBO_DTYPE_I32,
    TURBO_DTYPE_I64,
};

use super::{Expected, STATIC_ROLES};

/// A tensor found in one of the artifact's files.
pub(crate) struct Placed {
    pub(super) name: CString,
    pub(crate) file: usize,
    pub(crate) range: std::ops::Range<usize>,
    pub(super) shape: Vec<u64>,
    /// TURBO_DTYPE_* it is stored in.
    pub(crate) dtype: u32,
}

/// The dtypes a static model's tensor in each role may be stored in.
fn static_dtypes(role: TensorRole) -> &'static [safetensors::Dtype] {
    use safetensors::Dtype::*;
    match role {
        TensorRole::StaticEmbeddings => &[F32, F16, Bf16, F64, I8],
        TensorRole::StaticWeights => &[F32, F16, F64],
        _ => &[I32, I64],
    }
}

/// Rule 8 for a static model: each role the artifact names, found in its
/// files with the shape the static_embedding block implies, a dtype the
/// role takes and an aligned start; a role it does not name is None. Every
/// mapping value is checked to be a row of the table.
pub(super) fn place_static(
    index: usize,
    art: &Artifact,
    files: &[AlignedBytes],
    st: &StaticEmbedding,
    dim: u32,
) -> Result<Vec<Option<Placed>>> {
    let parsed = art
        .files
        .iter()
        .zip(files)
        .map(|(name, bytes)| safetensors::File::parse(name, &bytes[..]))
        .collect::<Result<Vec<_>>>()?;
    let rows = if st.rows == 0 { st.vocab_size } else { st.rows };
    let mut out = Vec::with_capacity(STATIC_ROLES.len());
    for role in STATIC_ROLES {
        let Some(name) = art.tensor_names.get(&role) else {
            out.push(None);
            continue;
        };
        let what = role_name(role);
        let mut found = parsed.iter().enumerate().filter_map(|(i, f)| f.tensor(name).map(|t| (i, t)));
        let Some((file, t)) = found.next() else {
            return Err(invalid(format!("{}: no tensor {name:?} ({what})", art.files.join(", "))));
        };
        if let Some((other, _)) = found.next() {
            return Err(invalid(format!("{name:?} ({what}) is in both {} and {}", art.files[file], art.files[other])));
        }
        let at = &art.files[file];
        let allowed = static_dtypes(role);
        if !allowed.contains(&t.dtype) {
            let names: Vec<&str> = allowed.iter().map(|d| d.name()).collect();
            return Err(invalid(format!("{at}: {name} ({what}) is {}; it may be {}", t.dtype_name, names.join(", "))));
        }
        let shape = match role {
            TensorRole::StaticEmbeddings => vec![rows as u64, dim as u64],
            _ => vec![st.vocab_size as u64],
        };
        if t.shape != shape {
            return Err(invalid(format!(
                "{at}: {name} ({what}) has shape {:?}; static_embedding implies {shape:?}",
                t.shape
            )));
        }
        let begin = t.data.as_ptr() as usize - files[file].as_ptr() as usize;
        let size = t.dtype.size().expect("a static dtype has a size");
        if !begin.is_multiple_of(size) {
            return Err(invalid(format!(
                "{at}: {name} ({what}) starts at byte {begin}, not a multiple of its {size}-byte elements"
            )));
        }
        if role == TensorRole::StaticMapping {
            let bad = match t.dtype {
                safetensors::Dtype::I32 => t
                    .data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .position(|b| !(0..rows as i64).contains(&(i32::from_le_bytes(*b) as i64))),
                _ => t.data.as_chunks::<8>().0.iter().position(|b| !(0..rows as i64).contains(&i64::from_le_bytes(*b))),
            };
            if let Some(id) = bad {
                return Err(invalid(format!(
                    "{at}: {name} ({what}) maps token {id} outside the table's {rows} rows (artifacts[{index}])"
                )));
            }
        }
        out.push(Some(Placed {
            name: CString::new(name.as_str()).map_err(|_| invalid(format!("{name:?}: a NUL in the name")))?,
            file,
            range: begin..begin + t.data.len(),
            shape: t.shape.clone(),
            dtype: backend_dtype(t.dtype),
        }));
    }
    Ok(out)
}

/// TURBO_DTYPE_* for a safetensors dtype the core hands a backend.
fn backend_dtype(d: safetensors::Dtype) -> u32 {
    use safetensors::Dtype::*;
    match d {
        I8 => TURBO_DTYPE_I8,
        I32 => TURBO_DTYPE_I32,
        I64 => TURBO_DTYPE_I64,
        F16 => TURBO_DTYPE_F16,
        Bf16 => TURBO_DTYPE_BF16,
        F64 => TURBO_DTYPE_F64,
        _ => TURBO_DTYPE_F32,
    }
}

/// Rule 8: find each of `expected` in `files`, the verified bytes of
/// `art`'s files, with its shape, one dtype for all, and an aligned start.
pub(crate) fn place(art: &Artifact, files: &[AlignedBytes], expected: &[Expected]) -> Result<(Vec<Placed>, u32)> {
    let parsed = art
        .files
        .iter()
        .zip(files)
        .map(|(name, bytes)| safetensors::File::parse(name, &bytes[..]))
        .collect::<Result<Vec<_>>>()?;

    let mut dtype = None;
    let mut tensors = Vec::with_capacity(expected.len());
    for e in expected {
        let mut found = parsed.iter().enumerate().filter_map(|(i, f)| f.tensor(&e.name).map(|t| (i, t)));
        let Some((file, t)) = found.next() else {
            return Err(invalid(format!("{}: no tensor {:?} ({})", art.files.join(", "), e.name, e.what)));
        };
        if let Some((other, _)) = found.next() {
            return Err(invalid(format!(
                "{:?} ({}) is in both {} and {}",
                e.name, e.what, art.files[file], art.files[other]
            )));
        }
        let at = &art.files[file];
        let d = match t.dtype {
            safetensors::Dtype::F32 => TURBO_DTYPE_F32,
            safetensors::Dtype::F16 => TURBO_DTYPE_F16,
            safetensors::Dtype::Bf16 => TURBO_DTYPE_BF16,
            _ => {
                return Err(invalid(format!(
                    "{at}: {} ({}) is {}; weights are F32, F16 or BF16",
                    e.name, e.what, t.dtype_name
                )));
            }
        };
        let want = *dtype.get_or_insert(d);
        if d != want {
            return Err(invalid(format!(
                "{at}: {} ({}) is {}; the other weights are {}",
                e.name,
                e.what,
                t.dtype_name,
                dtype_name(want)
            )));
        }
        if t.shape != e.shape {
            return Err(invalid(format!(
                "{at}: {} ({}) has shape {:?}; the architecture implies {:?}",
                e.name, e.what, t.shape, e.shape
            )));
        }
        // The file starts on a 64-byte boundary, so an offset that is a
        // multiple of the element size is an aligned address.
        let begin = t.data.as_ptr() as usize - files[file].as_ptr() as usize;
        let size = t.dtype.size().expect("a weights dtype has a size");
        if !begin.is_multiple_of(size) {
            return Err(invalid(format!(
                "{at}: {} ({}) starts at byte {begin}, not a multiple of its {size}-byte elements",
                e.name, e.what
            )));
        }
        tensors.push(Placed {
            name: CString::new(e.name.as_str()).map_err(|_| invalid(format!("{:?}: a NUL in the name", e.name)))?,
            file,
            range: begin..begin + t.data.len(),
            shape: t.shape.clone(),
            dtype: d,
        });
    }
    Ok((tensors, dtype.unwrap_or(0)))
}

fn dtype_name(d: u32) -> &'static str {
    match d {
        TURBO_DTYPE_F16 => "F16",
        TURBO_DTYPE_BF16 => "BF16",
        _ => "F32",
    }
}
