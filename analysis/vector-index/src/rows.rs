//! Rows as the kernels read them: blocks of 16 `f32`, one cache line
//! each, a row padded with zeros to whole blocks.

use crate::{IndexError, MAX_LENGTH, Metric, Normalization, Space, UNIT_TOLERANCE, Which};

/// The width of a block, and the number of lanes a dot product is summed
/// in.
pub(crate) const LANES: usize = 16;

#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(64))]
pub(crate) struct Block(pub(crate) [f32; LANES]);

/// The blocks a row of `dim` values takes.
pub(crate) fn blocks(dim: u32) -> usize {
    (dim as usize).div_ceil(LANES)
}

/// Refuses `other` unless it is `index`'s space: the same model, dimension
/// and normalization.
pub(crate) fn same_space(index: &Space, other: &Space) -> Result<(), IndexError> {
    if other.model != index.model {
        return Err(IndexError::Model { index: Box::new(index.model.clone()), other: Box::new(other.model.clone()) });
    }
    if other.dim != index.dim {
        return Err(IndexError::Dimension { index: index.dim, other: other.dim as usize });
    }
    if other.normalization != index.normalization {
        return Err(IndexError::Normalization { index: index.normalization, other: other.normalization });
    }
    Ok(())
}

/// Checks `values` as a vector of `space` and writes it into `out` as the
/// index scores it: as given for [`Metric::Dot`], divided by its length
/// for [`Metric::Cosine`], zeros after the last value. `out` holds
/// `blocks(space.dim)` blocks and `values` has `space.dim` values.
pub(crate) fn prepare(
    values: &[f32],
    space: &Space,
    metric: Metric,
    which: Which,
    out: &mut [Block],
) -> Result<(), IndexError> {
    let mut sum = 0.0f64;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(IndexError::NonFinite(which, i));
        }
        sum += f64::from(v) * f64::from(v);
    }
    if sum == 0.0 {
        return Err(IndexError::Zero(which));
    }
    let len = sum.sqrt();
    if len > MAX_LENGTH {
        return Err(IndexError::TooLong(which, len));
    }
    if space.normalization == Normalization::Unit && (len - 1.0).abs() > UNIT_TOLERANCE {
        return Err(IndexError::NotUnit(which, len));
    }
    out.fill(Block::default());
    let flat = flat_mut(out);
    match metric {
        Metric::Dot => flat[..values.len()].copy_from_slice(values),
        Metric::Cosine => {
            for (o, &v) in flat.iter_mut().zip(values) {
                *o = (f64::from(v) / len) as f32;
            }
        }
    }
    Ok(())
}

pub(crate) fn flat(blocks: &[Block]) -> &[f32] {
    // SAFETY: a Block is 16 f32 with no padding (repr(C), size 64), so a
    // slice of n blocks is 16n contiguous f32.
    unsafe { std::slice::from_raw_parts(blocks.as_ptr().cast::<f32>(), blocks.len() * LANES) }
}

pub(crate) fn flat_mut(blocks: &mut [Block]) -> &mut [f32] {
    // SAFETY: as in `flat`, and the borrow is unique.
    unsafe { std::slice::from_raw_parts_mut(blocks.as_mut_ptr().cast::<f32>(), blocks.len() * LANES) }
}

pub(crate) fn write_space(space: &Space, out: &mut Vec<u8>) {
    write_str(&space.model.id, out);
    write_str(&space.model.revision, out);
    write_str(&space.model.sha256, out);
    out.extend_from_slice(&space.dim.to_le_bytes());
    let norm = match space.normalization {
        Normalization::Unit => 0u8,
        Normalization::None => 1u8,
    };
    out.push(norm);
}

pub(crate) fn read_space(r: &mut ByteReader<'_>) -> Result<Space, String> {
    let id = r.str()?;
    let revision = r.str()?;
    let sha256 = r.str()?;
    let dim = r.u32()?;
    if dim == 0 {
        return Err("snapshot space has dimension 0".into());
    }
    let normalization = match r.u8()? {
        0 => Normalization::Unit,
        1 => Normalization::None,
        b => return Err(format!("invalid normalization tag {b}")),
    };
    Ok(Space::new(turbo_analysis::ModelRef { id, revision, sha256 }, dim, normalization))
}

pub(crate) fn write_str(s: &str, out: &mut Vec<u8>) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

pub(crate) struct ByteReader<'a>(pub(crate) &'a [u8]);

impl<'a> ByteReader<'a> {
    pub(crate) fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.0.len() < n {
            return Err("truncated vector-index snapshot".into());
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }

    pub(crate) fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub(crate) fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub(crate) fn str(&mut self) -> Result<String, String> {
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes).map(str::to_owned).map_err(|_| "invalid UTF-8 in snapshot".into())
    }
}
