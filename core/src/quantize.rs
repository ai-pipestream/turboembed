//! int8 and ubinary vector codes, the layout sentence-transformers'
//! `quantize_embeddings` makes (`sentence_transformers/util/quantization.py`
//! at 145ef47d, Apache-2.0): every step in f32, as numpy computes it there.
//!
//! - int8: `ranges` is `[2, dim]`, the minimums then the maximums. A step
//!   is `(max - min) / 255`, or 1 when that is 0, and a value's code is
//!   `clamp(floor((x - min) / step), 0, 255) - 128`.
//! - ubinary: bit `x > 0`, eight to a byte, the first value in the most
//!   significant bit (numpy's `packbits`); a last byte the width does not
//!   fill is padded with 0 bits.
//!
//! turbo.h's turbo_embed_options fields 7 and 8 select these; the session
//! applies them on the host to every backend's vectors, after the cut and
//! the normalization.

/// What a write's vectors are returned as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantization {
    F32,
    Int8,
    UBinary,
}

impl Quantization {
    /// The bytes one row of `dim` values takes.
    pub fn row_bytes(self, dim: usize) -> usize {
        match self {
            Quantization::F32 => dim * 4,
            Quantization::Int8 => dim,
            Quantization::UBinary => dim.div_ceil(8),
        }
    }
}

/// Why a `[2, dim]` table of int8 ranges cannot be used.
pub fn check_int8_ranges(ranges: &[f32]) -> Result<(), String> {
    if !ranges.len().is_multiple_of(2) {
        return Err(format!("int8 ranges hold {} values, not 2 per dimension", ranges.len()));
    }
    let (min, max) = ranges.split_at(ranges.len() / 2);
    for (d, (&lo, &hi)) in min.iter().zip(max).enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            return Err(format!("int8 ranges: dimension {d} has minimum {lo} and maximum {hi}, not both finite"));
        }
        if lo > hi {
            return Err(format!("int8 ranges: dimension {d}'s minimum {lo} is over its maximum {hi}"));
        }
    }
    Ok(())
}

/// One row's int8 codes, as two's-complement bytes. `ranges` is `[2, dim]`
/// for `x`'s `dim` values, already checked.
pub fn int8_row(x: &[f32], ranges: &[f32], out: &mut [u8]) {
    let (min, max) = ranges.split_at(x.len());
    for (((o, &v), &lo), &hi) in out.iter_mut().zip(x).zip(min).zip(max) {
        let step = (hi - lo) / 255.0;
        let step = if step == 0.0 { 1.0 } else { step };
        // NaN clamps to NaN and casts to 0, the bottom bucket; vectors a
        // backend returns are finite.
        let q = ((v - lo) / step).floor().clamp(0.0, 255.0) as i32;
        *o = (q - 128) as i8 as u8;
    }
}

/// One row's sign bits, packed most significant bit first.
pub fn ubinary_row(x: &[f32], out: &mut [u8]) {
    for (o, chunk) in out.iter_mut().zip(x.chunks(8)) {
        let mut byte = 0u8;
        for (i, &v) in chunk.iter().enumerate() {
            if v > 0.0 {
                byte |= 0x80 >> i;
            }
        }
        *o = byte;
    }
}

/// `rows` rows of `dim` f32 values, as codes into `out`, which holds
/// `rows * q.row_bytes(dim)` bytes. `ranges` is used by int8 alone.
pub fn rows(q: Quantization, x: &[f32], dim: usize, ranges: &[f32], out: &mut [u8]) {
    let width = q.row_bytes(dim);
    for (row, o) in x.chunks_exact(dim).zip(out.chunks_exact_mut(width)) {
        match q {
            Quantization::F32 => {
                for (b, v) in o.as_chunks_mut::<4>().0.iter_mut().zip(row) {
                    b.copy_from_slice(&v.to_le_bytes());
                }
            }
            Quantization::Int8 => int8_row(row, ranges, o),
            Quantization::UBinary => ubinary_row(row, o),
        }
    }
}

/// The code bytes a device quantize call of `rows` by `dim` writes, or why
/// the call cannot be made: the mode is not int8 or ubinary, a dimension
/// is negative, the shape overflows, or a slice (`x` F32 values, `ranges`
/// F32 values, `out` bytes) is shorter than the shape needs. The device
/// wrappers check this before any pointer reaches the device.
pub fn check_device_call(
    quantization: u32,
    x: usize,
    ranges: usize,
    rows: i32,
    dim: i32,
    out: usize,
) -> Result<usize, String> {
    let q = match quantization {
        crate::TURBO_QUANTIZE_INT8 => Quantization::Int8,
        crate::TURBO_QUANTIZE_UBINARY => Quantization::UBinary,
        other => return Err(format!("quantization {other} is not int8 or ubinary")),
    };
    let (Ok(r), Ok(d)) = (usize::try_from(rows), usize::try_from(dim)) else {
        return Err(format!("rows {rows} and dim {dim} must not be negative"));
    };
    let floats = r.checked_mul(d).ok_or_else(|| format!("{rows} by {dim} values overflow"))?;
    let codes = r.checked_mul(q.row_bytes(d)).ok_or_else(|| format!("{rows} by {dim} codes overflow"))?;
    if x < floats {
        return Err(format!("x holds {x} values, under the {floats} of {rows} by {dim}"));
    }
    if q == Quantization::Int8 && ranges < 2 * d {
        return Err(format!("ranges holds {ranges} values, under the {} int8 needs", 2 * d));
    }
    if out < codes {
        return Err(format!("out holds {out} bytes, under the {codes} codes"));
    }
    Ok(codes)
}

#[cfg(test)]
mod tests;
