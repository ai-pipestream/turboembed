//! The finiteness check on embedding rows (issue #2517). A NaN or an
//! infinity in a vector is a broken run, never an answer: the rerank
//! path already refuses non-finite encoder output, and the embed paths
//! check every float row before it is sent.

use turbo::{TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F32};

/// The index of the first value in `bytes`, little-endian values of
/// `dtype`, that is NaN or infinite. Integer codes are always finite.
pub(crate) fn first_non_finite(bytes: &[u8], dtype: u32) -> Option<usize> {
    match dtype {
        TURBO_DTYPE_F32 => {
            bytes.chunks_exact(4).position(|b| !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite())
        }
        // Half and bfloat16 are not finite when every exponent bit is set.
        TURBO_DTYPE_F16 => bytes.chunks_exact(2).position(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7c00 == 0x7c00),
        TURBO_DTYPE_BF16 => bytes.chunks_exact(2).position(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80),
        _ => None,
    }
}

/// The width in bytes of one value of `dtype`, for a float dtype.
fn width(dtype: u32) -> Option<usize> {
    match dtype {
        TURBO_DTYPE_F32 => Some(4),
        TURBO_DTYPE_F16 | TURBO_DTYPE_BF16 => Some(2),
        _ => None,
    }
}

/// The first non-finite value of `data`, `rows` rows of `dim` values of
/// `dtype`, as (row, dimension), or None.
pub(crate) fn first_bad_row(data: &[u8], dtype: u32, dim: u32) -> Option<(usize, usize)> {
    let row = width(dtype)? * dim as usize;
    if row == 0 {
        return None;
    }
    data.chunks(row).enumerate().find_map(|(r, b)| first_non_finite(b, dtype).map(|d| (r, d)))
}

/// What a row whose dimension `d` is not finite is told.
pub(crate) fn message(row: Option<usize>, d: usize) -> String {
    match row {
        Some(r) => format!("the model's output for row {r} is not finite at dimension {d}: the run is broken"),
        None => format!("the model's output for this document is not finite at dimension {d}: the run is broken"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_float_width_finds_nan_and_infinity() {
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        assert_eq!(first_non_finite(&f32s(&[0.0, 1.0, -2.5]), TURBO_DTYPE_F32), None);
        assert_eq!(first_non_finite(&f32s(&[0.0, f32::NAN]), TURBO_DTYPE_F32), Some(1));
        assert_eq!(first_non_finite(&f32s(&[f32::NEG_INFINITY]), TURBO_DTYPE_F32), Some(0));
        // F16: 1.0 is 0x3c00, +inf 0x7c00, NaN 0x7e00.
        assert_eq!(first_non_finite(&[0x00, 0x3c, 0x00, 0x7e], TURBO_DTYPE_F16), Some(1));
        assert_eq!(first_non_finite(&[0x00, 0x7c], TURBO_DTYPE_F16), Some(0));
        assert_eq!(first_non_finite(&[0x00, 0x3c], TURBO_DTYPE_F16), None);
        // BF16: 1.0 is 0x3f80, NaN 0x7fc0.
        assert_eq!(first_non_finite(&[0x80, 0x3f, 0xc0, 0x7f], TURBO_DTYPE_BF16), Some(1));
        assert_eq!(first_non_finite(&[0xff, 0xff], turbo::TURBO_DTYPE_I8), None);
        let two_rows = f32s(&[1.0, 2.0, 3.0, f32::NAN]);
        assert_eq!(first_bad_row(&two_rows, TURBO_DTYPE_F32, 2), Some((1, 1)));
    }
}
