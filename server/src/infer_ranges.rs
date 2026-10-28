//! The optional `int8_ranges` input of a ModelInfer (docs/kserve.md,
//! Quantization): FP32 `[2, width]`, each dimension's minimum, then each
//! one's maximum, for turbo_embed_options field 8. The server reads it into
//! memory of its own, aligned, and checks its width against the write's
//! resolved output_dim before the library reads 2 x output_dim values.

use crate::api::{Failure, Result, TURBO_E_INVALID_ARGUMENT};
use crate::infer::{EMPTY, bad, bad_shape, only, shape};
use crate::proto::ModelInferRequest;

/// turbo_embed_options' field number of int8_ranges.
pub(crate) const FIELD: u32 = 8;

/// The input's name.
pub(crate) const NAME: &str = "int8_ranges";

/// The caller's table: its values, minimums then maximums, and its width.
pub struct Int8Ranges {
    pub values: Vec<f32>,
    pub width: u32,
}

impl Int8Ranges {
    /// The table, if its width is the write's: 2 x output_dim values are
    /// what the library reads.
    pub fn for_width(&self, output_dim: u32) -> Result<*const f32> {
        if self.width != output_dim {
            return Err(Failure::new(
                TURBO_E_INVALID_ARGUMENT,
                FIELD,
                format!("input `{NAME}` is [2, {}] and the vectors are {output_dim} wide", self.width),
            ));
        }
        Ok(self.values.as_ptr())
    }
}

/// The request's `int8_ranges` input at `k`, read.
pub(crate) fn read(req: &ModelInferRequest, k: usize) -> Result<Int8Ranges> {
    let i = &req.inputs[k];
    if i.datatype != "FP32" {
        return Err(bad(format!("input `{NAME}` is {}; it takes FP32, and nothing is converted", i.datatype)));
    }
    let extents = shape(i, 2)?;
    if extents[0] != 2 {
        return Err(bad_shape(format!("input `{NAME}` is shaped {extents:?}; it is [2, width]")));
    }
    let width = extents[1];
    let count = 2 * width as usize;
    let values: Vec<f32> = if req.raw_input_contents.is_empty() {
        let c = i.contents.as_ref().unwrap_or(&EMPTY);
        only(i, c, "fp32_contents")?;
        c.fp32_contents.clone()
    } else {
        let b = &req.raw_input_contents[k];
        if b.len() != count * 4 {
            return Err(bad_shape(format!(
                "input `{NAME}` is shaped [2, {width}] and its raw contents are {} bytes, not {}",
                b.len(),
                count * 4
            )));
        }
        b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
    };
    if values.len() != count {
        return Err(bad_shape(format!("input `{NAME}` is shaped [2, {width}] and has {} elements", values.len())));
    }
    Ok(Int8Ranges { values, width })
}
