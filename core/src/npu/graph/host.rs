//! Host-side writes of a graph's inputs: gathered word rows, the bias and token ids.

use std::ffi::c_void;

use super::args::*;
use super::*;
use crate::npu::ze;
use crate::status::{INVALID_ARGUMENT, RUNTIME, UNSUPPORTED_OPTION};

/// The bias a dropped key gets. exp(-100) underflows to 0 in the softmax,
/// the value the Hailo cut is checked with (core/hailo/embed.cpp).
pub(super) const MASKED: f32 = -100.0;

/// Why a non-zero token type is refused, or None when the graph takes the
/// type ids and will run them.
#[inline]
pub(super) fn type_refusal(inputs: &GraphInputs) -> Option<&'static str> {
    match inputs {
        GraphInputs::Tokens { types: None, .. } => Some("the graph takes no token type input and computes type 0 only"),
        GraphInputs::Embeddings { .. } => Some("an INPUT_EMBEDDINGS graph computes token type 0 only"),
        GraphInputs::Tokens { types: Some(_), .. } => None,
    }
}

pub(super) fn type_error(row: usize, at: usize, ty: i32, why: &str) -> Fail {
    fail(UNSUPPORTED_OPTION, format!("npu: token type {ty} in row {row} position {at}: {why}"))
}

/// F32 as F16, round to nearest even, for writing a gathered tensor the
/// compiler kept in FP16.
#[inline]
pub(super) fn f32_to_half(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;
    if exp == 0xff {
        let payload = if frac == 0 { 0 } else { 0x200 };
        return sign | 0x7c00 | payload;
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let frac = frac | 0x0080_0000;
        let shift = (1 - half_exp) as u32;
        let mut half_frac = frac >> (shift + 13);
        let round = (frac >> (shift + 12)) & 1;
        let sticky = (frac & ((1 << (shift + 12)) - 1)) != 0;
        if round == 1 && (sticky || (half_frac & 1) == 1) {
            half_frac += 1;
        }
        return sign | (half_frac as u16);
    }
    let mut out = ((half_exp as u16) << 10) | ((frac >> 13) as u16);
    let round = (frac >> 12) & 1;
    let sticky = (frac & 0xfff) != 0;
    if round == 1 && (sticky || (out & 1) == 1) {
        out += 1;
    }
    sign | out
}

/// `src` into `dst` at the argument's float precision.
///
/// # Safety
/// `dst` holds `src.len()` elements of that precision.
pub(super) unsafe fn write_floats(dst: *mut c_void, precision: u32, src: &[f32]) -> Res<()> {
    unsafe {
        match precision {
            ze::GRAPH_ARGUMENT_PRECISION_FP32 => {
                (dst as *mut f32).copy_from_nonoverlapping(src.as_ptr(), src.len());
            }
            ze::GRAPH_ARGUMENT_PRECISION_FP16 => {
                let d = dst as *mut u16;
                for (i, &v) in src.iter().enumerate() {
                    d.add(i).write(f32_to_half(v));
                }
            }
            p => {
                return Err(fail(
                    RUNTIME,
                    format!("npu: a gathered tensor is argument precision 0x{p:02x}; FP32 and FP16 are written"),
                ));
            }
        }
    }
    Ok(())
}

/// One written row, as the Hailo backend gathers it.
pub(super) struct Gather<'a> {
    pub(super) table: &'a [f32],
    pub(super) vocab: usize,
    pub(super) hidden: usize,
    pub(super) heads: usize,
    /// The compiled frame length.
    pub(super) model_seq: usize,
    /// Tokens `embed_write` copied. Past this the frame repeats the first id.
    pub(super) written: usize,
    pub(super) ids: &'a [i32],
    pub(super) mask: &'a [i32],
}

/// `rows_out` is `[model_seq, hidden]`. `bias_out` is `[heads, model_seq, model_seq]`,
/// and every query sees the same keys: 0 where the mask keeps a key, MASKED
/// where it drops one.
pub(super) fn gather_row(g: &Gather<'_>, rows_out: &mut [f32], bias_out: &mut [f32]) -> Res<()> {
    if g.written == 0 || g.written > g.model_seq || g.ids.len() < g.written || g.mask.len() < g.written {
        return Err(fail(INVALID_ARGUMENT, "npu: a gathered row is longer than the frame or empty"));
    }
    if rows_out.len() != g.model_seq * g.hidden || bias_out.len() != g.heads * g.model_seq * g.model_seq {
        return Err(fail(INVALID_ARGUMENT, "npu: the gather buffers are not the frame"));
    }
    for t in 0..g.model_seq {
        let id = if t < g.written { g.ids[t] } else { g.ids[0] };
        if id < 0 || id as usize >= g.vocab {
            return Err(fail(
                INVALID_ARGUMENT,
                format!("npu: token id {id} is outside the word table of {} rows", g.vocab),
            ));
        }
        let src = id as usize * g.hidden;
        let dst = t * g.hidden;
        rows_out[dst..dst + g.hidden].copy_from_slice(&g.table[src..src + g.hidden]);
    }
    let plane = g.model_seq * g.model_seq;
    for k in 0..g.model_seq {
        let v = if k < g.written && g.mask[k] == 1 { 0.0 } else { MASKED };
        for head in 0..g.heads {
            let base = head * plane;
            for q in 0..g.model_seq {
                bias_out[base + q * g.model_seq + k] = v;
            }
        }
    }
    Ok(())
}

/// An i32 row written into a graph input at the argument's precision.
///
/// # Safety
/// `dst` holds `n` elements of the argument's precision at `at`.
#[inline]
pub(super) unsafe fn write_tokens(dst: *mut c_void, precision: u32, at: usize, src: &[i32]) {
    unsafe {
        match precision {
            ze::GRAPH_ARGUMENT_PRECISION_INT64 | ze::GRAPH_ARGUMENT_PRECISION_UINT64 => {
                let d = (dst as *mut i64).add(at);
                for (i, &v) in src.iter().enumerate() {
                    d.add(i).write(v as i64);
                }
            }
            _ => {
                let d = (dst as *mut i32).add(at);
                d.copy_from_nonoverlapping(src.as_ptr(), src.len());
            }
        }
    }
}
