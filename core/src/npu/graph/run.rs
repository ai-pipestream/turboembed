//! A run: each frame through the graph, pooled and normalized on the host.

use std::ffi::c_void;

use super::args::*;
use super::buffers::*;
use super::host::*;
use super::session::*;
use super::*;
use crate::backend::turbo_backend_run;
use crate::npu::ze;
use crate::status::{INVALID_STATE, UNSUPPORTED};
use crate::{
    TURBO_EMBED_STAGE_DOWNLOAD, TURBO_EMBED_STAGE_ENCODE, TURBO_EMBED_STAGE_LOOKUP, TURBO_EMBED_STAGE_NORMALIZE,
    TURBO_EMBED_STAGE_POOL, TURBO_EMBED_STAGE_UPLOAD, TURBO_NORMALIZE_L2, TURBO_PLACE_HOST, TURBO_POOLING_CLS,
    TURBO_POOLING_LAST, TURBO_POOLING_MEAN, TURBO_STAGE_DEVICE, TURBO_STAGE_HOST, turbo_error,
};

/// F16's bits as F32, for reading hidden states back.
pub(super) fn half_to_f32(h: u16) -> f32 {
    let (sign, exp, frac) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
    let bits = match (exp, frac) {
        (0, 0) => sign << 31,
        (0, f) => {
            // Subnormal: normalize it.
            let shift = f.leading_zeros() - 21;
            (sign << 31) | ((127 - 15 - shift + 1) << 23) | ((f << (shift + 13)) & 0x007f_ffff)
        }
        (0x1f, 0) => (sign << 31) | 0x7f80_0000,
        (0x1f, f) => (sign << 31) | 0x7f80_0000 | (f << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

/// half_to_f32 over a row. With F16C the CPU widens eight at a time,
/// exactly, so a value is the same either way; a NaN stays a NaN.
pub(super) fn widen_f16(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("f16c") {
        // SAFETY: f16c was detected, and both slices are dst.len() long.
        return unsafe { widen_f16c(src, dst) };
    }
    for (d, &h) in dst.iter_mut().zip(src) {
        *d = half_to_f32(h);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
pub(super) unsafe fn widen_f16c(src: &[u16], dst: &mut [f32]) {
    use std::arch::x86_64::{__m128i, _mm_loadu_si128, _mm256_cvtph_ps, _mm256_storeu_ps};
    let whole = src.len() / 8 * 8;
    for i in (0..whole).step_by(8) {
        unsafe {
            let h = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
        }
    }
    for i in whole..src.len() {
        dst[i] = half_to_f32(src[i]);
    }
}

/// # Safety
/// As turbo_backend.h says for session_run.
pub(crate) unsafe extern "C" fn session_run(
    session: *mut c_void,
    out: *mut turbo_backend_run,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        guarded(err, || {
            let s = &*(session as *const Session);
            let mut rows = s.rows.lock().unwrap_or_else(|p| p.into_inner());
            if !rows.state.written {
                return Err(fail(INVALID_STATE, "npu: no rows are written"));
            }
            run(s, &mut rows, &mut *out)
        })
    }
}

/// One embeddings frame: the compiled arguments, the host buffers, and the
/// word table the gather reads.
pub(super) struct EmbedFrame<'a> {
    pub(super) rows_arg: &'a Arg,
    pub(super) bias_arg: &'a Arg,
    pub(super) rows_buf: &'a Buffer,
    pub(super) bias_buf: &'a Buffer,
    pub(super) table: *const f32,
    pub(super) vocab: u32,
    pub(super) heads: u32,
    pub(super) hidden: usize,
    pub(super) model_seq: usize,
    pub(super) written: usize,
}

/// Word rows gathered from the table, and the bias, written at the
/// precision the compiler kept.
pub(super) fn write_embedding_frame(
    frame: &EmbedFrame<'_>,
    live: usize,
    first: usize,
    stride: usize,
    rows: &mut Rows,
) -> Res<()> {
    let vocab = frame.vocab as usize;
    let heads = frame.heads as usize;
    let table = unsafe { std::slice::from_raw_parts(frame.table, vocab * frame.hidden) };
    rows.gathered_rows.fill(0.0);
    rows.gathered_bias.fill(MASKED);
    let row_elems = frame.model_seq * frame.hidden;
    let bias_elems = heads * frame.model_seq * frame.model_seq;
    for r in 0..live {
        let src = (first + r) * stride;
        gather_row(
            &Gather {
                table,
                vocab,
                hidden: frame.hidden,
                heads,
                model_seq: frame.model_seq,
                written: frame.written,
                ids: &rows.ids[src..src + frame.written],
                mask: &rows.mask[src..src + frame.written],
            },
            &mut rows.gathered_rows[r * row_elems..(r + 1) * row_elems],
            &mut rows.gathered_bias[r * bias_elems..(r + 1) * bias_elems],
        )?;
    }
    unsafe {
        write_floats(frame.rows_buf.ptr, frame.rows_arg.precision, &rows.gathered_rows)?;
        write_floats(frame.bias_buf.ptr, frame.bias_arg.precision, &rows.gathered_bias)?;
    }
    Ok(())
}

/// One frame of the graph's hidden states, [frame_batch, model_seq,
/// hidden] at the output's precision, read in place.
pub(super) struct FrameStates<'a> {
    pub(super) values: Hidden<'a>,
    pub(super) model_seq: usize,
    pub(super) hidden: usize,
}

pub(super) enum Hidden<'a> {
    F16(&'a [u16]),
    F32(&'a [f32]),
}

impl<'a> FrameStates<'a> {
    /// The frame in `buf`, as halves for an FP16 output and as f32
    /// otherwise.
    ///
    /// # Safety
    /// Nothing writes `buf` while the frame is read.
    pub(super) unsafe fn read(buf: &'a Buffer, precision: u32, model_seq: usize, hidden: usize) -> FrameStates<'a> {
        let values = match precision {
            ze::GRAPH_ARGUMENT_PRECISION_FP16 => {
                Hidden::F16(unsafe { std::slice::from_raw_parts(buf.ptr as *const u16, buf.bytes as usize / 2) })
            }
            _ => Hidden::F32(unsafe { std::slice::from_raw_parts(buf.ptr as *const f32, buf.bytes as usize / 4) }),
        };
        FrameStates { values, model_seq, hidden }
    }

    /// Token `t` of the frame's row `r` added into `acc` in f64, widened
    /// through `token` when the frame is FP16.
    pub(super) fn add(&self, r: usize, t: usize, acc: &mut [f64], token: &mut [f32]) {
        let at = (r * self.model_seq + t) * self.hidden;
        let values: &[f32] = match self.values {
            Hidden::F16(h) => {
                widen_f16(&h[at..at + self.hidden], token);
                token
            }
            Hidden::F32(v) => &v[at..at + self.hidden],
        };
        for (a, &v) in acc.iter_mut().zip(values) {
            *a += v as f64;
        }
    }

    /// The mean of the frame's row `r` over the tokens from column
    /// `start` whose entry in `mask` is 1, into `acc`. `mask` is the
    /// row's mask from `start` on, as long as the columns pooled. The
    /// tokens are summed in column order, so a row and a range over the
    /// same columns give the same bits.
    pub(super) fn mean(&self, r: usize, start: usize, mask: &[i32], acc: &mut [f64], token: &mut [f32]) {
        acc.fill(0.0);
        let mut live_tokens = 0f64;
        for (k, &v) in mask.iter().enumerate() {
            if v == 1 {
                live_tokens += 1.0;
                self.add(r, start + k, acc, token);
            }
        }
        for a in acc.iter_mut() {
            *a /= live_tokens;
        }
    }
}

/// The vectors whose tokens are in one frame, rows `first` to
/// `first + live`, pooled, cut and normalized into `result`. A write of
/// rows gives each live row its vector at the row's index. A write of
/// ranges gives each range whose row is in the frame its vector at the
/// range's index: ranges come in any order, and each frame pools the
/// ones it holds, since the hidden-state buffer holds one frame at a
/// time.
pub(super) fn pool_frame(
    states: &FrameStates<'_>,
    first: usize,
    live: usize,
    stride: usize,
    rows: &mut Rows,
    result: &mut [f32],
) {
    let Rows { mask, acc, token, ranges, state, .. } = rows;
    let seq = state.seq as usize;
    let dim = state.output_dim as usize;
    match state.vectors {
        Vectors::Rows => {
            for r in 0..live {
                let row = first + r;
                let mask = &mask[row * stride..row * stride + seq];
                match state.pooling {
                    TURBO_POOLING_CLS => {
                        acc.fill(0.0);
                        states.add(r, 0, acc, token);
                    }
                    TURBO_POOLING_LAST => {
                        acc.fill(0.0);
                        states.add(r, mask.iter().rposition(|&v| v == 1).unwrap_or(0), acc, token);
                    }
                    _ => {
                        debug_assert_eq!(state.pooling, TURBO_POOLING_MEAN);
                        states.mean(r, 0, mask, acc, token);
                    }
                }
                write_vector(&acc[..dim], state.normalize, &mut result[row * dim..(row + 1) * dim]);
            }
        }
        Vectors::Ranges => {
            for (i, g) in ranges.iter().enumerate() {
                let row = g.row as usize;
                if row < first || row >= first + live {
                    continue;
                }
                let (start, end) = (g.start as usize, g.end as usize);
                let at = row * stride;
                states.mean(row - first, start, &mask[at + start..at + end], acc, token);
                write_vector(&acc[..dim], state.normalize, &mut result[i * dim..(i + 1) * dim]);
            }
        }
    }
}

/// A pooled vector, already cut to its length, normalized as `normalize`
/// says and written to `dst` as f32.
pub(super) fn write_vector(vector: &[f64], normalize: u32, dst: &mut [f32]) {
    let norm = match normalize {
        // The same floor the cpu uses: a zero vector stays finite.
        TURBO_NORMALIZE_L2 => vector.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12),
        _ => 1.0,
    };
    for (d, v) in dst.iter_mut().zip(vector) {
        *d = (v / norm) as f32;
    }
}

pub(super) fn run(s: &Session, rows: &mut Rows, out: &mut turbo_backend_run) -> Res<()> {
    let m = s.model();
    let ctx = m.ctx();
    let set = m
        .ctx()
        .ext
        .set_argument_value()
        .ok_or_else(|| fail(UNSUPPORTED, "npu: the driver's graph extension has no pfnSetArgumentValue"))?;
    let _one_run_at_a_time = m.run.lock().unwrap_or_else(|p| p.into_inner());

    // The session's buffers onto the graph's arguments: values live on
    // the graph, and another session's are whatever it set last.
    let bind = |arg: &Arg, buf: &Buffer| ze_res("pfnSetArgumentValue", unsafe { set(m.graph, arg.index, buf.ptr) });
    let h2d_frame = match (&m.inputs, &s.frames) {
        (GraphInputs::Tokens { ids, mask, types }, Frames::Tokens { ids: ib, mask: mb, types: tb }) => {
            bind(ids, ib)?;
            bind(mask, mb)?;
            if let (Some(t), Some(b)) = (types, tb) {
                bind(t, b)?;
            }
            ib.bytes + mb.bytes + tb.as_ref().map_or(0, |b| b.bytes)
        }
        (GraphInputs::Embeddings { rows, bias, .. }, Frames::Embeddings { rows: rb, bias: bb }) => {
            bind(rows, rb)?;
            bind(bias, bb)?;
            rb.bytes + bb.bytes
        }
        _ => return Err(fail(INVALID_STATE, "npu: the session was not made for this model")),
    };
    bind(&m.output, &s.out_buf)?;

    let (batch, seq) = (rows.state.batch as usize, rows.state.seq as usize);
    let normalize = rows.state.normalize;
    let (frame_batch, model_seq, hidden) = (m.frame_batch as usize, m.seq as usize, m.hidden as usize);
    let stride = s.max_seq as usize;
    let frames = batch.div_ceil(frame_batch);
    let mut h2d = 0u64;
    let mut d2h = 0u64;
    // SAFETY: the session's vectors, made for max(max_batch, max_ranges)
    // rows of hidden F32, and read by the caller only after the run.
    let result = unsafe { std::slice::from_raw_parts_mut(s.result.ptr as *mut f32, s.result.bytes as usize / 4) };

    for frame in 0..frames {
        let first = frame * frame_batch;
        let live = (batch - first).min(frame_batch);
        // The frame's inputs. Token ids are the live tokens and zeros
        // after. An embeddings graph is gathered here: each token's word
        // row, and the bias the mask asks for. Rows past the batch stay
        // zero (and, for the bias, dropped), and nothing reads them.
        match (&m.inputs, &s.frames) {
            (GraphInputs::Tokens { ids, mask, types }, Frames::Tokens { ids: ib, mask: mb, types: tb }) => unsafe {
                std::ptr::write_bytes(ib.ptr as *mut u8, 0, ib.bytes as usize);
                std::ptr::write_bytes(mb.ptr as *mut u8, 0, mb.bytes as usize);
                if let Some(b) = tb {
                    std::ptr::write_bytes(b.ptr as *mut u8, 0, b.bytes as usize);
                }
                for r in 0..live {
                    let src = (first + r) * stride;
                    let dst = r * model_seq;
                    write_tokens(ib.ptr, ids.precision, dst, &rows.ids[src..src + seq]);
                    write_tokens(mb.ptr, mask.precision, dst, &rows.mask[src..src + seq]);
                    if let (Some(t), Some(b)) = (types, tb) {
                        write_tokens(b.ptr, t.precision, dst, &rows.types[src..src + seq]);
                    }
                }
            },
            (
                GraphInputs::Embeddings { rows: rows_arg, bias: bias_arg, heads, table, vocab },
                Frames::Embeddings { rows: rb, bias: bb },
            ) => {
                write_embedding_frame(
                    &EmbedFrame {
                        rows_arg,
                        bias_arg,
                        rows_buf: rb,
                        bias_buf: bb,
                        table: *table,
                        vocab: *vocab,
                        heads: *heads,
                        hidden,
                        model_seq,
                        written: seq,
                    },
                    live,
                    first,
                    stride,
                    rows,
                )?;
            }
            _ => return Err(fail(INVALID_STATE, "npu: the session was not made for this model")),
        }
        h2d += h2d_frame;

        ctx.execute(m.graph)?;
        d2h += s.out_buf.bytes;

        // Pool, cut and normalize on the host, reading the hidden states
        // where the device left them.
        // SAFETY: the execute above returned, and out_buf holds the frame.
        let states = unsafe { FrameStates::read(&s.out_buf, m.output.precision, model_seq, hidden) };
        pool_frame(&states, first, live, stride, rows, result);
    }

    out.placement = TURBO_PLACE_HOST;
    out.output = &*s.result as *const Buffer as *mut c_void;
    out.host = s.result.ptr;
    out.h2d_bytes = h2d;
    out.d2h_bytes = d2h;
    out.host_allocs = 0;
    out.device_allocs = 0;
    out.stage[TURBO_EMBED_STAGE_UPLOAD] = TURBO_STAGE_DEVICE;
    out.stage[TURBO_EMBED_STAGE_LOOKUP] = lookup_stage(&m.inputs);
    out.stage[TURBO_EMBED_STAGE_ENCODE] = TURBO_STAGE_DEVICE;
    out.stage[TURBO_EMBED_STAGE_POOL] = TURBO_STAGE_HOST;
    if normalize == TURBO_NORMALIZE_L2 {
        out.stage[TURBO_EMBED_STAGE_NORMALIZE] = TURBO_STAGE_HOST;
    }
    out.stage[TURBO_EMBED_STAGE_DOWNLOAD] = TURBO_STAGE_DEVICE;
    Ok(())
}

/// Where the word-row lookup ran. An embeddings graph gathers on the
/// host. A token-id graph hands ids to the device, and the graph does
/// the lookup.
pub(super) fn lookup_stage(inputs: &GraphInputs) -> u32 {
    match inputs {
        GraphInputs::Embeddings { .. } => TURBO_STAGE_HOST,
        GraphInputs::Tokens { .. } => TURBO_STAGE_DEVICE,
    }
}
