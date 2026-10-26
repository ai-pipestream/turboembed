//! DeBERTa-v2 (TURBO_FAMILY_DEBERTA_V2) on the device: the parts that
//! differ from the BERT encoder. The CPU encoder (core/src/deberta) is the
//! reference every kernel here is held to.
//!
//! The model's tensors are TURBO_DEBERTA_*, the word table, its LayerNorm,
//! the relative table and its LayerNorm, then each layer's in TURBO_BERT_*
//! order. Each layer reads the relative table through its own query and
//! key projections (shared keys); those rows depend on the weights alone,
//! so the first session makes them once for the model, beside the fused
//! projections: the table LayerNormed, and each layer's fused query, key
//! and value projection of it, [2 * buckets, 3 * hidden]. A session holds
//! the bucket of every distance its rows can span, made when it is made.
//! The lookup and attention are deberta_embed and deberta_attention in
//! encoder.cl; the linear layers, the LayerNorms after them and the
//! pooling are the BERT encoder's, in F32: FASTEST is refused.

use std::ffi::c_void;

use super::model::Model;
use super::*;
use crate::TURBO_PRECISION_FASTEST;
use crate::backend::{
    TURBO_BERT_EMBEDDING_TENSORS, TURBO_BERT_LAYER_TENSORS, TURBO_DEBERTA_EMBEDDING_TENSORS,
    TURBO_DEBERTA_RELATIVE_EMBEDDINGS, TURBO_DEBERTA_RELATIVE_LN_BIAS, TURBO_DEBERTA_RELATIVE_LN_WEIGHT,
    TURBO_FAMILY_DEBERTA_V2, turbo_backend_model,
};
use crate::deberta::{Config, bucket};
use crate::levelzero::gpu::{Arg, Context, Kernel, Res, fail, fail_field};
use crate::status::{INVALID_ARGUMENT, INVALID_STRUCT_SIZE, UNSUPPORTED_OPTION};

/// The tensors before the first layer's, by family.
pub(super) fn layer_base(d: &turbo_backend_model) -> u32 {
    if d.family == TURBO_FAMILY_DEBERTA_V2 { TURBO_DEBERTA_EMBEDDING_TENSORS } else { TURBO_BERT_EMBEDDING_TENSORS }
}

/// What model_load checks of a DeBERTa-v2 description, as the CPU's does:
/// the struct reaches max_relative_positions, the table holds the
/// embeddings and each layer's tensors, and the shape is one the CPU
/// encoder takes.
///
/// # Safety
/// `d` points to a description of d.struct_size bytes at least.
pub(super) unsafe fn check_load(d: *const turbo_backend_model) -> Res<()> {
    let need = std::mem::offset_of!(turbo_backend_model, max_relative_positions) + size_of::<u32>();
    let size = unsafe { (*d).struct_size } as usize;
    if size < need {
        return Err(fail(
            INVALID_STRUCT_SIZE,
            format!("turbo_backend_model of {size} bytes: a DeBERTa-v2 model needs {need}"),
        ));
    }
    let d = unsafe { &*d };
    let expected = TURBO_DEBERTA_EMBEDDING_TENSORS as u64 + d.layers as u64 * TURBO_BERT_LAYER_TENSORS as u64;
    if d.tensor_count as u64 != expected || d.tensors.is_null() {
        return Err(fail(
            INVALID_ARGUMENT,
            format!("tensor_count {}: a DeBERTa-v2 of {} layers has {expected}", d.tensor_count, d.layers),
        ));
    }
    config(d).check().map_err(|e| fail(INVALID_ARGUMENT, e.message))
}

fn config(d: &turbo_backend_model) -> Config {
    Config {
        hidden: d.hidden as usize,
        intermediate: d.intermediate as usize,
        heads: d.heads as usize,
        layers: d.layers as usize,
        vocabulary: d.vocab_size as usize,
        max_sequence: d.max_positions as usize,
        position_buckets: d.position_buckets as usize,
        max_relative_positions: d.max_relative_positions as usize,
        layer_norm_eps: d.layer_norm_eps,
    }
}

/// A session at `precision` on a DeBERTa-v2 model: MODEL and EXACT run in
/// F32; FASTEST is refused, never run in F32 in its place.
pub(super) fn check_precision(precision: u32) -> Res<()> {
    if precision == TURBO_PRECISION_FASTEST {
        return Err(fail_field(
            UNSUPPORTED_OPTION,
            3,
            "precision: FASTEST on FAMILY_DEBERTA_V2: the levelzero backend runs a DeBERTa-v2 encoder in F32 \
             alone; ask for MODEL or EXACT",
        ));
    }
    Ok(())
}

/// Queries a tiled attention group takes for a head width, as
/// encoder.cl's instances of DEBERTA_FLASH; None for the widths the
/// general kernel takes.
fn tiled_queries(head_dim: u32) -> Option<u32> {
    match head_dim {
        32 => Some(128),
        64 => Some(64),
        128 => Some(32),
        _ => None,
    }
}

/// The tiled attention for `head_dim` and its queries a group, where one
/// is built, its local memory fits the device, and a test has not forced
/// the general kernel.
pub(super) fn tiled(c: &Context, head_dim: u32) -> Res<Option<(Kernel, u32)>> {
    #[cfg(feature = "internals")]
    if GENERAL.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(None);
    }
    let Some(queries) = tiled_queries(head_dim) else { return Ok(None) };
    let k = c.kernel(&format!("deberta_attention_{head_dim}"), [queries, 1, 1])?;
    Ok((k.local_bytes()? <= c.max_local).then_some((k, queries)))
}

/// A test's choice of the general attention for the sessions made after
/// it, in place of the tiled one (use_general_attention).
#[cfg(feature = "internals")]
static GENERAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "internals")]
pub(crate) fn use_general_attention(on: bool) {
    GENERAL.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// The bucket of each distance i - j between positions of a row of
/// max_seq tokens, at i - j + max_seq - 1.
pub(super) fn buckets(d: &turbo_backend_model, max_seq: u32) -> Vec<i32> {
    let span = max_seq as isize - 1;
    let (b, most) = (d.position_buckets as usize, d.max_relative_positions as usize);
    (-span..=span).map(|k| bucket(k, b, most) as i32).collect()
}

impl Model {
    /// Each layer's relative rows through its fused projection, from the
    /// F32 tensors and fused projections: the addresses, and the one
    /// allocation that holds them with the LayerNormed table.
    pub(super) fn relative(&self, tensors: &[u64], qkv: &[(u64, u64)]) -> Res<(Vec<u64>, *mut c_void)> {
        let c = self.ctx();
        let d = &self.desc;
        let (h, rows) = (d.hidden as usize, 2 * d.position_buckets as usize);
        let table = round_up(rows * h * 4, DEVICE_ALIGN);
        let per_layer = round_up(rows * 3 * h * 4, DEVICE_ALIGN);
        let alloc = c.alloc_device(table + d.layers as usize * per_layer)?;
        let at: Vec<u64> =
            (0..d.layers as u64).map(|l| alloc as u64 + (table + l as usize * per_layer) as u64).collect();
        let made = (|| {
            let norm = c.kernel("layer_norm_rows", [BLOCK, 1, 1])?;
            let linear = c.kernel("linear", [16, 16, 1])?;
            let mut q = c.lock_queue()?;
            let appended = (|| {
                let src = tensors[TURBO_DEBERTA_RELATIVE_EMBEDDINGS as usize] as usize as *const c_void;
                unsafe { c.copy(&mut q, alloc, src, rows * h * 4)? };
                let args = [
                    Arg::Ptr(alloc as u64),
                    Arg::Ptr(tensors[TURBO_DEBERTA_RELATIVE_LN_WEIGHT as usize]),
                    Arg::Ptr(tensors[TURBO_DEBERTA_RELATIVE_LN_BIAS as usize]),
                    Arg::F32(d.layer_norm_eps as f32),
                    Arg::I32(h as i32),
                ];
                norm.launch(c, &mut q, "the relative table's LayerNorm", &args, [rows as u32, 1, 1])?;
                for (l, &(w, b)) in qkv.iter().enumerate() {
                    let n_out = 3 * h as u32;
                    let args = [
                        Arg::Ptr(alloc as u64),
                        Arg::Ptr(w),
                        Arg::Ptr(b),
                        Arg::Ptr(at[l]),
                        Arg::I32(rows as i32),
                        Arg::I32(n_out as i32),
                        Arg::I32(h as i32),
                        Arg::I32(LINEAR_BIAS),
                        Arg::I32(h as i32),
                    ];
                    let groups = [n_out.div_ceil(TILE), (rows as u32).div_ceil(TILE), 1];
                    linear.launch(c, &mut q, "the relative projections", &args, groups)?;
                }
                Ok(())
            })();
            let synced = c.sync(&mut q);
            appended?;
            synced
        })();
        if let Err(e) = made {
            c.free(alloc, "the relative projections");
            return Err(e);
        }
        Ok((at, alloc))
    }
}
