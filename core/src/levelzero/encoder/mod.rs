//! Models and embed sessions on a listed GPU: the BERT encoder in the
//! kernels of core/levelzero/encoder.cl.
//!
//! Loading copies each tensor, in the dtype it is stored in, into one
//! device allocation: that copy is the load, and the core's host bytes are
//! not read again. The first session makes, on the device, what the
//! sessions compute from, shared by every later one and freed with the
//! model: an F16 or BF16 model's weights widened to F32; each layer's Q, K
//! and V weights and biases side by side; and at FASTEST the linear
//! layers' weights in F16.
//!
//! A session is device scratch for its largest batch, host staging the
//! device reads, and the buffer its vectors are written to, all allocated
//! when it is made; a run allocates nothing. embed_write packs the rows on
//! the host into the staging: each row's positions through its last live
//! token, one after another, and a table of where each row starts. The
//! padding after a row's last live token is never computed: attention
//! skips masked keys and no pooling reads past the last live token, so no
//! output depends on it. The run's lookup kernel reads the packed rows
//! over the bus, and the run leaves the vectors on the device, in the
//! session's DEVICE buffer, which turbo_result_read copies back through
//! buffer_read.
//!
//! embed_write_ranges packs the rows the same way and copies its ranges
//! into the staging after the row table, where the pooling kernel reads
//! them over the bus as the lookup kernel reads the rows. The run then
//! writes one vector per range in place of one per row, so the output
//! buffer holds max(max_batch, max_ranges) vectors of hidden F32.

use super::gpu::{Context, Kernel, Res};
use crate::{TURBO_DTYPE_F16, TURBO_DTYPE_F32, turbo_token_range};

/// TURBO_BERT_* in turbo_backend.h: the embedding tensors, then each
/// layer's at these offsets.
const WORD: usize = 0;
const POSITION: usize = 1;
const TOKEN_TYPE: usize = 2;
const EMB_LN_W: usize = 3;
const EMB_LN_B: usize = 4;
const Q_WEIGHT: u32 = 0;
const Q_BIAS: u32 = 1;
const K_WEIGHT: u32 = 2;
const K_BIAS: u32 = 3;
const V_WEIGHT: u32 = 4;
const V_BIAS: u32 = 5;
const ATTN_OUT_WEIGHT: u32 = 6;
const ATTN_OUT_BIAS: u32 = 7;
const ATTN_LN_WEIGHT: u32 = 8;
const ATTN_LN_BIAS: u32 = 9;
const FFN_IN_WEIGHT: u32 = 10;
const FFN_IN_BIAS: u32 = 11;
const FFN_OUT_WEIGHT: u32 = 12;
const FFN_OUT_BIAS: u32 = 13;
const FFN_LN_WEIGHT: u32 = 14;
const FFN_LN_BIAS: u32 = 15;

/// Tensors and scratch in one device allocation each start on 256 bytes.
const DEVICE_ALIGN: usize = 256;
/// Tokens per group of the LayerNorm kernels, a sub-group each, as
/// encoder.cl's ROWS; below FEW_TOKENS a group takes one token instead.
const ROWS: u32 = 8;
const FEW_TOKENS: u32 = 256;
/// Sub-groups of a pooling group, each summing every so-many tokens, as
/// encoder.cl's POOL_SLICES.
const POOL_SLICES: u32 = 8;
/// Bytes of one turbo_token_range in the staging: row, start and end,
/// three int32 as the pool_ranges kernel reads them.
const RANGE_BYTES: usize = size_of::<turbo_token_range>();
const _: () = assert!(RANGE_BYTES == 12);
/// Work-items per group for the row kernels, as encoder.cl's BLOCK.
const BLOCK: u32 = 128;
/// Work-items per group for the elementwise kernels.
const WIDE: u32 = 256;
/// The linear kernel's output tile, as encoder.cl's TILE.
const TILE: u32 = 64;

fn round_up(n: usize, a: usize) -> usize {
    n.div_ceil(a) * a
}

fn dtype_name(d: u32) -> &'static str {
    match d {
        TURBO_DTYPE_F32 => "F32",
        TURBO_DTYPE_F16 => "F16",
        _ => "BF16",
    }
}

/// Groups for an elementwise pass over n values: enough to fill the
/// device, each work-item striding over the rest.
fn elementwise_groups(n: u64) -> u32 {
    n.div_ceil(WIDE as u64).clamp(1, 65535) as u32
}

mod deberta;
mod encode;
mod kernels;
mod model;
mod session;
mod stages;
mod tune;
mod write;

#[cfg(feature = "internals")]
pub(crate) use model::widened;
pub(crate) use model::{model_load, model_release};
#[cfg(feature = "internals")]
pub(crate) use session::use_choices;
pub(crate) use session::{session_create, session_open, session_release};
pub(crate) use tune::session_create_tuned;
pub(crate) use write::{embed_write, embed_write_ranges, session_run};

/// In F32, the parts the feed-forward output's sums are split into, over
/// its terms, so its few output tiles still fill the device; the LayerNorm
/// after it adds them. FASTEST never splits.
const SPLITS: u32 = 4;

/// The F32 sub-group linear kernel's tile, as encoder.cl's SG_N and SG_T.
const SG_N: u32 = 64;
const SG_T: u32 = 8;

/// The XMX linear kernels' tiles, as encoder.cl's instances of
/// LINEAR_DPAS: a sub-group's DPAS_TM tokens by DPAS_TN outputs, and a
/// group's DPAS_WM by DPAS_WN sub-groups; for at most FEW_TOKENS_DPAS
/// tokens, FEW_TM by DPAS_TN, and FEW_WM by FEW_WN. The terms go DPAS_K a
/// step.
const DPAS_TM: u32 = 16;
const DPAS_TN: u32 = 32;
const DPAS_WM: u32 = 8;
const DPAS_WN: u32 = 2;
const FEW_TOKENS_DPAS: u32 = 8;
const FEW_TM: u32 = 8;
const FEW_WM: u32 = 1;
const FEW_WN: u32 = 4;
const DPAS_K: u32 = 32;
/// From this hidden width (a multiple of 64), the XMX linear kernels take
/// the wide tiles: a sub-group's WIDE_TM tokens by WIDE_TN outputs, WIDE_WM
/// by WIDE_WN sub-groups a group; the LayerNorm-fused kernel WIDE_LN_TM
/// tokens by WIDE_TN outputs a sub-group; and the feed-forward block runs
/// as two kernels at every batch. The narrow models keep the tiles above:
/// on the wide ones they lose to the weights' traffic, on the narrow ones
/// the wide tiles leave the device half filled (docs/levelzero.md).
const WIDE_HIDDEN: u32 = 768;
const WIDE_TM: u32 = 32;
const WIDE_TN: u32 = 64;
const WIDE_WM: u32 = 4;
const WIDE_WN: u32 = 1;
const WIDE_LN_TM: u32 = 32;

/// oneDNN's primitives are built for a shape, tokens included, and
/// building one takes the host's memory and up to tens of milliseconds. So
/// oneDNN runs a batch at its tokens rounded up to a bin, a multiple of a
/// 128th of the next power of two and of ONEDNN_STEP at least, and at most
/// the session's tokens, which bounds the primitives a context keeps and
/// the runs that build one: the first at each bin, which counts them in
/// host_allocs. (Built when the session is made instead, the same
/// primitives ran 0.6% to 1% slower on a B70.) The rows past the batch's
/// tokens are scratch the other kernels never read, each computed on its
/// own.
const ONEDNN_STEP: u32 = 16;

#[cfg_attr(not(feature = "levelzero-onednn"), allow(dead_code))]
fn onednn_tokens(tokens: u32, most: u32) -> u32 {
    tokens.next_multiple_of((tokens.next_power_of_two() / 128).max(ONEDNN_STEP)).min(most)
}

fn wide(hidden: u32) -> bool {
    hidden >= WIDE_HIDDEN && hidden.is_multiple_of(64)
}
/// The LayerNorm-fused XMX kernel's tokens a group, and its most
/// sub-groups, DPAS_TN outputs each, as encoder.cl's.
const DPAS_LN_TM: u32 = 16;
const DPAS_LN_SUBGROUPS: u32 = 64;
const DPAS_LN_BLOCKS: u32 = 4;

/// Blocks of tokens a LayerNorm-fused group of `cols` sub-groups a block
/// takes: as many as its sub-groups and the kernel's largest group (`most`
/// blocks) allow, up to DPAS_LN_BLOCKS, so more tokens share each read of
/// the weights.
fn dpas_ln_blocks(cols: u32, most: Option<u32>) -> u32 {
    (1..=DPAS_LN_BLOCKS).rev().find(|b| cols * b <= DPAS_LN_SUBGROUPS && most.is_none_or(|m| *b <= m)).unwrap_or(1)
}
/// linear_sgemm's tiles, as encoder.cl's: a sub-group's SGEMM_TM tokens by
/// SGEMM_TN outputs, a group's SGEMM_WM by SGEMM_WN sub-groups.
const SGEMM_TM: u32 = 8;
const SGEMM_TN: u32 = 32;
const SGEMM_WM: u32 = 4;
const SGEMM_WN: u32 = 2;
/// The kernels for 8 tokens at most: GEMV_SUBGROUPS sub-groups a group,
/// each summing a share of the terms, as encoder.cl's GV_KS and GV.
const GEMV_SUBGROUPS: u32 = 8;

/// The terms each of a GEMV group's sub-groups sums: an equal share, a
/// multiple of 16, of k_len. None when no such share covers k_len exactly
/// with the sub-groups there are.
fn gemv_share(k_len: u32) -> Option<u32> {
    let share = k_len.div_ceil(16 * GEMV_SUBGROUPS) * 16;
    (k_len.is_multiple_of(share) && k_len / share <= GEMV_SUBGROUPS).then_some(share)
}

/// Tokens from which FASTEST runs the whole feed-forward block in one
/// kernel, where its 2-byte middle, intermediate values a token, would
/// otherwise go through memory; fewer tokens fill too few of its groups.
const MLP_TOKENS: u32 = 4096;

/// Bytes of local memory the feed-forward kernel's middle takes: blocks of
/// DPAS_LN_TM tokens by hidden values in F16.
fn mlp_local_bytes(hidden: u32, blocks: u32) -> u32 {
    DPAS_LN_TM * blocks * hidden * 2
}

/// The one-kernel feed-forward block and its blocks of tokens a group: as
/// many as dpas_ln_blocks gives, fewer where its middle and the kernel's
/// own local memory do not fit the device's; None where no block fits, or
/// the intermediate width is not a multiple of the hidden width, a chunk
/// of the middle at a time.
fn mlp_kernel(c: &Context, hidden: u32, intermediate: u32) -> Res<Option<(Kernel, u32)>> {
    if hidden / DPAS_TN > DPAS_LN_SUBGROUPS || intermediate == 0 || !intermediate.is_multiple_of(hidden) {
        return Ok(None);
    }
    let cols = hidden / DPAS_TN;
    let one = c.kernel("linear_dpas_mlp", [16 * cols, 1, 1])?;
    let most = dpas_ln_blocks(cols, one.max_group_size()?.map(|m| m / (16 * cols)));
    let k = if most == 1 { one } else { c.kernel("linear_dpas_mlp", [16 * cols * most, 1, 1])? };
    let own = k.local_bytes()?;
    match (1..=most).rev().find(|&b| own + mlp_local_bytes(hidden, b) <= c.max_local) {
        Some(b) if b == most => Ok(Some((k, b))),
        Some(b) => Ok(Some((c.kernel("linear_dpas_mlp", [16 * cols * b, 1, 1])?, b))),
        None => Ok(None),
    }
}

/// Sub-groups of an XMX attention group, 16 queries each, as encoder.cl's
/// ATT_SUBGROUPS.
const ATT_SUBGROUPS: u32 = 4;
/// The staged XMX attention's sub-groups a group (encoder.cl's ATT2_SG);
/// it runs where the longest row fills a group.
const ATT2_SUBGROUPS: u32 = 8;

/// Queries a tiled attention group takes, as encoder.cl's QUERIES.
const QUERIES: u32 = 256;

/// The linear kernel's epilogue, as encoder.cl's LINEAR_*.
const LINEAR_BIAS: i32 = 1;
const LINEAR_GELU: i32 = 2;

/// Bytes of local memory the general attention kernel takes for rows of
/// seq tokens: the query's head, the row's scores and the partial
/// contexts.
fn attention_local_bytes(seq: u32, head_dim: u32) -> u64 {
    let part = if head_dim <= BLOCK { (BLOCK / head_dim) * head_dim } else { 0 };
    4 * (head_dim as u64 + seq as u64 + part as u64)
}

/// Whose kernels run a session's linear layers past a few tokens: the
/// backend's own, or oneDNN's (the levelzero-onednn feature).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Linear {
    Own,
    #[cfg_attr(not(feature = "levelzero-onednn"), allow(dead_code))]
    Onednn,
}

impl Linear {
    fn name(self) -> &'static str {
        match self {
            Linear::Own => "own",
            Linear::Onednn => "onednn",
        }
    }
}
