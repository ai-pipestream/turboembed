//! The encoder's kernel objects, one set per session.

use super::*;
use crate::levelzero::gpu::{Context, Kernel, Res};

// ---- Sessions ------------------------------------------------------------------

/// The encoder's kernel objects, one set per session.
pub(super) struct Kernels {
    pub(super) linear: Kernel,
    /// The linear layers in F32 by sub-group, where the terms come in
    /// sixteens.
    pub(super) linear_sg: Kernel,
    /// The same for 8 tokens at most, a group's sub-groups sharing the
    /// terms.
    pub(super) linear_gemv: Kernel,
    /// In F32 for more than a handful of tokens, from transposed weights.
    pub(super) linear_sgemm: Kernel,
    /// The linear layers on the matrix engines, for a session at FASTEST,
    /// from F16 activations: to F32 and to F16, then the same for at most
    /// 8 tokens.
    pub(super) linear_dpas: Option<[Kernel; 4]>,
    /// The main XMX linear tile: tokens and outputs a sub-group, sub-groups a group each way.
    pub(super) dpas: (u32, u32, u32, u32),
    /// At FASTEST, the attention output and the feed-forward output with
    /// the LayerNorm after them, for a hidden width a group spans: groups of
    /// one block of tokens, and of as many as dpas_ln_blocks gives.
    pub(super) linear_dpas_layer_norm: Option<[Kernel; 2]>,
    /// Its tokens a sub-group and blocks a group.
    pub(super) ln: (u32, u32),
    /// At FASTEST, the whole feed-forward block in one kernel for large
    /// batches, and its blocks of tokens a group.
    pub(super) linear_dpas_mlp: Option<(Kernel, u32)>,
    pub(super) embed_layer_norm: Kernel,
    /// F16 hidden states widened to F32, for the pooling after oneDNN's last LayerNorm.
    #[cfg_attr(not(feature = "levelzero-onednn"), allow(dead_code))]
    pub(super) widen_f16: Kernel,
    pub(super) add_layer_norm: Kernel,
    /// The same with each row held in registers, for a hidden width that
    /// is a multiple of 64 up to 1024.
    pub(super) add_layer_norm_v4: Option<Kernel>,
    /// The same, a group per token, for few tokens.
    pub(super) embed_layer_norm_group: Kernel,
    pub(super) add_layer_norm_group: Kernel,
    pub(super) attention: Attention,
    pub(super) pool: Kernel,
    /// The mean over each range of embed_write_ranges, in place of pool.
    pub(super) pool_ranges: Kernel,
    pub(super) normalize: Kernel,
}

/// Attention for the model's head width: a tiled kernel where one is
/// built for it, else the general one.
pub(super) enum Attention {
    Tiled(Kernel),
    General(Kernel),
    /// At FASTEST, on the matrix engines, from F16 projections to an F16
    /// context.
    /// The plain kernel, and for head width 64 the one with K and V
    /// staged in local memory, for the runs whose longest row fills its
    /// group.
    Xmx(Kernel, Option<Kernel>),
}

impl Kernels {
    pub(super) fn new(c: &Context, hidden: u32, intermediate: u32, head_dim: u32, xmx: bool) -> Res<Kernels> {
        let row = [BLOCK, 1, 1];
        // At FASTEST the context is the next layer's F16 operand.
        let attention = match head_dim {
            32 | 64 if xmx => {
                let staged = if head_dim == 64 {
                    Some(c.kernel("attention_xmx2_64", [16 * ATT2_SUBGROUPS, 1, 1])?)
                } else {
                    None
                };
                Attention::Xmx(c.kernel(&format!("attention_xmx_{head_dim}"), [16 * ATT_SUBGROUPS, 1, 1])?, staged)
            }
            128 if xmx => Attention::Tiled(c.kernel(&format!("attention_{head_dim}_to_half"), [QUERIES, 1, 1])?),
            32 | 64 | 128 => Attention::Tiled(c.kernel(&format!("attention_{head_dim}"), [QUERIES, 1, 1])?),
            _ => Attention::General(c.kernel("attention", row)?),
        };
        let wide = wide(hidden);
        let dpas = if wide { (WIDE_TM, WIDE_TN, WIDE_WM, WIDE_WN) } else { (DPAS_TM, DPAS_TN, DPAS_WM, DPAS_WN) };
        let (ln_tm, ln_tn, ln_name) = if wide {
            (WIDE_LN_TM, WIDE_TN, "linear_dpas_layer_norm_32x64")
        } else {
            (DPAS_LN_TM, DPAS_TN, "linear_dpas_layer_norm")
        };
        let mut ln_blocks = 1;
        Ok(Kernels {
            linear: c.kernel("linear", [16, 16, 1])?,
            linear_sg: c.kernel("linear_sg", [16, 1, 1])?,
            linear_gemv: c.kernel("linear_gemv", [16 * GEMV_SUBGROUPS, 1, 1])?,
            linear_sgemm: c.kernel("linear_sgemm", [16 * SGEMM_WM * SGEMM_WN, 1, 1])?,
            linear_dpas_layer_norm: if xmx && hidden.is_multiple_of(ln_tn) && hidden / ln_tn <= DPAS_LN_SUBGROUPS {
                let cols = hidden / ln_tn;
                let one = c.kernel(ln_name, [16 * cols, 1, 1])?;
                ln_blocks = dpas_ln_blocks(cols, one.max_group_size()?.map(|m| m / (16 * cols)));
                Some([one, c.kernel(ln_name, [16 * cols * ln_blocks, 1, 1])?])
            } else {
                None
            },
            ln: (ln_tm, ln_blocks),
            linear_dpas_mlp: if xmx && !wide { mlp_kernel(c, hidden, intermediate)? } else { None },
            dpas,
            linear_dpas: if xmx {
                let (main, few) = ([16 * dpas.2 * dpas.3, 1, 1], [16 * FEW_WM * FEW_WN, 1, 1]);
                let name = if wide { "linear_dpas_32x64" } else { "linear_dpas" };
                Some([
                    c.kernel(name, main)?,
                    c.kernel(&format!("{name}_to_half"), main)?,
                    c.kernel("linear_dpas_few", few)?,
                    c.kernel("linear_dpas_few_to_half", few)?,
                ])
            } else {
                None
            },
            embed_layer_norm: c.kernel("embed_layer_norm", [16 * ROWS, 1, 1])?,
            widen_f16: c.kernel("widen_f16", [WIDE, 1, 1])?,
            add_layer_norm: c.kernel("add_layer_norm", [16 * ROWS, 1, 1])?,
            add_layer_norm_v4: if hidden.is_multiple_of(64) && hidden <= 1024 {
                Some(c.kernel("add_layer_norm_v4", [16 * ROWS, 1, 1])?)
            } else {
                None
            },
            embed_layer_norm_group: c.kernel("embed_layer_norm_group", row)?,
            add_layer_norm_group: c.kernel("add_layer_norm_group", row)?,
            attention,
            pool: c.kernel("pool", [16 * POOL_SLICES, 1, 1])?,
            pool_ranges: c.kernel("pool_ranges", [16 * POOL_SLICES, 1, 1])?,
            normalize: c.kernel("normalize", row)?,
        })
    }
}
