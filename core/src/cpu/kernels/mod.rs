//! The encoder's arithmetic: the matrix product of the linear layers over
//! weights packed once per model, and the row-wise functions around it
//! (GELU, LayerNorm, attention within one row and head).
//!
//! Every function here computes each output from its inputs in one fixed
//! order, whatever part of the output a call is asked for: that is what
//! lets the encoder split a run over threads and still give the same bits
//! for the same input on any number of them. In particular no sum over the
//! reduction dimension of a product is split: each output of a linear
//! layer is one chain of multiply-adds over k = 0, 1, .. n_in - 1, then
//! its bias. Attention is written once over sixteen lanes (Simd) and keeps
//! the same rule: each score and each context value is one chain in a
//! fixed order. The AVX-512 and AVX2 kernels, chosen at run time, fuse
//! each multiply-add, so they give the same bits as each other. The
//! portable kernel fuses only where it is compiled for aarch64, where
//! every processor has FMA: a compile-time choice, not a detected one; on
//! x86_64 it multiplies and adds in two roundings.

mod attend;
mod dense;
mod gemm;
mod micro;
mod rowwise;
mod simd;

pub(super) use attend::{Attend, LANES};
pub(crate) use dense::softmax;
pub use dense::{Matrix, Mode};
pub(super) use gemm::{Epilogue, Gemm};
#[cfg(target_arch = "x86_64")]
pub(super) use micro::{Avx2, Avx512};
pub(super) use micro::{Micro, Portable};
pub(crate) use rowwise::gelu;
pub(super) use rowwise::layer_norm;

use crate::backend::turbo_backend_model;

// ---- Instruction sets -------------------------------------------------------

/// The instruction set a model's weights are packed for and its sessions
/// compute with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Isa {
    /// Plain Rust, which the compiler vectorizes as far as the baseline
    /// target lets it; FMA where every processor of the target has it.
    Portable,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Isa {
    /// The widest this processor runs.
    pub(super) fn detect() -> Isa {
        #[cfg(target_arch = "x86_64")]
        {
            if Isa::Avx512.available() {
                return Isa::Avx512;
            }
            if Isa::Avx2.available() {
                return Isa::Avx2;
            }
        }
        Isa::Portable
    }

    /// Whether this processor runs the instruction set. Only x86_64 has
    /// more than one to choose from; elsewhere the tests alone ask.
    #[cfg(any(target_arch = "x86_64", test))]
    pub(super) fn available(self) -> bool {
        match self {
            Isa::Portable => true,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => {
                Isa::Avx2.available()
                    && std::arch::is_x86_feature_detected!("avx512f")
                    && std::arch::is_x86_feature_detected!("avx512vl")
            }
        }
    }

    /// The micro-kernel's rows and columns: weights are packed in panels
    /// of its columns.
    fn tile(self) -> (usize, usize) {
        match self {
            Isa::Portable => (Portable::MR, Portable::NR),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => (Avx2::MR, Avx2::NR),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => (Avx512::MR, Avx512::NR),
        }
    }
}

// ---- Packed weights ---------------------------------------------------------

/// A weight matrix W [n, k] (y = x W^T) laid out for the micro-kernel:
/// panels of NR output columns, each [k][NR] contiguous, so a step of k
/// reads NR weights in one run. Columns past n in the last panel are 0.
pub(super) struct Panels {
    buf: Vec<f32>,
    /// Floats from buf's start to its first 64-byte boundary.
    off: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    mr: usize,
    nr: usize,
}

/// Zeroed memory, or the bytes that could not be had.
pub(super) fn zeroed<T: Clone + Default>(n: usize) -> Result<Vec<T>, usize> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| n.saturating_mul(size_of::<T>()))?;
    v.resize(n, T::default());
    Ok(v)
}

impl Panels {
    /// The rows of `parts`, each [n_i, k] row-major, one after another as
    /// one matrix of sum n_i rows. Err is the bytes it could not allocate.
    pub(super) fn pack(parts: &[&[f32]], k: usize, isa: Isa) -> Result<Panels, usize> {
        let (mr, nr) = isa.tile();
        let n: usize = parts.iter().map(|p| p.len() / k).sum();
        let floats = n.div_ceil(nr).checked_mul(k * nr).ok_or(usize::MAX)?;
        let mut buf = zeroed::<f32>(floats + 16)?;
        let off = buf.as_ptr().align_offset(64).min(16);
        let dst = &mut buf[off..off + floats];
        let mut o = 0;
        for part in parts {
            for row in part.chunks_exact(k) {
                let (p, j) = (o / nr, o % nr);
                for (kk, &w) in row.iter().enumerate() {
                    dst[(p * k + kk) * nr + j] = w;
                }
                o += 1;
            }
        }
        Ok(Panels { buf, off, n, k, mr, nr })
    }

    #[inline]
    fn panel(&self, p: usize) -> *const f32 {
        self.buf[self.off + p * self.k * self.nr..].as_ptr()
    }
}

/// A BERT's linear layers packed for one instruction set: the loaded
/// model's, shared by its sessions.
pub(super) struct Packed {
    pub(super) isa: Isa,
    pub(super) layers: Vec<PackedLayer>,
}

pub(super) struct PackedLayer {
    /// Q, K and V as one matrix of 3 * hidden columns, and their biases.
    pub(super) qkv: Panels,
    pub(super) qkv_bias: Vec<f32>,
    pub(super) out: Panels,
    pub(super) ffn_in: Panels,
    pub(super) ffn_out: Panels,
}

impl Packed {
    /// `layer(l, r)` is layer l's tensor r in TURBO_BERT_* order.
    pub(super) fn new<'a>(
        d: &turbo_backend_model,
        layer: impl Fn(usize, usize) -> &'a [f32],
        isa: Isa,
    ) -> Result<Packed, usize> {
        use super::encoder::{FFN_IN_W, FFN_OUT_W, K_B, K_W, O_W, Q_B, Q_W, V_B, V_W};
        let (h, i) = (d.hidden as usize, d.intermediate as usize);
        let mut layers = Vec::new();
        layers.try_reserve_exact(d.layers as usize).map_err(|_| usize::MAX)?;
        for l in 0..d.layers as usize {
            let w = |r| layer(l, r);
            let mut qkv_bias = zeroed(0)?;
            qkv_bias.try_reserve_exact(3 * h).map_err(|_| 12 * h)?;
            qkv_bias.extend([w(Q_B), w(K_B), w(V_B)].iter().flat_map(|b| b[..h].iter()));
            layers.push(PackedLayer {
                qkv: Panels::pack(&[w(Q_W), w(K_W), w(V_W)], h, isa)?,
                qkv_bias,
                out: Panels::pack(&[w(O_W)], h, isa)?,
                ffn_in: Panels::pack(&[w(FFN_IN_W)], h, isa)?,
                ffn_out: Panels::pack(&[w(FFN_OUT_W)], i, isa)?,
            });
        }
        Ok(Packed { isa, layers })
    }
}

/// Every instruction set this processor has.
#[cfg(test)]
pub(super) fn isas() -> Vec<Isa> {
    #[cfg(target_arch = "x86_64")]
    let all = vec![Isa::Portable, Isa::Avx2, Isa::Avx512];
    #[cfg(not(target_arch = "x86_64"))]
    let all = vec![Isa::Portable];
    all.into_iter().filter(|i| i.available()).collect()
}

/// Test values in [-scale, scale], spread by `seed`.
#[cfg(test)]
pub(super) fn values(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (((i * 7919 + seed * 104_729) % 2001) as f32 / 1000.0 - 1.0) * scale).collect()
}
