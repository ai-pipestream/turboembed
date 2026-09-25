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

// ---- Micro-kernels ----------------------------------------------------------

/// The most accumulators a micro-kernel tile holds: MR x NR.
pub(super) const ACC: usize = 8 * 32;

/// A register tile of the product: `rows` (at most MR) input rows by one
/// panel of NR columns, each a chain of FMAs over k from 0.
pub(super) trait Micro {
    const MR: usize;
    const NR: usize;
    /// Whether madd fuses.
    const FMA: bool;
    /// Sixteen lanes of the same instruction set, for attention.
    type V: Simd;

    /// acc[i * NR + j] = sum over kk of x[i * ldx + kk] * p[kk * NR + j],
    /// for i under rows.
    ///
    /// # Safety
    /// x holds rows rows of k floats at stride ldx, p holds k * NR floats,
    /// and the processor has the kernel's instruction set.
    unsafe fn tile(rows: usize, x: *const f32, ldx: usize, p: *const f32, k: usize, acc: &mut [f32; ACC]);
}

#[inline(always)]
pub(super) fn madd(fma: bool, a: f32, b: f32, c: f32) -> f32 {
    if fma { a.mul_add(b, c) } else { a * b + c }
}

pub(super) struct Portable;

impl Micro for Portable {
    const MR: usize = 4;
    const NR: usize = 8;
    const FMA: bool = cfg!(target_arch = "aarch64");
    type V = P16;

    #[inline(always)]
    unsafe fn tile(rows: usize, x: *const f32, ldx: usize, p: *const f32, k: usize, acc: &mut [f32; ACC]) {
        // SAFETY: as the trait says, for each arm.
        unsafe {
            match rows {
                1 => portable_tile::<1>(x, ldx, p, k, acc),
                2 => portable_tile::<2>(x, ldx, p, k, acc),
                3 => portable_tile::<3>(x, ldx, p, k, acc),
                _ => portable_tile::<4>(x, ldx, p, k, acc),
            }
        }
    }
}

#[inline(always)]
unsafe fn portable_tile<const R: usize>(x: *const f32, ldx: usize, p: *const f32, k: usize, acc: &mut [f32; ACC]) {
    const NR: usize = Portable::NR;
    let mut c = [[0.0f32; NR]; R];
    for kk in 0..k {
        // SAFETY: kk < k, so the panel has these NR floats, and each row
        // under R has float kk.
        let b: [f32; NR] = unsafe { *(p.add(kk * NR) as *const [f32; NR]) };
        for (i, ci) in c.iter_mut().enumerate() {
            let a = unsafe { *x.add(i * ldx + kk) };
            for j in 0..NR {
                ci[j] = madd(Portable::FMA, a, b[j], ci[j]);
            }
        }
    }
    for (i, ci) in c.iter().enumerate() {
        acc[i * NR..][..NR].copy_from_slice(ci);
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) struct Avx2;

#[cfg(target_arch = "x86_64")]
impl Micro for Avx2 {
    const MR: usize = 6;
    const NR: usize = 16;
    const FMA: bool = true;
    type V = Y16;

    #[inline(always)]
    unsafe fn tile(rows: usize, x: *const f32, ldx: usize, p: *const f32, k: usize, acc: &mut [f32; ACC]) {
        // SAFETY: as the trait says, for each arm.
        unsafe {
            match rows {
                1 => avx2_tile::<1>(x, ldx, p, k, acc),
                2 => avx2_tile::<2>(x, ldx, p, k, acc),
                3 => avx2_tile::<3>(x, ldx, p, k, acc),
                4 => avx2_tile::<4>(x, ldx, p, k, acc),
                5 => avx2_tile::<5>(x, ldx, p, k, acc),
                _ => avx2_tile::<6>(x, ldx, p, k, acc),
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn avx2_tile<const R: usize>(x: *const f32, ldx: usize, p: *const f32, k: usize, acc: &mut [f32; ACC]) {
    use std::arch::x86_64::*;
    let mut c0 = [_mm256_setzero_ps(); R];
    let mut c1 = [_mm256_setzero_ps(); R];
    for kk in 0..k {
        // SAFETY: kk < k: the panel has 16 floats at kk * 16, and each
        // row under R has float kk.
        let (b0, b1) = unsafe { (_mm256_loadu_ps(p.add(kk * 16)), _mm256_loadu_ps(p.add(kk * 16 + 8))) };
        for i in 0..R {
            let a = _mm256_set1_ps(unsafe { *x.add(i * ldx + kk) });
            c0[i] = _mm256_fmadd_ps(a, b0, c0[i]);
            c1[i] = _mm256_fmadd_ps(a, b1, c1[i]);
        }
    }
    for i in 0..R {
        // SAFETY: R * 16 <= ACC.
        unsafe {
            _mm256_storeu_ps(acc.as_mut_ptr().add(i * 16), c0[i]);
            _mm256_storeu_ps(acc.as_mut_ptr().add(i * 16 + 8), c1[i]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) struct Avx512;

#[cfg(target_arch = "x86_64")]
impl Micro for Avx512 {
    const MR: usize = 8;
    const NR: usize = 32;
    const FMA: bool = true;
    type V = Z16;

    #[inline(always)]
    unsafe fn tile(rows: usize, x: *const f32, ldx: usize, p: *const f32, k: usize, acc: &mut [f32; ACC]) {
        // SAFETY: as the trait says, for each arm.
        unsafe {
            match rows {
                1 => avx512_tile::<1>(x, ldx, p, k, acc),
                2 => avx512_tile::<2>(x, ldx, p, k, acc),
                3 => avx512_tile::<3>(x, ldx, p, k, acc),
                4 => avx512_tile::<4>(x, ldx, p, k, acc),
                5 => avx512_tile::<5>(x, ldx, p, k, acc),
                6 => avx512_tile::<6>(x, ldx, p, k, acc),
                7 => avx512_tile::<7>(x, ldx, p, k, acc),
                _ => avx512_tile::<8>(x, ldx, p, k, acc),
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn avx512_tile<const R: usize>(x: *const f32, ldx: usize, p: *const f32, k: usize, acc: &mut [f32; ACC]) {
    use std::arch::x86_64::*;
    let mut c0 = [_mm512_setzero_ps(); R];
    let mut c1 = [_mm512_setzero_ps(); R];
    for kk in 0..k {
        // SAFETY: kk < k: the panel has 32 floats at kk * 32, and each
        // row under R has float kk.
        let (b0, b1) = unsafe { (_mm512_loadu_ps(p.add(kk * 32)), _mm512_loadu_ps(p.add(kk * 32 + 16))) };
        for i in 0..R {
            let a = _mm512_set1_ps(unsafe { *x.add(i * ldx + kk) });
            c0[i] = _mm512_fmadd_ps(a, b0, c0[i]);
            c1[i] = _mm512_fmadd_ps(a, b1, c1[i]);
        }
    }
    for i in 0..R {
        // SAFETY: R * 32 <= ACC.
        unsafe {
            _mm512_storeu_ps(acc.as_mut_ptr().add(i * 32), c0[i]);
            _mm512_storeu_ps(acc.as_mut_ptr().add(i * 32 + 16), c1[i]);
        }
    }
}

// ---- Sixteen lanes -----------------------------------------------------------

/// Sixteen F32 lanes, in the registers of one instruction set: what the
/// attention kernel is written in, so the compiler need not find the
/// vectors itself. Every operation is lane by lane, and madd fuses exactly
/// where the instruction set's matrix kernel does.
///
/// # Safety
/// Every method may be called only on a processor with the instruction
/// set, and load and store only where 16 floats may be read or written.
pub(super) trait Simd: Copy {
    /// Queries attention takes at once: as many as keep its sums in the
    /// instruction set's registers.
    const GROUP: usize;

    unsafe fn zero() -> Self;
    unsafe fn splat(x: f32) -> Self;
    unsafe fn load(p: *const f32) -> Self;
    unsafe fn store(self, p: *mut f32);
    /// self * b + c
    unsafe fn madd(self, b: Self, c: Self) -> Self;
    unsafe fn add(self, b: Self) -> Self;
    unsafe fn mul(self, b: Self) -> Self;
    /// The larger, lane by lane, of values that are not NaN.
    unsafe fn max(self, b: Self) -> Self;
}

#[derive(Clone, Copy)]
pub(super) struct P16([f32; 16]);

impl Simd for P16 {
    const GROUP: usize = 4;
    #[inline(always)]
    unsafe fn zero() -> Self {
        P16([0.0; 16])
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        P16([x; 16])
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        // SAFETY: the trait's contract.
        P16(unsafe { *(p as *const [f32; 16]) })
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        // SAFETY: the trait's contract.
        unsafe { *(p as *mut [f32; 16]) = self.0 }
    }
    #[inline(always)]
    unsafe fn madd(self, b: Self, c: Self) -> Self {
        P16(std::array::from_fn(|l| madd(Portable::FMA, self.0[l], b.0[l], c.0[l])))
    }
    #[inline(always)]
    unsafe fn add(self, b: Self) -> Self {
        P16(std::array::from_fn(|l| self.0[l] + b.0[l]))
    }
    #[inline(always)]
    unsafe fn mul(self, b: Self) -> Self {
        P16(std::array::from_fn(|l| self.0[l] * b.0[l]))
    }
    #[inline(always)]
    unsafe fn max(self, b: Self) -> Self {
        P16(std::array::from_fn(|l| if b.0[l] > self.0[l] { b.0[l] } else { self.0[l] }))
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(super) struct Y16(std::arch::x86_64::__m256, std::arch::x86_64::__m256);

// SAFETY, for every method: the trait's contract; the intrinsics need AVX2
// and FMA, which the processor has.
#[cfg(target_arch = "x86_64")]
impl Simd for Y16 {
    const GROUP: usize = 2;
    #[inline(always)]
    unsafe fn zero() -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_setzero_ps(), _mm256_setzero_ps()) }
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_set1_ps(x), _mm256_set1_ps(x)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8))) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        use std::arch::x86_64::*;
        unsafe {
            _mm256_storeu_ps(p, self.0);
            _mm256_storeu_ps(p.add(8), self.1);
        }
    }
    #[inline(always)]
    unsafe fn madd(self, b: Self, c: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_fmadd_ps(self.0, b.0, c.0), _mm256_fmadd_ps(self.1, b.1, c.1)) }
    }
    #[inline(always)]
    unsafe fn add(self, b: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_add_ps(self.0, b.0), _mm256_add_ps(self.1, b.1)) }
    }
    #[inline(always)]
    unsafe fn mul(self, b: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_mul_ps(self.0, b.0), _mm256_mul_ps(self.1, b.1)) }
    }
    #[inline(always)]
    unsafe fn max(self, b: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Y16(_mm256_max_ps(self.0, b.0), _mm256_max_ps(self.1, b.1)) }
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
pub(super) struct Z16(std::arch::x86_64::__m512);

// SAFETY, for every method: the trait's contract; the intrinsics need
// AVX-512F, which the processor has.
#[cfg(target_arch = "x86_64")]
impl Simd for Z16 {
    const GROUP: usize = 4;
    #[inline(always)]
    unsafe fn zero() -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_setzero_ps()) }
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_set1_ps(x)) }
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_loadu_ps(p)) }
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        use std::arch::x86_64::*;
        unsafe { _mm512_storeu_ps(p, self.0) }
    }
    #[inline(always)]
    unsafe fn madd(self, b: Self, c: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_fmadd_ps(self.0, b.0, c.0)) }
    }
    #[inline(always)]
    unsafe fn add(self, b: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_add_ps(self.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn mul(self, b: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_mul_ps(self.0, b.0)) }
    }
    #[inline(always)]
    unsafe fn max(self, b: Self) -> Self {
        use std::arch::x86_64::*;
        unsafe { Z16(_mm512_max_ps(self.0, b.0)) }
    }
}

// ---- Linear layers ----------------------------------------------------------

/// What is done with each output of a product once its sum is complete.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Epilogue {
    /// y = gelu(sum + b)
    Gelu,
    /// y = y + (sum + b): the residual, in place.
    Residual,
    /// y = sum + b, but each column c in lo..hi goes transposed to
    /// t_out[(c - lo) * stride + slot[t]] instead. So K lands a column at
    /// a time, each row's tokens one after another from its own slot, as
    /// attention reads it.
    Split { lo: usize, hi: usize, t_out: *mut f32, stride: usize, slot: *const u32 },
}

/// y[t, o] = epilogue(sum_i x[t, i] w[o, i], b[o]) for t under rows, split
/// into tiles of MB rows by NB columns, one tile per task.
#[derive(Clone, Copy)]
pub(super) struct Gemm<'a> {
    pub(super) x: *const f32,
    pub(super) ldx: usize,
    pub(super) rows: usize,
    pub(super) w: &'a Panels,
    pub(super) bias: &'a [f32],
    pub(super) y: *mut f32,
    pub(super) ldy: usize,
    pub(super) epilogue: Epilogue,
}

/// Micro-kernel tiles of rows per task.
const MB_TILES: usize = 4;
/// Panels per task.
const NB_PANELS: usize = 2;

impl Gemm<'_> {
    pub(super) fn tasks(&self) -> usize {
        let (mb, nb) = self.block();
        self.rows.div_ceil(mb) * self.w.n.div_ceil(nb)
    }

    fn block(&self) -> (usize, usize) {
        (self.w.mr * MB_TILES, self.w.nr * NB_PANELS)
    }

    /// One tile.
    ///
    /// # Safety
    /// x holds rows x ldx floats and y rows x ldy; w was packed for M;
    /// no other thread touches this task's tile of y while it runs; the
    /// processor has M's instruction set.
    #[inline(always)]
    pub(super) unsafe fn task<M: Micro>(&self, task: usize) {
        debug_assert_eq!((self.w.mr, self.w.nr), (M::MR, M::NR));
        let (mb, nb) = self.block();
        let ncb = self.w.n.div_ceil(nb);
        let (rb, cb) = (task / ncb, task % ncb);
        let (r_end, c_end) = (((rb + 1) * mb).min(self.rows), ((cb + 1) * nb).min(self.w.n));
        let mut acc = [0.0f32; ACC];
        let mut col = cb * nb;
        while col < c_end {
            let p = self.w.panel(col / M::NR);
            let cols = (c_end - col).min(M::NR);
            let mut r = rb * mb;
            while r < r_end {
                let rows = (r_end - r).min(M::MR);
                // SAFETY: rows rows of x from r, the panel of k * NR.
                unsafe { M::tile(rows, self.x.add(r * self.ldx), self.ldx, p, self.w.k, &mut acc) };
                let b = &self.bias[col..][..cols];
                match self.epilogue {
                    Epilogue::Split { lo, hi, t_out, stride, slot } if col < hi && col + cols > lo => {
                        // A column at a time, so each column's tokens go
                        // to consecutive slots.
                        for j in 0..cols {
                            let c = col + j;
                            // SAFETY: tokens r..r + rows and their slots;
                            // column c of y or of t_out, in this task's
                            // tile.
                            unsafe {
                                for i in 0..rows {
                                    let v = acc[i * M::NR + j] + b[j];
                                    if (lo..hi).contains(&c) {
                                        *t_out.add((c - lo) * stride + *slot.add(r + i) as usize) = v;
                                    } else {
                                        *self.y.add((r + i) * self.ldy + c) = v;
                                    }
                                }
                            }
                        }
                    }
                    epilogue => {
                        for i in 0..rows {
                            // SAFETY: row r + i under rows, columns col..col
                            // + cols under n, in this task's tile.
                            let y =
                                unsafe { std::slice::from_raw_parts_mut(self.y.add((r + i) * self.ldy + col), cols) };
                            let a = &acc[i * M::NR..][..cols];
                            match epilogue {
                                Epilogue::Gelu => {
                                    for j in 0..cols {
                                        y[j] = gelu(a[j] + b[j]);
                                    }
                                }
                                Epilogue::Residual => {
                                    for j in 0..cols {
                                        y[j] += a[j] + b[j];
                                    }
                                }
                                // The columns of a panel outside lo..hi.
                                Epilogue::Split { .. } => {
                                    for j in 0..cols {
                                        y[j] = a[j] + b[j];
                                    }
                                }
                            }
                        }
                    }
                }
                r += rows;
            }
            col += cols;
        }
    }
}

// ---- Row-wise functions -----------------------------------------------------

/// The sum of eight lanes, in one fixed order.
#[inline(always)]
fn lanes8(s: [f64; 8]) -> f64 {
    ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]))
}

/// y = (y - mean) / sqrt(var + eps) * w + b, with the mean and the biased
/// variance summed in F64, in eight lanes: element i into lane i % 8.
#[inline(always)]
pub(super) fn layer_norm(y: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = y.len() as f64;
    let mut s = [0.0f64; 8];
    for (i, &v) in y.iter().enumerate() {
        s[i % 8] += v as f64;
    }
    let mean = lanes8(s) / n;
    let mut s = [0.0f64; 8];
    for (i, &v) in y.iter().enumerate() {
        let c = v as f64 - mean;
        s[i % 8] += c * c;
    }
    let var = lanes8(s) / n;
    let inv = 1.0 / (var + eps as f64).sqrt();
    for ((y, &w), &b) in y.iter_mut().zip(w).zip(b) {
        *y = ((*y as f64 - mean) * inv) as f32 * w + b;
    }
}

/// e^x in F32 to about an ulp, with no branch, so a loop of it vectorizes:
/// Cephes' expf. Under -80 it is 0. e^-80 is 1.8e-35: its callers only
/// weigh it against terms near 1 (a softmax's largest, 2 - erfc), and a
/// value that reached the subnormal range would send every later multiply
/// of it down the processor's slow path.
#[inline(always)]
pub(super) fn exp(x: f32) -> f32 {
    const LO: f32 = -80.0;
    let c = x.clamp(LO, 88.0);
    let n = (c * std::f32::consts::LOG2_E + 0.5).floor();
    let r = c - n * 0.693_359_4;
    let r = r - n * -2.121_944_4e-4;
    let r2 = r * r;
    let p =
        ((((1.987_569_1e-4 * r + 1.398_2e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r + 0.166_666_65) * r + 0.5;
    let e = p * r2 + r + 1.0;
    // 2^n from n's bits: adding 1.5 * 2^23 puts the integer n in the low
    // bits of the sum exactly (|n| < 2^22), with no conversion the
    // compiler would take out of the vector registers.
    let n_int = (n + 12_582_912.0).to_bits().wrapping_sub(0x4b40_0000);
    let scale = f32::from_bits(n_int.wrapping_add(127) << 23);
    if x < LO { 0.0 } else { e * scale }
}

/// GELU with the error function, as upstream's "gelu": 0.5 x (1 + erf(x /
/// sqrt 2)) = 0.5 x (2 - erfc(x / sqrt 2)), with erfc(z) = t exp(-z^2 +
/// P(t)) by Numerical Recipes' erfcc Chebyshev fit, whose fractional error
/// is under 1.2e-7 everywhere. In F32, but for its one sensitive part: the
/// exponent's -z^2 = -x^2 / 2, which moves erfc by 2z^2 times its own
/// relative error, is carried exactly as a sum of two floats into the
/// exponential. For x < 0 it is 0.5 x erfc(|z|), with no cancellation.
/// Its relative error, fit and rounding together, is under 5e-7 (tested).
#[inline(always)]
pub(super) fn gelu(x: f32) -> f32 {
    // Past 16, erfc(|z|) is far under what exp keeps: the clamp keeps x^2
    // finite and changes nothing.
    let xc = x.clamp(-16.0, 16.0);
    let z = xc.abs() * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.5 * z);
    let p = -1.265_512_2
        + t * (1.000_023_7
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_204 + t * (1.488_515_9 + t * (-0.822_152_2 + t * 0.170_872_77))))))));
    // -x^2 / 2 = hi + lo exactly, and hi + p = a + e exactly.
    let (sq, sq_lo) = two_product(xc, xc);
    let (hi, lo) = (-0.5 * sq, -0.5 * sq_lo);
    let (a, e) = two_sum(hi, p);
    let r = t * (exp(a) * (1.0 + (e + lo)));
    0.5 * x * if x >= 0.0 { 2.0 - r } else { r }
}

/// a * b = p + e exactly (Dekker, with Veltkamp's split), in plain
/// multiplies and adds: the same bits with or without FMA.
#[inline(always)]
fn two_product(a: f32, b: f32) -> (f32, f32) {
    let split = |v: f32| {
        let c = 4097.0 * v;
        let hi = c - (c - v);
        (hi, v - hi)
    };
    let p = a * b;
    let ((ah, al), (bh, bl)) = (split(a), split(b));
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

/// a + b = s + e exactly (Knuth).
#[inline(always)]
fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// The sum of LANES lanes, in one fixed order: halves added pairwise.
#[inline(always)]
fn tree(mut s: [f32; LANES]) -> f32 {
    let mut w = LANES / 2;
    while w > 0 {
        for l in 0..w {
            s[l] += s[l + w];
        }
        w /= 2;
    }
    s[0]
}

/// Lanes of the attention's vector loops.
pub(super) const LANES: usize = 16;

/// Softmax attention of queries q0..q1 of one row's tokens over its keys,
/// in one head.
pub(super) struct Attend<'a> {
    /// The row's queries and values, [len, ld], the head's d columns of
    /// each from q_at and v_at.
    pub(super) qv: *const f32,
    pub(super) ld: usize,
    pub(super) q_at: usize,
    pub(super) v_at: usize,
    /// The head's keys transposed: column c of key j at c * stride + j,
    /// for j under len rounded up to LANES.
    pub(super) kt: *const f32,
    pub(super) stride: usize,
    pub(super) d: usize,
    pub(super) len: usize,
    pub(super) mask: &'a [u8],
    pub(super) scale: f32,
    /// Whether the plain Rust fallback fuses multiply and add: as V does.
    pub(super) fma: bool,
    /// [len, ldc], the head's columns from c_at.
    pub(super) ctx: *mut f32,
    pub(super) ldc: usize,
    pub(super) c_at: usize,
}

impl Attend<'_> {
    /// Per-thread floats the scratch needs for rows of up to `seq` in
    /// heads of `d`.
    pub(super) fn scratch(seq: usize, d: usize) -> usize {
        let lp = seq.next_multiple_of(LANES);
        4 * lp + lp + 4 * d
    }

    /// Queries are taken a group at a time, which shares each key and
    /// value load between them; a query's results do not depend on its
    /// group. Each score is a chain of FMAs over the head's d columns, in
    /// order. The softmax's sum is taken in LANES lanes, key j into lane
    /// j % LANES, then added in a fixed tree. Each context value is a
    /// chain of FMAs over the keys, in order.
    ///
    /// # Safety
    /// qv and ctx hold len rows at ld and ldc, kt d rows at stride of len
    /// rounded up to LANES finite values; no other thread touches ctx's
    /// rows q0..q1 in this head's columns; `scratch` has Attend::scratch
    /// floats. The processor has V's instruction set.
    #[inline(always)]
    pub(super) unsafe fn run<V: Simd>(&self, q0: usize, q1: usize, scratch: &mut [f32]) {
        let (len, lp) = (self.len, self.len.next_multiple_of(LANES));
        let (sc, rest) = scratch.split_at_mut(4 * lp);
        let (masked, acc) = rest.split_at_mut(lp);
        // What each key's score is offset by: 0, or minus infinity for a
        // key that is masked or past len, which leaves it out of the max
        // and makes its exponential 0.
        for (j, m) in masked[..lp].iter_mut().enumerate() {
            *m = if j < len && self.mask[j] != 0 { 0.0 } else { f32::NEG_INFINITY };
        }
        let mut i = q0;
        // SAFETY: as run's contract.
        unsafe {
            while i < q1 {
                let g = (q1 - i).min(V::GROUP);
                match g {
                    4 => self.group::<V, 4>(i, sc, masked, acc),
                    3 => self.group::<V, 3>(i, sc, masked, acc),
                    2 => self.group::<V, 2>(i, sc, masked, acc),
                    _ => self.group::<V, 1>(i, sc, masked, acc),
                }
                i += g;
            }
        }
    }

    /// Queries i..i + G.
    ///
    /// # Safety
    /// As run.
    #[inline(always)]
    unsafe fn group<V: Simd, const G: usize>(&self, i: usize, sc: &mut [f32], masked: &[f32], acc: &mut [f32]) {
        let (d, lp) = (self.d, self.len.next_multiple_of(LANES));
        let (sp, mp) = (sc.as_mut_ptr(), masked.as_ptr());
        // SAFETY: every pointer below is within sc (G rows of lp),
        // masked, the head's columns of qv's rows under len, kt's d rows
        // of lp, or the head's columns of ctx's rows i..i + G; the
        // processor has V's instruction set.
        unsafe {
            let q: [*const f32; G] = std::array::from_fn(|g| self.qv.add((i + g) * self.ld + self.q_at));
            let mut jb = 0;
            while jb + 2 * LANES <= lp {
                self.scores::<V, G, 2>(q, jb, sp, lp, mp);
                jb += 2 * LANES;
            }
            if jb < lp {
                self.scores::<V, G, 1>(q, jb, sp, lp, mp);
            }
            for g in 0..G {
                let row = &mut sc[g * lp..][..lp];
                let rp = row.as_mut_ptr();
                let mut m = V::splat(f32::NEG_INFINITY);
                for jb in (0..lp).step_by(LANES) {
                    m = m.max(V::load(rp.add(jb)));
                }
                let mut lanes = [0.0f32; LANES];
                m.store(lanes.as_mut_ptr());
                let max = lanes.iter().fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m });
                for s in row.iter_mut() {
                    *s = exp(*s - max);
                }
                let mut sum = V::zero();
                for jb in (0..lp).step_by(LANES) {
                    sum = sum.add(V::load(rp.add(jb)));
                }
                sum.store(lanes.as_mut_ptr());
                let inv = V::splat(1.0 / tree(lanes));
                for jb in (0..lp).step_by(LANES) {
                    V::load(rp.add(jb)).mul(inv).store(rp.add(jb));
                }
            }
            let out: [*mut f32; G] = std::array::from_fn(|g| self.ctx.add((i + g) * self.ldc + self.c_at));
            if d % LANES != 0 {
                self.context_any::<G>(sc, lp, out, acc);
                return;
            }
            let mut col = 0;
            while col < d {
                match (d - col) / LANES {
                    1 => self.context::<V, G, 1>(sp, lp, col, out),
                    _ => self.context::<V, G, 2>(sp, lp, col, out),
                }
                col += 2 * LANES;
            }
        }
    }

    /// Scores of G queries for the N * LANES keys from jb: sc[g * lp + j]
    /// = q_g . k_j * scale + masked[j].
    ///
    /// # Safety
    /// As run.
    // Register tiles, indexed as the sums are laid out.
    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    unsafe fn scores<V: Simd, const G: usize, const N: usize>(
        &self,
        q: [*const f32; G],
        jb: usize,
        sc: *mut f32,
        lp: usize,
        masked: *const f32,
    ) {
        // SAFETY: as run.
        unsafe {
            let mut acc = [[V::zero(); N]; G];
            for c in 0..self.d {
                let kc = self.kt.add(c * self.stride + jb);
                let k: [V; N] = std::array::from_fn(|n| V::load(kc.add(n * LANES)));
                for g in 0..G {
                    let a = V::splat(*q[g].add(c));
                    for n in 0..N {
                        acc[g][n] = a.madd(k[n], acc[g][n]);
                    }
                }
            }
            let scale = V::splat(self.scale);
            for g in 0..G {
                for n in 0..N {
                    let at = jb + n * LANES;
                    acc[g][n].mul(scale).add(V::load(masked.add(at))).store(sc.add(g * lp + at));
                }
            }
        }
    }

    /// Context of G queries, columns col..col + N * LANES of the head:
    /// out_g = sum over keys of p_gj v_j.
    ///
    /// # Safety
    /// As run.
    // Register tiles, indexed as the sums are laid out.
    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    unsafe fn context<V: Simd, const G: usize, const N: usize>(
        &self,
        p: *const f32,
        lp: usize,
        col: usize,
        out: [*mut f32; G],
    ) {
        // SAFETY: as run.
        unsafe {
            let mut acc = [[V::zero(); N]; G];
            for j in 0..self.len {
                let vj = self.qv.add(j * self.ld + self.v_at + col);
                let v: [V; N] = std::array::from_fn(|n| V::load(vj.add(n * LANES)));
                for g in 0..G {
                    let pj = V::splat(*p.add(g * lp + j));
                    for n in 0..N {
                        acc[g][n] = pj.madd(v[n], acc[g][n]);
                    }
                }
            }
            for g in 0..G {
                for n in 0..N {
                    acc[g][n].store(out[g].add(col + n * LANES));
                }
            }
        }
    }

    /// As context, for a head whose width is not a multiple of LANES, in
    /// plain Rust with its sums in `acc`: the same sums in the same
    /// order, with FMA where V has it.
    ///
    /// # Safety
    /// As run.
    #[inline(always)]
    unsafe fn context_any<const G: usize>(&self, p: &[f32], lp: usize, out: [*mut f32; G], acc: &mut [f32]) {
        let d = self.d;
        acc[..G * d].fill(0.0);
        for j in 0..self.len {
            // SAFETY: row j < len, at the head's value columns.
            let v = unsafe { std::slice::from_raw_parts(self.qv.add(j * self.ld + self.v_at), d) };
            for g in 0..G {
                let pj = p[g * lp + j];
                for (a, &v) in acc[g * d..][..d].iter_mut().zip(v) {
                    *a = madd(self.fma, pj, v, *a);
                }
            }
        }
        for g in 0..G {
            // SAFETY: row i + g of ctx, at the head's columns.
            unsafe { std::slice::from_raw_parts_mut(out[g], d) }.copy_from_slice(&acc[g * d..][..d]);
        }
    }
}
