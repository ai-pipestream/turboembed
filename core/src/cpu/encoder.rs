//! A BERT encoder in F32 on the host: the family TURBO_FAMILY_BERT names.
//!
//! Embeddings are the sum of the word, position and token type rows,
//! LayerNorm'd. Each layer is self-attention over the row's live tokens
//! (Q, K and V projections, scaled dot products, softmax, the output
//! projection), a residual and LayerNorm, then a feed-forward block with
//! GELU (erf), a residual and LayerNorm. Then each row is pooled (mean over
//! its mask, its first token, or its last live token), cut to output_dim,
//! and L2-normalized when asked.
//!
//! The rows of a batch are packed: row r's positions up to its last live
//! token sit one after another with the other rows', so padding past that
//! token is never computed. No output depends on it: attention skips
//! masked keys, and no pooling reads past the last live token. Positions
//! are the row's own column indices, as upstream numbers them.
//!
//! Every buffer is allocated by `new`, for the session's largest batch;
//! `write` and `run` allocate nothing.

use crate::backend::{
    TURBO_BERT_EMBEDDING_TENSORS, TURBO_BERT_LAYER_TENSORS, turbo_backend_embed_rows, turbo_backend_model,
};
use crate::{TURBO_NORMALIZE_L2, TURBO_POOLING_CLS, TURBO_POOLING_LAST};

// TURBO_BERT_* in turbo_backend.h.
const WORD: usize = 0;
const POSITION: usize = 1;
const TOKEN_TYPE: usize = 2;
const EMB_LN_W: usize = 3;
const EMB_LN_B: usize = 4;
const Q_W: usize = 0;
const Q_B: usize = 1;
const K_W: usize = 2;
const K_B: usize = 3;
const V_W: usize = 4;
const V_B: usize = 5;
const O_W: usize = 6;
const O_B: usize = 7;
const ATTN_LN_W: usize = 8;
const ATTN_LN_B: usize = 9;
const FFN_IN_W: usize = 10;
const FFN_IN_B: usize = 11;
const FFN_OUT_W: usize = 12;
const FFN_OUT_B: usize = 13;
const FFN_LN_W: usize = 14;
const FFN_LN_B: usize = 15;

/// A linear layer's kernel: y[t, o] = b[o] + sum_i x[t, i] w[o, i] for t
/// under rows, with w [n_out, n_in].
type Linear = fn(x: &[f32], rows: usize, n_in: usize, w: &[f32], b: &[f32], n_out: usize, y: &mut [f32]);

pub(super) struct Encoder {
    layers: usize,
    hidden: usize,
    heads: usize,
    intermediate: usize,
    eps: f32,
    /// Every tensor in F32, in TURBO_BERT_* order: pointers into the
    /// core's weights, or into the model's converted copy, both of which
    /// outlive the session.
    tensors: Vec<(*const f32, usize)>,
    linear: Linear,

    max_seq: usize,
    // The written rows, [batch, seq] packed with no stride.
    batch: usize,
    seq: usize,
    ids: Vec<u32>,
    types: Vec<u32>,
    mask: Vec<u8>,
    pooling: u32,
    normalize: u32,
    output_dim: usize,

    /// Each row's first packed token and its length through its last
    /// live token.
    rows: Vec<(usize, usize)>,
    // Scratch: [tokens, hidden] for the hidden states, the projections and
    // the attention context; [tokens, intermediate] for the feed-forward
    // block; one row of attention scores.
    x: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    ctx: Vec<f32>,
    ffn: Vec<f32>,
    scores: Vec<f32>,
}

/// Zeroed memory, or the bytes that could not be had.
fn zeroed<T: Clone + Default>(n: usize) -> Result<Vec<T>, usize> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).map_err(|_| n.saturating_mul(size_of::<T>()))?;
    v.resize(n, T::default());
    Ok(v)
}

impl Encoder {
    /// An encoder for `max_batch` rows of `max_seq` tokens over `tensors`,
    /// which stay where they are while it lives. Err is the bytes it could
    /// not allocate.
    pub(super) fn new(
        d: &turbo_backend_model,
        tensors: Vec<&[f32]>,
        max_batch: usize,
        max_seq: usize,
    ) -> Result<Encoder, usize> {
        debug_assert_eq!(tensors.len(), (TURBO_BERT_EMBEDDING_TENSORS + d.layers * TURBO_BERT_LAYER_TENSORS) as usize);
        let (h, i) = (d.hidden as usize, d.intermediate as usize);
        let tokens = max_batch.checked_mul(max_seq).ok_or(usize::MAX)?;
        let wide = |n: usize| tokens.checked_mul(n).ok_or(usize::MAX);
        Ok(Encoder {
            layers: d.layers as usize,
            hidden: h,
            heads: d.heads as usize,
            intermediate: i,
            eps: d.layer_norm_eps as f32,
            tensors: tensors.iter().map(|t| (t.as_ptr(), t.len())).collect(),
            linear: kernel(),
            max_seq,
            batch: 0,
            seq: 0,
            ids: zeroed(tokens)?,
            types: zeroed(tokens)?,
            mask: zeroed(tokens)?,
            pooling: 0,
            normalize: 0,
            output_dim: 0,
            rows: zeroed(max_batch)?,
            x: zeroed(wide(h)?)?,
            q: zeroed(wide(h)?)?,
            k: zeroed(wide(h)?)?,
            v: zeroed(wide(h)?)?,
            ctx: zeroed(wide(h)?)?,
            ffn: zeroed(wide(i)?)?,
            scores: zeroed(max_seq)?,
        })
    }

    fn tensor(&self, i: usize) -> &[f32] {
        let (p, n) = self.tensors[i];
        unsafe { std::slice::from_raw_parts(p, n) }
    }

    fn layer(&self, l: usize, r: usize) -> &[f32] {
        self.tensor(TURBO_BERT_EMBEDDING_TENSORS as usize + l * TURBO_BERT_LAYER_TENSORS as usize + r)
    }

    pub(super) fn normalize(&self) -> u32 {
        self.normalize
    }

    /// Copy in the rows the core checked, dropping their stride.
    pub(super) fn write(&mut self, r: &turbo_backend_embed_rows, ids: &[i32], mask: &[i32], types: Option<&[i32]>) {
        let (b, s, stride) = (r.batch as usize, r.seq as usize, r.row_stride as usize);
        debug_assert!(s <= self.max_seq && b * s <= self.ids.len());
        for row in 0..b {
            let (src, dst) = (row * stride, row * s);
            for p in 0..s {
                self.ids[dst + p] = ids[src + p] as u32;
                self.mask[dst + p] = mask[src + p] as u8;
                self.types[dst + p] = types.map_or(0, |t| t[src + p] as u32);
            }
        }
        self.batch = b;
        self.seq = s;
        self.pooling = r.pooling;
        self.normalize = r.normalize;
        self.output_dim = r.output_dim as usize;
    }

    /// Run the written rows into `out`, [batch, output_dim] packed.
    pub(super) fn run(&mut self, out: &mut [f32]) {
        let tokens = self.pack();
        self.lookup();
        for l in 0..self.layers {
            self.layer_forward(l, tokens);
        }
        self.pool(out);
    }

    /// Each row's place among the packed tokens, and their count.
    fn pack(&mut self) -> usize {
        let mut t = 0;
        for r in 0..self.batch {
            let m = &self.mask[r * self.seq..(r + 1) * self.seq];
            let len = m.iter().rposition(|&v| v != 0).map_or(0, |p| p + 1);
            self.rows[r] = (t, len);
            t += len;
        }
        t
    }

    fn lookup(&mut self) {
        let h = self.hidden;
        let mut x = std::mem::take(&mut self.x);
        let (word, pos, ty) = (self.tensor(WORD), self.tensor(POSITION), self.tensor(TOKEN_TYPE));
        let (ln_w, ln_b) = (self.tensor(EMB_LN_W), self.tensor(EMB_LN_B));
        for r in 0..self.batch {
            let (start, len) = self.rows[r];
            for p in 0..len {
                let at = r * self.seq + p;
                let (id, t) = (self.ids[at] as usize, self.types[at] as usize);
                let (w, ps, tt) = (&word[id * h..][..h], &pos[p * h..][..h], &ty[t * h..][..h]);
                let dst = &mut x[(start + p) * h..][..h];
                for i in 0..h {
                    dst[i] = w[i] + ps[i] + tt[i];
                }
                layer_norm(dst, ln_w, ln_b, self.eps);
            }
        }
        self.x = x;
    }

    fn layer_forward(&mut self, l: usize, tokens: usize) {
        let (h, n_i) = (self.hidden, self.intermediate);
        let lin = self.linear;
        let mut x = std::mem::take(&mut self.x);
        let mut q = std::mem::take(&mut self.q);
        let mut k = std::mem::take(&mut self.k);
        let mut v = std::mem::take(&mut self.v);
        let mut ctx = std::mem::take(&mut self.ctx);
        let mut ffn = std::mem::take(&mut self.ffn);
        let mut scores = std::mem::take(&mut self.scores);
        {
            let w = |r| self.layer(l, r);
            lin(&x, tokens, h, w(Q_W), w(Q_B), h, &mut q);
            lin(&x, tokens, h, w(K_W), w(K_B), h, &mut k);
            lin(&x, tokens, h, w(V_W), w(V_B), h, &mut v);
            self.attention(&q, &k, &v, &mut ctx, &mut scores);
            // The attention output projection, into q, which is done with.
            lin(&ctx, tokens, h, w(O_W), w(O_B), h, &mut q);
            add_layer_norm(&mut x[..tokens * h], &q[..tokens * h], h, w(ATTN_LN_W), w(ATTN_LN_B), self.eps);
            lin(&x, tokens, h, w(FFN_IN_W), w(FFN_IN_B), n_i, &mut ffn);
            for y in &mut ffn[..tokens * n_i] {
                *y = gelu(*y);
            }
            lin(&ffn, tokens, n_i, w(FFN_OUT_W), w(FFN_OUT_B), h, &mut q);
            add_layer_norm(&mut x[..tokens * h], &q[..tokens * h], h, w(FFN_LN_W), w(FFN_LN_B), self.eps);
        }
        (self.x, self.q, self.k, self.v, self.ctx, self.ffn, self.scores) = (x, q, k, v, ctx, ffn, scores);
    }

    /// Scaled dot-product attention within each row, head by head, over
    /// the keys whose mask is 1.
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], ctx: &mut [f32], scores: &mut [f32]) {
        let h = self.hidden;
        let d = h / self.heads;
        let scale = 1.0 / (d as f32).sqrt();
        for r in 0..self.batch {
            let (start, len) = self.rows[r];
            let mask = &self.mask[r * self.seq..][..len];
            for head in 0..self.heads {
                let col = head * d;
                for i in 0..len {
                    let qi = &q[(start + i) * h + col..][..d];
                    let mut max = f32::NEG_INFINITY;
                    for j in 0..len {
                        if mask[j] == 0 {
                            continue;
                        }
                        let kj = &k[(start + j) * h + col..][..d];
                        let s = dot(qi, kj) * scale;
                        scores[j] = s;
                        max = max.max(s);
                    }
                    let mut sum = 0.0f32;
                    for j in 0..len {
                        if mask[j] != 0 {
                            let e = (scores[j] - max).exp();
                            scores[j] = e;
                            sum += e;
                        }
                    }
                    let inv = 1.0 / sum;
                    let c = &mut ctx[(start + i) * h + col..][..d];
                    c.fill(0.0);
                    for j in 0..len {
                        if mask[j] != 0 {
                            let p = scores[j] * inv;
                            let vj = &v[(start + j) * h + col..][..d];
                            for (c, &vv) in c.iter_mut().zip(vj) {
                                *c += p * vv;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Pool each row, cut it to output_dim, and normalize it. The cut
    /// comes first, so an L2-normalized vector is unit length at the
    /// width the caller asked for.
    fn pool(&self, out: &mut [f32]) {
        let (h, od) = (self.hidden, self.output_dim);
        for r in 0..self.batch {
            let (start, len) = self.rows[r];
            let dst = &mut out[r * od..][..od];
            match self.pooling {
                TURBO_POOLING_CLS => dst.copy_from_slice(&self.x[start * h..][..od]),
                TURBO_POOLING_LAST => dst.copy_from_slice(&self.x[(start + len - 1) * h..][..od]),
                // Mean over the tokens whose mask is 1, as upstream's
                // mean pooling divides the masked sum by the mask's sum.
                _ => {
                    dst.fill(0.0);
                    let mask = &self.mask[r * self.seq..][..len];
                    let mut n = 0u32;
                    for (p, _) in mask.iter().enumerate().filter(|(_, m)| **m != 0) {
                        for (d, &x) in dst.iter_mut().zip(&self.x[(start + p) * h..][..od]) {
                            *d += x;
                        }
                        n += 1;
                    }
                    let inv = 1.0 / n as f32;
                    for d in dst.iter_mut() {
                        *d *= inv;
                    }
                }
            }
            if self.normalize == TURBO_NORMALIZE_L2 {
                // As upstream: divided by the norm, or by 1e-12 when the
                // norm is smaller.
                let norm = dst.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt().max(1e-12);
                let inv = (1.0 / norm) as f32;
                for d in dst.iter_mut() {
                    *d *= inv;
                }
            }
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// y = (y - mean) / sqrt(var + eps) * w + b, with the mean and the biased
/// variance summed in F64.
fn layer_norm(y: &mut [f32], w: &[f32], b: &[f32], eps: f32) {
    let n = y.len() as f64;
    let mean = y.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = y.iter().map(|&v| (v as f64 - mean) * (v as f64 - mean)).sum::<f64>() / n;
    let inv = 1.0 / (var + eps as f64).sqrt();
    for ((y, &w), &b) in y.iter_mut().zip(w).zip(b) {
        *y = ((*y as f64 - mean) * inv) as f32 * w + b;
    }
}

/// x = LayerNorm(x + y), row by row.
fn add_layer_norm(x: &mut [f32], y: &[f32], h: usize, w: &[f32], b: &[f32], eps: f32) {
    for (xr, yr) in x.chunks_exact_mut(h).zip(y.chunks_exact(h)) {
        for (a, &c) in xr.iter_mut().zip(yr) {
            *a += c;
        }
        layer_norm(xr, w, b, eps);
    }
}

/// GELU with the error function, as upstream's "gelu".
fn gelu(x: f32) -> f32 {
    let x = x as f64;
    (0.5 * x * (2.0 - erfc(x * std::f64::consts::FRAC_1_SQRT_2))) as f32
}

/// The complementary error function to a fractional error under 1.2e-7
/// everywhere: the Chebyshev fit of Numerical Recipes' erfcc, in F64. GELU
/// reads it as 1 + erf = 2 - erfc, so its error is relative to a value
/// near 1 and below what F32 holds.
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let p = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98 + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let r = t * p.exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

// ---- Linear layers ---------------------------------------------------------
//
// Each output is the dot product of an input row and a weight row, both
// contiguous. A tile of 4 input rows by 2 weight rows keeps eight dot
// products in flight, each over 8 lanes the compiler holds in vector
// registers; weights are walked in chunks that stay in cache while every
// tile of input rows passes over them.

/// Lanes per dot product.
const LANES: usize = 8;
/// Floats of weights walked per chunk: 128 KiB.
const CHUNK: usize = 32 * 1024;

/// The fastest kernel this processor has, chosen once per session.
#[cfg(target_arch = "x86_64")]
fn kernel() -> Linear {
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return linear_avx2;
    }
    linear_impl::<false>
}

/// Fused multiply-add is part of every aarch64 processor.
#[cfg(target_arch = "aarch64")]
fn kernel() -> Linear {
    linear_impl::<true>
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn kernel() -> Linear {
    linear_impl::<false>
}

#[cfg(target_arch = "x86_64")]
fn linear_avx2(x: &[f32], rows: usize, n_in: usize, w: &[f32], b: &[f32], n_out: usize, y: &mut [f32]) {
    #[target_feature(enable = "avx2,fma")]
    fn go(x: &[f32], rows: usize, n_in: usize, w: &[f32], b: &[f32], n_out: usize, y: &mut [f32]) {
        linear_impl::<true>(x, rows, n_in, w, b, n_out, y)
    }
    // kernel() returns this only where both features were detected.
    unsafe { go(x, rows, n_in, w, b, n_out, y) }
}

#[inline(always)]
fn linear_impl<const FMA: bool>(
    x: &[f32],
    rows: usize,
    n_in: usize,
    w: &[f32],
    b: &[f32],
    n_out: usize,
    y: &mut [f32],
) {
    let (x, w, b, y) = (&x[..rows * n_in], &w[..n_out * n_in], &b[..n_out], &mut y[..rows * n_out]);
    let chunk = (CHUNK / n_in.max(1)).clamp(2, n_out.max(2)) & !1;
    let mut o0 = 0;
    while o0 < n_out {
        let o1 = (o0 + chunk).min(n_out);
        let mut r = 0;
        while r + 4 <= rows {
            tile::<4, FMA>(x, r, n_in, w, b, o0, o1, y, n_out);
            r += 4;
        }
        while r < rows {
            tile::<1, FMA>(x, r, n_in, w, b, o0, o1, y, n_out);
            r += 1;
        }
        o0 = o1;
    }
}

/// Outputs o0..o1 of input rows r..r+R.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn tile<const R: usize, const FMA: bool>(
    x: &[f32],
    r: usize,
    n_in: usize,
    w: &[f32],
    b: &[f32],
    o0: usize,
    o1: usize,
    y: &mut [f32],
    n_out: usize,
) {
    let xs: [&[f32]; R] = std::array::from_fn(|i| &x[(r + i) * n_in..][..n_in]);
    let mut o = o0;
    while o + 2 <= o1 {
        let ws = [&w[o * n_in..][..n_in], &w[(o + 1) * n_in..][..n_in]];
        let d = dots::<R, 2, FMA>(xs, ws, n_in);
        for i in 0..R {
            y[(r + i) * n_out + o] = d[i][0] + b[o];
            y[(r + i) * n_out + o + 1] = d[i][1] + b[o + 1];
        }
        o += 2;
    }
    if o < o1 {
        let d = dots::<R, 1, FMA>(xs, [&w[o * n_in..][..n_in]], n_in);
        for i in 0..R {
            y[(r + i) * n_out + o] = d[i][0] + b[o];
        }
    }
}

#[inline(always)]
fn madd<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// R x C dot products of length n.
#[inline(always)]
fn dots<const R: usize, const C: usize, const FMA: bool>(xs: [&[f32]; R], ws: [&[f32]; C], n: usize) -> [[f32; C]; R] {
    let mut acc = [[[0.0f32; LANES]; C]; R];
    let main = n - n % LANES;
    let mut k = 0;
    while k < main {
        let xv: [&[f32; LANES]; R] = std::array::from_fn(|i| xs[i][k..k + LANES].try_into().unwrap());
        let wv: [&[f32; LANES]; C] = std::array::from_fn(|j| ws[j][k..k + LANES].try_into().unwrap());
        for i in 0..R {
            for j in 0..C {
                for l in 0..LANES {
                    acc[i][j][l] = madd::<FMA>(xv[i][l], wv[j][l], acc[i][j][l]);
                }
            }
        }
        k += LANES;
    }
    let mut out = [[0.0f32; C]; R];
    for i in 0..R {
        for j in 0..C {
            let a = &acc[i][j];
            let mut s = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
            for t in main..n {
                s = madd::<FMA>(xs[i][t], ws[j][t], s);
            }
            out[i][j] = s;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// erf by its Taylor series in F64, which converges everywhere and is
    /// exact to rounding for |x| under 3.
    fn erf_series(x: f64) -> f64 {
        let z = x * x;
        let (mut term, mut sum, mut n) = (x, x, 0.0);
        loop {
            n += 1.0;
            term *= -z / n;
            let t = term / (2.0 * n + 1.0);
            sum += t;
            if t.abs() < 1e-17 * sum.abs().max(1e-300) {
                break;
            }
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    }

    #[test]
    fn erfc_is_within_its_bound_of_the_series() {
        for (x, want) in
            [(0.5, 0.520_499_877_813_046_5), (1.0, 0.842_700_792_949_714_9), (2.0, 0.995_322_265_018_952_7)]
        {
            assert!((erf_series(x) - want).abs() < 1e-15, "the series at {x}");
        }
        let mut worst = 0.0f64;
        for i in -3000..=3000 {
            let x = i as f64 / 1000.0;
            let exact = 1.0 - erf_series(x);
            let rel = ((erfc(x) - exact) / exact).abs();
            worst = worst.max(rel);
        }
        assert!(worst < 1.2e-7, "erfc's fractional error reaches {worst}");
    }

    #[test]
    fn gelu_matches_its_definition() {
        for i in -600..=600 {
            let x = i as f64 / 100.0;
            let want = 0.5 * x * (1.0 + erf_series(x / std::f64::consts::SQRT_2));
            let got = gelu(x as f32) as f64;
            assert!((got - want).abs() <= 1e-6 * want.abs().max(1e-3), "gelu({x}) is {got}, not {want}");
        }
    }

    /// Every kernel against the plain sum, for shapes with and without
    /// remainders in each dimension.
    #[test]
    fn linear_kernels_match_the_plain_sum() {
        for (rows, n_in, n_out) in [(1, 1, 1), (3, 7, 5), (4, 8, 2), (9, 37, 11), (13, 64, 130), (5, 1536, 9)] {
            let x: Vec<f32> = (0..rows * n_in).map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0).collect();
            let w: Vec<f32> = (0..n_out * n_in).map(|i| ((i * 53 % 97) as f32 - 48.0) / 48.0).collect();
            let b: Vec<f32> = (0..n_out).map(|i| i as f32 / 10.0).collect();
            let mut want = vec![0.0f64; rows * n_out];
            for t in 0..rows {
                for o in 0..n_out {
                    want[t * n_out + o] =
                        b[o] as f64 + (0..n_in).map(|i| x[t * n_in + i] as f64 * w[o * n_in + i] as f64).sum::<f64>();
                }
            }
            let mut kernels: Vec<Linear> = vec![linear_impl::<false>, kernel()];
            if cfg!(target_arch = "aarch64") {
                kernels.push(linear_impl::<true>);
            }
            for k in kernels {
                let mut y = vec![f32::NAN; rows * n_out + 3];
                k(&x, rows, n_in, &w, &b, n_out, &mut y);
                for (i, (&g, &e)) in y.iter().zip(&want).enumerate() {
                    assert!((g as f64 - e).abs() < 1e-4 * (1.0 + e.abs()), "{rows}x{n_in}x{n_out} [{i}]: {g} vs {e}");
                }
                assert!(y[rows * n_out..].iter().all(|v| v.is_nan()), "nothing past the output is written");
            }
        }
    }
}
