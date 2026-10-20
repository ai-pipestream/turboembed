//! SPLADE pooling of masked language model logits into weighted
//! vocabulary terms: the sparse half of `embed.sparse`. As SPLADE pools
//! (SPLADE-v3, arXiv 2403.06789), a text's sparse vector is, over its
//! tokens, the max of `log(1 + relu(logit))` per vocabulary term, kept
//! here as `(term id, weight)` pairs sorted by id, cut by an optional
//! top-k or weight floor.
//!
//! The pooling is two pure functions over logits and a pooled vector, so
//! a reference implementation can check them without a model: the golden
//! test does (testdata/splade-golden), the bundle's MLM head the rest.

use crate::mlm::{MlmHead, layer_norm};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result};

/// One term of a learned sparse vector: a model vocabulary id and the
/// weight the text gives it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseTerm {
    pub term: u32,
    pub weight: f32,
}

/// How a pooled vector is cut into terms: a weight under `floor` drops,
/// and at most `top_k` of the rest, the heaviest, are kept. Both 0 keep
/// every positive weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseOptions {
    pub top_k: u32,
    /// The least weight a term keeps; 0 keeps every positive weight.
    pub floor: f32,
}

impl SparseOptions {
    /// No cut: every positive weight stays.
    pub fn all() -> SparseOptions {
        SparseOptions { top_k: 0, floor: 0.0 }
    }

    fn check(&self) -> Result<()> {
        if !self.floor.is_finite() || self.floor < 0.0 {
            return Err(Error::new(
                INVALID_ARGUMENT,
                format!("sparse floor {} is not a finite number over 0", self.floor),
            ));
        }
        Ok(())
    }
}

/// `log(1 + relu(x))`, the per-logit transform SPLADE max-pools.
#[inline]
pub fn maxlog(x: f32) -> f32 {
    x.max(0.0).ln_1p()
}

/// The max pool of `logits` (`[tokens, vocab]`, the MLM head's output)
/// accumulated as each term's peak `relu(logit)`: `log(1 + x)` is
/// monotonic, so max-pooling the weights is max-pooling the peaks and
/// transforming once. Keeping the peak turns a `log1p` per token and term
/// into one per term.
#[derive(Debug, Clone)]
pub struct MaxLogPool {
    /// Each term's peak `relu(logit)` over the tokens observed so far.
    peak: Vec<f32>,
}

impl MaxLogPool {
    pub fn new(vocab: usize) -> MaxLogPool {
        MaxLogPool { peak: vec![0.0f32; vocab] }
    }

    pub fn vocab(&self) -> usize {
        self.peak.len()
    }

    /// Observe one token's logits (`[vocab]`), keeping each term's peak.
    /// A non-finite logit is `INVALID_ARGUMENT`, naming its term.
    pub fn observe(&mut self, logits: &[f32]) -> Result<()> {
        self.observe_at(0, logits)
    }

    /// As `observe`, for the terms `at..at + logits.len()`.
    pub fn observe_at(&mut self, at: usize, logits: &[f32]) -> Result<()> {
        let end = at.checked_add(logits.len()).ok_or_else(|| {
            Error::new(
                INVALID_SHAPE,
                format!("logits of {} terms at {at} overflow the pool of {}", logits.len(), self.peak.len()),
            )
        })?;
        if end > self.peak.len() {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("logits of {} terms at {at} overflow the pool of {}", logits.len(), self.peak.len()),
            ));
        }
        for (k, &x) in logits.iter().enumerate() {
            if !x.is_finite() {
                return Err(Error::new(INVALID_ARGUMENT, format!("MLM logit of term {} is not finite", at + k)));
            }
            let r = x.max(0.0);
            let p = &mut self.peak[at + k];
            if r > *p {
                *p = r;
            }
        }
        Ok(())
    }

    /// The pooled weights: `log(1 + peak)` per term.
    pub fn weights(&self) -> Vec<f32> {
        self.peak.iter().map(|&p| p.ln_1p()).collect()
    }

    /// The sparse terms of the pooled weights, cut as `o` says
    /// (select_terms).
    pub fn terms(&self, o: &SparseOptions) -> Result<Vec<SparseTerm>> {
        select_terms(&self.weights(), o)
    }
}

/// Accumulate the SPLADE max pool of `logits` (`[tokens, vocab]`, the MLM
/// head's output) into `pool` (`[vocab]`): each term's pool entry is the
/// max of what it held and the max over the tokens of
/// `log(1 + relu(logit))`, so batches pool into the same vector.
/// A token count of 0 leaves `pool` as it was. A non-finite logit is
/// `INVALID_ARGUMENT`, naming its token and term.
pub fn pool_logits_maxlog(logits: &[f32], vocab: usize, pool: &mut [f32]) -> Result<()> {
    if vocab == 0 || !logits.len().is_multiple_of(vocab) {
        return Err(Error::new(
            INVALID_SHAPE,
            format!("logits length {} is not a multiple of vocab {vocab}", logits.len()),
        ));
    }
    if pool.len() != vocab {
        return Err(Error::new(INVALID_SHAPE, format!("pool has {} values, expected {vocab}", pool.len())));
    }
    if logits.is_empty() {
        return Ok(());
    }
    let mut p = MaxLogPool::new(vocab);
    for row in logits.chunks_exact(vocab) {
        p.observe(row)?;
    }
    for (dst, &w) in pool.iter_mut().zip(&p.weights()) {
        *dst = dst.max(w);
    }
    Ok(())
}

/// The sparse terms of a pooled vector: `(id, weight)` pairs sorted by
/// id, finite and over `floor`, at most `top_k` of them, the heaviest
/// (ties by id, so the cut is deterministic).
pub fn select_terms(pool: &[f32], o: &SparseOptions) -> Result<Vec<SparseTerm>> {
    o.check()?;
    let mut keep: Vec<SparseTerm> = Vec::new();
    for (v, &w) in pool.iter().enumerate() {
        if !w.is_finite() {
            return Err(Error::new(INVALID_ARGUMENT, format!("pooled weight of term {v} is not finite")));
        }
        if w < 0.0 {
            return Err(Error::new(INVALID_ARGUMENT, format!("pooled weight of term {v} is under 0")));
        }
        if w > o.floor {
            keep.push(SparseTerm {
                term: u32::try_from(v).map_err(|_| Error::new(INVALID_SHAPE, "term id over u32"))?,
                weight: w,
            });
        }
    }
    if o.top_k > 0 && keep.len() as u64 > o.top_k as u64 {
        keep.sort_by(|a, b| b.weight.total_cmp(&a.weight).then(a.term.cmp(&b.term)));
        keep.truncate(o.top_k as usize);
    }
    keep.sort_by_key(|t| t.term);
    Ok(keep)
}

impl MlmHead {
    /// The learned sparse vector of `states` (`[tokens, hidden]`, the
    /// encoder's final states for the text's tokens): the MLM head's
    /// logits pooled as SPLADE pools them, cut as `o` says.
    pub fn sparse_terms(&self, states: &[f32], o: &SparseOptions) -> Result<Vec<SparseTerm>> {
        self.validate()?;
        o.check()?;
        let mut pool = MaxLogPool::new(self.vocab);
        self.relu_peak_into(states, &mut pool)?;
        pool.terms(o)
    }

    /// Accumulate the SPLADE pool of `states` (`[tokens, hidden]`) into
    /// `pool`: each term's peak `relu(logit)` over the tokens. The hot
    /// path: the decoder runs as a blocked GEMM over its transposed
    /// weight, and the transform runs once per token; `log(1 + x)` waits
    /// for the selection (MaxLogPool).
    pub fn relu_peak_into(&self, states: &[f32], pool: &mut MaxLogPool) -> Result<()> {
        self.validate()?;
        if pool.vocab() != self.vocab {
            return Err(Error::new(INVALID_SHAPE, format!("pool has {} terms, expected {}", pool.vocab(), self.vocab)));
        }
        if !states.len().is_multiple_of(self.hidden) {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("states length {} is not a multiple of hidden width {}", states.len(), self.hidden),
            ));
        }
        if states.is_empty() {
            return Ok(());
        }
        let (h, v) = (self.hidden, self.vocab);
        // The transform, the GELU and the LayerNorm, one vector per token.
        let mut t = vec![0.0f32; h];
        let mut normed = vec![0.0f32; states.len()];
        for (r, state) in states.chunks_exact(h).enumerate() {
            if !state.iter().all(|x| x.is_finite()) {
                return Err(Error::new(INVALID_ARGUMENT, "hidden state contains non-finite value"));
            }
            self.transform.apply(state, &mut t);
            for x in t.iter_mut() {
                *x = crate::cpu::gelu(*x);
            }
            layer_norm(&t, &self.ln_weight, &self.ln_bias, self.ln_eps, &mut normed[r * h..(r + 1) * h])?;
        }
        // The decoder, transposed so a block of tokens streams its weight
        // once: out[t, i] = b[i] + sum_j x[t, j] * wt[j, i], summed over
        // j in the order Linear::apply sums it. FMA where the processor
        // has it, as the encoder's kernels detect (cpu/kernels/micro.rs).
        #[cfg(target_arch = "x86_64")]
        let avx2 = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let avx2 = false;
        let mut wt = vec![0.0f32; h * v];
        for (i, w) in self.decoder.weight.chunks_exact(h).enumerate() {
            for (j, &x) in w.iter().enumerate() {
                wt[j * v + i] = x;
            }
        }
        const BLOCK: usize = 48;
        // Columns at a time, so a block's output rows and the weight's
        // column chunk both stay in cache: the weight streams once per
        // block, the output writes once.
        const CHUNK: usize = 2048;
        let mut out = vec![0.0f32; BLOCK * CHUNK];
        for block in normed.chunks(BLOCK * h) {
            let n = block.len() / h;
            let pool = &mut *pool;
            for i0 in (0..v).step_by(CHUNK) {
                let w = CHUNK.min(v - i0);
                let out = &mut out[..n * w];
                for (t, o) in out.chunks_exact_mut(w).enumerate() {
                    o.copy_from_slice(&self.decoder.bias[i0..i0 + w]);
                    let x = &block[t * h..(t + 1) * h];
                    if avx2 {
                        // SAFETY: avx2 says the processor has AVX2 and FMA.
                        unsafe { gemm_avx2(x, &wt, o, v, i0) };
                    } else {
                        for (j, &xj) in x.iter().enumerate() {
                            let wt = &wt[j * v + i0..j * v + i0 + w];
                            for i in 0..w {
                                o[i] += xj * wt[i];
                            }
                        }
                    }
                }
                for o in out.chunks_exact(w) {
                    pool.observe_at(i0, o)?;
                }
            }
        }
        Ok(())
    }
}

/// `o[i] += x[j] * wt[j * v + i0 + i]` over the `w` columns, with FMA.
///
/// # Safety
/// The processor has AVX2 and FMA, and `wt` holds rows of `v` floats with
/// `i0 + o.len() <= v`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_avx2(x: &[f32], wt: &[f32], o: &mut [f32], v: usize, i0: usize) {
    use std::arch::x86_64::*;
    let w = o.len();
    let mut i = 0;
    while i + 32 <= w {
        // Four 8-lane accumulators cover 32 columns a step, unrolled so
        // the loads and FMAs pipeline.
        let (mut c0, mut c1, mut c2, mut c3) = unsafe {
            (
                _mm256_loadu_ps(o.as_ptr().add(i)),
                _mm256_loadu_ps(o.as_ptr().add(i + 8)),
                _mm256_loadu_ps(o.as_ptr().add(i + 16)),
                _mm256_loadu_ps(o.as_ptr().add(i + 24)),
            )
        };
        for (j, &xj) in x.iter().enumerate() {
            let a = _mm256_set1_ps(xj);
            // SAFETY: i + 32 <= w and i0 + w <= v, so the row has these
            // columns.
            let row = unsafe { wt.as_ptr().add(j * v + i0 + i) };
            // SAFETY: i + 32 <= w <= v - i0, so the row has these columns.
            c0 = unsafe { _mm256_fmadd_ps(a, _mm256_loadu_ps(row), c0) };
            c1 = unsafe { _mm256_fmadd_ps(a, _mm256_loadu_ps(row.add(8)), c1) };
            c2 = unsafe { _mm256_fmadd_ps(a, _mm256_loadu_ps(row.add(16)), c2) };
            c3 = unsafe { _mm256_fmadd_ps(a, _mm256_loadu_ps(row.add(24)), c3) };
        }
        unsafe {
            _mm256_storeu_ps(o.as_mut_ptr().add(i), c0);
            _mm256_storeu_ps(o.as_mut_ptr().add(i + 8), c1);
            _mm256_storeu_ps(o.as_mut_ptr().add(i + 16), c2);
            _mm256_storeu_ps(o.as_mut_ptr().add(i + 24), c3);
        }
        i += 32;
    }
    // The tail under 32 columns, where the chunk ends: scalar, the same
    // sum order.
    for i in i..w {
        let mut s = o[i];
        for (j, &xj) in x.iter().enumerate() {
            s += xj * wt[j * v + i0 + i];
        }
        o[i] = s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool_of(logits: &[f32], vocab: usize) -> Vec<f32> {
        let mut pool = vec![0.0f32; vocab];
        pool_logits_maxlog(logits, vocab, &mut pool).unwrap();
        pool
    }

    #[test]
    fn the_pool_is_the_max_of_log_one_plus_relu_over_tokens() {
        // token 0: [1, -2, 0.5], token 1: [-1, 3, 0.25]
        let logits = [1.0f32, -2.0, 0.5, -1.0, 3.0, 0.25];
        let pool = pool_of(&logits, 3);
        let want = [maxlog(1.0f32), maxlog(3.0), maxlog(0.5)];
        assert_eq!(pool, want);
        // A token whose logits are all negative adds nothing.
        let pool = pool_of(&[-5.0f32, -1.0, -0.25], 3);
        assert_eq!(pool, [0.0; 3]);
    }

    #[test]
    fn empty_logits_leave_the_pool_zeroed() {
        let mut pool = vec![0.0f32; 4];
        pool_logits_maxlog(&[], 4, &mut pool).unwrap();
        assert_eq!(pool, [0.0; 4]);
        // No tokens is not a shape error; empty output follows in select.
        assert!(select_terms(&pool, &SparseOptions::all()).unwrap().is_empty());
    }

    #[test]
    fn pooling_merges_with_the_pool_a_batch_at_a_time() {
        let mut pool = vec![0.0f32; 2];
        pool_logits_maxlog(&[3.0, 1.0], 2, &mut pool).unwrap();
        pool_logits_maxlog(&[1.0, 0.0], 2, &mut pool).unwrap();
        assert_eq!(pool, [4.0f32.ln(), 2.0f32.ln()]);
        // A batch with no tokens leaves a non-zero pool as it was.
        pool_logits_maxlog(&[], 2, &mut pool).unwrap();
        assert_eq!(pool, [4.0f32.ln(), 2.0f32.ln()]);
    }

    #[test]
    fn observing_past_the_pool_is_an_error_not_a_panic() {
        let mut p = MaxLogPool::new(2);
        assert_eq!(p.observe_at(usize::MAX, &[1.0]).unwrap_err().code, INVALID_SHAPE);
        assert_eq!(p.observe_at(2, &[1.0]).unwrap_err().code, INVALID_SHAPE);
        assert_eq!(p.weights(), [0.0; 2]);
    }

    #[test]
    fn selection_sorts_by_id_and_applies_floor_and_top_k() {
        // Terms 1 and 3 tie at weight maxlog(2); term 2 is heavier; term 0
        // and 4 are negative, so they pool to 0.
        let logits = [-1.0f32, 2.0, 3.0, 2.0, -1.0];
        let pool = pool_of(&logits, 5);
        let terms = select_terms(&pool, &SparseOptions::all()).unwrap();
        assert_eq!(
            terms.iter().map(|t| (t.term, t.weight)).collect::<Vec<_>>(),
            vec![(1, maxlog(2.0)), (2, maxlog(3.0)), (3, maxlog(2.0))]
        );
        let terms = select_terms(&pool, &SparseOptions { top_k: 0, floor: maxlog(2.5) }).unwrap();
        assert_eq!(terms.iter().map(|t| t.term).collect::<Vec<_>>(), [2]);
        // top_k 1 takes the heaviest; top_k past the count keeps all.
        let terms = select_terms(&pool, &SparseOptions { top_k: 1, floor: 0.0 }).unwrap();
        assert_eq!(terms.iter().map(|t| t.term).collect::<Vec<_>>(), [2]);
        let terms = select_terms(&pool, &SparseOptions { top_k: 99, floor: 0.0 }).unwrap();
        assert_eq!(terms.len(), 3);
        // A tie at the cut takes the lower id.
        let terms = select_terms(&pool, &SparseOptions { top_k: 2, floor: 0.0 }).unwrap();
        assert_eq!(terms.iter().map(|t| t.term).collect::<Vec<_>>(), [1, 2]);
    }

    #[test]
    fn boundary_cases_are_refused_or_empty() {
        let pool = vec![0.0f32; 3];
        assert_eq!(
            select_terms(&pool, &SparseOptions { top_k: 0, floor: f32::NAN }).unwrap_err().code,
            INVALID_ARGUMENT
        );
        assert_eq!(select_terms(&pool, &SparseOptions { top_k: 0, floor: -1.0 }).unwrap_err().code, INVALID_ARGUMENT);
        assert_eq!(
            select_terms(&pool, &SparseOptions { top_k: 0, floor: f32::INFINITY }).unwrap_err().code,
            INVALID_ARGUMENT
        );
        // top_k 0 keeps every positive weight; none are, here.
        assert!(select_terms(&pool, &SparseOptions::all()).unwrap().is_empty());
        assert_eq!(pool_logits_maxlog(&[1.0, 2.0], 3, &mut pool.clone()).unwrap_err().code, INVALID_SHAPE);
        assert_eq!(pool_logits_maxlog(&[f32::NAN, 2.0, 3.0], 3, &mut pool.clone()).unwrap_err().code, INVALID_ARGUMENT);
        assert_eq!(pool_logits_maxlog(&[1.0, 2.0, 3.0], 0, &mut Vec::new()).unwrap_err().code, INVALID_SHAPE);
        let mut bad = vec![f32::NAN; 3];
        assert_eq!(select_terms(&bad, &SparseOptions::all()).unwrap_err().code, INVALID_ARGUMENT);
        bad.fill(1.0);
        assert_eq!(select_terms(&bad, &SparseOptions::all()).unwrap().len(), 3);
        let mut neg = vec![-0.5f32; 2];
        assert_eq!(select_terms(&neg, &SparseOptions::all()).unwrap_err().code, INVALID_ARGUMENT);
        neg.fill(0.0);
        assert!(select_terms(&neg, &SparseOptions::all()).unwrap().is_empty());
    }
}
