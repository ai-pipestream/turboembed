//! Dot products of blocked rows. Every path sums a product in the order
//! the crate documentation states, so the paths give the same bits; they
//! differ in how many rows and queries they carry at once.

use crate::rows::{Block, LANES};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Path {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

/// The fastest path this CPU runs.
pub(crate) fn detect() -> Path {
    *available().last().unwrap_or(&Path::Scalar)
}

/// Every path this CPU runs, the scalar one first.
pub(crate) fn available() -> Vec<Path> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::is_x86_feature_detected as has;
        paths_for(has!("avx512f"), has!("avx2"), has!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    vec![Path::Scalar]
}

/// The paths a CPU with these features runs, the scalar one first. The
/// AVX-512 path folds its lanes with AVX2 code and the AVX2 path uses FMA,
/// so each needs every feature it calls.
#[cfg(target_arch = "x86_64")]
pub(crate) fn paths_for(avx512f: bool, avx2: bool, fma: bool) -> Vec<Path> {
    let mut out = vec![Path::Scalar];
    if avx2 && fma {
        out.push(Path::Avx2);
        if avx512f {
            out.push(Path::Avx512);
        }
    }
    out
}

/// Scores each of the `nq` queries in `qs` against each row of `rows`,
/// `bpr` blocks a query and a row, into `out[q * n + r]` for `n` rows.
pub(crate) fn scores(path: Path, qs: &[Block], rows: &[Block], bpr: usize, out: &mut [f32]) {
    let nq = qs.len() / bpr;
    let n = rows.len() / bpr;
    assert!(nq * bpr == qs.len() && n * bpr == rows.len() && out.len() == nq * n);
    match path {
        Path::Scalar => scalar(qs, rows, bpr, out),
        // SAFETY: detect() and available() return a path only where the
        // CPU has its features, and the lengths are checked above.
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => unsafe { x86::avx2(qs.as_ptr(), nq, rows.as_ptr(), n, bpr, out.as_mut_ptr()) },
        #[cfg(target_arch = "x86_64")]
        Path::Avx512 => unsafe { x86::avx512(qs.as_ptr(), nq, rows.as_ptr(), n, bpr, out.as_mut_ptr()) },
    }
}

fn scalar(qs: &[Block], rows: &[Block], bpr: usize, out: &mut [f32]) {
    let n = rows.len() / bpr;
    for (qi, q) in qs.chunks_exact(bpr).enumerate() {
        for (ri, r) in rows.chunks_exact(bpr).enumerate() {
            let mut acc = [0.0f32; LANES];
            for (a, b) in q.iter().zip(r) {
                for ((s, &x), &y) in acc.iter_mut().zip(&a.0).zip(&b.0) {
                    *s = x.mul_add(y, *s);
                }
            }
            for half in [8, 4, 2, 1] {
                for j in 0..half {
                    acc[j] += acc[j + half];
                }
            }
            out[qi * n + ri] = acc[0];
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::Block;

    /// Adds lane `j + 4`, then `j + 2`, then `j + 1` into lane 0.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn sum8(s8: __m256) -> f32 {
        let s4 = _mm_add_ps(_mm256_castps256_ps128(s8), _mm256_extractf128_ps::<1>(s8));
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps::<1>(s2, s2));
        _mm_cvtss_f32(s1)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    fn sum16(acc: __m512) -> f32 {
        let lo = _mm512_castps512_ps256(acc);
        let hi = _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(acc)));
        sum8(_mm256_add_ps(lo, hi))
    }

    /// `Q` queries against `R` rows, one `zmm` accumulator each.
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn tile512<const Q: usize, const R: usize>(
        q: *const Block,
        r: *const Block,
        bpr: usize,
        out: *mut f32,
        stride: usize,
    ) {
        let mut acc = [[_mm512_setzero_ps(); R]; Q];
        for b in 0..bpr {
            // SAFETY: the caller passes Q queries and R rows of bpr blocks.
            let rv: [__m512; R] = std::array::from_fn(|i| unsafe { _mm512_load_ps(r.add(i * bpr + b).cast()) });
            for (qi, row) in acc.iter_mut().enumerate() {
                let qv = unsafe { _mm512_load_ps(q.add(qi * bpr + b).cast()) };
                for (a, &v) in row.iter_mut().zip(&rv) {
                    *a = _mm512_fmadd_ps(qv, v, *a);
                }
            }
        }
        if Q == 4 && R == 4 {
            let flat: [__m512; 16] = std::array::from_fn(|k| acc[k / R][k % R]);
            let mut sums = [0.0f32; 16];
            // SAFETY: sums holds 16 floats.
            unsafe { _mm512_storeu_ps(sums.as_mut_ptr(), sum16x16(&flat)) };
            for qi in 0..4 {
                for ri in 0..4 {
                    // SAFETY: out has Q lines of stride floats, R used.
                    unsafe { *out.add(qi * stride + ri) = sums[4 * ri + qi] };
                }
            }
            return;
        }
        for (qi, row) in acc.iter().enumerate() {
            for (ri, &a) in row.iter().enumerate() {
                // SAFETY: out has Q lines of stride floats, R used.
                unsafe { *out.add(qi * stride + ri) = sum16(a) };
            }
        }
    }

    /// The sums of 16 accumulators at once, each folded as [`sum16`] folds
    /// one (lane `j + 8`, then `j + 4`, `j + 2`, `j + 1`), so the bits are
    /// the same; the shuffles move four accumulators' lanes into one
    /// register at each step. Lane `4 * g + e` holds the sum of `acc[4 * e +
    /// g]`.
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn sum16x16(acc: &[__m512; 16]) -> __m512 {
        // Groups of four lanes: [a 0..4, a 4..8, b 0..4, b 4..8] each the
        // sum of lanes j and j + 8.
        let s8: [__m512; 8] = std::array::from_fn(|m| {
            let (a, b) = (acc[2 * m], acc[2 * m + 1]);
            _mm512_add_ps(_mm512_shuffle_f32x4::<0x44>(a, b), _mm512_shuffle_f32x4::<0xEE>(a, b))
        });
        // [a, b, c, d], four lanes each: lanes j and j + 4 of s8.
        let s4: [__m512; 4] = std::array::from_fn(|p| {
            let (x, y) = (s8[2 * p], s8[2 * p + 1]);
            _mm512_add_ps(_mm512_shuffle_f32x4::<0x88>(x, y), _mm512_shuffle_f32x4::<0xDD>(x, y))
        });
        // Within each group of four: lanes j and j + 2.
        let s2: [__m512; 2] = std::array::from_fn(|r| {
            let (p, q) = (s4[2 * r], s4[2 * r + 1]);
            _mm512_add_ps(_mm512_shuffle_ps::<0x44>(p, q), _mm512_shuffle_ps::<0xEE>(p, q))
        });
        // Lanes j and j + 1.
        let (r, s) = (s2[0], s2[1]);
        _mm512_add_ps(_mm512_shuffle_ps::<0x88>(r, s), _mm512_shuffle_ps::<0xDD>(r, s))
    }

    /// `Q` queries against `R` rows, two `ymm` accumulators each (lanes 0
    /// to 7 and 8 to 15).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn tile256<const Q: usize, const R: usize>(
        q: *const Block,
        r: *const Block,
        bpr: usize,
        out: *mut f32,
        stride: usize,
    ) {
        let mut lo = [[_mm256_setzero_ps(); R]; Q];
        let mut hi = [[_mm256_setzero_ps(); R]; Q];
        for b in 0..bpr {
            // SAFETY: the caller passes Q queries and R rows of bpr blocks.
            let rp: [*const f32; R] = std::array::from_fn(|i| unsafe { r.add(i * bpr + b).cast::<f32>() });
            let rl: [__m256; R] = std::array::from_fn(|i| unsafe { _mm256_load_ps(rp[i]) });
            let rh: [__m256; R] = std::array::from_fn(|i| unsafe { _mm256_load_ps(rp[i].add(8)) });
            for qi in 0..Q {
                let qp = unsafe { q.add(qi * bpr + b).cast::<f32>() };
                let (ql, qh) = unsafe { (_mm256_load_ps(qp), _mm256_load_ps(qp.add(8))) };
                for ri in 0..R {
                    lo[qi][ri] = _mm256_fmadd_ps(ql, rl[ri], lo[qi][ri]);
                    hi[qi][ri] = _mm256_fmadd_ps(qh, rh[ri], hi[qi][ri]);
                }
            }
        }
        for qi in 0..Q {
            for ri in 0..R {
                // SAFETY: out has Q lines of stride floats, R used.
                unsafe { *out.add(qi * stride + ri) = sum8(_mm256_add_ps(lo[qi][ri], hi[qi][ri])) };
            }
        }
    }

    /// Runs `tile::<Q, R>` over every row for one group of `Q` queries,
    /// then the rows left over one at a time.
    macro_rules! sweep {
        ($tile:ident, $q:expr, $r:expr, $qs:expr, $rows:expr, $n:expr, $bpr:expr, $out:expr) => {{
            let mut ri = 0;
            while ri + $r <= $n {
                unsafe { $tile::<$q, $r>($qs, $rows.add(ri * $bpr), $bpr, $out.add(ri), $n) };
                ri += $r;
            }
            while ri < $n {
                unsafe { $tile::<$q, 1>($qs, $rows.add(ri * $bpr), $bpr, $out.add(ri), $n) };
                ri += 1;
            }
        }};
    }

    /// One query against one row at a time, since one stream of rows reads
    /// memory fastest; four queries against four rows at a time when there
    /// are several.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn avx512(qs: *const Block, nq: usize, rows: *const Block, n: usize, bpr: usize, out: *mut f32) {
        let mut qi = 0;
        while qi + 4 <= nq {
            let (q, o) = unsafe { (qs.add(qi * bpr), out.add(qi * n)) };
            sweep!(tile512, 4, 4, q, rows, n, bpr, o);
            qi += 4;
        }
        while qi < nq {
            let (q, o) = unsafe { (qs.add(qi * bpr), out.add(qi * n)) };
            for ri in 0..n {
                unsafe { tile512::<1, 1>(q, rows.add(ri * bpr), bpr, o.add(ri), n) };
            }
            qi += 1;
        }
    }

    /// One query against four rows at a time; two queries against two rows
    /// at a time when there are several.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn avx2(qs: *const Block, nq: usize, rows: *const Block, n: usize, bpr: usize, out: *mut f32) {
        let mut qi = 0;
        while qi + 2 <= nq {
            let (q, o) = unsafe { (qs.add(qi * bpr), out.add(qi * n)) };
            sweep!(tile256, 2, 2, q, rows, n, bpr, o);
            qi += 2;
        }
        while qi < nq {
            let (q, o) = unsafe { (qs.add(qi * bpr), out.add(qi * n)) };
            sweep!(tile256, 1, 4, q, rows, n, bpr, o);
            qi += 1;
        }
    }
}
