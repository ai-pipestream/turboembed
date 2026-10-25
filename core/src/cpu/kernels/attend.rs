//! Softmax attention within one row and head.

use super::micro::madd;
use super::rowwise::exp;
use super::simd::Simd;

/// The sum of LANES lanes, in one fixed order: halves added pairwise.
#[inline(always)]
pub(super) fn tree(mut s: [f32; LANES]) -> f32 {
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
pub(in crate::cpu) const LANES: usize = 16;

/// Softmax attention of queries q0..q1 of one row's tokens over its keys,
/// in one head.
pub(in crate::cpu) struct Attend<'a> {
    /// The row's queries and values, [len, ld], the head's d columns of
    /// each from q_at and v_at.
    pub(in crate::cpu) qv: *const f32,
    pub(in crate::cpu) ld: usize,
    pub(in crate::cpu) q_at: usize,
    pub(in crate::cpu) v_at: usize,
    /// The head's keys transposed: column c of key j at c * stride + j,
    /// for j under len rounded up to LANES.
    pub(in crate::cpu) kt: *const f32,
    pub(in crate::cpu) stride: usize,
    pub(in crate::cpu) d: usize,
    pub(in crate::cpu) len: usize,
    pub(in crate::cpu) mask: &'a [u8],
    /// Limited lookahead: query i attends to keys up to i + ahead, and
    /// usize::MAX leaves the whole row. A banded key's score is minus
    /// infinity, as a masked key's is.
    pub(in crate::cpu) ahead: usize,
    pub(in crate::cpu) scale: f32,
    /// Whether the plain Rust fallback fuses multiply and add: as V does.
    pub(in crate::cpu) fma: bool,
    /// [len, ldc], the head's columns from c_at.
    pub(in crate::cpu) ctx: *mut f32,
    pub(in crate::cpu) ldc: usize,
    pub(in crate::cpu) c_at: usize,
}

impl Attend<'_> {
    /// Per-thread floats the scratch needs for rows of up to `seq` in
    /// heads of `d`.
    pub(in crate::cpu) fn scratch(seq: usize, d: usize) -> usize {
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
    pub(in crate::cpu) unsafe fn run<V: Simd>(&self, q0: usize, q1: usize, scratch: &mut [f32]) {
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
            // The band: only a model with limited lookahead has keys past
            // it, so the whole-row path is one comparison per group.
            if self.ahead < self.len {
                for g in 0..G {
                    let from = (i + g).saturating_add(self.ahead).saturating_add(1).min(self.len);
                    sc[g * lp + from..g * lp + self.len].fill(f32::NEG_INFINITY);
                }
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
