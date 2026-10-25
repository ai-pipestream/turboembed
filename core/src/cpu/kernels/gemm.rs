//! The linear layers: a product over packed weights, split into tiles,
//! and what is done with each output once its sum is complete.

use super::Panels;
use super::micro::{ACC, Micro};
use super::rowwise::gelu;

/// What is done with each output of a product once its sum is complete.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::cpu) enum Epilogue {
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
pub(in crate::cpu) struct Gemm<'a> {
    pub(in crate::cpu) x: *const f32,
    pub(in crate::cpu) ldx: usize,
    pub(in crate::cpu) rows: usize,
    pub(in crate::cpu) w: &'a Panels,
    pub(in crate::cpu) bias: &'a [f32],
    pub(in crate::cpu) y: *mut f32,
    pub(in crate::cpu) ldy: usize,
    pub(in crate::cpu) epilogue: Epilogue,
}

/// Micro-kernel tiles of rows per task.
pub(super) const MB_TILES: usize = 4;
/// Panels per task.
pub(super) const NB_PANELS: usize = 2;

impl Gemm<'_> {
    pub(in crate::cpu) fn tasks(&self) -> usize {
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
    pub(in crate::cpu) unsafe fn task<M: Micro>(&self, task: usize) {
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

#[cfg(test)]
mod tests {
    use super::super::{Isa, isas, values};
    use super::*;

    /// Every kernel and epilogue against the plain sum, for shapes with
    /// and without remainders in each dimension, over every task.
    #[test]
    fn linear_kernels_match_the_plain_sum() {
        for (rows, n_in, n_out) in
            [(1, 1, 1), (3, 7, 5), (4, 8, 2), (9, 37, 11), (13, 64, 130), (5, 1536, 9), (70, 40, 97)]
        {
            let x = values(rows * n_in, 1, 1.0);
            let w = values(n_out * n_in, 2, 1.0);
            let b = values(n_out, 3, 1.0);
            let y0 = values(rows * n_out, 4, 1.0);
            for isa in isas() {
                let panels = Panels::pack(&[&w], n_in, isa).unwrap();
                // Split's transposed columns, a third to two thirds: each
                // row's slot one past its index.
                let (lo, hi, stride) = (n_out / 3, 2 * n_out / 3, rows + 3);
                let slot: Vec<u32> = (1..=rows as u32).collect();
                let mut kt = vec![f32::NAN; (hi - lo) * stride];
                let split = Epilogue::Split { lo, hi, t_out: kt.as_mut_ptr(), stride, slot: slot.as_ptr() };
                for epilogue in [Epilogue::Gelu, Epilogue::Residual, split] {
                    let mut y = y0.clone();
                    y.extend([f32::NAN; 3]);
                    let g = Gemm {
                        x: x.as_ptr(),
                        ldx: n_in,
                        rows,
                        w: &panels,
                        bias: &b,
                        y: y.as_mut_ptr(),
                        ldy: n_out,
                        epilogue,
                    };
                    for t in 0..g.tasks() {
                        // SAFETY: the buffers are the shapes given, and
                        // the kernel is one this processor has.
                        unsafe { run_task(isa, &g, t) };
                    }
                    for t in 0..rows {
                        for o in 0..n_out {
                            let s = b[o] as f64
                                + (0..n_in).map(|i| x[t * n_in + i] as f64 * w[o * n_in + i] as f64).sum::<f64>();
                            let (want, got) = match epilogue {
                                Epilogue::Gelu => (gelu(s as f32) as f64, y[t * n_out + o]),
                                Epilogue::Residual => (y0[t * n_out + o] as f64 + s, y[t * n_out + o]),
                                Epilogue::Split { lo, hi, .. } if (lo..hi).contains(&o) => {
                                    assert_eq!(y[t * n_out + o], y0[t * n_out + o], "these go to t_out only");
                                    (s, kt[(o - lo) * stride + t + 1])
                                }
                                Epilogue::Split { .. } => (s, y[t * n_out + o]),
                            };
                            let got = got as f64;
                            let tol = 1e-4 * (1.0 + want.abs()) + if epilogue == Epilogue::Gelu { 1e-4 } else { 0.0 };
                            assert!(
                                (got - want).abs() < tol,
                                "{isa:?} {epilogue:?} {rows}x{n_in}x{n_out} [{t}, {o}]: {got} vs {want}"
                            );
                        }
                    }
                    assert!(y[rows * n_out..].iter().all(|v| v.is_nan()), "nothing past the output is written");
                }
            }
        }
    }

    unsafe fn run_task(isa: Isa, g: &Gemm, t: usize) {
        unsafe {
            match isa {
                Isa::Portable => g.task::<super::super::Portable>(t),
                #[cfg(target_arch = "x86_64")]
                Isa::Avx2 => avx2(g, t),
                #[cfg(target_arch = "x86_64")]
                Isa::Avx512 => avx512(g, t),
                #[cfg(target_arch = "aarch64")]
                Isa::Neon => neon(g, t),
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn avx2(g: &Gemm, t: usize) {
        unsafe { g.task::<super::super::Avx2>(t) }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512vl,avx2,fma")]
    unsafe fn avx512(g: &Gemm, t: usize) {
        unsafe { g.task::<super::super::Avx512>(t) }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn neon(g: &Gemm, t: usize) {
        unsafe { g.task::<super::super::Neon>(t) }
    }
}
