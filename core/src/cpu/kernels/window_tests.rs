//! The sliding window: Attend over a group's span of keys gives the bits
//! of Attend over the whole row with the same band set to minus infinity,
//! however the queries are split into tasks, and both are the windowed
//! softmax attention.

use super::super::{Isa, isas, values};
use super::*;

/// One row and head: queries and values [len, 2d] (Q's columns first),
/// keys transposed [d, len rounded up to LANES].
struct Row {
    len: usize,
    d: usize,
    qv: Vec<f32>,
    kt: Vec<f32>,
    mask: Vec<u8>,
}

impl Row {
    fn new(len: usize, d: usize, seed: usize) -> Row {
        let lp = len.next_multiple_of(LANES);
        let mut kt = values(d * lp, seed + 1, 1.0);
        for c in 0..d {
            kt[c * lp + len..(c + 1) * lp].fill(0.0);
        }
        // Every seventh key masked, as padding inside a row would be.
        let mask = (0..len).map(|j| u8::from(j % 7 != 3)).collect();
        Row { len, d, qv: values(len * 2 * len.max(d), seed, 1.0)[..len * 2 * d].to_vec(), kt, mask }
    }

    /// Every query's context, taken `block` queries a task.
    fn attend(&self, isa: Isa, window: usize, span: bool, block: usize) -> Vec<f32> {
        let (len, d) = (self.len, self.d);
        let mut ctx = vec![f32::NAN; len * d];
        let a = Attend {
            qv: self.qv.as_ptr(),
            ld: 2 * d,
            q_at: 0,
            v_at: d,
            kt: self.kt.as_ptr(),
            stride: len.next_multiple_of(LANES),
            d,
            len,
            mask: &self.mask,
            ahead: usize::MAX,
            window,
            span,
            scale: 1.0 / (d as f32).sqrt(),
            fma: isa != Isa::Portable || cfg!(target_arch = "aarch64"),
            ctx: ctx.as_mut_ptr(),
            ldc: d,
            c_at: 0,
        };
        let mut scratch = vec![0.0; Attend::scratch(len, d)];
        for q0 in (0..len).step_by(block) {
            // SAFETY: the buffers are the shapes Attend's contract asks,
            // and the instruction set is one this processor has.
            unsafe { run(isa, &a, q0, (q0 + block).min(len), &mut scratch) };
        }
        ctx
    }

    /// The windowed attention in F64.
    fn plain(&self, window: usize) -> Vec<f64> {
        let (len, d, lp) = (self.len, self.d, self.len.next_multiple_of(LANES));
        let mut out = vec![0.0; len * d];
        for i in 0..len {
            let keys: Vec<usize> = (0..len).filter(|&j| self.mask[j] != 0 && i.abs_diff(j) <= window).collect();
            let s: Vec<f64> = keys
                .iter()
                .map(|&j| {
                    (0..d).map(|c| self.qv[i * 2 * d + c] as f64 * self.kt[c * lp + j] as f64).sum::<f64>()
                        / (d as f64).sqrt()
                })
                .collect();
            let max = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = s.iter().map(|v| (v - max).exp()).collect();
            let sum: f64 = e.iter().sum();
            for (&j, p) in keys.iter().zip(&e) {
                for c in 0..d {
                    out[i * d + c] += p / sum * self.qv[j * 2 * d + d + c] as f64;
                }
            }
        }
        out
    }
}

unsafe fn run(isa: Isa, a: &Attend, q0: usize, q1: usize, scratch: &mut [f32]) {
    unsafe {
        match isa {
            Isa::Portable => a.run::<super::super::simd::P16>(q0, q1, scratch),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => avx2(a, q0, q1, scratch),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => avx512(a, q0, q1, scratch),
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => neon(a, q0, q1, scratch),
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2(a: &Attend, q0: usize, q1: usize, scratch: &mut [f32]) {
    unsafe { a.run::<super::super::simd::Y16>(q0, q1, scratch) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vl,avx2,fma")]
unsafe fn avx512(a: &Attend, q0: usize, q1: usize, scratch: &mut [f32]) {
    unsafe { a.run::<super::super::simd::Z16>(q0, q1, scratch) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neon(a: &Attend, q0: usize, q1: usize, scratch: &mut [f32]) {
    unsafe { a.run::<super::super::simd::N16>(q0, q1, scratch) }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn the_window_s_span_gives_the_whole_row_s_bits() {
    for (len, d) in [(5, 16), (17, 16), (40, 32), (100, 16), (130, 24)] {
        let row = Row::new(len, d, len);
        for window in [1, 4, 8, 20, 64, usize::MAX] {
            let want = row.plain(window);
            for isa in isas() {
                let full = row.attend(isa, window, false, 32);
                for block in [1, 3, 7, 32] {
                    let span = row.attend(isa, window, true, block);
                    assert_eq!(bits(&span), bits(&full), "{isa:?} len {len} d {d} window {window} block {block}");
                }
                for (i, (&got, &w)) in full.iter().zip(&want).enumerate() {
                    assert!((got as f64 - w).abs() < 1e-5, "{isa:?} len {len} window {window} [{i}]: {got} vs {w}");
                }
            }
        }
    }
}
