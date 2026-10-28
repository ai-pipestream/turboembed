use super::super::gemm::tests::run_task;
use super::super::{Gemm, isas, values};
use super::*;
use crate::cpu::kernels::Epilogue;

/// Every task of `g` on `isa`.
fn run(isa: Isa, g: &Gemm) {
    for t in 0..g.tasks() {
        // SAFETY: the buffers are the shapes g gives, and the kernel is
        // one this processor has.
        unsafe { run_task(isa, g, t) };
    }
}

/// x W^T in F64.
fn plain(x: &[f32], w: &[f32], k: usize) -> Vec<f64> {
    let rows = x.len() / k;
    let n = w.len() / k;
    let mut y = vec![0.0; rows * n];
    for t in 0..rows {
        for o in 0..n {
            y[t * n + o] = (0..k).map(|i| x[t * k + i] as f64 * w[o * k + i] as f64).sum();
        }
    }
    y
}

/// The QKV product over pair-adjacent weights, rotated in its epilogue,
/// against the stored product rotated by rotate_half in F64: each packed
/// Q and K column holds the stored column it was packed from, turned.
#[test]
fn the_pair_adjacent_rotation_is_rotate_half() {
    let (h, d, tokens, seq) = (64, 16, 23, 12);
    let half = d / 2;
    let w = values(3 * h * h, 1, 0.3);
    let x = values(tokens * h, 2, 1.0);
    let positions = 16;
    let table = crate::rope::table(10000.0, 10000.0, positions, half).unwrap();
    let (cos, sin) = (table.as_ptr(), table[positions * half..].as_ptr());
    // Two rows of seq tokens, the second cut short: token t is at place
    // at[t] of the rows, its position at[t] % seq.
    let at: Vec<u32> = (0..tokens as u32).collect();
    let slot: Vec<u32> = (0..tokens as u32).map(|t| t + 1).collect();
    let stride = tokens + 3;
    let want = plain(&x, &w, h);
    let turned = |t: usize, c: usize| {
        let pos = t % seq;
        let (head, j) = (c / d * d, c % d);
        let p = j % half;
        let (cs, sn) = (table[pos * half + p] as f64, table[(positions + pos) * half + p] as f64);
        let (x1, x2) = (want[t * 3 * h + head + p], want[t * 3 * h + head + p + half]);
        if j < half { x1 * cs - x2 * sn } else { x2 * cs + x1 * sn }
    };
    for isa in isas() {
        let panels = Panels::pack(&qkv_rows(&w, h, d), h, isa).unwrap();
        let bias = vec![0.0; 3 * h];
        let mut y = vec![f32::NAN; tokens * 3 * h];
        let mut kt = vec![f32::NAN; h * stride];
        let rope =
            Rope { hidden: h, d, cos, sin, at: at.as_ptr(), seq, t_out: kt.as_mut_ptr(), stride, slot: slot.as_ptr() };
        let g = Gemm {
            x: x.as_ptr(),
            ldx: h,
            rows: tokens,
            w: &panels,
            bias: &bias,
            y: y.as_mut_ptr(),
            ldy: 3 * h,
            epilogue: Epilogue::Rope(rope),
        };
        run(isa, &g);
        for t in 0..tokens {
            for c in 0..3 * h {
                // The stored column packed column c holds.
                let (third, local) = (c / h, c % h);
                let (head, j) = (local / d * d, local % d);
                let stored = if third == 2 { c } else { third * h + head + j / 2 + (j % 2) * half };
                let (got, wanted) = match third {
                    0 => (y[t * 3 * h + c], turned(t, stored)),
                    1 => (kt[local * stride + t + 1], turned(t, stored)),
                    _ => (y[t * 3 * h + c], want[t * 3 * h + c]),
                };
                let err = (got as f64 - wanted).abs();
                assert!(err <= 1e-6 * (1.0 + wanted.abs()) * 4.0, "{isa:?} token {t} column {c}: {got} vs {wanted}");
            }
            assert!(y[t * 3 * h + h..t * 3 * h + 2 * h].iter().all(|v| v.is_nan()), "K goes to kt alone");
        }
    }
}

/// The gated epilogue over gate and up rows packed side by side, against
/// the two products taken apart, then act(gate) * up.
#[test]
fn the_gated_epilogue_is_the_two_products() {
    for (rows, k, n) in [(1, 8, 4), (13, 40, 24), (30, 64, 48), (7, 33, 70)] {
        let (gate, up) = (values(n * k, 3, 0.4), values(n * k, 4, 0.4));
        let x = values(rows * k, 5, 1.0);
        let (wg, wu) = (plain(&x, &gate, k), plain(&x, &up, k));
        for isa in isas() {
            let panels = Panels::pack(&gated_rows(&gate, &up, k), k, isa).unwrap();
            let bias = vec![0.0; 2 * n];
            for act in [Act::GeluErf, Act::Silu] {
                let mut y = vec![f32::NAN; rows * n + 3];
                let g = Gemm {
                    x: x.as_ptr(),
                    ldx: k,
                    rows,
                    w: &panels,
                    bias: &bias,
                    y: y.as_mut_ptr(),
                    ldy: n,
                    epilogue: Epilogue::Gated(act),
                };
                run(isa, &g);
                for t in 0..rows {
                    for o in 0..n {
                        let a = (wg[t * n + o] as f32, wu[t * n + o]);
                        let f = match act {
                            Act::GeluErf => gelu(a.0),
                            Act::Silu => silu(a.0),
                        };
                        let want = f as f64 * a.1;
                        let got = y[t * n + o] as f64;
                        assert!(
                            (got - want).abs() <= 1e-5 * (1.0 + want.abs()),
                            "{isa:?} {act:?} {rows}x{k}x{n} [{t}, {o}]: {got} vs {want}"
                        );
                    }
                }
                assert!(y[rows * n..].iter().all(|v| v.is_nan()), "nothing past the output is written");
            }
        }
    }
}
