use super::super::kernels::Isa;
use super::round::{pairwise_squares, round_f16, round_f16_f16c};
use super::sum::{F16, mean, mean_f16c, mean_quantized, mean_quantized_avx2};
use super::*;

/// numpy's pairwise sum of squares as its loops read, one running sum
/// at a time.
fn numpy_squares(x: &[f32]) -> f32 {
    let n = x.len();
    if n < 8 {
        x.iter().fold(0f32, |res, &v| res + v * v)
    } else if n <= 128 {
        let mut r = [0f32; 8];
        for k in 0..8 {
            r[k] = x[k] * x[k];
        }
        let mut i = 8;
        while i < n - n % 8 {
            for k in 0..8 {
                r[k] += x[i + k] * x[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for &v in &x[i..] {
            res += v * v;
        }
        res
    } else {
        let n2 = n / 2 - n / 2 % 8;
        numpy_squares(&x[..n2]) + numpy_squares(&x[n2..])
    }
}

/// The norm's sum of squares is numpy's, bit for bit, at every length
/// up to well past a split.
#[test]
fn the_sum_of_squares_is_numpy_s() {
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let x: Vec<f32> = (0..700)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect();
    for n in 0..x.len() {
        assert_eq!(pairwise_squares(&x[..n]).to_bits(), numpy_squares(&x[..n]).to_bits(), "{n} values");
    }
}

/// F16C's rounding to F16 against the portable one, bit for bit, over
/// values of every exponent F16 has and beyond.
#[test]
fn the_f16c_rounding_is_the_portable_rounding() {
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
        return;
    }
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut values: Vec<f32> = (0..100_003)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            // Exponents from 2^-30 to 2^20, either sign, any mantissa.
            let bits = (seed as u32 & 0x807f_ffff) | ((97 + (seed >> 32) % 51) as u32) << 23;
            f32::from_bits(bits)
        })
        .collect();
    values.extend([0.0, -0.0, 65504.0, 65519.0, 65520.0, 1e-8, 5.96e-8, 2.98e-8, 2.99e-8, f32::INFINITY]);
    let mut want = values.clone();
    round_f16(&mut want);
    unsafe { round_f16_f16c(&mut values) };
    for (g, w) in values.iter().zip(&want) {
        assert_eq!(g.to_bits(), w.to_bits(), "{g} vs {w}");
    }
}

/// The AVX2 I8 mean against the portable one, with and without
/// weights, on widths that take each of its loops. They differ only by
/// the fused multiply-add's roundings.
#[test]
fn the_avx2_quantized_mean_is_the_portable_one() {
    if !Isa::Avx2.available() {
        return;
    }
    for dim in [1, 7, 8, 63, 64, 77, 136, 256, 300] {
        let vocab = 40;
        let values: Vec<f32> = (0..vocab * dim).map(|i| ((i * 7919 % 1013) as f32 - 506.0) / 97.0).collect();
        for weights in [None, Some((0..vocab).map(|i| 0.25 + i as f64 / 16.0).collect::<Box<[f64]>>())] {
            let t = Table {
                rows: values.as_ptr() as *const u8,
                dtype: crate::TURBO_DTYPE_F32,
                dim,
                height: vocab,
                weights,
                mapping: None,
                wide: false,
                _copy: None,
            };
            let q = Quantized::new(&t, false).unwrap();
            let ids: Vec<u32> = (0..23).map(|i| (i * 17 % vocab) as u32).collect();
            let (mut want, mut got) = (vec![0f32; dim], vec![0f32; dim]);
            mean_quantized(&t, &q, &ids, &mut want);
            unsafe { mean_quantized_avx2(&t, &q, &ids, &mut got) };
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-5 * (1.0 + w.abs()), "dim {dim}: {g} vs {w}");
            }
        }
    }
}

/// The F16C mean against the portable one, on widths that take each of
/// its loops, over halves of every exponent: the same bits.
#[test]
fn the_f16c_mean_is_the_portable_mean() {
    if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("f16c") {
        return;
    }
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for dim in [1, 7, 8, 63, 64, 77, 136, 256, 300] {
        let vocab = 50;
        // Finite halves only: exponent 31 is Inf and NaN.
        let halves: Vec<u16> = (0..vocab * dim)
            .map(|_| {
                let h = next() as u16;
                if (h >> 10) & 0x1f == 0x1f { h & !0x4000 } else { h }
            })
            .collect();
        let t = Table {
            rows: halves.as_ptr() as *const u8,
            dtype: TURBO_DTYPE_F16,
            dim,
            height: vocab,
            weights: None,
            mapping: None,
            wide: false,
            _copy: None,
        };
        for len in [1, 2, 5, 33] {
            let ids: Vec<u32> = (0..len).map(|_| (next() % vocab as u64) as u32).collect();
            let (mut want, mut got) = (vec![0f32; dim], vec![0f32; dim]);
            mean::<F16>(&t, &ids, &mut want);
            unsafe { mean_f16c(&t, &ids, &mut got) };
            let (w, g): (Vec<u32>, Vec<u32>) =
                (want.iter().map(|v| v.to_bits()).collect(), got.iter().map(|v| v.to_bits()).collect());
            assert_eq!(w, g, "dim {dim}, {len} ids");
        }
    }
}
