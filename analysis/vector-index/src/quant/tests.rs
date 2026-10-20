use super::encode::pack;
use super::hadamard::{Rotation, padded};

/// Values from a fixed LCG in `[-1, 1)`.
fn values(seed: u64, n: usize) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (s >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        })
        .collect()
}

#[test]
fn padding_is_a_power_of_two_of_at_least_8() {
    assert_eq!([1, 7, 8, 9, 384, 512, 513].map(padded), [8, 8, 8, 16, 512, 512, 1024]);
}

#[test]
fn the_rotation_is_orthogonal_and_inverts() {
    for n in [8, 64, 512] {
        let r = Rotation::new(n, 42);
        let x = values(n as u64, n);
        let mut y = x.clone();
        r.rotate(&mut y);
        let len = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>();
        assert!((len(&x) - len(&y)).abs() < 1e-12 * len(&x));
        r.inverse(&mut y);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }
}

#[test]
fn seeds_pick_different_flips() {
    let a = Rotation::new(64, 1);
    let b = Rotation::new(64, 2);
    let mut x = vec![0.0; 64];
    x[0] = 1.0;
    let (mut ya, mut yb) = (x.clone(), x);
    a.rotate(&mut ya);
    b.rotate(&mut yb);
    assert_ne!(ya, yb);
}

#[test]
fn the_flip_stream_is_splitmix64() {
    // SplitMix64 from state 0: its first output is 0xE220A8397B1DCDAF, so
    // the flips of seed 0 are that word's bits, low bit first.
    let r = Rotation::new(64, 0);
    let mut x = vec![0.0; 64];
    x[0] = 1.0;
    let mut y = x.clone();
    r.rotate(&mut y);
    // A unit vector at coordinate 0 rotates to +-1/8 everywhere; its sign
    // is the flip of coordinate 0 (bit 0 of the word is 1: flipped).
    assert!(y.iter().all(|&v| (v + 0.125).abs() < 1e-15));
    let word = 0xE220_A839_7B1D_CDAFu64;
    for i in 0..64 {
        let mut e = vec![0.0; 64];
        e[i] = 1.0;
        r.rotate(&mut e);
        let flipped = word >> i & 1 == 1;
        assert_eq!(e[0] < 0.0, flipped, "coordinate {i}");
    }
}

#[test]
fn packing_is_little_endian_and_dense() {
    let mut out = [0u8; 3];
    pack(&[1, 2, 3, 4, 5, 6, 7, 0], 3, &mut out);
    // 001 010 011 100 101 110 111 000, first code in the lowest bits.
    let bits = u32::from(out[0]) | u32::from(out[1]) << 8 | u32::from(out[2]) << 16;
    for (i, c) in [1, 2, 3, 4, 5, 6, 7, 0].iter().enumerate() {
        assert_eq!(bits >> (3 * i) & 7, *c);
    }
    let mut out = [0u8; 1];
    pack(&[1, 0, 1, 1, 0, 0, 0, 1], 1, &mut out);
    assert_eq!(out[0], 0b1000_1101);
}
