//! SplitMix64 (Steele, Lea and Flood 2014): a small seeded generator
//! with the same stream on every platform.

pub(super) struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `0..n`, `n` positive, by the high bits of a 128-bit
    /// product.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn stream_is_fixed_by_the_seed() {
        // SplitMix64's published first output for seed 0.
        assert_eq!(Rng::new(0).next_u64(), 0xe220_a839_7b1d_cdaf);
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let mut r = Rng::new(3);
        for _ in 0..1000 {
            assert!(r.below(5) < 5);
            let u = r.unit();
            assert!((0.0..1.0).contains(&u));
        }
    }
}
