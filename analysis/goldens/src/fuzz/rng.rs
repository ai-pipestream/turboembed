//! The generator's own random source: SplitMix64, a few lines long, so the
//! corpus depends on nothing outside this crate and the same seed gives
//! the same texts on every machine and every build.

/// A SplitMix64 generator. The state advances by the golden ratio
/// constant and each output is the mixed state, as in the reference
/// (Steele, Lea and Flood, 2014).
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    /// The generator for input `index` of a corpus made with `seed`: the
    /// two are mixed so that neighbouring indices do not give overlapping
    /// sequences, and so that one input can be made again without the
    /// ones before it.
    pub fn for_input(seed: u64, index: u64) -> Rng {
        let mut r = Rng::new(seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(23));
        r.next_u64();
        r
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in `0..n`. `n` is never zero here; the bias of the modulo
    /// is of no account for a text generator.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0)");
        (self.next_u64() % n as u64) as usize
    }

    /// A number in `low..=high`.
    pub fn range(&mut self, low: usize, high: usize) -> usize {
        low + self.below(high - low + 1)
    }

    /// True with the given probability in percent.
    pub fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_gives_the_same_sequence() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    }

    #[test]
    fn the_reference_value_of_splitmix64_from_seed_zero() {
        // The first output of SplitMix64 seeded with 0 in the reference
        // implementation, which most ports print as a check.
        assert_eq!(Rng::new(0).next_u64(), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn inputs_of_one_corpus_do_not_share_a_stream() {
        let a: Vec<u64> = (0..8).map(|_| Rng::for_input(7, 0).next_u64()).collect();
        let b: Vec<u64> = (0..8).map(|_| Rng::for_input(7, 1).next_u64()).collect();
        assert_ne!(a, b);
        assert!(Rng::for_input(7, 3).below(10) < 10);
        assert!((5..=9).contains(&Rng::for_input(7, 3).range(5, 9)));
    }
}
