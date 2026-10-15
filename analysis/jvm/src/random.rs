//! `java.util.Random` and `Collections.shuffle(List, Random)`.
//!
//! OpenNLP shuffles with a seeded `java.util.Random` in its readers,
//! cross-validation and trainers, so reproducing a training run or an
//! evaluation split needs the same sequence. [`JavaRandom`] is the JDK 21
//! generator: a 48-bit linear congruential generator (multiplier
//! 0x5DEECE66D, addend 11) and the methods built on it, each written as the
//! JDK writes it, with Java's 32- and 64-bit wrapping arithmetic.

use crate::strict_math;

const MULTIPLIER: u64 = 0x5_DEEC_E66D;
const ADDEND: u64 = 0xB;
const MASK: u64 = (1 << 48) - 1;
/// 2^-53, the step between the doubles `next_double` returns.
const DOUBLE_UNIT: f64 = 1.0 / (1u64 << 53) as f64;
/// 2^-24, the step between the floats `next_float` returns.
const FLOAT_UNIT: f32 = 1.0 / (1u32 << 24) as f32;

/// A `java.util.Random` made with a seed (`new Random(seed)`): the same
/// seed gives the same values from each method as the JVM does.
#[derive(Clone, Debug)]
pub struct JavaRandom {
    /// The 48-bit state.
    seed: u64,
    /// The second value of the last pair `next_gaussian` made, while it
    /// has not been returned.
    next_next_gaussian: Option<f64>,
}

impl JavaRandom {
    /// `new Random(seed)`.
    pub fn new(seed: i64) -> JavaRandom {
        JavaRandom { seed: scramble(seed), next_next_gaussian: None }
    }

    /// `setSeed(seed)`: the state of `new Random(seed)`, including
    /// forgetting a cached Gaussian.
    pub fn set_seed(&mut self, seed: i64) {
        self.seed = scramble(seed);
        self.next_next_gaussian = None;
    }

    /// `next(bits)`: advances the state and returns its top `bits` bits
    /// (of 48), as an `i32` the way Java's `(int)` cast makes it, so
    /// `next(32)` can be negative.
    ///
    /// # Panics
    ///
    /// In debug builds, if `bits` is not in 1..=32.
    #[inline]
    pub fn next(&mut self, bits: u32) -> i32 {
        debug_assert!((1..=32).contains(&bits), "next({bits})");
        self.seed = (self.seed.wrapping_mul(MULTIPLIER).wrapping_add(ADDEND)) & MASK;
        (self.seed >> (48 - bits)) as i32
    }

    /// `nextInt()`: 32 random bits.
    pub fn next_int(&mut self) -> i32 {
        self.next(32)
    }

    /// `nextInt(bound)`: a value in `0..bound`.
    ///
    /// When `bound` is a power of two it takes the high bits of one
    /// `next(31)`. Otherwise it takes `next(31) % bound` and draws again
    /// while the draw falls in the last, incomplete multiple of `bound`
    /// (detected, as in the JDK, by `u - r + (bound - 1)` overflowing).
    ///
    /// # Panics
    ///
    /// If `bound` is not positive, where Java throws
    /// `IllegalArgumentException`.
    pub fn next_int_bounded(&mut self, bound: i32) -> i32 {
        assert!(bound > 0, "bound must be positive, got {bound}");
        let mut r = self.next(31);
        let m = bound - 1;
        if bound & m == 0 {
            // A power of two.
            r = ((bound as i64 * r as i64) >> 31) as i32;
        } else {
            let mut u = r;
            loop {
                r = u % bound;
                if u.wrapping_sub(r).wrapping_add(m) >= 0 {
                    break;
                }
                u = self.next(31);
            }
        }
        r
    }

    /// `nextLong()`: two `next(32)` values, the first shifted into the high
    /// half and the second, sign extended, added to it.
    pub fn next_long(&mut self) -> i64 {
        let high = (self.next(32) as i64) << 32;
        high.wrapping_add(self.next(32) as i64)
    }

    /// `nextBoolean()`: one random bit.
    pub fn next_boolean(&mut self) -> bool {
        self.next(1) != 0
    }

    /// `nextFloat()`: 24 random bits times 2^-24, in [0, 1).
    pub fn next_float(&mut self) -> f32 {
        self.next(24) as f32 * FLOAT_UNIT
    }

    /// `nextDouble()`: 53 random bits (26 and then 27) times 2^-53, in
    /// [0, 1).
    pub fn next_double(&mut self) -> f64 {
        let high = (self.next(26) as i64) << 27;
        (high + self.next(27) as i64) as f64 * DOUBLE_UNIT
    }

    /// `nextGaussian()`: a standard normal value, by the polar method
    /// (Knuth, TAOCP vol. 2, 3.4.1, algorithm C) as the JDK does it. Each
    /// accepted pair gives two values; the second is returned by the next
    /// call.
    ///
    /// The JDK computes the scale with `StrictMath.log` and
    /// `StrictMath.sqrt`. Square root is correctly rounded in both, so
    /// `f64::sqrt` gives the same bits; the logarithm is
    /// [`strict_math::log`].
    pub fn next_gaussian(&mut self) -> f64 {
        if let Some(g) = self.next_next_gaussian.take() {
            return g;
        }
        let (mut v1, mut v2, mut s);
        loop {
            v1 = 2.0 * self.next_double() - 1.0; // between -1 and 1
            v2 = 2.0 * self.next_double() - 1.0; // between -1 and 1
            s = v1 * v1 + v2 * v2;
            if !(s >= 1.0 || s == 0.0) {
                break;
            }
        }
        let multiplier = (-2.0 * strict_math::log(s) / s).sqrt();
        self.next_next_gaussian = Some(v2 * multiplier);
        v1 * multiplier
    }

    /// `Collections.shuffle(list, random)`: from the last position down to
    /// the second, swaps each item with one at a position drawn by
    /// `nextInt(i + 1)` (the item itself included).
    ///
    /// # Panics
    ///
    /// If `items` has more than `i32::MAX` items, more than a Java list can
    /// hold.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        let size = i32::try_from(items.len()).expect("a Java list holds at most i32::MAX items");
        for i in (2..=size).rev() {
            let j = self.next_int_bounded(i);
            items.swap((i - 1) as usize, j as usize);
        }
    }
}

/// The JDK's `initialScramble`: the seed XOR the multiplier, kept to 48
/// bits.
fn scramble(seed: i64) -> u64 {
    (seed as u64 ^ MULTIPLIER) & MASK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_first_values() {
        // What new Random(42).nextInt(), new Random(0).nextInt(10) and
        // new Random(42).nextInt(10) return on JDK 21.
        assert_eq!(JavaRandom::new(42).next_int(), -1170105035);
        assert_eq!(JavaRandom::new(0).next_int_bounded(10), 0);
        assert_eq!(JavaRandom::new(42).next_int_bounded(10), 0);
    }

    #[test]
    fn set_seed_restarts_and_forgets_the_cached_gaussian() {
        let mut a = JavaRandom::new(7);
        let first = a.next_gaussian();
        a.set_seed(7);
        assert_eq!(a.next_gaussian().to_bits(), first.to_bits());
    }

    #[test]
    fn gaussian_returns_the_second_value_of_a_pair_next() {
        let mut a = JavaRandom::new(3);
        let mut b = a.clone();
        a.next_gaussian();
        // The second value takes no draws: b, after two doubles per try,
        // is where a is now.
        let before = a.seed;
        a.next_gaussian();
        assert_eq!(a.seed, before);
        b.next_gaussian();
        b.next_gaussian();
        assert_eq!(a.seed, b.seed);
    }

    #[test]
    fn bounded_values_are_in_range() {
        let mut r = JavaRandom::new(1);
        for bound in [1, 2, 3, 7, 16, 1000, (1 << 30) + 1, i32::MAX] {
            for _ in 0..1000 {
                let v = r.next_int_bounded(bound);
                assert!((0..bound).contains(&v));
            }
        }
    }

    #[test]
    fn shuffle_gives_the_jdk_s_permutations() {
        // Collections.shuffle of the lists 0..10, 0..20 and 0..5 with
        // new Random(42), new Random(429) and new Random(9) on JDK 21.
        for (seed, len, jdk) in [
            (42, 10, &[4, 6, 2, 1, 7, 9, 8, 5, 3, 0][..]),
            (429, 20, &[17, 0, 6, 8, 7, 1, 13, 9, 16, 2, 15, 3, 5, 12, 14, 4, 11, 19, 10, 18][..]),
            (9, 5, &[0, 2, 3, 1, 4][..]),
        ] {
            let mut items: Vec<u32> = (0..len).collect();
            JavaRandom::new(seed).shuffle(&mut items);
            assert_eq!(items, jdk, "seed {seed}");
        }
    }

    #[test]
    fn shuffle_is_a_permutation_and_leaves_short_lists() {
        let mut items: Vec<u32> = (0..50).collect();
        JavaRandom::new(9).shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort();
        assert_eq!(sorted, (0..50).collect::<Vec<_>>());
        assert_ne!(items, sorted);
        let mut one = [5];
        let mut r = JavaRandom::new(9);
        r.shuffle(&mut one);
        r.shuffle::<u8>(&mut []);
        assert_eq!(one, [5]);
        // Nothing was drawn.
        assert_eq!(r.seed, scramble(9));
    }
}
