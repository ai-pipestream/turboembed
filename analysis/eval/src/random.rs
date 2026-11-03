//! The seeded 48-bit linear congruential generator the significance
//! tests have always drawn from.
//!
//! This is the `java.util.Random` algorithm (multiplier `0x5DEECE66D`,
//! addend 11, 48-bit state) for the bounded-integer and boolean draws
//! the bootstrap and permutation tests use. Keeping the sequence keeps
//! every committed record reproducible to the bit after the dependency
//! is dropped; it is not a JVM parity claim about any model.

const MULTIPLIER: u64 = 0x5_DEEC_E66D;
const ADDEND: u64 = 0xB;
const MASK: u64 = (1 << 48) - 1;

/// A seeded generator with the historical draw sequence.
#[derive(Clone, Debug)]
pub struct JavaRandom {
    seed: u64,
}

impl JavaRandom {
    pub fn new(seed: i64) -> JavaRandom {
        JavaRandom { seed: (seed as u64 ^ MULTIPLIER) & MASK }
    }

    fn next(&mut self, bits: u32) -> i32 {
        assert!((1..=32).contains(&bits), "next({bits})");
        self.seed = (self.seed.wrapping_mul(MULTIPLIER).wrapping_add(ADDEND)) & MASK;
        (self.seed >> (48 - bits)) as i32
    }

    /// A value in `0..bound`, drawn as the JDK draws it.
    pub fn next_int_bounded(&mut self, bound: i32) -> i32 {
        assert!(bound > 0, "bound must be positive, got {bound}");
        let mut r = self.next(31);
        let m = bound - 1;
        if bound & m == 0 {
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

    /// One random bit.
    pub fn next_boolean(&mut self) -> bool {
        self.next(1) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_first_values() {
        // `new Random(0).nextInt(10)` and `new Random(42).nextInt(10)`.
        assert_eq!(JavaRandom::new(0).next_int_bounded(10), 0);
        assert_eq!(JavaRandom::new(42).next_int_bounded(10), 0);
    }

    #[test]
    fn bounded_values_are_in_range() {
        let mut r = JavaRandom::new(1);
        for bound in [1, 2, 3, 7, 16, 1000] {
            for _ in 0..1000 {
                assert!((0..bound).contains(&r.next_int_bounded(bound)));
            }
        }
    }
}
