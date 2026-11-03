//! Seeded 48-bit linear congruential generator used for neural weight
//! initialization, dropout masks, and sample shuffling.

const MULTIPLIER: u64 = 0x5_DEEC_E66D;
const ADDEND: u64 = 0xB;
const MASK: u64 = (1 << 48) - 1;
const DOUBLE_UNIT: f64 = 1.0 / (1u64 << 53) as f64;
const FLOAT_UNIT: f32 = 1.0 / (1u32 << 24) as f32;

/// A seeded 48-bit linear congruential random number generator.
#[derive(Clone, Debug)]
pub struct JavaRng {
    seed: u64,
    next_next_gaussian: Option<f64>,
}

impl JavaRng {
    /// Creates a new generator with the given seed.
    pub fn new(seed: i64) -> JavaRng {
        JavaRng { seed: scramble(seed), next_next_gaussian: None }
    }

    /// Resets the state to the given seed, clearing any cached Gaussian.
    pub fn set_seed(&mut self, seed: i64) {
        self.seed = scramble(seed);
        self.next_next_gaussian = None;
    }

    /// Advances the state and returns its top `bits` bits (of 48) as an `i32`.
    ///
    /// # Panics
    ///
    /// If `bits` is not in `1..=32`.
    #[inline]
    pub fn next(&mut self, bits: u32) -> i32 {
        assert!((1..=32).contains(&bits), "next({bits})");
        self.seed = (self.seed.wrapping_mul(MULTIPLIER).wrapping_add(ADDEND)) & MASK;
        (self.seed >> (48 - bits)) as i32
    }

    /// Returns 32 random bits.
    #[inline]
    pub fn next_int(&mut self) -> i32 {
        self.next(32)
    }

    /// Returns a random value in `0..bound`.
    ///
    /// # Panics
    ///
    /// If `bound` is not positive.
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

    /// Returns 64 random bits.
    #[inline]
    pub fn next_long(&mut self) -> i64 {
        let high = (self.next(32) as i64) << 32;
        high.wrapping_add(self.next(32) as i64)
    }

    /// Returns one random bit.
    #[inline]
    pub fn next_boolean(&mut self) -> bool {
        self.next(1) != 0
    }

    /// Returns a random `f32` in `[0, 1)`.
    #[inline]
    pub fn next_float(&mut self) -> f32 {
        self.next(24) as f32 * FLOAT_UNIT
    }

    /// Returns a random `f64` in `[0, 1)`.
    #[inline]
    pub fn next_double(&mut self) -> f64 {
        let high = (self.next(26) as i64) << 27;
        (high + self.next(27) as i64) as f64 * DOUBLE_UNIT
    }

    /// Returns a standard normal value by the polar method.
    pub fn next_gaussian(&mut self) -> f64 {
        if let Some(g) = self.next_next_gaussian.take() {
            return g;
        }
        let (mut v1, mut v2, mut s);
        loop {
            v1 = 2.0 * self.next_double() - 1.0;
            v2 = 2.0 * self.next_double() - 1.0;
            s = v1 * v1 + v2 * v2;
            if !(s >= 1.0 || s == 0.0) {
                break;
            }
        }
        let multiplier = (-2.0 * s.ln() / s).sqrt();
        self.next_next_gaussian = Some(v2 * multiplier);
        v1 * multiplier
    }

    /// Shuffles `items` in place from the last position down to the second.
    ///
    /// # Panics
    ///
    /// If `items` has more than `i32::MAX` items.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        let size = i32::try_from(items.len()).expect("a list holds at most i32::MAX items");
        for i in (2..=size).rev() {
            let j = self.next_int_bounded(i);
            items.swap((i - 1) as usize, j as usize);
        }
    }
}

fn scramble(seed: i64) -> u64 {
    (seed as u64 ^ MULTIPLIER) & MASK
}
