//! Deterministic 48-bit linear congruential generator used for coordinate
//! descent shuffles in the dual and L1 solvers.

const MULTIPLIER: u64 = 0x5_DEEC_E66D;
const ADDEND: u64 = 0xB;
const MASK: u64 = (1 << 48) - 1;

#[derive(Clone, Debug)]
pub(crate) struct Lcg {
    seed: u64,
}

impl Lcg {
    pub(crate) fn new(seed: i64) -> Lcg {
        Lcg { seed: (seed as u64 ^ MULTIPLIER) & MASK }
    }

    #[inline]
    fn next(&mut self, bits: u32) -> i32 {
        debug_assert!((1..=32).contains(&bits));
        self.seed = (self.seed.wrapping_mul(MULTIPLIER).wrapping_add(ADDEND)) & MASK;
        (self.seed >> (48 - bits)) as i32
    }

    pub(crate) fn next_int_bounded(&mut self, bound: i32) -> i32 {
        assert!(bound > 0, "bound must be positive, got {bound}");
        let mut r = self.next(31);
        let m = bound - 1;
        if bound & m == 0 {
            r = ((i64::from(bound) * i64::from(r)) >> 31) as i32;
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
}
