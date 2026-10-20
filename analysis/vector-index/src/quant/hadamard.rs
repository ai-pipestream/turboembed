//! A seeded randomized Hadamard rotation: flip the sign of some
//! coordinates, then apply the Walsh-Hadamard transform scaled by
//! `1 / sqrt(n)`, an orthogonal map on a power-of-two length `n`. The
//! flips are the low bits of SplitMix64 outputs, 64 coordinates an
//! output, so a seed names the same rotation on every machine.

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

fn splitmix64(state: u64) -> u64 {
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The length a vector of `dim` values is padded to with zeros before it
/// is rotated: the next power of two, and at least 8 so a row's codes
/// fill whole bytes.
pub(crate) fn padded(dim: u32) -> usize {
    (dim as usize).next_power_of_two().max(8)
}

#[derive(Clone, Debug)]
pub(crate) struct Rotation {
    flip: Vec<bool>,
    scale: f64,
}

impl Rotation {
    pub(crate) fn new(n: usize, seed: u64) -> Rotation {
        debug_assert!(n.is_power_of_two());
        let mut state = seed;
        let mut bits = 0u64;
        let flip = (0..n)
            .map(|i| {
                if i % 64 == 0 {
                    state = state.wrapping_add(GOLDEN_GAMMA);
                    bits = splitmix64(state);
                }
                let f = bits & 1 == 1;
                bits >>= 1;
                f
            })
            .collect();
        Rotation { flip, scale: 1.0 / (n as f64).sqrt() }
    }

    /// Rotates `v` in place: flips, then the scaled transform.
    pub(crate) fn rotate(&self, v: &mut [f64]) {
        debug_assert_eq!(v.len(), self.flip.len());
        for (x, &f) in v.iter_mut().zip(&self.flip) {
            if f {
                *x = -*x;
            }
        }
        self.transform(v);
    }

    /// The inverse of [`Rotation::rotate`]: the transform is its own
    /// inverse, so the flips come after it.
    #[cfg(test)]
    pub(crate) fn inverse(&self, v: &mut [f64]) {
        self.transform(v);
        for (x, &f) in v.iter_mut().zip(&self.flip) {
            if f {
                *x = -*x;
            }
        }
    }

    fn transform(&self, v: &mut [f64]) {
        let n = v.len();
        let mut half = 1;
        while half < n {
            for block in v.chunks_exact_mut(2 * half) {
                let (lo, hi) = block.split_at_mut(half);
                for (a, b) in lo.iter_mut().zip(hi) {
                    let (x, y) = (*a, *b);
                    *a = x + y;
                    *b = x - y;
                }
            }
            half *= 2;
        }
        for x in v.iter_mut() {
            *x *= self.scale;
        }
    }
}
