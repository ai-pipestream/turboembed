//! Inverted dropout with a seeded mask: each value is dropped with
//! probability p, the rest scaled by 1 / (1 - p), so the expectation
//! stays. The mask is a SplitMix64 stream from a key the caller derives
//! from the run's seed, the example's place in the run, the layer and
//! the branch, so it is the same bits on any thread, in any lane, and
//! backward draws it again instead of keeping it.

use crate::real::Real;
use crate::rng::Rng;

/// The rates of the three places ModernBERT drops: after the embedding
/// norm, on attention's output projection, and on the MLP's activated
/// hidden before its output projection; and the run's seed. A rate of 0
/// drops nothing there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dropout {
    pub embedding: f64,
    pub attention: f64,
    pub mlp: f64,
    pub seed: u64,
}

impl Dropout {
    /// Whether every rate is a probability below 1.
    pub fn valid(&self) -> bool {
        [self.embedding, self.attention, self.mlp].iter().all(|p| (0.0..1.0).contains(p))
    }
}

/// The key of one mask: the run's seed, the example's place `at`, the
/// layer (the embedding is u32::MAX) and the branch, mixed so that
/// neighbouring places give unrelated streams.
pub fn key(seed: u64, at: u64, layer: u32, branch: u32) -> u64 {
    let mut r = Rng::new(seed ^ at.rotate_left(17) ^ ((layer as u64) << 40) ^ ((branch as u64) << 56));
    r.next_u64()
}

/// Drops `x` in place under `p` and `key`; nothing happens at p = 0.
pub fn forward<R: Real>(x: &mut [R], p: f64, key: u64) {
    if p <= 0.0 {
        return;
    }
    let mut rng = Rng::new(key);
    let scale = R::from_f64(1.0 / (1.0 - p));
    for v in x {
        *v = if rng.unit() < p { R::ZERO } else { *v * scale };
    }
}

/// The gradient of `forward`: the same mask over `dx`, in place.
pub fn backward<R: Real>(dx: &mut [R], p: f64, key: u64) {
    forward(dx, p, key);
}
