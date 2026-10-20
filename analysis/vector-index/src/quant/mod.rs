//! TurboQuant storage: rows kept as a few bits a coordinate and scored
//! against a full-precision query without being decoded (Zandieh et al.,
//! "TurboQuant: Online Vector Quantization with Near-optimal Distortion
//! Rate", arXiv 2504.19874).
//!
//! A row is padded with zeros to a power of two `n` (at least 8) and
//! rotated by a seeded randomized Hadamard transform, which spreads its
//! length evenly so every rotated coordinate is close to normal. Each
//! coordinate is then coded independently against the Lloyd-Max grid for
//! a standard normal value, with one least-squares scale a row.
//!
//! - [`Estimator::Mse`] spends all `bits` on the grid. Its scores are the
//!   query's dot product with the decoded row, or for [`Metric::Cosine`]
//!   its cosine with the decoded row, so the shrinkage a coarse grid
//!   causes does not reorder rows of equal length.
//! - [`Estimator::InnerProduct`] spends `bits - 1` on the grid and one bit
//!   on the residual: the residual is rotated by a second, independent
//!   seeded Hadamard transform and only its signs and its length are kept
//!   (a QJL sketch). The sketch term corrects the grid's bias, so the
//!   score estimates the dot product without bias; at one bit the grid is
//!   absent and the sketch alone estimates it.
//!
//! The paper's sketch is a dense Gaussian matrix; this one is a randomized
//! Hadamard transform, `O(n log n)` instead of `O(n^2)` per row and per
//! query, whose rows are Rademacher rather than Gaussian. The record in
//! `docs/turboquant.md` measures the bias that leaves.
//!
//! Encoding is `f64` arithmetic in a fixed order and the grids are
//! constants, so the same rows, metric and [`TurboQuant`] settings give
//! the same bytes on every machine.

mod encode;
mod hadamard;
mod index;
mod levels;
mod scan;

pub use index::QuantizedIndex;

use crate::IndexError;

/// What a quantized score estimates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Estimator {
    /// All bits on the Lloyd-Max grid: the least squared error per row.
    Mse,
    /// `bits - 1` on the grid and one sign bit a coordinate on the
    /// residual: an unbiased estimate of the dot product.
    InnerProduct,
}

/// How a [`QuantizedIndex`] codes its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TurboQuant {
    /// Bits a padded coordinate, 1 to 4.
    pub bits: u8,
    pub estimator: Estimator,
    /// Names both rotations; record it with the index to reproduce it.
    pub seed: u64,
}

impl TurboQuant {
    pub fn new(bits: u8, estimator: Estimator, seed: u64) -> Result<TurboQuant, IndexError> {
        if !(1..=4).contains(&bits) {
            return Err(IndexError::Bits(bits));
        }
        Ok(TurboQuant { bits, estimator, seed })
    }

    /// The bits a coordinate spends on the grid.
    fn grid_bits(&self) -> u32 {
        match self.estimator {
            Estimator::Mse => u32::from(self.bits),
            Estimator::InnerProduct => u32::from(self.bits) - 1,
        }
    }
}

/// Where a row's bytes go: the grid codes, then the sign bits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub padded: usize,
    pub estimator: Estimator,
    pub cosine: bool,
    pub grid_bits: u32,
    pub grid_bytes: usize,
    pub row_bytes: usize,
}

impl Layout {
    fn new(dim: u32, config: TurboQuant, cosine: bool) -> Layout {
        let padded = hadamard::padded(dim);
        let grid_bits = config.grid_bits();
        let grid_bytes = padded * grid_bits as usize / 8;
        let sign_bytes = if config.estimator == Estimator::InnerProduct { padded / 8 } else { 0 };
        Layout {
            padded,
            estimator: config.estimator,
            cosine,
            grid_bits,
            grid_bytes,
            row_bytes: grid_bytes + sign_bytes,
        }
    }
}

/// The sketch's seed: the rotation's with fixed bits flipped, so the two
/// rotations draw their signs from different SplitMix64 streams.
fn sketch_seed(seed: u64) -> u64 {
    seed ^ 0xD1B5_4A32_D192_ED03
}

#[cfg(test)]
mod tests;
