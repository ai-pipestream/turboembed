//! TurboQuant storage: rows kept as 2 to 4 bits a coordinate and scored
//! against a full-precision query without being decoded (Zandieh et al.,
//! "TurboQuant: Online Vector Quantization with Near-optimal Distortion
//! Rate", arXiv 2504.19874). Built with the `turboquant` feature.
//!
//! The codes, their layout and the search kernels are the `turbovec`
//! crate's (pinned to one commit): a fixed random rotation, Lloyd-Max
//! codes for the rotated coordinates, a stored length that removes the
//! codes' shrinkage from the inner product, and SIMD scans on AVX-512,
//! AVX2 and NEON. This module keeps what the exact index promises around
//! it: the [`crate::Space`] of the rows and the refusal of a query from
//! another one, the rows' metadata, and the vectors the builder refuses.
//!
//! With [`TurboQuant::calibrate`] the index first fits turbovec's
//! per-coordinate calibration (TQ+) on up to 1024 of its rows, taken at an
//! even stride, so the same rows always give the same calibration.

mod index;

pub use index::QuantizedIndex;

use crate::IndexError;

/// How a [`QuantizedIndex`] codes its rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TurboQuant {
    /// Bits a coordinate, 2 to 4.
    pub bits: u8,
    /// Fit the per-coordinate calibration on the rows before coding them.
    pub calibrate: bool,
}

impl TurboQuant {
    pub fn new(bits: u8, calibrate: bool) -> Result<TurboQuant, IndexError> {
        if !(2..=4).contains(&bits) {
            return Err(IndexError::Bits(bits));
        }
        Ok(TurboQuant { bits, calibrate })
    }
}

/// The most rows the calibration is fitted on.
const CALIBRATION_ROWS: usize = 1024;
