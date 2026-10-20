//! Encoding one row into codes, in `f64` with every sum in a fixed
//! order, so a row's bytes are the same on every machine.
//!
//! A row of `n` padded values is rotated, and each rotated coordinate,
//! standardized to unit variance (`x_i * sqrt(n) / |x|`), is coded to the
//! nearest level of the grid. One scale per row then fits the levels to
//! the row by least squares. With the inner-product estimator the
//! residual of that fit is rotated again by an independent rotation and
//! kept as one sign bit a coordinate, with its length.

use super::hadamard::Rotation;
use super::levels::Grid;
use super::{Estimator, Layout};

/// The scratch space of an encoder, reused across rows.
pub(crate) struct Encoder<'a> {
    pub layout: &'a Layout,
    pub grid: Option<&'a Grid>,
    pub rotation: &'a Rotation,
    pub sketch: &'a Rotation,
    rotated: Vec<f64>,
    residual: Vec<f64>,
    codes: Vec<u32>,
}

impl<'a> Encoder<'a> {
    pub(crate) fn new(
        layout: &'a Layout,
        grid: Option<&'a Grid>,
        rotation: &'a Rotation,
        sketch: &'a Rotation,
    ) -> Self {
        let n = layout.padded;
        Encoder { layout, grid, rotation, sketch, rotated: vec![0.0; n], residual: vec![0.0; n], codes: vec![0; n] }
    }

    /// Encodes `row`, nonzero and finite, into `out` (`layout.row_bytes`
    /// bytes, zeroed) and returns the row's two factors: what the grid sum
    /// is multiplied by, and what the sign sum is multiplied by.
    pub(crate) fn encode(&mut self, row: &[f32], out: &mut [u8]) -> [f32; 2] {
        let l = self.layout;
        let n = l.padded;
        self.rotated.fill(0.0);
        let mut squares = 0.0f64;
        for (r, &v) in self.rotated.iter_mut().zip(row) {
            *r = f64::from(v);
            squares += f64::from(v) * f64::from(v);
        }
        let norm = squares.sqrt();
        self.rotation.rotate(&mut self.rotated);
        let root_n = (n as f64).sqrt();
        let mut grid_factor = 0.0f64;
        self.residual.copy_from_slice(&self.rotated);
        if let Some(grid) = self.grid {
            let standardize = root_n / norm;
            let levels = grid.levels();
            let (mut dot, mut squares) = (0.0f64, 0.0f64);
            for (c, &x) in self.codes.iter_mut().zip(&self.rotated) {
                let z = x * standardize;
                *c = grid.encode(z);
                let g = f64::from(levels[*c as usize]);
                dot += z * g;
                squares += g * g;
            }
            let fitted = if dot > 0.0 { dot / squares } else { 1.0 };
            let scale = norm / root_n * fitted;
            for (e, &c) in self.residual.iter_mut().zip(&self.codes) {
                *e -= scale * f64::from(levels[c as usize]);
            }
            pack(&self.codes, l.grid_bits, &mut out[..l.grid_bytes]);
            grid_factor = match l.estimator {
                // The cosine of the query with the decoded row, whose length
                // is `scale * sqrt(squares)`; the scale cancels.
                Estimator::Mse if l.cosine => 1.0 / squares.sqrt(),
                _ => scale,
            };
        }
        let mut sign_factor = 0.0f64;
        if l.estimator == Estimator::InnerProduct {
            let length = self.residual.iter().map(|e| e * e).sum::<f64>().sqrt();
            self.sketch.rotate(&mut self.residual);
            let signs = &mut out[l.grid_bytes..];
            for (j, &w) in self.residual.iter().enumerate() {
                if w >= 0.0 {
                    signs[j / 8] |= 1 << (j % 8);
                }
            }
            // <q, e> is estimated by |e| sqrt(pi / 2) / sqrt(n) times the sum
            // over the sketch rows of sign(<s_j, e>) <s_j, q>.
            sign_factor = length * (std::f64::consts::PI / 2.0).sqrt() / root_n;
        }
        [grid_factor as f32, sign_factor as f32]
    }
}

/// Packs `codes` of `bits` bits each, little-endian: code `i` occupies bits
/// `[i * bits, (i + 1) * bits)` of `out`.
pub(crate) fn pack(codes: &[u32], bits: u32, out: &mut [u8]) {
    let mut acc = 0u64;
    let mut filled = 0u32;
    let mut at = 0;
    for &c in codes {
        acc |= u64::from(c) << filled;
        filled += bits;
        while filled >= 8 {
            out[at] = acc as u8;
            at += 1;
            acc >>= 8;
            filled -= 8;
        }
    }
    debug_assert_eq!(filled, 0);
    debug_assert_eq!(at, out.len());
}
