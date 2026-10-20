//! Scoring a query against packed codes without decoding them. For each
//! query the scan builds lookup tables: for every byte (or 6-bit field)
//! position of a row's grid codes, the query's partial dot product with
//! the levels each value of that byte stands for; and for every byte of
//! sign bits, the signed sum of the sketched query over those 8 signs. A
//! row's score is then one table read per byte, summed in four lanes
//! (byte `j` in lane `j % 4`, lanes added `(0 + 1) + (2 + 3)`).

use super::hadamard::Rotation;
use super::levels::Grid;
use super::{Estimator, Layout};

/// A query as the scan reads it.
pub(crate) struct Tables {
    /// For 1, 2 and 4 grid bits: 256 entries a byte. For 3 bits: 64 entries
    /// a 6-bit field of two codes, four fields to three bytes.
    grid: Vec<f32>,
    signs: Vec<[f32; 256]>,
}

impl Tables {
    /// The tables of `query`, `dim` values already prepared as the index
    /// stores its rows.
    pub(crate) fn new(
        layout: &Layout,
        grid: Option<&Grid>,
        rotation: &Rotation,
        sketch: &Rotation,
        query: &[f32],
    ) -> Tables {
        let n = layout.padded;
        let mut q = vec![0.0f64; n];
        for (x, &v) in q.iter_mut().zip(query) {
            *x = f64::from(v);
        }
        rotation.rotate(&mut q);
        let mut out = Tables { grid: Vec::new(), signs: Vec::new() };
        if let Some(grid) = grid {
            let levels = grid.levels();
            let bits = layout.grid_bits as usize;
            // Codes a field and the field's width in bits.
            let (per, width) = if bits == 3 { (2, 6) } else { (8 / bits, 8) };
            let mask = (1usize << bits) - 1;
            out.grid.reserve((n / per) << width);
            for coords in q.chunks_exact(per) {
                for value in 0..1usize << width {
                    let mut s = 0.0f64;
                    for (t, &x) in coords.iter().enumerate() {
                        s += x * f64::from(levels[(value >> (t * bits)) & mask]);
                    }
                    out.grid.push(s as f32);
                }
            }
        }
        if layout.estimator == Estimator::InnerProduct {
            sketch.rotate(&mut q);
            for coords in q.as_chunks::<8>().0 {
                let mut table = [0.0f32; 256];
                for (value, slot) in table.iter_mut().enumerate() {
                    let mut s = 0.0f64;
                    for (t, &x) in coords.iter().enumerate() {
                        s += if value >> t & 1 == 1 { x } else { -x };
                    }
                    *slot = s as f32;
                }
                out.signs.push(table);
            }
        }
        out
    }

    /// The score of one row: its packed bytes and its two factors.
    #[inline]
    pub(crate) fn score(&self, layout: &Layout, bytes: &[u8], factors: [f32; 2]) -> f32 {
        let (grid, signs) = bytes.split_at(layout.grid_bytes);
        let mut s = 0.0f32;
        if layout.grid_bits == 3 {
            s = factors[0] * six_bit_sum(self.grid.as_chunks::<64>().0, grid);
        } else if layout.grid_bits > 0 {
            s = factors[0] * byte_sum(self.grid.as_chunks::<256>().0, grid);
        }
        if layout.estimator == Estimator::InnerProduct {
            s += factors[1] * byte_sum(&self.signs, signs);
        }
        s
    }
}

#[inline]
fn byte_sum(tables: &[[f32; 256]], bytes: &[u8]) -> f32 {
    let mut lane = [0.0f32; 4];
    let (quads, rest) = bytes.as_chunks::<4>();
    let (table_quads, table_rest) = tables.as_chunks::<4>();
    for (b, t) in quads.iter().zip(table_quads) {
        lane[0] += t[0][usize::from(b[0])];
        lane[1] += t[1][usize::from(b[1])];
        lane[2] += t[2][usize::from(b[2])];
        lane[3] += t[3][usize::from(b[3])];
    }
    // Fewer than 4 bytes are left only in rows of 8 or 16 padded values.
    for (j, (&b, t)) in rest.iter().zip(table_rest).enumerate() {
        lane[j] += t[usize::from(b)];
    }
    (lane[0] + lane[1]) + (lane[2] + lane[3])
}

/// Four 6-bit fields to three bytes; a 3-bit row always has a multiple of
/// three bytes, since its padded length is a multiple of 8.
#[inline]
fn six_bit_sum(tables: &[[f32; 64]], bytes: &[u8]) -> f32 {
    let mut lane = [0.0f32; 4];
    for (b, t) in bytes.as_chunks::<3>().0.iter().zip(tables.as_chunks::<4>().0) {
        let x = u32::from(b[0]) | u32::from(b[1]) << 8 | u32::from(b[2]) << 16;
        lane[0] += t[0][(x & 63) as usize];
        lane[1] += t[1][(x >> 6 & 63) as usize];
        lane[2] += t[2][(x >> 12 & 63) as usize];
        lane[3] += t[3][(x >> 18 & 63) as usize];
    }
    (lane[0] + lane[1]) + (lane[2] + lane[3])
}
