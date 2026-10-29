//! Rotary position embeddings (rotate_half) over the heads of Q and K,
//! from the core's own cos/sin table (`turbo::rope::table`), so the
//! trainer turns each pair by the same bits the backends do. Column c of
//! a head rotates with column c + d / 2: y1 = x1 cos - x2 sin, y2 = x2 cos
//! + x1 sin, at the token's position and the pair's frequency.

use turbo::status::{Result, invalid};

use crate::real::Real;

/// The table as `R`: `[2 (global, local)][2 (cos, sin)][positions][pairs]`.
#[derive(Debug, Clone)]
pub struct Table<R> {
    positions: usize,
    pairs: usize,
    data: Vec<R>,
}

impl<R: Real> Table<R> {
    /// The table of `positions` positions for heads `d` wide, with the
    /// global base and the local one (the global again without
    /// sliding-window layers).
    pub fn new(global: f64, local: f64, positions: usize, d: usize) -> Result<Table<R>> {
        if d == 0 || !d.is_multiple_of(2) {
            return Err(invalid(format!("rope: a head of width {d} has no pairs to rotate")));
        }
        let pairs = d / 2;
        let data = turbo::rope::table(global, local, positions, pairs)
            .map_err(|bytes| invalid(format!("rope: could not allocate {bytes} bytes for the table")))?;
        Ok(Table { positions, pairs, data: data.into_iter().map(|v| R::from_f64(v as f64)).collect() })
    }

    /// cos and sin at position `p`, pair `i`, under the global or local base.
    fn at(&self, global: bool, p: usize, i: usize) -> (R, R) {
        let half = self.positions * self.pairs;
        let base = if global { 0 } else { 2 * half };
        (self.data[base + p * self.pairs + i], self.data[base + half + p * self.pairs + i])
    }

    /// Rotates columns `0..cols` of each row of x [rows, ld] in place, in
    /// blocks of `2 * pairs` (one head each), row t at position t.
    pub fn apply(&self, global: bool, x: &mut [R], ld: usize, cols: usize) {
        self.turn(global, x, ld, cols, false);
    }

    /// The transpose of [`apply`](Self::apply): turns a gradient of the
    /// rotated columns into the gradient of the columns before rotation.
    pub fn apply_grad(&self, global: bool, dx: &mut [R], ld: usize, cols: usize) {
        self.turn(global, dx, ld, cols, true);
    }

    fn turn(&self, global: bool, x: &mut [R], ld: usize, cols: usize, back: bool) {
        let d = 2 * self.pairs;
        debug_assert!(cols.is_multiple_of(d) && cols <= ld);
        for (t, row) in x.chunks_exact_mut(ld).enumerate() {
            debug_assert!(t < self.positions);
            for head in row[..cols].chunks_exact_mut(d) {
                let (lo, hi) = head.split_at_mut(self.pairs);
                for (i, (a, b)) in lo.iter_mut().zip(hi).enumerate() {
                    let (c, s) = self.at(global, t, i);
                    let (x1, x2) = (*a, *b);
                    if back {
                        (*a, *b) = (x1 * c + x2 * s, x2 * c - x1 * s);
                    } else {
                        (*a, *b) = (x1 * c - x2 * s, x2 * c + x1 * s);
                    }
                }
            }
        }
    }
}
