//! Sentence vectors as directions: each row scaled to length one, and
//! running sums of those over a run of sentences, so the sum over any
//! stretch of it costs one subtraction per dimension, whatever its length.
//! Drift compares the sums of two small windows; cohesion measures how far
//! the sentences of a chunk spread around their sum.

use std::ops::Range;

use turbo_analysis::{Ix, Layer};
use turbo_analysis_layers::{Sentences, Vectors};

/// The rows of a vectors layer with the length of each, read in place.
/// A sentence with no vector (one `Vectors::empty` lists, or a row of
/// zeros or of non-finite numbers) has no direction to compare: it is
/// left out of `present` and scaled by zero.
#[derive(Clone, Debug, PartialEq)]
pub struct Rows<'a> {
    dim: usize,
    data: &'a [f32],
    /// One over each row's length, or zero for a sentence with no vector.
    scale: Vec<f64>,
    present: Vec<u32>,
}

impl<'a> Rows<'a> {
    /// From rows of `dim` floats, one per sentence; `empty(i)` says
    /// sentence i has no vector.
    pub fn new(dim: usize, data: &'a [f32], empty: impl Fn(usize) -> bool) -> Rows<'a> {
        let n = data.len().checked_div(dim).unwrap_or(0);
        let mut scale = vec![0.0f64; n];
        let mut present = Vec::new();
        for (i, s) in scale.iter_mut().enumerate() {
            let row = &data[i * dim..(i + 1) * dim];
            let norm = row.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
            if !empty(i) && norm > 0.0 && norm.is_finite() {
                *s = 1.0 / norm;
                present.push(i as u32);
            }
        }
        Rows { dim, data, scale, present }
    }

    /// From a sentence vectors layer, honouring its `empty` column.
    pub fn of(v: &'a Vectors<Sentences>) -> Rows<'a> {
        Rows::new(v.dim as usize, &v.data, |i| v.is_empty(Ix::new(i as u32)))
    }

    /// How many sentences there are.
    pub fn len(&self) -> usize {
        self.scale.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scale.is_empty()
    }

    /// The sentences that have a vector, in order.
    pub fn present(&self) -> &[u32] {
        &self.present
    }

    /// Adds sentence i's unit vector to `out`.
    fn add_to(&self, i: usize, out: &mut [f64]) {
        let s = self.scale[i];
        // A row with no vector may hold NaN, which times zero is NaN.
        if s == 0.0 {
            return;
        }
        for (o, &x) in out.iter_mut().zip(&self.data[i * self.dim..(i + 1) * self.dim]) {
            *o += f64::from(x) * s;
        }
    }

    /// The cosine between the sum of the unit vectors of sentences `a`
    /// and that of sentences `b`, or `None` when either sum is zero. It
    /// sums in the order given, so the result is the same on every run.
    pub fn cosine(&self, a: &[u32], b: &[u32]) -> Option<f64> {
        let (mut x, mut y) = (vec![0.0f64; self.dim], vec![0.0f64; self.dim]);
        a.iter().for_each(|&i| self.add_to(i as usize, &mut x));
        b.iter().for_each(|&i| self.add_to(i as usize, &mut y));
        let (mut dot, mut nx, mut ny) = (0.0f64, 0.0f64, 0.0f64);
        for (&p, &q) in x.iter().zip(&y) {
            dot += p * q;
            nx += p * p;
            ny += q * q;
        }
        (nx > 0.0 && ny > 0.0).then(|| (dot / (nx.sqrt() * ny.sqrt())).clamp(-1.0, 1.0))
    }
}

/// Check the layer before Rows can discard a partial or non-finite row.
pub(crate) fn checked_rows(v: &Vectors<Sentences>) -> Result<Rows<'_>, String> {
    v.check()?;
    for (i, row) in v.data.chunks_exact(v.dim as usize).enumerate() {
        if let Some(coordinate) = row.iter().position(|x| !x.is_finite()) {
            return Err(format!("row {i} coordinate {coordinate} is not finite"));
        }
    }
    Ok(Rows::of(v))
}

/// The unit vectors of a run of sentences summed from the run's start, in
/// f64, with a count of the sentences that have one. It holds
/// `(len + 1) * dim` f64s, so it is built per run (a top-level section),
/// not for the whole document.
#[derive(Clone, Debug, PartialEq)]
pub struct Sums {
    dim: usize,
    /// `sum[i * dim..(i + 1) * dim]`: the unit vectors of the run's first
    /// i sentences.
    sum: Vec<f64>,
    count: Vec<u32>,
}

impl Sums {
    /// The sums over sentences `run`, indexed from `run.start`.
    pub fn new(rows: &Rows<'_>, run: Range<usize>) -> Sums {
        let d = rows.dim;
        let mut sum = vec![0.0f64; (run.len() + 1) * d];
        let mut count = vec![0u32; run.len() + 1];
        for (k, i) in run.enumerate() {
            let (done, rest) = sum.split_at_mut((k + 1) * d);
            rest[..d].copy_from_slice(&done[k * d..]);
            rows.add_to(i, &mut rest[..d]);
            count[k + 1] = count[k] + u32::from(rows.scale[i] > 0.0);
        }
        Sums { dim: d, sum, count }
    }

    /// How many sentences there are.
    pub fn len(&self) -> usize {
        self.count.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many of sentences `[i, j)` have a vector.
    pub fn count(&self, i: usize, j: usize) -> u32 {
        self.count[j] - self.count[i]
    }

    /// The length of the sum of the unit vectors of sentences `[i, j)`.
    /// One pass over the dimensions.
    pub fn norm(&self, i: usize, j: usize) -> f64 {
        let d = self.dim;
        let (a, b) = (&self.sum[i * d..(i + 1) * d], &self.sum[j * d..(j + 1) * d]);
        a.iter().zip(b).map(|(&x, &y)| (y - x) * (y - x)).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_rows_add_nothing_and_are_not_counted() {
        // Sentence 1 is listed empty, sentence 2 is zeros, sentence 3 is NaN.
        let data = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, f32::NAN, 1.0, 0.0, 2.0];
        let rows = Rows::new(2, &data, |i| i == 1);
        assert_eq!((rows.len(), rows.present()), (5, &[0u32, 4][..]));
        let s = Sums::new(&rows, 0..5);
        assert_eq!(s.count(0, 5), 2);
        assert!((s.norm(0, 4) - 1.0).abs() < 1e-12);
        assert_eq!(rows.cosine(&[1, 2, 3], &[0]), None);
        assert!((rows.cosine(&[0], &[4]).unwrap() - 0.8).abs() < 1e-12);
        // A run is indexed from its own start.
        let tail = Sums::new(&rows, 3..5);
        assert_eq!((tail.len(), tail.count(0, 2)), (2, 1));
        assert!((tail.norm(0, 2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_sentences_no_sums() {
        let rows = Rows::new(4, &[], |_| false);
        assert!(rows.is_empty());
        assert!(Sums::new(&rows, 0..0).is_empty());
        assert!(Rows::new(0, &[], |_| false).is_empty());
    }
}
