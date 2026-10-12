//! Running sums of unit sentence vectors, so the sum over any run of
//! sentences costs one subtraction per dimension, whatever its length.
//! Drift compares the sums of two windows; cohesion measures how far the
//! sentences of a chunk spread around their sum.

use turbo_analysis::Ix;
use turbo_analysis_layers::{Sentences, Vectors};

/// Each sentence's vector scaled to length one, summed from the start of
/// the document, in f64. A sentence with no vector (one `Vectors::empty`
/// lists, or a row of zeros or of non-finite numbers) adds nothing and is
/// not counted: it has no direction to compare.
///
/// Memory is `(n + 1) * dim` f64s, twice the vectors' own f32 block.
#[derive(Clone, Debug, PartialEq)]
pub struct Sums {
    dim: usize,
    /// `sum[i * dim..(i + 1) * dim]`: the unit vectors of sentences `[0, i)`.
    sum: Vec<f64>,
    /// `count[i]`: how many of sentences `[0, i)` have a vector.
    count: Vec<u32>,
    /// The sentences that have a vector, in order.
    present: Vec<u32>,
}

impl Sums {
    /// From rows of `dim` floats, one per sentence; `empty(i)` says
    /// sentence i has no vector.
    pub fn new(dim: usize, data: &[f32], empty: impl Fn(usize) -> bool) -> Sums {
        let n = data.len().checked_div(dim).unwrap_or(0);
        let mut sum = vec![0.0f64; (n + 1) * dim];
        let mut count = vec![0u32; n + 1];
        let mut present = Vec::new();
        for i in 0..n {
            let row = &data[i * dim..(i + 1) * dim];
            let norm = row.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
            let has = !empty(i) && norm > 0.0 && norm.is_finite();
            let (done, rest) = sum.split_at_mut((i + 1) * dim);
            let before = &done[i * dim..];
            for (k, out) in rest[..dim].iter_mut().enumerate() {
                *out = before[k] + if has { f64::from(row[k]) / norm } else { 0.0 };
            }
            count[i + 1] = count[i] + u32::from(has);
            if has {
                present.push(i as u32);
            }
        }
        Sums { dim, sum, count, present }
    }

    /// From a sentence vectors layer, honouring its `empty` column.
    pub fn of(v: &Vectors<Sentences>) -> Sums {
        Sums::new(v.dim as usize, &v.data, |i| v.is_empty(Ix::new(i as u32)))
    }

    /// How many sentences there are.
    pub fn len(&self) -> usize {
        self.count.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The sentences that have a vector, in order.
    pub fn present(&self) -> &[u32] {
        &self.present
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

    /// The cosine between the sums of sentences `[i, j)` and `[k, l)`, or
    /// `None` when either sum is zero. One pass over the dimensions.
    pub fn cosine(&self, (i, j): (usize, usize), (k, l): (usize, usize)) -> Option<f64> {
        let d = self.dim;
        let row = |r: usize| &self.sum[r * d..(r + 1) * d];
        let (a0, a1, b0, b1) = (row(i), row(j), row(k), row(l));
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for x in 0..d {
            let (a, b) = (a1[x] - a0[x], b1[x] - b0[x]);
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        (na > 0.0 && nb > 0.0).then(|| (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_rows_add_nothing_and_are_not_counted() {
        // Sentence 1 is listed empty, sentence 2 is zeros, sentence 3 is NaN.
        let data = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, f32::NAN, 1.0, 0.0, 2.0];
        let s = Sums::new(2, &data, |i| i == 1);
        assert_eq!((s.len(), s.present(), s.count(0, 5)), (5, &[0u32, 4][..], 2));
        assert!((s.norm(0, 4) - 1.0).abs() < 1e-12);
        assert_eq!(s.cosine((1, 4), (0, 1)), None);
        let c = s.cosine((0, 1), (4, 5)).unwrap();
        assert!((c - 0.8).abs() < 1e-12);
    }

    #[test]
    fn no_sentences_no_sums() {
        let s = Sums::new(4, &[], |_| false);
        assert!(s.is_empty());
        assert_eq!(s.count(0, 0), 0);
        assert!(Sums::new(0, &[], |_| false).is_empty());
    }
}
