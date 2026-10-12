//! Cohesion: a chunk term for the optimizer from how tightly a chunk's
//! sentence vectors cluster.
//!
//! A chunk pays, for each of its sentences with a vector, one less the
//! cosine between that sentence and the chunk's centroid. With unit vectors
//! u_k summing to S over m sentences, the cosine of u_k to the centroid is
//! u_k . S / |S|, and the sum over the chunk is |S|, so the chunk pays
//! `m - |S|`: zero when all its sentences point the same way, up to m when
//! they cancel. The running sums of `Sums` give S for any chunk in one
//! pass over the dimensions, whatever its length, so the optimizer's time
//! is its usual count of chunks tried times the dimension. They are built
//! for one run (a top-level section) at a time.
//!
//! Splitting a chunk never raises the total (|A| + |B| >= |A + B|), and
//! lowers it by exactly how much the two parts point apart. So the term
//! pulls toward cuts between sentences that differ, and the boundary
//! costs and the size penalty decide how many cuts that is worth.

use turbo_analysis::{Annotator, Error, Layer, Refs};
use turbo_analysis_layers::{Boundaries, Chunks, Sentences, Structure, Subwords, Vectors};

use crate::sums::{Rows, Sums};
use crate::{Chunker, TERM_MAX};

/// What a chunk pays for its spread.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Cohesion {
    /// What a sentence at right angles to its chunk's centroid costs, in
    /// thousandths of a `Boundaries::UNIT` (the units costs are written
    /// in); a sentence at the centroid costs nothing.
    pub weight: u32,
}

impl Default for Cohesion {
    /// Untuned: a sentence at right angles to its chunk costs as much as
    /// a plain sentence boundary, 1000.
    fn default() -> Cohesion {
        Cohesion { weight: Boundaries::UNIT as u32 }
    }
}

impl Cohesion {
    /// The term of sentences `[i, j)`: `weight * (m - |S|)`, computed in
    /// f64, rounded once to the nearest integer, halves away from zero,
    /// and clamped to `TERM_MAX`.
    pub fn term(&self, sums: &Sums, i: usize, j: usize) -> i64 {
        let spread = (f64::from(sums.count(i, j)) - sums.norm(i, j)).max(0.0);
        let t = (spread * f64::from(self.weight)).round();
        t.clamp(-(TERM_MAX as f64), TERM_MAX as f64) as i64
    }
}

/// The chunker with cohesion as its chunk term. It reads the sentence
/// vectors besides what `Chunker` reads, so it is a step of its own rather
/// than an option of `Chunker`: a chunker that only might read them would
/// still have to run after them, and a pipeline that pools chunk vectors
/// in the same step as sentence vectors (`SentenceChunkVectors`) would be a
/// cycle.
///
/// Off unless chosen: no measurement yet says it helps.
pub struct CohesiveChunker {
    pub chunker: Chunker,
    pub cohesion: Cohesion,
}

impl Annotator for CohesiveChunker {
    const ID: &'static str = "turbo:cohesive-chunker";
    type Reads = (Sentences, Subwords, Boundaries, Option<Structure>, Vectors<Sentences>);
    type Writes = (Chunks,);

    fn annotate(&self, (sentences, subwords, bounds, structure, vectors): Refs<'_, Self>) -> Result<(Chunks,), Error> {
        if vectors.len() != sentences.spans.len() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("{} vectors for {} sentences", vectors.len(), sentences.spans.len()),
            });
        }
        let rows = Rows::of(vectors);
        let c = self.cohesion;
        let term = |run: std::ops::Range<usize>| {
            let sums = Sums::new(&rows, run.clone());
            move |i: usize, j: usize| c.term(&sums, i - run.start, j - run.start)
        };
        Ok((self.chunker.chunk(Self::ID, sentences, subwords, bounds, structure, term)?,))
    }

    fn config_hash(&self) -> u64 {
        let h = self.chunker.config_hash();
        (h ^ u64::from(self.cohesion.weight)).wrapping_mul(0x0100_0000_01b3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::brute;
    use crate::{Budget, Lengths, segment};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self, m: u64) -> u64 {
            self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % m
        }

        /// `n` rows of `dim` numbers in [-1, 1), a few of them zeros
        /// listed empty.
        fn vectors(&mut self, n: usize, dim: usize) -> (Vec<f32>, Vec<usize>) {
            let mut data: Vec<f32> = (0..n * dim).map(|_| self.next(2001) as f32 / 1000.0 - 1.0).collect();
            let empty: Vec<usize> = (0..n).filter(|_| self.next(6) == 0).collect();
            for &i in &empty {
                data[i * dim..(i + 1) * dim].fill(0.0);
            }
            (data, empty)
        }
    }

    /// The spread of sentences `[i, j)` computed directly: each vector
    /// normalized, the centroid summed, and one less each cosine to it.
    fn direct(data: &[f32], dim: usize, empty: &[usize], i: usize, j: usize) -> f64 {
        let unit = |k: usize| {
            let r: Vec<f64> = data[k * dim..(k + 1) * dim].iter().map(|&x| f64::from(x)).collect();
            let n = r.iter().map(|x| x * x).sum::<f64>().sqrt();
            (n > 0.0).then(|| r.into_iter().map(|x| x / n).collect::<Vec<f64>>())
        };
        // A row of zeros not listed empty has no direction either.
        let rows: Vec<Vec<f64>> = (i..j).filter(|k| !empty.contains(k)).filter_map(unit).collect();
        let mut c = vec![0.0; dim];
        for r in &rows {
            for (a, b) in c.iter_mut().zip(r) {
                *a += b;
            }
        }
        let cn = c.iter().map(|x| x * x).sum::<f64>().sqrt();
        if cn == 0.0 {
            return rows.len() as f64;
        }
        rows.iter().map(|r| 1.0 - r.iter().zip(&c).map(|(a, b)| a * b).sum::<f64>() / cn).sum()
    }

    #[test]
    fn the_running_sums_give_the_direct_spread() {
        let mut rng = Rng(0x5851_f42d_4c95_7f2d);
        let w = Cohesion { weight: 1_000_000 };
        for case in 0..300 {
            let (n, dim) = (1 + rng.next(12) as usize, 1 + rng.next(9) as usize);
            let (data, empty) = rng.vectors(n, dim);
            let sums = Sums::new(&Rows::new(dim, &data, |k| empty.contains(&k)), 0..n);
            for i in 0..n {
                for j in i + 1..=n {
                    let want = direct(&data, dim, &empty, i, j);
                    let got = f64::from(sums.count(i, j)) - sums.norm(i, j);
                    assert!((want - got).abs() < 1e-9, "case {case} [{i}, {j}): {want} {got}");
                    // As an integer term it is off by at most the rounding.
                    let t = w.term(&sums, i, j);
                    assert!((t as f64 - want * 1e6).abs() <= 1.0 + 1e-3, "case {case} [{i}, {j}): {t} {want}");
                }
            }
        }
    }

    #[test]
    fn a_tight_chunk_costs_nothing_and_two_clusters_cost_their_angle() {
        // Three vectors one way, three at right angles, and an empty row.
        let data = [1.0, 0.0, 2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 1.0];
        let sums = Sums::new(&Rows::new(2, &data, |k| k == 3), 0..7);
        let c = Cohesion::default();
        assert_eq!((c.term(&sums, 0, 3), c.term(&sums, 4, 7), c.term(&sums, 0, 4)), (0, 0, 0));
        // Six sentences, the sum of length sqrt(18): 6 - 4.243 units.
        assert_eq!(c.term(&sums, 0, 7), 1757);
        // The empty row alone is not a sentence to spread.
        assert_eq!(c.term(&sums, 3, 4), 0);
    }

    #[test]
    fn the_optimizer_with_cohesion_agrees_with_trying_every_cut() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for case in 0..1500 {
            let n = 1 + rng.next(10) as usize;
            let dim = 1 + rng.next(4) as usize;
            let (data, empty) = rng.vectors(n, dim);
            let sums = Sums::new(&Rows::new(dim, &data, |k| empty.contains(&k)), 0..n);
            let c = Cohesion { weight: rng.next(3000) as u32 };
            let tokens: Vec<u32> = (0..n).map(|_| rng.next(60) as u32).collect();
            let cut: Vec<i32> = (0..n).map(|_| rng.next(3001) as i32 - 1000).collect();
            let b = Budget {
                max: 1 + rng.next(150) as u32,
                target: rng.next(120) as u32,
                size_weight: [0, 1000, 100_000][rng.next(3) as usize],
                overhead: rng.next(3) as u32,
            };
            let term = |i: usize, j: usize| c.term(&sums, i, j);
            let l = Lengths::of(&tokens).unwrap();
            let segments = segment(&cut, |i, j| l.run(i, j), b, term);
            let total: i128 = segments.iter().map(|s| s.cost).sum();
            let ends: Vec<usize> = segments.iter().map(|s| s.end).collect();
            assert_eq!((total, ends), brute(&cut, &tokens, b, &term), "case {case}");
        }
    }

    #[test]
    fn cohesion_moves_a_cut_to_where_the_vectors_part() {
        // Six sentences of 100 tokens, every boundary the same cost; the
        // vectors turn after the second. The size alone picks 3 + 3.
        let b = Budget { max: 400, target: 300, size_weight: 1000, overhead: 0 };
        let l = Lengths::of(&[100; 6]).unwrap();
        let tokens = |i: usize, j: usize| l.run(i, j);
        let ends = |t: &dyn Fn(usize, usize) -> i64| -> Vec<usize> {
            segment(&[1000; 6], tokens, b, t).iter().map(|s| s.end).collect()
        };
        assert_eq!(ends(&crate::no_term), [3, 6]);
        let data = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let sums = Sums::new(&Rows::new(2, &data, |_| false), 0..6);
        let c = Cohesion::default();
        assert_eq!(ends(&|i, j| c.term(&sums, i, j)), [2, 6]);
    }
}
