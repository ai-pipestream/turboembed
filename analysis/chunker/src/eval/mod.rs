//! Evaluating a chunking against a gold one, and sweeping the chunker's
//! settings for the ones no other setting beats.
//!
//! A chunking is a [`Segmentation`]: `units` units (sentences or tokens)
//! and the boundaries between them, boundary `b` a cut before unit `b`.
//! The measures:
//!
//! - [`boundary`]: precision, recall and F1 of the boundaries, exact and
//!   within one or two units; Beeferman's Pk and Pevzner and Hearst's
//!   WindowDiff.
//! - [`coverage`]: how tightly chunks wrap gold evidence: span IoU,
//!   containment recall and redundancy, over bytes or tokens.
//! - [`budget`]: how the chunks sit in the token budget, and which
//!   protected spans (code blocks and tables, coreference chains, clause
//!   pairs) no cut parts.
//! - [`sweep`]: the chunker run over a grid of budgets and signal weights,
//!   every setting scored, and the settings ranked into Pareto fronts.
//!
//! Every measure is computed from integers in a fixed order, so the same
//! inputs give the same numbers on every machine.

pub mod boundary;
pub mod budget;
pub mod coverage;
pub mod identity;
pub use identity::{CutDifference, CutIdentityReport, CutRecord, compare_chunk_cuts};
pub mod sweep;

#[cfg(test)]
mod sweep_tests;
#[cfg(test)]
mod tests;

use turbo_analysis::Error;
use turbo_analysis_layers::Chunks;

/// The step name evaluation errors carry.
pub const ID: &str = "turbo:chunker-eval";

pub(crate) fn invalid(detail: String) -> Error {
    Error::Annotator { annotator: ID, detail }
}

/// `units` units cut into segments: each boundary `b` is a cut before
/// unit `b`, so it lies in `1..units`, and the boundaries are strictly
/// increasing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segmentation {
    units: usize,
    boundaries: Vec<usize>,
}

impl Segmentation {
    /// The segmentation, or an error when a boundary is out of `1..units`
    /// or the boundaries are not strictly increasing.
    pub fn new(units: usize, boundaries: Vec<usize>) -> Result<Segmentation, Error> {
        if let Some(&b) = boundaries.iter().find(|&&b| b == 0 || b >= units) {
            return Err(invalid(format!("boundary {b} is outside 1..{units}")));
        }
        if boundaries.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid(format!("boundaries {boundaries:?} are not strictly increasing")));
        }
        Ok(Segmentation { units, boundaries })
    }

    /// Segments of these lengths, in order; a zero length, or lengths
    /// past `usize::MAX` in all, is refused.
    pub fn of_lengths(lengths: &[usize]) -> Result<Segmentation, Error> {
        if lengths.contains(&0) {
            return Err(invalid(format!("segment lengths {lengths:?} hold a zero")));
        }
        let mut at = 0usize;
        let mut boundaries = Vec::with_capacity(lengths.len().saturating_sub(1));
        for &l in lengths {
            if at > 0 {
                boundaries.push(at);
            }
            at = at.checked_add(l).ok_or_else(|| invalid(format!("segment lengths {lengths:?} overflow usize")))?;
        }
        Segmentation::new(at, boundaries)
    }

    /// The leaf chunks (level 0 rows) of a chunks layer over `sentences`
    /// sentences, as a segmentation of the sentences: a boundary where
    /// each leaf after the first starts. Leaves that start at the same
    /// sentence make one boundary.
    pub fn of_chunks(chunks: &Chunks, sentences: usize) -> Result<Segmentation, Error> {
        let mut starts: Vec<usize> = (0..chunks.level.len())
            .filter(|&i| chunks.level[i] == 0)
            .map(|i| chunks.sentences[i].start.get())
            .filter(|&s| s > 0)
            .collect();
        starts.sort_unstable();
        starts.dedup();
        Segmentation::new(sentences, starts)
    }

    /// Every unit a segment of its own.
    pub fn all_cut(units: usize) -> Segmentation {
        Segmentation { units, boundaries: (1..units).collect() }
    }

    /// One segment holding every unit.
    pub fn no_cut(units: usize) -> Segmentation {
        Segmentation { units, boundaries: Vec::new() }
    }

    pub fn units(&self) -> usize {
        self.units
    }

    pub fn boundaries(&self) -> &[usize] {
        &self.boundaries
    }

    /// How many segments: one more than the boundaries, none when there
    /// are no units.
    pub fn segments(&self) -> usize {
        if self.units == 0 { 0 } else { self.boundaries.len() + 1 }
    }

    /// The boundaries in `(i, j]`: between unit i and unit j; none when
    /// j <= i. Two binary searches.
    pub fn between(&self, i: usize, j: usize) -> usize {
        let at = |x: usize| self.boundaries.partition_point(|&b| b <= x);
        at(j).saturating_sub(at(i))
    }

    /// `at[x]`: the boundaries at or before unit x, for x in `0..=units`,
    /// so `at[j] - at[i]` is `between(i, j)` in one subtraction.
    pub(crate) fn counts(&self) -> Vec<usize> {
        let mut at = vec![0usize; self.units + 1];
        for &b in &self.boundaries {
            at[b] = 1;
        }
        let mut sum = 0;
        for x in at.iter_mut() {
            sum += *x;
            *x = sum;
        }
        at
    }
}
