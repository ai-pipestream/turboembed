//! The hard limits: no chunk boundary inside a sentence, a table row, a
//! code block or a quote, unless that unit alone is over the budget.
//! Boundaries fall only between sentences, so a sentence is never cut;
//! the rest is a matter of which gaps between sentences may be cut.

use std::ops::Range;

use crate::gaps::Gap;

/// Where a run of sentences may be cut: sentence indices from
/// `run.start` to `run.end`, both included, so each pair of neighbours
/// is an atom the optimizer never splits. A gap inside a unit is no
/// place to cut when the unit `fits` the budget. Units nest, and the gap
/// names the smallest unit around it: when that one is over the budget,
/// every unit around it is too, so the gap may be cut.
pub fn atoms(gaps: &[Gap], run: Range<usize>, fits: impl Fn(&Range<usize>) -> bool) -> Vec<usize> {
    let mut out = vec![run.start];
    // The gaps between the run's sentences: not the one after its last.
    let inner = run.start..run.end.saturating_sub(1).max(run.start);
    for (g, gap) in inner.clone().zip(&gaps[inner]) {
        if gap.inside.as_ref().is_none_or(|u| !fits(&u.sentences)) {
            out.push(g + 1);
        }
    }
    out.push(run.end);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaps::Unit;
    use turbo_analysis_layers::{Block, Cut};

    fn gaps(inside: &[Option<Range<usize>>]) -> Vec<Gap> {
        inside
            .iter()
            .map(|r| Gap {
                reason: Cut::Sentence,
                after_heading: false,
                inside: r.clone().map(|sentences| Unit { kind: Block::Code, sentences }),
            })
            .collect()
    }

    #[test]
    fn a_unit_that_fits_is_one_atom() {
        // A unit over sentences 1 to 3 of five.
        let g = gaps(&[None, Some(1..4), Some(1..4), None, None]);
        assert_eq!(atoms(&g, 0..5, |_| true), [0, 1, 4, 5]);
        assert_eq!(atoms(&g, 0..5, |_| false), [0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn a_unit_over_the_budget_frees_only_its_own_gaps() {
        // A quote over every sentence holds a code block over 1 and 2; the
        // quote is over the budget and the code block is not.
        let g = gaps(&[Some(0..4), Some(1..3), Some(0..4), None]);
        assert_eq!(atoms(&g, 0..4, |r| r.len() < 4), [0, 1, 3, 4]);
    }

    #[test]
    fn a_run_cuts_only_its_own_gaps() {
        let g = gaps(&[None, None, None, None]);
        assert_eq!(atoms(&g, 1..3, |_| true), [1, 2, 3]);
        assert_eq!(atoms(&g, 2..3, |_| true), [2, 3]);
    }
}
