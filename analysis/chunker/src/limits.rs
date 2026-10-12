//! The hard limits: no chunk boundary inside a sentence, a table row, a
//! code block or a quote, unless that unit alone is over the budget.
//! Boundaries fall only between sentences, so a sentence is never cut;
//! the rest is a matter of which gaps between sentences may be cut.

use std::ops::Range;

use turbo_analysis_layers::{Block, Cut};

use crate::gaps::Unit;

/// What the units say about cutting at one gap.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Limit {
    /// No unit holds sentences on both sides.
    Free,
    /// A unit around it fits the budget: the smallest such unit's kind.
    Held(Block),
    /// Units are around it and none fits: the smallest one's kind.
    Inside(Block),
}

/// The limit at every gap of `n` sentences. A gap is held when any unit
/// around it fits, whatever else is around it: units nest, and they can
/// also share a sentence without nesting, when a unit's edge falls inside
/// a sentence. Time is the sum of the units' lengths in sentences.
pub fn limits(units: &[Unit], n: usize, fits: impl Fn(&Range<usize>) -> bool) -> Vec<Limit> {
    // The smallest unit around each gap, and the smallest that fits: by
    // sentences, then bytes, then the first in structure order.
    type Smallest = Option<(usize, u32, Block)>;
    let (mut any, mut fit): (Vec<Smallest>, Vec<Smallest>) = (vec![None; n], vec![None; n]);
    for u in units {
        let key = (u.sentences.len(), u.bytes, u.kind);
        let smaller = |c: &Smallest| c.is_none_or(|(s, b, _)| (key.0, key.1) < (s, b));
        let f = fits(&u.sentences);
        for g in u.sentences.start..u.sentences.end - 1 {
            if smaller(&any[g]) {
                any[g] = Some(key);
            }
            if f && smaller(&fit[g]) {
                fit[g] = Some(key);
            }
        }
    }
    any.iter()
        .zip(&fit)
        .map(|(a, f)| match (a, f) {
            (_, Some((.., k))) => Limit::Held(*k),
            (Some((.., k)), None) => Limit::Inside(*k),
            (None, None) => Limit::Free,
        })
        .collect()
}

/// Where a run of sentences may be cut: sentence indices from `run.start`
/// to `run.end`, each with the reason a chunk that starts there gives in
/// place of the boundary's own (`None` for an ordinary gap, and at both
/// ends). Each pair of neighbours is an atom the optimizer never splits.
///
/// Held gaps join their sentences into one atom. Units that share
/// sentences can chain into an atom that is over the budget though each
/// fits (`too_long` says so); such an atom must be cut somewhere, so every
/// gap in it is freed and a chunk that starts at one says `Cut::Overlap`.
/// A run's ends are always cuts: a unit that crosses one is cut there.
pub fn atoms(
    limits: &[Limit],
    run: Range<usize>,
    too_long: impl Fn(Range<usize>) -> bool,
) -> Vec<(usize, Option<Cut>)> {
    let mut out = vec![(run.start, None)];
    let mut start = run.start;
    // Closes the atom `start..end`, freeing its gaps when it is too long.
    let close = |out: &mut Vec<(usize, Option<Cut>)>, start: usize, end: usize| {
        if end - start > 1 && too_long(start..end) {
            for (g, l) in (start..end - 1).zip(&limits[start..end - 1]) {
                if let Limit::Held(k) = *l {
                    out.push((g + 1, Some(Cut::Overlap(k))));
                }
            }
        }
    };
    let inner = run.start..run.end.saturating_sub(1).max(run.start);
    for (g, l) in inner.clone().zip(&limits[inner]) {
        let forced = match *l {
            Limit::Held(_) => continue,
            Limit::Free => None,
            Limit::Inside(k) => Some(Cut::Inside(k)),
        };
        close(&mut out, start, g + 1);
        out.push((g + 1, forced));
        start = g + 1;
    }
    close(&mut out, start, run.end);
    out.push((run.end, None));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(kind: Block, sentences: Range<usize>) -> Unit {
        let bytes = 10 * sentences.len() as u32;
        Unit { kind, sentences, bytes }
    }

    fn at(a: &[(usize, Option<Cut>)]) -> Vec<usize> {
        a.iter().map(|x| x.0).collect()
    }

    #[test]
    fn a_unit_that_fits_is_one_atom() {
        // A unit over sentences 1 to 3 of five.
        let u = [unit(Block::Code, 1..4)];
        let held = limits(&u, 5, |_| true);
        assert_eq!(held, [Limit::Free, Limit::Held(Block::Code), Limit::Held(Block::Code), Limit::Free, Limit::Free]);
        assert_eq!(at(&atoms(&held, 0..5, |_| false)), [0, 1, 4, 5]);
        let over = limits(&u, 5, |_| false);
        let a = atoms(&over, 0..5, |_| false);
        assert_eq!(at(&a), [0, 1, 2, 3, 4, 5]);
        let inside = Some(Cut::Inside(Block::Code));
        assert_eq!(a.iter().map(|x| x.1).collect::<Vec<_>>(), [None, None, inside, inside, None, None]);
    }

    #[test]
    fn a_unit_over_the_budget_frees_only_its_own_gaps() {
        // A quote over every sentence holds a code block over 1 and 2; the
        // quote is over the budget and the code block is not.
        let u = [unit(Block::Quote, 0..4), unit(Block::Code, 1..3)];
        let l = limits(&u, 4, |r| r.len() < 4);
        let q = Limit::Inside(Block::Quote);
        assert_eq!(l, [q, Limit::Held(Block::Code), q, Limit::Free]);
        assert_eq!(at(&atoms(&l, 0..4, |_| false)), [0, 1, 3, 4]);
    }

    #[test]
    fn a_gap_is_held_when_any_unit_around_it_fits() {
        // A quote over 0 to 2 that is over the budget, and a code block
        // over 1 to 4 that fits and shares sentences 1 and 2 with it. The
        // quote is the smaller in sentences; the code block still holds.
        let u = [unit(Block::Quote, 0..3), unit(Block::Code, 1..5)];
        let l = limits(&u, 5, |r| r.start == 1);
        let c = Limit::Held(Block::Code);
        assert_eq!(l, [Limit::Inside(Block::Quote), c, c, c, Limit::Free]);
    }

    #[test]
    fn units_that_chain_over_the_budget_are_freed() {
        // Three rows that each fit share a sentence with the next, making
        // one run of five sentences that does not.
        let u = [unit(Block::TableRow, 0..2), unit(Block::TableRow, 1..3), unit(Block::TableRow, 2..5)];
        let l = limits(&u, 6, |_| true);
        assert_eq!(at(&atoms(&l, 0..6, |_| false)), [0, 5, 6]);
        let a = atoms(&l, 0..6, |r| r.len() > 3);
        assert_eq!(at(&a), [0, 1, 2, 3, 4, 5, 6]);
        let o = Some(Cut::Overlap(Block::TableRow));
        assert_eq!(a.iter().map(|x| x.1).collect::<Vec<_>>(), [None, o, o, o, o, None, None]);
    }

    #[test]
    fn a_run_cuts_only_its_own_gaps() {
        let l = limits(&[unit(Block::Code, 1..4)], 5, |_| true);
        // The run starts inside the unit: its start is a cut all the same.
        assert_eq!(at(&atoms(&l, 2..5, |_| false)), [2, 4, 5]);
        assert_eq!(at(&atoms(&l, 2..3, |_| false)), [2, 3]);
        assert_eq!(at(&atoms(&[Limit::Free; 4], 1..3, |_| false)), [1, 2, 3]);
    }
}
