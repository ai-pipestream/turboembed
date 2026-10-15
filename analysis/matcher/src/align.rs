//! Which pattern token took each token of a match: spaCy's
//! `with_alignments`, as a separate pass over a match already found, so
//! the matcher itself keeps state sets and no paths.
//!
//! Where a pattern can take a run of tokens more than one way, the way
//! chosen is a backtracking regular expression's with greedy operators:
//! from the first pattern token on, each takes as many tokens as it can
//! while the rest of the pattern can still take the rest of the run.
//! A table says, for each pattern token `k` and each place `j` in the
//! run, whether tokens `k..` can take the run from `j` on; it is filled
//! from the last pattern token back, then read forward. It costs a few
//! words per pattern token and place, and no recursion, so a long match
//! cannot overflow the stack.

use crate::input::{Columns, Input};
use crate::nfa::Nfa;

/// One pattern token: its compiled token's number and its repeat.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Step {
    pub(crate) tok: u32,
    pub(crate) min: u32,
    pub(crate) max: Option<u32>,
}

const NONE: u32 = u32::MAX;

/// The pattern token index of each token of the run `base..base + width`
/// (columns built over that run), or `None` when the pattern cannot take
/// exactly that run.
pub(crate) fn align(
    nfa: &Nfa,
    steps: &[Step],
    cols: &Columns,
    input: &Input<'_>,
    base: usize,
    width: usize,
) -> Option<Vec<u32>> {
    let m = steps.len();
    let row = width + 1;
    // run[k * row + j]: how many tokens from `j` on pass pattern token `k`.
    let mut run = vec![0u32; m * row];
    // last[k * row + x]: the largest `y <= x` from which tokens `k..` take
    // the rest of the run, or NONE.
    let mut last = vec![NONE; (m + 1) * row];
    last[m * row + width] = width as u32;
    // first: the smallest `y >= x` from which tokens `k + 1..` can, for
    // the row below the one being filled.
    let mut first = vec![NONE; row + 1];
    for k in (0..m).rev() {
        let below = &last[(k + 1) * row..(k + 2) * row];
        first[row] = NONE;
        for x in (0..row).rev() {
            first[x] = if below[x] == x as u32 { x as u32 } else { first[x + 1] };
        }
        let step = steps[k];
        let runs = &mut run[k * row..(k + 1) * row];
        for j in (0..width).rev() {
            runs[j] = if nfa.passes(step.tok, cols, input, j, base + j) { runs[j + 1] + 1 } else { 0 };
        }
        let mut prev = NONE;
        for j in 0..row {
            let lo = j + step.min as usize;
            let most = step.max.map_or(runs[j], |max| max.min(runs[j]));
            if lo <= width && step.min <= runs[j] && first[lo] != NONE && first[lo] as usize <= j + most as usize {
                prev = j as u32;
            }
            last[k * row + j] = prev;
        }
    }
    if last[0] != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(width);
    let mut j = 0usize;
    for (k, step) in steps.iter().enumerate() {
        // The most tokens this one can take with the rest still taking
        // the rest of the run.
        let most = step.max.map_or(run[k * row + j], |max| max.min(run[k * row + j])) as usize;
        let to = last[(k + 1) * row + j + most] as usize;
        debug_assert!(to != NONE as usize && to >= j + step.min as usize);
        out.extend(std::iter::repeat_n(k as u32, to - j));
        j = to;
    }
    debug_assert_eq!(j, width);
    Some(out)
}
