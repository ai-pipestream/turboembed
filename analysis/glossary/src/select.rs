//! Choosing the matches to keep among overlapping candidates, and merging
//! the results of several matchers by priority.

use std::cmp::Reverse;

use turbo_analysis::Span;
use turbo_glossary_layers::EntryIx;

/// One glossary term found in a text: where, and which entry matched.
/// OpenNLP's `GlossaryMatch`, whose identifier and term are the entry's.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GlossaryMatch {
    pub span: Span,
    pub entry: EntryIx,
}

/// A raw occurrence before overlaps are resolved: its span, how many
/// tokens it matched (zero for a character match) and its entry.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub span: Span,
    pub tokens: u32,
    pub entry: EntryIx,
}

/// OpenNLP's `Span.intersects`: whether either span contains the other,
/// or either one's start lies inside the other. Two non-empty spans
/// intersect when they share a byte; an empty span intersects a span that
/// covers its offset, its end included. That last case is where it
/// differs from [`Span::overlaps`], which leaves out an empty span at the
/// other's end.
pub fn intersects(a: Span, b: Span) -> bool {
    a.contains(b) || b.contains(a) || (a.start <= b.start && b.start < a.end) || (b.start <= a.start && a.start < b.end)
}

/// Sorts candidates by start, then the longest span first, then the most
/// tokens, then registration order, and keeps each one that starts at or
/// after the end of the last kept and does not intersect it: OpenNLP's
/// `GlossaryMatchSelection.select` with the order both of its matchers
/// sort by (the character matcher has no token count, so that key ties).
/// The matches come out in text order without overlaps.
pub fn select(candidates: &mut [Candidate], out: &mut Vec<GlossaryMatch>) {
    candidates.sort_unstable_by_key(|c| (c.span.start, Reverse(c.span.end), Reverse(c.tokens), c.entry));
    let first = out.len();
    let mut last_end = 0;
    for c in candidates.iter() {
        if c.span.start < last_end {
            continue;
        }
        if out.len() > first && intersects(out[out.len() - 1].span, c.span) {
            continue;
        }
        out.push(GlossaryMatch { span: c.span, entry: c.entry });
        last_end = c.span.end;
    }
}

/// Adds `candidates`, the matches of a lower priority matcher, to
/// `accepted`, the matches kept so far: a candidate is kept when it
/// intersects neither an accepted match nor the candidate kept before it.
/// Both lists are in text order without overlaps, so one pass over each
/// suffices; the result is in text order, an accepted match before an
/// added one that starts at the same offset. OpenNLP's
/// `CompositeGlossaryMatcher.filterOverlaps` and `mergeByStart`.
pub fn merge(accepted: &mut Vec<GlossaryMatch>, candidates: &[GlossaryMatch]) {
    if candidates.is_empty() {
        return;
    }
    let mut additions: Vec<GlossaryMatch> = Vec::with_capacity(candidates.len());
    let mut a = 0;
    for &c in candidates {
        // An accepted match that ends before the candidate starts cannot
        // intersect it nor any later candidate; one that starts after it
        // ends cannot either.
        while a < accepted.len() && accepted[a].span.end < c.span.start {
            a += 1;
        }
        let hits_accepted =
            accepted[a..].iter().take_while(|x| x.span.start <= c.span.end).any(|x| intersects(x.span, c.span));
        let hits_previous = additions.last().is_some_and(|p| intersects(p.span, c.span));
        if !hits_accepted && !hits_previous {
            additions.push(c);
        }
    }
    if additions.is_empty() {
        return;
    }
    if accepted.is_empty() {
        *accepted = additions;
        return;
    }
    let old = std::mem::take(accepted);
    accepted.reserve(old.len() + additions.len());
    let (mut i, mut j) = (0, 0);
    while i < old.len() && j < additions.len() {
        if old[i].span.start <= additions[j].span.start {
            accepted.push(old[i]);
            i += 1;
        } else {
            accepted.push(additions[j]);
            j += 1;
        }
    }
    accepted.extend_from_slice(&old[i..]);
    accepted.extend_from_slice(&additions[j..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(start: u32, end: u32, tokens: u32, entry: u32) -> Candidate {
        Candidate { span: Span::new(start, end), tokens, entry: EntryIx(entry) }
    }

    fn m(start: u32, end: u32, entry: u32) -> GlossaryMatch {
        GlossaryMatch { span: Span::new(start, end), entry: EntryIx(entry) }
    }

    #[test]
    fn intersects_as_opennlp_span() {
        let s = Span::new;
        assert!(intersects(s(0, 3), s(2, 5)));
        assert!(!intersects(s(0, 3), s(3, 5)));
        assert!(intersects(s(3, 5), s(5, 5)), "an empty span at the end");
        assert!(!s(3, 5).overlaps(s(5, 5)));
        assert!(intersects(s(5, 5), s(5, 5)));
        assert!(intersects(s(3, 5), s(3, 3)));
        assert!(!intersects(s(3, 5), s(6, 6)));
    }

    #[test]
    fn leftmost_then_longest_then_tokens_then_first_registered() {
        let mut cands = vec![c(4, 8, 1, 0), c(0, 5, 2, 3), c(0, 5, 2, 1), c(0, 3, 1, 0), c(6, 9, 1, 2), c(0, 5, 3, 4)];
        let mut out = Vec::new();
        select(&mut cands, &mut out);
        assert_eq!(out, [m(0, 5, 4), m(6, 9, 2)]);
    }

    #[test]
    fn an_empty_candidate_at_the_last_end_is_dropped() {
        let mut cands = vec![c(0, 5, 0, 0), c(5, 5, 0, 1), c(5, 7, 0, 2)];
        let mut out = Vec::new();
        select(&mut cands, &mut out);
        // [5, 7) sorts before [5, 5) and is kept; [5, 5) then intersects it.
        assert_eq!(out, [m(0, 5, 0), m(5, 7, 2)]);
    }

    #[test]
    fn merge_keeps_priority_and_text_order() {
        let mut accepted = vec![m(4, 8, 0), m(10, 12, 1)];
        merge(&mut accepted, &[m(0, 3, 2), m(3, 5, 3), m(8, 10, 4), m(9, 11, 5), m(12, 14, 6)]);
        assert_eq!(accepted, [m(0, 3, 2), m(4, 8, 0), m(8, 10, 4), m(10, 12, 1), m(12, 14, 6)]);
        let mut none = Vec::new();
        merge(&mut none, &[m(1, 2, 0)]);
        assert_eq!(none, [m(1, 2, 0)]);
        // A candidate holding an accepted empty match intersects it.
        let mut accepted = vec![m(2, 2, 0)];
        merge(&mut accepted, &[m(0, 1, 1), m(2, 4, 2)]);
        assert_eq!(accepted, [m(0, 1, 1), m(2, 2, 0)]);
        // An empty candidate at the end of one accepted match and the
        // start of the next intersects the first.
        let mut accepted = vec![m(0, 5, 0), m(5, 8, 1)];
        merge(&mut accepted, &[m(5, 5, 2)]);
        assert_eq!(accepted, [m(0, 5, 0), m(5, 8, 1)]);
    }

    /// `CompositeGlossaryMatcher` as written: each candidate against every
    /// accepted match and the last one added, then a stable merge by start.
    fn merge_by_rule(accepted: &[GlossaryMatch], candidates: &[GlossaryMatch]) -> Vec<GlossaryMatch> {
        let mut added: Vec<GlossaryMatch> = Vec::new();
        for &c in candidates {
            if accepted.iter().any(|a| intersects(a.span, c.span)) {
                continue;
            }
            if added.last().is_some_and(|p| intersects(p.span, c.span)) {
                continue;
            }
            added.push(c);
        }
        let mut all: Vec<(u32, usize, GlossaryMatch)> =
            accepted.iter().map(|&a| (a.span.start, 0, a)).chain(added.iter().map(|&a| (a.span.start, 1, a))).collect();
        all.sort_by_key(|&(start, side, _)| (start, side));
        all.into_iter().map(|(_, _, m)| m).collect()
    }

    #[test]
    fn merge_agrees_with_the_rule_on_random_lists() {
        let mut seed = 202u64;
        let mut next = |n: u32| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) % u64::from(n)) as u32
        };
        for _ in 0..2000 {
            let mut lists = Vec::new();
            for _ in 0..3 {
                let mut cands: Vec<Candidate> = (0..next(8))
                    .map(|i| {
                        let start = next(20);
                        c(start, start + next(5), 0, i)
                    })
                    .collect();
                let mut out = Vec::new();
                select(&mut cands, &mut out);
                lists.push(out);
            }
            let mut fast = Vec::new();
            let mut slow = Vec::new();
            for list in &lists {
                merge(&mut fast, list);
                slow = merge_by_rule(&slow, list);
                assert_eq!(fast, slow, "{lists:?}");
            }
        }
    }
}
