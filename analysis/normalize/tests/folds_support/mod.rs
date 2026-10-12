//! Shared checks for the tests of the whitespace, punctuation, invisible
//! character and dehyphenation normalizers: the alignment invariants every
//! output must satisfy, a deterministic random generator, and helpers to
//! read spans the way the OpenNLP tests do.

#![allow(dead_code)]

use turbo_analysis::{Alignment, Span};
use turbo_normalize::{Normalizer, normalize};

/// The source text a span of the output maps back to.
pub fn covered<'a>(src: &'a str, a: &Alignment, start: usize, end: usize) -> &'a str {
    &src[a.span_to_parent(Span::new(start as u32, end as u32)).range()]
}

/// The output span a source span maps forward to.
pub fn forward(a: &Alignment, start: usize, end: usize) -> (u32, u32) {
    let s = a.span_to_child(Span::new(start as u32, end as u32));
    (s.start, s.end)
}

/// The output span a source span maps back from.
pub fn back(a: &Alignment, start: usize, end: usize) -> (u32, u32) {
    let s = a.span_to_parent(Span::new(start as u32, end as u32));
    (s.start, s.end)
}

/// Runs `n` on `src`, checks every alignment invariant, and returns the
/// output. `edit_ok(parent, child)` says whether one edit of the alignment
/// is a change this normalizer may make.
pub fn run_checked<N: Normalizer>(n: &N, src: &str, edit_ok: impl Fn(&str, &str) -> bool) -> String {
    let (out, a) = normalize(n, src);
    check_alignment(src, &out, &a, edit_ok);
    out
}

/// The invariants of an alignment between `src` and `out`:
///
/// - the edits are in order and do not overlap, on both sides, and the
///   text between two edits is the same on both sides, so `src` with every
///   edit applied is `out`;
/// - every edit is one `edit_ok` accepts, and none is a no-op;
/// - every character of `out` maps back to the source text that produced
///   it: itself when it lies outside the edits, the whole parent of the
///   edit it lies in otherwise;
/// - every character of `src` maps forward and back to a span that covers
///   it, and any span of whole characters does too.
pub fn check_alignment(src: &str, out: &str, a: &Alignment, edit_ok: impl Fn(&str, &str) -> bool) {
    let edits = a.edits();
    let mut rebuilt = String::new();
    let (mut p, mut c) = (0usize, 0usize);
    for e in edits {
        let (ps, pe) = (e.parent.start as usize, e.parent.end as usize);
        let (cs, ce) = (e.child.start as usize, e.child.end as usize);
        assert!(ps >= p && cs >= c, "edits out of order in {src:?}: {edits:?}");
        assert_eq!(ps - p, cs - c, "unequal kept stretch before {e:?} in {src:?}");
        rebuilt.push_str(&src[p..ps]);
        rebuilt.push_str(&out[cs..ce]);
        assert_ne!(&src[ps..pe], &out[cs..ce], "no-op edit {e:?} in {src:?}");
        assert!(
            edit_ok(&src[ps..pe], &out[cs..ce]),
            "unexpected edit {:?} -> {:?} in {src:?}",
            &src[ps..pe],
            &out[cs..ce]
        );
        (p, c) = (pe, ce);
    }
    rebuilt.push_str(&src[p..]);
    assert_eq!(rebuilt, out, "edits do not rebuild the output of {src:?}");

    for (i, ch) in out.char_indices() {
        let span = Span::new(i as u32, (i + ch.len_utf8()) as u32);
        let parent = a.span_to_parent(span);
        match edits.iter().find(|e| e.child.contains(span)) {
            Some(e) => assert_eq!(parent, e.parent, "output {ch:?} at {i} of {out:?} from {src:?}"),
            None => assert_eq!(&src[parent.range()], &out[span.range()], "output {ch:?} at {i} of {out:?}"),
        }
    }

    let bounds: Vec<usize> = src.char_indices().map(|(i, _)| i).chain([src.len()]).collect();
    for (k, &start) in bounds.iter().enumerate() {
        for &end in &bounds[k..] {
            let s = Span::new(start as u32, end as u32);
            let child = a.span_to_child(s);
            assert!(child.end as usize <= out.len() && out.is_char_boundary(child.start as usize));
            assert!(out.is_char_boundary(child.end as usize));
            // A span that maps to an empty output span is mapped forward only:
            // mapping an empty output span back is checked by
            // `empty_output_span_at_a_deletion` in tests/whitespace.rs.
            if !child.is_empty() {
                let round_trip = a.span_to_parent(child);
                let kept = without_edge_deletions(a, s);
                assert!(round_trip.contains(kept), "{s:?} came back as {round_trip:?} in {src:?}");
                assert!(widened(a, s).contains(round_trip), "{s:?} came back as {round_trip:?} in {src:?}");
            }
        }
    }
}

/// `s` without the deleted bytes at its ends. Deleted bytes belong to
/// neither neighbour, so a span that starts or ends inside a deletion
/// comes back without that part.
fn without_edge_deletions(a: &Alignment, s: Span) -> Span {
    let (mut start, mut end) = (s.start, s.end);
    let deletions: Vec<_> = a.edits().iter().filter(|e| e.child.is_empty()).collect();
    // Each deletion is its own edit, so a span edge inside a run of
    // deleted characters moves past one deletion at a time.
    for e in &deletions {
        if e.parent.start <= start && start < e.parent.end {
            start = e.parent.end;
        }
    }
    for e in deletions.iter().rev() {
        if e.parent.start < end && end <= e.parent.end {
            end = e.parent.start;
        }
    }
    Span::new(start, end.max(start))
}

/// `s` widened to every edit it overlaps: the most a mapping may add.
fn widened(a: &Alignment, s: Span) -> Span {
    let (mut start, mut end) = (s.start, s.end);
    for e in a.edits() {
        if e.parent.start < s.end && s.start < e.parent.end {
            start = start.min(e.parent.start);
            end = end.max(e.parent.end);
        }
    }
    Span::new(start, end)
}

/// An edit that replaces each source character by `fold(c)`, where `fold`
/// returns `None` for a character the normalizer keeps. Each replaced
/// character is its own edit, so the check maps every character.
pub fn per_char_edit(fold: impl Fn(char) -> Option<&'static str>) -> impl Fn(&str, &str) -> bool {
    move |parent, child| {
        let mut expected = String::new();
        for c in parent.chars() {
            match fold(c) {
                Some(s) => expected.push_str(s),
                None => return false,
            }
        }
        expected == child
    }
}

/// xorshift64*, enough for reproducible test inputs.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// A string of up to `max_pieces` pieces drawn from `pool`.
    pub fn text(&mut self, pool: &[&str], max_pieces: usize) -> String {
        let n = self.below(max_pieces + 1);
        let mut s = String::new();
        for _ in 0..n {
            s.push_str(pool[self.below(pool.len())]);
        }
        s
    }
}
