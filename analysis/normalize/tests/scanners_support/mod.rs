//! Shared checks for the tests of the scanning normalizers (URLs, social
//! media, numbers, shrink, contractions, symbols): the alignment
//! invariants, placeholder checks and a deterministic random
//! generator.

#![allow(dead_code)]

use turbo_analysis::{Alignment, Document, Key, Refs, Span, View};
use turbo_normalize::{Normalizer, normalize};
use turbo_normalize_layers::{PlaceholderKind, Placeholders};

/// Runs `n` on `src`, checks the alignment, and returns the output.
pub fn run<N: Normalizer>(n: &N, src: &str) -> String {
    let (out, a) = normalize(n, src);
    check_alignment(src, &out, &a);
    out
}

/// The invariants of an alignment between `src` and `out`: the edits are
/// ordered and rebuild `out` from `src`, none is a no-op, and every
/// character of `out` maps back to the source bytes that produced it
/// (itself outside the edits, the whole parent of its edit inside one).
pub fn check_alignment(src: &str, out: &str, a: &Alignment) {
    let edits = a.edits();
    let mut rebuilt = String::new();
    let (mut p, mut c) = (0usize, 0usize);
    for e in edits {
        let (ps, pe, cs, ce) =
            (e.parent.start as usize, e.parent.end as usize, e.child.start as usize, e.child.end as usize);
        assert!(ps >= p && cs >= c, "edits out of order in {src:?}: {edits:?}");
        assert_eq!(ps - p, cs - c, "unequal kept stretch before {e:?} in {src:?}");
        assert_ne!(&src[ps..pe], &out[cs..ce], "no-op edit {e:?} in {src:?}");
        rebuilt.push_str(&src[p..ps]);
        rebuilt.push_str(&out[cs..ce]);
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
    // The whole source maps forward to the whole output. (Backward, the
    // output may leave out bytes deleted at its edges.)
    assert_eq!(a.span_to_child(Span::new(0, src.len() as u32)), Span::new(0, out.len() as u32));
}

/// Runs a placeholder annotator through a document, checks the view's
/// alignment and that every placeholder is a source stretch whose output
/// is the one space it became, and returns the output text and the
/// placeholders as (source text, kind).
pub fn placeholders<A>(annotator: A, src: &str) -> (String, Vec<(String, PlaceholderKind)>)
where
    A: turbo_analysis::Annotator<Reads = (View,), Writes = (View, Placeholders)>,
{
    let doc = Document::new(src);
    let view = doc.get::<View>().expect("the source view");
    let input: Refs<'_, A> = (view,);
    let (out_view, found) = annotator.annotate(input).expect("annotate");
    turbo_analysis::Layer::check(&found).expect("placeholder layer invariants");
    let out = out_view.text().to_string();
    let a = out_view.alignment();
    check_alignment(src, &out, a);
    assert_eq!(found.spans.view(), view.id(), "placeholders are in the view read");
    let mut rows = Vec::new();
    for (ix, s) in found.spans.iter() {
        let child = a.span_to_child(s);
        let landed = &out[child.range()];
        assert!(landed.contains(' '), "{:?} landed on {landed:?} in {out:?}", &src[s.range()]);
        rows.push((src[s.range()].to_string(), found.kind[ix.get()]));
    }
    let mut doc = doc;
    let p = turbo_analysis::Provenance::by("test");
    doc.add(&Key::<View>::named("out"), out_view, p.clone()).expect("view accepted");
    doc.add(&Key::<Placeholders>::DEFAULT, found, p).expect("placeholders accepted");
    (out, rows)
}

/// xorshift64*: a deterministic generator with no dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }

    pub fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize % n
    }

    /// Up to 23 pieces from `pool`, as CharacterizationInputs.randomInput
    /// builds them.
    pub fn input(&mut self, pool: &[&str]) -> String {
        let pieces = self.below(24);
        (0..pieces).map(|_| pool[self.below(pool.len())]).collect()
    }
}
