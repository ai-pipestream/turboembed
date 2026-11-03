//! Term stacks over layers bound to another view, and the scale of
//! ProjectChunks' per-chunk and per-sentence work (#1220).
#[path = "inspect_safety/common.rs"]
#[allow(dead_code)]
mod common;
use common::*;
use std::time::Instant;
use turbo_analysis::{Document, Key, Provenance, Span, Spans, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{Chunks, Cut, Sentences, SparseTerms, Tokens};
use turbo_kserve::api::TURBO_E_INVALID_ARGUMENT;
use turbo_kserve::inspect::project_chunks;
use turbo_kserve::proto::result::OffsetMode;

/// `base()` ("é 😀", source Sentences and Tokens) plus a longer
/// normalized view and document SparseTerms bound to it, with an entry
/// span past the end of the source text.
fn sparse_on_a_longer_view() -> Document {
    let mut d = base();
    let source = d.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(source.text());
    b.replace(2, "e with acute");
    b.keep(source.text().len());
    let (text, alignment) = b.finish();
    let plain = Key::<View>::named("plain");
    d.add(&plain, View::derived(&source, text, alignment), Provenance::by("review-fixture")).unwrap();
    let mut terms = SparseTerms::<View>::new(model());
    terms.push(7, 1.5, Some(Span::new(13, 17)));
    terms.unit();
    let mut p = Provenance::by("review-fixture");
    p.on.push(plain.id());
    d.add(&Key::<SparseTerms<View>>::DEFAULT, terms, p).unwrap();
    d
}

#[test]
fn document_sparse_terms_on_another_view_are_refused_not_misplaced() {
    let bytes = graph(&sparse_on_a_longer_view());
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let mut req = project(bytes.clone());
        req.offsets = mode as i32;
        let out = std::panic::catch_unwind(|| project_chunks(&req));
        let Ok(out) = out else { panic!("{mode:?}: ProjectChunks panicked on a request") };
        let err = out.expect_err("view-1 spans placed on view-0 text");
        assert_eq!(err.code, TURBO_E_INVALID_ARGUMENT, "{mode:?}");
    }
}

/// `n` one-byte tokens "a" separated by spaces, one sentence and one
/// level-0 chunk per four tokens.
fn wide_document(n: u32) -> Document {
    let text: String = (0..n).map(|_| "a ").collect();
    let mut d = Document::new(text);
    let (mut s, mut c, mut t) = (Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE));
    for i in 0..n {
        t.push(Span::new(2 * i, 2 * i + 1));
    }
    for g in 0..n / 4 {
        s.push(Span::new(8 * g, 8 * g + 7));
        c.push(Span::new(8 * g, 8 * g + 7));
    }
    let units = (n / 4) as usize;
    d.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("review-fixture")).unwrap();
    d.add(&Key::DEFAULT, Tokens { spans: t, sentence: None }, Provenance::by("review-fixture")).unwrap();
    let chunks = Chunks {
        spans: c,
        level: vec![0; units],
        parent: vec![None; units],
        sentences: (0..units as u32).map(|g| turbo_analysis::IxRange::new(g, g + 1)).collect(),
        cut: (0..units).map(|g| if g == 0 { Cut::DocumentStart } else { Cut::Sentence }).collect(),
        cost: vec![0; units],
        model_tokens: vec![4; units],
        over_budget: vec![false; units],
        prompt_overhead: 0,
    };
    d.add(&Key::DEFAULT, chunks, Provenance::by("review-fixture")).unwrap();
    d
}

/// Best wall time of `rounds` runs per input, taken round-robin after one
/// warm-up pass. A burst of load on a shared runner then lands on every
/// size alike, and the minimum reflects the code rather than the neighbours.
fn interleaved_best<T>(inputs: &[T], rounds: usize, mut run: impl FnMut(&T)) -> Vec<f64> {
    inputs.iter().for_each(&mut run);
    let mut best = vec![f64::INFINITY; inputs.len()];
    for _ in 0..rounds {
        for (input, best) in inputs.iter().zip(&mut best) {
            let at = Instant::now();
            run(input);
            *best = best.min(at.elapsed().as_secs_f64());
        }
    }
    best
}

#[test]
fn term_stacks_scale_with_the_document_not_its_square() {
    // Two sizes eight times apart: linear work grows about eightfold, the
    // old per-chunk token and per-sentence chunk scans grew 64-fold. The
    // bound of 24 sits near the geometric middle, so neither runner load
    // (best of five, interleaved) nor cache effects on the large run cross
    // it. The old sizes were four apart with a 10x bound over 4x, and a
    // loaded runner measured 10.8x.
    let reqs: Vec<_> = [4_000, 32_000]
        .map(|n| {
            let mut r = project(graph(&wide_document(n)));
            r.include_drift_curve = true;
            r
        })
        .into();
    let best = interleaved_best(&reqs, 5, |req| {
        project_chunks(req).unwrap();
    });
    let (small, large) = (best[0], best[1]);
    eprintln!("4k tokens {small:.4}s, 32k tokens {large:.4}s");
    assert!(large < small * 24.0, "4k tokens {small:.4}s, 32k tokens {large:.4}s");
}
