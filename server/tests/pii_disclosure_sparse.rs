//! `ProjectChunks` sparse term stacks under `rewritten_text` alone (#1904,
//! #1882 part C review): a `SparseTerms` layer's term strings come only
//! through the vocabulary, so a vocabulary of the source view is left out
//! even when the sparse entries sit on the disclosed masked view, and
//! sparse entries placed in the source view are refused, not projected
//! onto the masked text. With `full_text` the source's words come back.

use prost::Message;
use turbo_analysis::{Analyzer, Annotator, Document, Key, ModelRef, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::{SparseTerms, Tokens};
use turbo_chunker::terms::{TermIndexer, Vocabulary};
use turbo_kserve::inspect::project::project_chunks_under;
use turbo_kserve::proto::result::{OffsetMode, PiiOutputOptions, ProjectChunksRequest, ProjectChunksResponse};
use turbo_kserve::results::disclosure::PiiOutputPolicy;
use turbo_kserve::results::document::codecs;
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::{MASKED, MaskedView};

const PLANTED: &str = "alice.planted@example.com";

fn words(text: &str, view: ViewId) -> Tokens {
    let mut spans = Spans::new(view);
    let mut at = 0u32;
    for w in text.split(' ') {
        spans.push(Span::new(at, at + w.len() as u32));
        at += w.len() as u32 + 1;
    }
    Tokens { spans, sentence: None }
}

/// "Write to <PLANTED> before noon.", masked, with a vocabulary of the
/// source view's words, document sparse terms naming every vocabulary
/// entry, made on the view `on` with spans in that view's text, and the
/// default tokens (which choose the projected view) in `projected`.
fn graph(on: ViewId, projected: ViewId) -> Vec<u8> {
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(MaskedView::by_kind('*').unwrap().bound()).build().unwrap();
    let mut docs = vec![Document::new(format!("Write to {PLANTED} before noon."))];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let mut doc = docs.pop().unwrap();
    let source = doc.get_key(&Key::<View>::DEFAULT).unwrap().clone();
    let (vocab, _) = TermIndexer.annotate((&source, &words(source.text(), ViewId::SOURCE))).unwrap();
    let anchor = if on == ViewId::SOURCE { Key::<View>::DEFAULT.id() } else { MASKED.id() };
    let text = doc.text(on).to_owned();
    let spans = words(&text, on).spans;
    let mut sparse =
        SparseTerms::<View>::new(ModelRef { id: "test/sparse".into(), revision: "1".into(), sha256: String::new() });
    for t in 0..vocab.len().min(spans.len()) {
        sparse.push(t as u32, 1.0, Some(spans.get(turbo_analysis::Ix::new(t as u32))));
    }
    sparse.unit();
    let shown = doc.text(projected).to_owned();
    doc.add(&Key::<Tokens>::DEFAULT, words(&shown, projected), Provenance::by("test:tokens")).unwrap();
    doc.add(&Key::<Vocabulary>::DEFAULT, vocab, Provenance::by("test:vocabulary")).unwrap();
    doc.add(&Key::DEFAULT, sparse, Provenance { on: vec![anchor], ..Provenance::by("test:sparse") }).unwrap();
    turbo_analysis::export::export(&doc, codecs()).unwrap().to_bytes().unwrap()
}

fn project(graph: Vec<u8>, full_text: bool) -> Result<ProjectChunksResponse, String> {
    let req = ProjectChunksRequest {
        graph,
        offsets: OffsetMode::Utf8 as i32,
        pca_dims: 2,
        max_stack_segments: 10,
        pii_output: Some(PiiOutputOptions { full_text, rewritten_text: !full_text, ..Default::default() }),
        ..Default::default()
    };
    project_chunks_under(&req, PiiOutputPolicy::ALL).map_err(|e| e.message)
}

fn holds(out: &ProjectChunksResponse) -> bool {
    out.encode_to_vec().windows(PLANTED.len()).any(|w| w == PLANTED.as_bytes())
}

fn terms(out: &ProjectChunksResponse) -> Vec<String> {
    out.document_term_stack.as_ref().map(|s| s.segments.iter().map(|s| s.term.clone()).collect()).unwrap_or_default()
}

#[test]
fn sparse_terms_on_the_masked_view_name_no_source_word() {
    // Positive control: with the source disclosed, sparse terms on the
    // source view name the vocabulary's words, the planted address among
    // them.
    let full = project(graph(ViewId::SOURCE, ViewId::SOURCE), true).unwrap();
    assert!(terms(&full).iter().any(|t| t == PLANTED), "positive control: {:?}", terms(&full));
    let out = project(graph(ViewId(1), ViewId(1)), false).unwrap();
    assert!(!holds(&out), "the rewritten-only projection holds the value: {:?}", terms(&out));
    assert!(!terms(&out).is_empty() && terms(&out).iter().all(|t| t.starts_with("term:")), "{:?}", terms(&out));
}

#[test]
fn sparse_terms_in_the_source_view_are_refused_under_rewritten_text() {
    let e = project(graph(ViewId::SOURCE, ViewId(1)), false).expect_err("source-view sparse terms on the masked text");
    assert!(e.contains("spans are in view 0"), "{e}");
    assert!(!e.contains(PLANTED), "{e}");
}
