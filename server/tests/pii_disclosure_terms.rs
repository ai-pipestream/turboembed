//! `ProjectChunks` term stacks under `rewritten_text` alone (#1904): the
//! masked view's text is disclosed, so its words may leave, but a
//! vocabulary and term statistics over the source view hold the source's
//! words, the planted address among them. They are left out unless the
//! vocabulary is shown to be the disclosed view's own; with `full_text`
//! they come back.

use prost::Message;
use turbo_analysis::{Analyzer, Annotator, Document, Key, KeyId, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_chunker::terms::{TermCounter, TermIndexer};
use turbo_kserve::inspect::project::project_chunks_under;
use turbo_kserve::proto::result::{OffsetMode, PiiOutputOptions, ProjectChunksRequest, ProjectChunksResponse};
use turbo_kserve::results::disclosure::PiiOutputPolicy;
use turbo_kserve::results::document::codecs;
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::{MASKED, MaskedView};

const PLANTED: &str = "alice.planted@example.com";
const SOURCE_TOKENS: Key<Tokens> = Key::named("source");

/// Whitespace tokens of `text` in `view`.
fn words(text: &str, view: ViewId) -> Tokens {
    let mut spans = Spans::new(view);
    let mut at = 0u32;
    for w in text.split(' ') {
        spans.push(Span::new(at, at + w.len() as u32));
        at += w.len() as u32 + 1;
    }
    Tokens { spans, sentence: None }
}

/// Where the vocabulary comes from.
#[derive(Clone, Copy)]
enum Vocab {
    /// The source view, with no provenance naming its inputs (the review
    /// probe).
    SourceUnnamed,
    /// The source view and source tokens, named in its provenance.
    SourceNamed,
    /// The masked view and its tokens, named in its provenance.
    Masked,
}

/// "Write to <PLANTED> before noon.", scanned and masked, with tokens on
/// the masked view (the projected view) and the vocabulary and document
/// term statistics from `vocab`.
fn graph(vocab: Vocab) -> Vec<u8> {
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(MaskedView::by_kind('*').unwrap().bound()).build().unwrap();
    let mut docs = vec![Document::new(format!("Write to {PLANTED} before noon."))];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let mut doc = docs.pop().unwrap();
    let masked_text = doc.text(ViewId(1)).to_owned();
    assert!(!masked_text.contains(PLANTED), "view 1 is the masked view");
    let masked = words(&masked_text, ViewId(1));
    let source_text = doc.text(ViewId::SOURCE).to_owned();
    let source = words(&source_text, ViewId::SOURCE);
    let (view_key, tokens, on): (Key<View>, &Tokens, Vec<KeyId>) = match vocab {
        Vocab::SourceUnnamed => (Key::DEFAULT, &source, Vec::new()),
        Vocab::SourceNamed => (Key::DEFAULT, &source, vec![Key::<View>::DEFAULT.id(), SOURCE_TOKENS.id()]),
        Vocab::Masked => (MASKED, &masked, vec![MASKED.id(), Key::<Tokens>::DEFAULT.id()]),
    };
    let view = doc.get_key(&view_key).unwrap().clone();
    let (vocabulary, indexed) = TermIndexer.annotate((&view, tokens)).unwrap();
    let (stats,) = TermCounter::<View>::new().annotate((tokens, &indexed, &vocabulary, &view)).unwrap();
    doc.add(&SOURCE_TOKENS, words(&source_text, ViewId::SOURCE), Provenance::by("test:source-tokens")).unwrap();
    doc.add(&Key::DEFAULT, words(&masked_text, ViewId(1)), Provenance::by("test:masked-tokens")).unwrap();
    doc.add(&Key::DEFAULT, vocabulary, Provenance { on: on.clone(), ..Provenance::by("test:vocabulary") }).unwrap();
    doc.add(&Key::DEFAULT, stats, Provenance { on, ..Provenance::by("test:terms") }).unwrap();
    turbo_analysis::export::export(&doc, codecs()).unwrap().to_bytes().unwrap()
}

fn project(graph: Vec<u8>, full_text: bool) -> ProjectChunksResponse {
    let req = ProjectChunksRequest {
        graph,
        offsets: OffsetMode::Utf8 as i32,
        pca_dims: 2,
        max_stack_segments: 10,
        pii_output: Some(PiiOutputOptions { full_text, rewritten_text: !full_text, ..Default::default() }),
        ..Default::default()
    };
    project_chunks_under(&req, PiiOutputPolicy::ALL).unwrap()
}

fn holds(out: &ProjectChunksResponse) -> bool {
    out.encode_to_vec().windows(PLANTED.len()).any(|w| w == PLANTED.as_bytes())
}

fn terms(out: &ProjectChunksResponse) -> Vec<String> {
    let stack = out.document_term_stack.as_ref().expect("a document term stack");
    stack.segments.iter().map(|s| s.term.clone()).collect()
}

fn id_of(out: &ProjectChunksResponse, term: &str) -> Option<u32> {
    let stack = out.document_term_stack.as_ref()?;
    stack.segments.iter().find(|s| s.term == term).map(|s| s.term_id)
}

#[test]
fn rewritten_only_projection_must_not_release_source_term_statistics() {
    for vocab in [Vocab::SourceUnnamed, Vocab::SourceNamed] {
        // Positive control: with the source disclosed the value is a term.
        let full = project(graph(vocab), true);
        assert!(terms(&full).iter().any(|t| t == PLANTED), "positive control: {:?}", terms(&full));
        // rewritten_text alone: the stack is the masked view's words.
        let out = project(graph(vocab), false);
        assert!(!holds(&out), "the rewritten-only projection holds the value: {:?}", terms(&out));
        // The token fallback over the masked view, which folds case.
        assert_eq!(id_of(&out, "write"), Some(3), "{:?}", terms(&out));
        assert_eq!(id_of(&out, "Write"), None, "{:?}", terms(&out));
    }
}

#[test]
fn a_vocabulary_of_the_disclosed_view_is_kept() {
    let out = project(graph(Vocab::Masked), false);
    assert!(!holds(&out));
    // The vocabulary keeps case ("Write"), where the token fallback folds
    // it ("write").
    assert_eq!(id_of(&out, "Write"), Some(0), "{:?}", terms(&out));
}
