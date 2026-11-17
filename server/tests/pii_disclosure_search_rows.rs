//! `ProjectSearchDocument` on a PII rewrite view under `rewritten_text`
//! alone: the rows and scalars that are not text still describe the text
//! they were measured on (#1932, #2345). Sentence and chunk rows, chunk
//! token counts and cut reasons, boilerplate marks, and the noise,
//! artifact and text quality figures of the source view tell two values
//! apart that the rewrite made alike, so each is read only when its whole
//! lineage may leave. The same layers made from the rewrite view are kept.

use prost::Message;
use turbo_analysis::{Analyzer, Bound, Document, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, View, ViewId};
use turbo_analysis_layers::{Chunks, Cut, Entities, Sentences, Tokens};
use turbo_entity_linker::LinkedEntities;
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{ProjectSearchDocumentRequest, SearchAssessmentState, SearchDocumentProjection};
use turbo_kserve::results::disclosure::{Disclosure, PiiOutputPolicy};
use turbo_kserve::results::document::codecs;
use turbo_kserve::results::search_projection::{
    document_to_search_projection_bound, project_search_document_under, validate_search_projection,
};
use turbo_normalize::artifacts::ArtifactDetector;
use turbo_normalize::noise::NoiseScorer;
use turbo_normalize_layers::TextQualityAnnotator;
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::{REDACTED, RedactedView, RewritePolicy, StaticReplacer};
use turbo_sentdetect::RuleSentenceDetector;
use turbo_tokenize::uax29::UntypedWordTokenizer;

const EMAIL: &str = "alice.planted@example.com";
/// Another address, of another length and letter mix.
const OTHER: &str = "QZXJ.KVWPQ.ZZZZ.XXXXXXXX.YYYYYYYY@EXAMPLE.ORG";
const REWRITE: ViewId = ViewId(1);
const REWRITTEN: Disclosure = Disclosure { rewritten_text: true, full_text: false, detail: false };

/// Which view the sentence, token and quality steps of a fixture read.
#[derive(Clone, Copy, PartialEq)]
enum On {
    Source,
    Rewrite,
}

/// "Write to <value> today. Then call the office." with its findings, a
/// view where each finding is `[E]`, and real sentence, token, noise,
/// artifact and text quality steps run over the view `on`; then chunks,
/// one a sentence, whose token counts are the word counts of that view.
fn analyzed(value: &str, on: On) -> Document {
    let name = if on == On::Rewrite { REDACTED.name } else { "" };
    let analyzer = Analyzer::builder()
        .add(PiiDetector::cursor())
        .step(RedactedView::new(RewritePolicy::new(StaticReplacer::new("[E]"))).bound())
        .step(Bound::new(RuleSentenceDetector::new("eng")).read(0, name))
        .step(Bound::new(UntypedWordTokenizer::default()).read(0, name))
        .step(Bound::new(TextQualityAnnotator).read(0, name))
        .step(Bound::new(NoiseScorer::new()).read(0, name))
        .step(Bound::new(ArtifactDetector::default()).read(0, name))
        .build()
        .unwrap();
    let mut docs = vec![Document::new(format!("Write to {value} today. Then call the office."))];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let mut doc = docs.pop().unwrap();
    let view = if on == On::Rewrite { REWRITE } else { ViewId::SOURCE };
    let sentences = doc.get::<Sentences>().unwrap().spans.clone();
    assert_eq!(sentences.view(), view, "the fixture's sentences sit on the view asked for");
    let text = doc.text(view).to_owned();
    let mut chunks = Chunks {
        spans: Spans::new(view),
        level: Vec::new(),
        parent: Vec::new(),
        sentences: Vec::new(),
        cut: Vec::new(),
        cost: Vec::new(),
        model_tokens: Vec::new(),
        over_budget: Vec::new(),
        prompt_overhead: 0,
    };
    for (i, _) in sentences.iter() {
        let span = sentences.get(i);
        chunks.spans.push(span);
        chunks.level.push(0);
        chunks.parent.push(None);
        chunks.sentences.push(IxRange::new(i.get() as u32, i.get() as u32 + 1));
        chunks.cut.push(if i.get() == 0 { Cut::DocumentStart } else { Cut::Sentence });
        chunks.cost.push(i64::from(span.end));
        chunks.model_tokens.push(text[span.range()].split(['.', '@', ' ']).count() as u32);
        chunks.over_budget.push(false);
    }
    let reads = vec![Key::<Sentences>::DEFAULT.id()];
    doc.add(&Key::DEFAULT, chunks, Provenance { reads, ..Provenance::by("test:chunks") }).unwrap();
    doc
}

fn projected(doc: &Document, disclosure: &Disclosure) -> SearchDocumentProjection {
    let p = document_to_search_projection_bound("d", doc, REWRITE, OffsetMode::Utf8, None, &[], disclosure).unwrap();
    validate_search_projection(&p).unwrap();
    p
}

/// The same projection through the request handler, with the server's
/// policy capped at `rewritten_text`.
fn handled(doc: &Document) -> SearchDocumentProjection {
    let request = ProjectSearchDocumentRequest {
        doc_id: "d".into(),
        graph: turbo_analysis::export::export(doc, codecs()).unwrap().to_bytes().unwrap(),
        view: u32::from(REWRITE.0),
        pii_output: Some(REWRITTEN.to_options()),
        ..Default::default()
    };
    let policy = PiiOutputPolicy { full_text: false, detail: false, ..PiiOutputPolicy::ALL };
    project_search_document_under(&request, policy).unwrap()
}

fn holds(m: &SearchDocumentProjection, needle: &str) -> bool {
    m.encode_to_vec().windows(needle.len()).any(|w| w == needle.as_bytes())
}

const UNASSESSED: i32 = SearchAssessmentState::Unassessed as i32;
const ASSESSED: i32 = SearchAssessmentState::Assessed as i32;

#[test]
fn quality_measured_on_the_source_is_left_out() {
    let doc = analyzed(EMAIL, On::Source);
    assert_eq!(doc.text(REWRITE), "Write to [E] today. Then call the office.");
    for (path, p) in [("function", projected(&doc, &REWRITTEN)), ("handler", handled(&doc))] {
        let q = p.quality.unwrap();
        assert_eq!(
            (q.noise_state, q.artifacts_state, q.text_quality_state),
            (UNASSESSED, UNASSESSED, UNASSESSED),
            "{path}: a withheld measurement must not read as measured"
        );
        assert_eq!((q.worst_noise_score, q.union_noise_chars, q.artifact_count), (0.0, 0, 0), "{path}");
        assert!(q.metrics.is_none(), "{path}: the source's text quality metrics left: {:?}", q.metrics);
        assert!(q.noise_error.is_none() && q.artifacts_error.is_none(), "{path}");
    }
    // With the full text asked for, the same document answers with them.
    let q = projected(&doc, &Disclosure::FULL).quality.unwrap();
    assert_eq!((q.noise_state, q.artifacts_state, q.text_quality_state), (ASSESSED, ASSESSED, ASSESSED));
    assert!(q.metrics.is_some());
}

#[test]
fn sentence_and_chunk_rows_made_on_the_source_are_left_out() {
    let doc = analyzed(EMAIL, On::Source);
    for (path, p) in [("function", projected(&doc, &REWRITTEN)), ("handler", handled(&doc))] {
        assert!(p.sentences.is_empty(), "{path}: source sentence rows left: {:?}", p.sentences);
        assert!(p.chunks.is_empty(), "{path}: source chunk rows left: {:?}", p.chunks);
    }
    let full = projected(&doc, &Disclosure::FULL);
    assert_eq!(full.sentences.len(), doc.get::<Sentences>().unwrap().spans.len());
    assert_eq!(full.chunks.len(), full.sentences.len());
    assert!(full.chunks.iter().all(|c| c.model_tokens > 0), "{:?}", full.chunks);
}

/// The property behind both: two values the rewrite makes alike answer
/// with the same bytes, though every source measurement of them differs.
#[test]
fn two_values_give_the_same_answer_whatever_was_measured_on_the_source() {
    let (a, b) = (analyzed(EMAIL, On::Source), analyzed(OTHER, On::Source));
    assert_eq!(a.text(REWRITE), b.text(REWRITE), "the fixture redacts both alike");
    let (fa, fb) = (projected(&a, &Disclosure::FULL), projected(&b, &Disclosure::FULL));
    let metrics = |p: &SearchDocumentProjection| p.quality.as_ref().unwrap().metrics.clone();
    assert_ne!(metrics(&fa), metrics(&fb), "the fixture's source metrics differ");
    let tokens = |p: &SearchDocumentProjection| p.chunks.iter().map(|c| c.model_tokens).collect::<Vec<_>>();
    assert_ne!(tokens(&fa), tokens(&fb), "the fixture's source chunk token counts differ");
    assert_eq!(projected(&a, &REWRITTEN).encode_to_vec(), projected(&b, &REWRITTEN).encode_to_vec());
    assert_eq!(handled(&a).encode_to_vec(), handled(&b).encode_to_vec());
}

#[test]
fn rows_and_quality_made_on_the_rewrite_view_are_kept() {
    let doc = analyzed(EMAIL, On::Rewrite);
    for (path, p) in [("function", projected(&doc, &REWRITTEN)), ("handler", handled(&doc))] {
        assert!(!holds(&p, EMAIL), "{path}");
        assert_eq!(p.sentences.len(), 2, "{path}: {:?}", p.sentences);
        assert_eq!((p.sentences[0].span_start, p.sentences[1].span_end), (0, 41), "{path}");
        assert_eq!(p.chunks.len(), 2, "{path}: {:?}", p.chunks);
        assert_eq!((p.chunks[0].cut_reason.as_str(), p.chunks[1].cut_reason.as_str()), ("document_start", "sentence"));
        assert!(p.chunks.iter().all(|c| c.model_tokens > 0), "{path}: {:?}", p.chunks);
        let q = p.quality.unwrap();
        assert_eq!((q.noise_state, q.artifacts_state, q.text_quality_state), (ASSESSED, ASSESSED, ASSESSED), "{path}");
        assert!(q.metrics.is_some(), "{path}");
    }
    // The two values answer alike here too: nothing read the source.
    let other = analyzed(OTHER, On::Rewrite);
    assert_eq!(projected(&doc, &REWRITTEN).encode_to_vec(), projected(&other, &REWRITTEN).encode_to_vec());
}

/// Linked names are joined into the entity layer they were made on. Read
/// from the source, they carry its words, so an entity layer of the
/// rewrite view is judged with them and neither leaves.
#[test]
fn linked_names_read_from_the_source_do_not_leave_with_rewrite_view_entities() {
    let mut doc = analyzed(EMAIL, On::Rewrite);
    let word = Span::new(0, 5);
    let mut spans = Spans::new(REWRITE);
    spans.push(word);
    let tokens = Key::<Tokens>::named("ner");
    let view = REDACTED.id();
    doc.add(&tokens, Tokens { spans, sentence: None }, Provenance { reads: vec![view], ..Provenance::by("test:tok") })
        .unwrap();
    let mut spans = Spans::new(REWRITE);
    spans.push(word);
    let entities = Entities {
        spans,
        tokens: vec![IxRange::new(0, 1)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("test", &["CONTACT"]),
        score: vec![0.9],
    };
    doc.add(
        &Key::<Entities>::named("ner"),
        entities,
        Provenance { on: vec![tokens.id()], ..Provenance::by("test:ner") },
    )
    .unwrap();
    let mut linked = LinkedEntities::with_capacity(1);
    linked.kb_id.push(Some("Q1".into()));
    linked.provider_id.push("test".into());
    linked.canonical_name.push(EMAIL.into());
    linked.matched_alias.push(EMAIL.into());
    linked.score.push(0.9);
    linked.nil_reason.push(None);
    linked.nil_cluster.push(None);
    linked.candidates.push(Vec::new());
    let (reads, on) = (vec![Key::<View>::DEFAULT.id()], vec![Key::<Entities>::named("ner").id()]);
    doc.add(&Key::<LinkedEntities>::named("ner"), linked, Provenance { reads, on, ..Provenance::by("test:linker") })
        .unwrap();
    for (path, p) in [("function", projected(&doc, &REWRITTEN)), ("handler", handled(&doc))] {
        assert!(!holds(&p, EMAIL), "{path}: a linked name read from the source left: {:?}", p.entities);
    }
    let full = projected(&doc, &Disclosure::FULL).entities.unwrap();
    assert_eq!(full.affirmed[0].canonical_name, EMAIL, "with the full text the name is joined: {full:?}");
}
