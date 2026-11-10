//! What the precision layer and its inputs are held to: every span the
//! span of its tokens in the text, trimming that leaves no bare article or
//! stray punctuation, and a PII layer over the same view.

use std::sync::Arc;

use turbo_analysis::{Document, Ix, IxRange, Key, Layer, Provenance, Span, Spans, TagSet, View, ViewId};
use turbo_analysis_layers::{Entities, Pii, PiiEvidence, PiiKind, PiiMention, Tokens};
use turbo_gliner::{Decisions, HybridPrecision, PrecisionCandidates, PrecisionDecision, PrecisionRejection};

/// Tokens: `The`, `!!!!!`, `the`, `"`, `Ácme`, `"`, `The`.
const TEXT: &str = "The !!!!! the \" Ácme \" The";
const TOKENS: [(u32, u32); 7] = [(0, 3), (4, 9), (10, 13), (14, 15), (16, 21), (22, 23), (24, 27)];

fn spans<L>(view: ViewId, ranges: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(view);
    for &(start, end) in ranges {
        out.push(Span::new(start, end));
    }
    out
}

fn document() -> Document {
    let kinds = TagSet::new("entity-kinds", &["ORG"]);
    let org = kinds.tag("ORG").unwrap();
    let entities = Entities {
        spans: spans(ViewId::SOURCE, &[(0, 9), (10, 23), (24, 27)]),
        tokens: vec![IxRange::new(0, 2), IxRange::new(2, 6), IxRange::new(6, 7)],
        kind: vec![org; 3],
        kinds,
        score: vec![0.9, 0.8, 0.7],
    };
    let mut doc = Document::new(TEXT);
    let tokens = Tokens { spans: spans(ViewId::SOURCE, &TOKENS), sentence: None };
    doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test:tokens")).unwrap();
    doc.add(&Key::<Entities>::DEFAULT, entities, Provenance::by("test:gliner")).unwrap();
    let decisions = Decisions { logits: vec![3.0, 2.0, 1.0], threshold: 0.0 };
    doc.add(&Key::<Decisions>::DEFAULT, decisions, Provenance::by("test:gliner")).unwrap();
    doc
}

fn filter(doc: &Document, stage: &HybridPrecision, pii: Option<&Pii>) -> Result<PrecisionCandidates, String> {
    stage
        .filter(
            doc.get::<View>().unwrap(),
            doc.require(&Key::<Tokens>::DEFAULT).unwrap(),
            doc.require(&Key::<Entities>::DEFAULT).unwrap(),
            doc.require(&Key::<Decisions>::DEFAULT).unwrap(),
            pii,
        )
        .map_err(|e| e.to_string())
}

fn provenance() -> Provenance {
    let reads = vec![Key::<Entities>::DEFAULT.id(), Key::<Decisions>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id()];
    Provenance { version: "1", reads, ..Provenance::by("turbo:gliner-precision") }
}

#[test]
fn trimming_leaves_no_bare_article_and_no_punctuation_an_article_hid() {
    let doc = document();
    let rows = filter(&doc, &HybridPrecision::new(0.2).unwrap(), None).unwrap();
    let non_entity = PrecisionDecision::Rejected(PrecisionRejection::NonEntity);
    // `The !!!!!`: the punctuation goes and an article alone is no entity.
    assert_eq!(rows.get(Ix::new(0)).decision, non_entity);
    // `the " Ácme "`: the quote the article stood before goes with it.
    let acme = rows.get(Ix::new(1));
    assert_eq!(acme.decision, PrecisionDecision::Changed);
    assert_eq!(&TEXT[acme.span.range()], "Ácme");
    assert_eq!(acme.tokens, IxRange::new(4, 5));
    assert_eq!(acme.original_span, Span::new(10, 23));
    // `The` on its own.
    assert_eq!(rows.get(Ix::new(2)).decision, non_entity);
    let accepted = rows.accepted_entities();
    assert_eq!(accepted.spans.iter().map(|(_, s)| &TEXT[s.range()]).collect::<Vec<_>>(), ["Ácme"]);
    // What the stage wrote passes the checks a document holds it to.
    let mut doc = document();
    doc.add(&Key::<PrecisionCandidates>::DEFAULT, rows, provenance()).unwrap();

    // With article trimming off, an article is left as the model gave it.
    let kept = filter(&doc, &HybridPrecision::new(0.2).unwrap().trim_articles(false), None).unwrap();
    assert_eq!(kept.get(Ix::new(2)).decision, PrecisionDecision::Accepted);
    assert_eq!(&TEXT[kept.get(Ix::new(1)).span.range()], "the \" Ácme");
}

/// One accepted row over token 4 (`Ácme`), then `edit` applied to it.
fn candidates(edit: impl FnOnce(&mut PrecisionCandidates)) -> PrecisionCandidates {
    let kinds = TagSet::new("entity-kinds", &["ORG"]);
    let mut rows = PrecisionCandidates::new(ViewId::SOURCE, Arc::clone(&kinds));
    rows.spans.push(Span::new(16, 21));
    rows.original_spans.push(Span::new(16, 21));
    rows.tokens.push(IxRange::new(4, 5));
    rows.original_tokens.push(IxRange::new(4, 5));
    rows.kind.push(kinds.tag("ORG").unwrap());
    rows.logits.push(1.0);
    rows.score.push(0.7);
    rows.decision.push(PrecisionDecision::Accepted);
    edit(&mut rows);
    rows
}

#[test]
fn a_document_refuses_spans_that_are_not_the_span_of_their_tokens() {
    let add = |rows: PrecisionCandidates| {
        document().add(&Key::<PrecisionCandidates>::DEFAULT, rows, provenance()).map_err(|e| e.to_string())
    };
    add(candidates(|_| {})).unwrap();
    // Past the end of the text, in both columns.
    let past = Span::new(16, 4096);
    let mut outside = candidates(|rows| rows.original_spans[0] = past);
    outside.spans = spans(ViewId::SOURCE, &[(16, 4096)]);
    assert!(add(outside).is_err());
    // Inside a character: `Á` is two bytes.
    let mut split = candidates(|rows| rows.original_spans[0] = Span::new(17, 21));
    split.spans = spans(ViewId::SOURCE, &[(17, 21)]);
    assert!(add(split).is_err());
    // An original span that is in the text but is not its tokens' span.
    let error = add(candidates(|rows| {
        rows.original_spans[0] = Span::new(10, 21);
        rows.decision[0] = PrecisionDecision::Changed;
    }))
    .unwrap_err();
    assert!(error.contains("original span is not the span of its tokens"), "{error}");
    // Another view than the tokens.
    let mut other = candidates(|_| {});
    other.spans = spans(ViewId(7), &[(16, 21)]);
    assert!(add(other).is_err());
}

#[test]
fn the_layer_check_refuses_empty_token_ranges_and_spans_outside_the_original() {
    candidates(|_| {}).check().unwrap();
    let empty = candidates(|rows| rows.tokens[0] = IxRange::new(4, 4));
    assert!(empty.check().unwrap_err().contains("empty or reversed token bounds"));
    let empty_original = candidates(|rows| {
        rows.tokens[0] = IxRange::new(4, 4);
        rows.original_tokens[0] = IxRange::new(4, 4);
    });
    assert!(empty_original.check().unwrap_err().contains("empty or reversed token bounds"));
    let mut wider = candidates(|rows| rows.decision[0] = PrecisionDecision::Changed);
    wider.spans = spans(ViewId::SOURCE, &[(14, 21)]);
    assert!(wider.check().unwrap_err().contains("expands beyond its original span"));
}

fn pii(view: ViewId, span: Span, normalized: &str) -> Pii {
    let mut pii = Pii::new(view);
    let detector = pii.detector("test:pii", "1").unwrap();
    pii.push(PiiMention {
        span,
        kind: PiiKind::Email,
        detector,
        evidence: &[PiiEvidence::Shape],
        score: None,
        normalized,
    });
    pii
}

#[test]
fn a_pii_layer_over_another_view_is_refused() {
    let doc = document();
    let stage = HybridPrecision::new(0.2).unwrap().with_negative_rule("ORG", PiiKind::Email);
    // Same bytes as the candidate, but offsets into a different text.
    let error = filter(&doc, &stage, Some(&pii(ViewId(7), Span::new(16, 21), "x"))).unwrap_err();
    assert!(error.contains("the PII layer must be over the view of the precision inputs"), "{error}");
    // Over the same view the rule applies.
    let rows = filter(&doc, &stage, Some(&pii(ViewId::SOURCE, Span::new(16, 21), "x"))).unwrap();
    let conflict = PrecisionDecision::Rejected(PrecisionRejection::PiiConflict(PiiKind::Email));
    assert_eq!(rows.get(Ix::new(1)).decision, conflict);
}
