//! A required link is a candidate of a compatible type, a PII layer handed
//! straight to the filter is held to the text, and a rejection for a
//! custom PII kind survives a graph round trip.

use std::sync::Arc;

use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Ix, IxRange, Key, Provenance, Span, Spans, TagSet, View, ViewId};
use turbo_analysis_layers::{CustomKinds, Entities, Pii, PiiEvidence, PiiKind, PiiMention, Tokens};
use turbo_entity_linker::{CatalogProvider, KbEntity, KnowledgeBase};
use turbo_gliner::{Decisions, HybridPrecision, PrecisionCandidates, PrecisionDecision, PrecisionRejection, codecs};

/// One `ORG` candidate over the whole text, `Ácme`: five bytes.
const TEXT: &str = "Ácme";

fn spans<L>(ranges: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(ViewId::SOURCE);
    for &(start, end) in ranges {
        out.push(Span::new(start, end));
    }
    out
}

fn document() -> Document {
    let kinds = TagSet::new("entity-kinds", &["ORG"]);
    let entities = Entities {
        spans: spans(&[(0, 5)]),
        tokens: vec![IxRange::new(0, 1)],
        kind: vec![kinds.tag("ORG").unwrap()],
        kinds,
        score: vec![0.9],
    };
    let mut doc = Document::new(TEXT);
    let tokens = Tokens { spans: spans(&[(0, 5)]), sentence: None };
    doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test:tokens")).unwrap();
    doc.add(&Key::<Entities>::DEFAULT, entities, Provenance::by("test:gliner")).unwrap();
    let decisions = Decisions { logits: vec![3.0], threshold: 0.0 };
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

fn linked(threshold: f64, entities: Vec<KbEntity>) -> PrecisionDecision {
    let provider = CatalogProvider::new(KnowledgeBase::from_entities("test:kb", entities).unwrap());
    let stage = HybridPrecision::new(threshold).unwrap().with_provider(Arc::new(provider)).require_link("ORG");
    filter(&document(), &stage, None).unwrap().get(Ix::new(0)).decision
}

#[test]
fn a_required_link_is_a_candidate_of_a_compatible_type() {
    let no_link = PrecisionDecision::Rejected(PrecisionRejection::NoCompatibleLink);
    let acme = |types: [&str; 1]| {
        KbEntity::new(format!("kb:acme:{}", types[0]), "Ácme").with_alias("Ácme", 1.0).with_types(types)
    };
    // No candidate is no link, even at a threshold of zero.
    assert_eq!(linked(0.0, Vec::new()), no_link);
    // A person of that name is not a link for an organisation, however
    // strong its prior.
    assert_eq!(linked(0.2, vec![acme(["PERSON"])]), no_link);
    assert_eq!(linked(0.0, vec![acme(["PERSON"])]), no_link);
    // An organisation of that name is.
    assert_eq!(linked(0.2, vec![acme(["ORG"])]), PrecisionDecision::Accepted);
    assert_eq!(linked(0.2, vec![acme(["PERSON"]), acme(["ORG"])]), PrecisionDecision::Accepted);
    // A label outside the required set keeps its model candidate.
    let empty = CatalogProvider::new(KnowledgeBase::from_entities("test:kb", Vec::new()).unwrap());
    let stage = HybridPrecision::new(0.2).unwrap().with_provider(Arc::new(empty)).require_link("LOC");
    assert_eq!(filter(&document(), &stage, None).unwrap().get(Ix::new(0)).decision, PrecisionDecision::Accepted);
}

fn pii(custom: CustomKinds, kind: PiiKind, span: Span) -> Pii {
    let mut pii = Pii::with_custom_kinds(ViewId::SOURCE, custom);
    let detector = pii.detector("test:pii", "1").unwrap();
    pii.push(PiiMention { span, kind, detector, evidence: &[PiiEvidence::Shape], score: None, normalized: "x" });
    pii
}

#[test]
fn a_pii_span_that_is_not_in_the_text_is_refused_by_the_filter() {
    let doc = document();
    let stage = HybridPrecision::new(0.2).unwrap().with_negative_rule("ORG", PiiKind::Email);
    let email = |span| pii(CustomKinds::new(), PiiKind::Email, span);
    // Past the end, and inside the two-byte `Á`: neither may veto a
    // candidate.
    for span in [Span::new(0, 4096), Span::new(1, 3)] {
        let error = filter(&doc, &stage, Some(&email(span))).unwrap_err();
        assert!(error.contains("outside the text or splits a character"), "{error}");
    }
    let conflict = PrecisionDecision::Rejected(PrecisionRejection::PiiConflict(PiiKind::Email));
    assert_eq!(filter(&doc, &stage, Some(&email(Span::new(0, 5)))).unwrap().get(Ix::new(0)).decision, conflict);
}

#[test]
fn a_rejection_for_a_custom_pii_kind_survives_a_graph_round_trip() {
    let custom = CustomKinds::from_names(["account-name"]).unwrap();
    let kind = custom.find("account-name").unwrap();
    let mut doc = document();
    let stage = HybridPrecision::new(0.2).unwrap().with_negative_rule("ORG", kind);
    let rows = filter(&doc, &stage, Some(&pii(custom, kind, Span::new(0, 5)))).unwrap();
    let conflict = PrecisionDecision::Rejected(PrecisionRejection::PiiConflict(kind));
    assert_eq!(rows.get(Ix::new(0)).decision, conflict);
    let reads = vec![Key::<Entities>::DEFAULT.id(), Key::<Decisions>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id()];
    let provenance = Provenance { version: "1", reads, ..Provenance::by("turbo:gliner-precision") };
    doc.add(&Key::<PrecisionCandidates>::DEFAULT, rows, provenance).unwrap();
    let bytes = export(&doc, &codecs()).unwrap().to_bytes().unwrap();
    let restored = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();
    let rows = restored.require(&Key::<PrecisionCandidates>::DEFAULT).unwrap();
    assert_eq!(rows.get(Ix::new(0)).decision, conflict);
}
