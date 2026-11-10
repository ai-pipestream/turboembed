use std::sync::Arc;

use turbo_analysis::export::{export, import};
use turbo_analysis::{Document, Ix, IxRange, Key, ModelRef, Provenance, Span, Spans, TagSet, ViewId};
use turbo_analysis_layers::{Entities, Pii, PiiEvidence, PiiKind, Tokens};
use turbo_entity_linker::{CatalogProvider, KbEntity, KnowledgeBase};
use turbo_gliner::{
    Decisions, HybridPrecision, PrecisionCandidates, PrecisionDecision, PrecisionRejection, codecs, precision_evidence,
};

const TEXT: &str = "The 東京 !!!!! Jobs Apple test@example.com";

fn spans<L>(ranges: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(ViewId::SOURCE);
    for &(start, end) in ranges {
        out.push(Span::new(start, end));
    }
    out
}

fn inputs() -> Document {
    let kinds = TagSet::new("entity-kinds", &["PERSON", "ORG", "LOC"]);
    let ranges = [(0, 10), (11, 16), (17, 21), (22, 27), (28, 44)];
    let entities = Entities {
        spans: spans(&ranges),
        tokens: [IxRange::new(0, 2), IxRange::new(2, 3), IxRange::new(3, 4), IxRange::new(4, 5), IxRange::new(5, 6)]
            .into_iter()
            .collect(),
        kind: [
            kinds.tag("LOC").unwrap(),
            kinds.tag("PERSON").unwrap(),
            kinds.tag("PERSON").unwrap(),
            kinds.tag("ORG").unwrap(),
            kinds.tag("PERSON").unwrap(),
        ]
        .into_iter()
        .collect(),
        kinds: Arc::clone(&kinds),
        score: vec![0.9, 0.8, 0.7, 0.6, 0.5],
    };
    let decisions = Decisions { logits: vec![4.0, 3.0, 2.0, 1.0, 0.0], threshold: 0.0 };
    let tokens = Tokens { spans: spans(&[(0, 3), (4, 10), (11, 16), (17, 21), (22, 27), (28, 44)]), sentence: None };
    let model = ModelRef { id: "test:gliner".into(), revision: "1".into(), sha256: "fixture".into() };
    let producer = Provenance { model: Some(model), ..Provenance::by("turbo:gliner") };
    let mut doc = Document::new(TEXT);
    doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test:tokens")).unwrap();
    doc.add(&Key::<Entities>::DEFAULT, entities, producer.clone()).unwrap();
    doc.add(&Key::<Decisions>::DEFAULT, decisions, producer).unwrap();
    let mut pii = Pii::new(ViewId::SOURCE);
    let detector = pii.detector("test:pii", "1").unwrap();
    pii.push(turbo_analysis_layers::PiiMention {
        span: Span::new(28, 44),
        kind: PiiKind::Email,
        detector,
        evidence: &[PiiEvidence::Shape],
        score: None,
        normalized: "test@example.com",
    });
    doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("test:pii")).unwrap();
    doc
}

fn provider() -> CatalogProvider {
    let entities = vec![
        KbEntity::new("loc:tokyo", "東京").with_alias("東京", 1.0).with_types(["LOC"]),
        KbEntity::new("person:jobs", "Jobs").with_alias("Jobs", 1.0).with_types(["PERSON"]),
        KbEntity::new("org:apple", "Apple").with_alias("Apple", 1.0).with_types(["ORG"]),
    ];
    CatalogProvider::new(KnowledgeBase::from_entities("test:kb", entities).unwrap())
}

#[test]
fn precision_records_accepted_changed_and_rejected_candidates_with_evidence() {
    let mut doc = inputs();
    let model = doc.provenance(Key::<Entities>::DEFAULT.id()).unwrap().model.clone();
    let stage = HybridPrecision::new(0.2)
        .unwrap()
        .with_provider(Arc::new(provider()))
        .require_link("LOC")
        .require_link("PERSON")
        .require_link("ORG")
        .with_negative_rule("PERSON", PiiKind::Email);
    let decisions = doc.require(&Key::<Decisions>::DEFAULT).unwrap();
    let filtered = stage
        .filter(
            doc.get::<turbo_analysis::View>().unwrap(),
            doc.require(&Key::<Tokens>::DEFAULT).unwrap(),
            doc.require(&Key::<Entities>::DEFAULT).unwrap(),
            decisions,
            doc.get::<Pii>(),
        )
        .unwrap();
    assert_eq!(filtered.spans.len(), 5);
    let rows: Vec<_> = (0..5).map(|i| filtered.get(Ix::new(i as u32))).collect();
    assert_eq!(rows[0].decision, PrecisionDecision::Changed);
    assert_eq!(rows[0].original_span, Span::new(0, 10));
    assert_eq!(rows[0].span, Span::new(4, 10));
    assert_eq!(rows[0].tokens, IxRange::new(1, 2));
    assert_eq!(rows[1].decision, PrecisionDecision::Rejected(PrecisionRejection::NonEntity));
    assert_eq!(rows[2].decision, PrecisionDecision::Accepted);
    assert_eq!(rows[3].decision, PrecisionDecision::Accepted);
    assert_eq!(rows[4].decision, PrecisionDecision::Rejected(PrecisionRejection::PiiConflict(PiiKind::Email)));
    assert_eq!(rows[2].logit, 2.0);
    assert_eq!(rows[3].score, 0.6);

    let accepted = filtered.accepted_entities();
    assert_eq!(accepted.spans.len(), 3);
    let source: Vec<_> = accepted.spans.iter().map(|(_, span)| doc.source()[span.range()].to_owned()).collect();
    assert_eq!(source, ["東京", "Jobs", "Apple"]);
    let reads = vec![Key::<Entities>::DEFAULT.id(), Key::<Decisions>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id()];
    let provenance = Provenance { version: "1", reads, ..Provenance::by("turbo:gliner-precision") };
    doc.add(&Key::<PrecisionCandidates>::DEFAULT, filtered, provenance).unwrap();
    let evidence = precision_evidence(&doc, Key::<PrecisionCandidates>::DEFAULT).unwrap();
    evidence.verify(&doc).unwrap();
    assert_eq!(evidence.raw_of(0), Some(4.0));
    assert_eq!(evidence.abstention_of(1), Some("non-entity"));
    assert_eq!(evidence.abstention_of(4), Some("pii-conflict:5"));
    assert_eq!(evidence.support_of(0).len(), 3);
    doc.add(&Key::DEFAULT, evidence, Provenance::by("turbo:evidence")).unwrap();

    let bytes = export(&doc, &codecs()).unwrap().to_bytes().unwrap();
    let graph = turbo_analysis::export::DocumentGraph::from_bytes(&bytes).unwrap();
    let restored = import(&graph, &codecs()).unwrap();
    assert_eq!(restored.provenance(Key::<Entities>::DEFAULT.id()).unwrap().model, model);
    let round_trip = restored.require(&Key::<PrecisionCandidates>::DEFAULT).unwrap();
    assert_eq!(round_trip.spans.len(), 5);
    for (i, row) in rows.iter().enumerate() {
        let actual = round_trip.get(Ix::new(i as u32));
        assert_eq!(actual.original_span, row.original_span);
        assert_eq!(actual.span, row.span);
        assert_eq!(actual.original_tokens, row.original_tokens);
        assert_eq!(actual.tokens, row.tokens);
        assert_eq!(actual.kind, row.kind);
        assert_eq!(actual.logit, row.logit);
        assert_eq!(actual.score, row.score);
        assert_eq!(actual.decision, row.decision);
    }
    restored
        .require(&Key::<turbo_analysis::Evidence<PrecisionCandidates>>::DEFAULT)
        .unwrap()
        .verify(&restored)
        .unwrap();
    let accepted_restored = round_trip.accepted_entities();
    assert_eq!(accepted_restored.spans.len(), 3);
    assert_eq!(
        accepted_restored.spans.iter().map(|(_, span)| &restored.source()[span.range()]).collect::<Vec<_>>(),
        source
    );
}

#[test]
fn precision_can_leave_boundaries_and_reject_every_linked_candidate() {
    let doc = inputs();
    let decisions = doc.require(&Key::<Decisions>::DEFAULT).unwrap();
    let empty = CatalogProvider::new(KnowledgeBase::from_entities("test:empty", Vec::new()).unwrap());
    let stage =
        HybridPrecision::new(1.0).unwrap().with_provider(Arc::new(empty)).trim_articles(false).reject_non_alpha(false);
    let filtered = stage.filter(doc.get().unwrap(), doc.get().unwrap(), doc.get().unwrap(), decisions, None).unwrap();
    for i in 0..filtered.spans.len() {
        let row = filtered.get(Ix::new(i as u32));
        assert_eq!(row.decision, PrecisionDecision::Rejected(PrecisionRejection::NoCompatibleLink));
        assert_eq!(row.original_span, row.span);
    }
    assert_eq!(filtered.accepted_entities().spans.len(), 0);
}

#[test]
fn precision_rejects_empty_and_out_of_bounds_token_ranges() {
    let doc = inputs();
    let stage = HybridPrecision::new(0.2).unwrap();
    for range in [IxRange::new(0, 0), IxRange::new(0, 7)] {
        let original = doc.require(&Key::<Entities>::DEFAULT).unwrap();
        let mut entities = Entities {
            spans: original.spans.clone(),
            tokens: original.tokens.clone(),
            kind: original.kind.clone(),
            kinds: Arc::clone(&original.kinds),
            score: original.score.clone(),
        };
        entities.tokens[0] = range;
        let error = stage
            .filter(
                doc.get().unwrap(),
                doc.require(&Key::<Tokens>::DEFAULT).unwrap(),
                &entities,
                doc.require(&Key::<Decisions>::DEFAULT).unwrap(),
                None,
            )
            .unwrap_err();
        assert!(error.to_string().contains("empty or out-of-bounds"), "{error}");
    }
}
