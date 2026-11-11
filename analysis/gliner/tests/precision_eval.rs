use std::sync::Arc;
use std::time::Instant;

use turbo_analysis::export::{export, import};
use turbo_analysis::{Document, IxRange, Key, Layer, Provenance, Span, Spans, TagSet, ViewId};
use turbo_analysis_layers::{Entities, Pii, PiiEvidence, PiiKind, Tokens};
use turbo_entity_linker::{CatalogProvider, KbEntity, KbProvider, KnowledgeBase};
use turbo_gliner::{
    Decisions, HybridPrecision, PrecisionCandidates, PrecisionDecision, PrecisionEvaluation, codecs, precision_evidence,
};

const TEXT: &str = "The 東京 !!!!! Jobs Apple test@example.com Museum 😀";
const TOKENS: [(u32, u32); 8] = [(0, 3), (4, 10), (11, 16), (17, 21), (22, 27), (28, 44), (45, 51), (52, 56)];
const MODEL: [(usize, usize, &str); 7] = [
    (0, 2, "LOC"),
    (2, 3, "PERSON"),
    (3, 4, "PERSON"),
    (4, 5, "ORG"),
    (5, 6, "PERSON"),
    (6, 7, "FAC"),
    (7, 8, "PERSON"),
];
const GOLD: [(usize, usize, &str); 4] = [(1, 2, "LOC"), (3, 4, "PERSON"), (4, 5, "ORG"), (6, 7, "FAC")];

struct Fixture {
    doc: Document,
    dictionary: Entities,
    gold: Entities,
}

fn push(entities: &mut Entities, span: Span, tokens: IxRange<Tokens>, label: &str, score: f64) {
    entities.spans.push(span);
    entities.tokens.push(tokens);
    entities.kind.push(entities.kinds.tag(label).unwrap());
    entities.score.push(score);
}

fn empty(kinds: &Arc<TagSet>) -> Entities {
    Entities {
        spans: Spans::new(ViewId::SOURCE),
        tokens: Vec::new(),
        kind: Vec::new(),
        kinds: Arc::clone(kinds),
        score: Vec::new(),
    }
}

fn fixture(repeats: usize) -> Fixture {
    let kinds = TagSet::new("entity-kinds", &["PERSON", "ORG", "LOC", "FAC"]);
    let mut text = String::new();
    let mut tokens = Tokens { spans: Spans::new(ViewId::SOURCE), sentence: None };
    let mut model = empty(&kinds);
    let mut dictionary = empty(&kinds);
    let mut gold = empty(&kinds);
    let mut pii = Pii::new(ViewId::SOURCE);
    let detector = pii.detector("test:pii", "1").unwrap();
    let mut logits = Vec::new();
    for repeat in 0..repeats {
        let base = (repeat * (TEXT.len() + 1)) as u32;
        text.push_str(TEXT);
        text.push('\n');
        for (start, end) in TOKENS {
            tokens.spans.push(Span::new(base + start, base + end));
        }
        let token_base = (repeat * TOKENS.len()) as u32;
        for (first, last, label) in MODEL {
            let span = Span::new(base + TOKENS[first].0, base + TOKENS[last - 1].1);
            let range = IxRange::new(token_base + first as u32, token_base + last as u32);
            push(&mut model, span, range, label, 0.8);
            logits.push(2.0);
        }
        for &(first, last, label) in &GOLD {
            let span = Span::new(base + TOKENS[first].0, base + TOKENS[last - 1].1);
            let range = IxRange::new(token_base + first as u32, token_base + last as u32);
            push(&mut gold, span, range, label, 1.0);
            if label != "FAC" {
                push(&mut dictionary, span, range, label, 1.0);
            }
        }
        pii.push(turbo_analysis_layers::PiiMention {
            span: Span::new(base + 28, base + 44),
            kind: PiiKind::Email,
            detector,
            evidence: &[PiiEvidence::Shape],
            score: None,
            normalized: "test@example.com",
        });
    }
    let mut doc = Document::new(text);
    doc.add(&Key::<Tokens>::DEFAULT, tokens, Provenance::by("test:tokens")).unwrap();
    doc.add(&Key::<Entities>::DEFAULT, model, Provenance::by("turbo:gliner")).unwrap();
    doc.add(&Key::<Decisions>::DEFAULT, Decisions { logits, threshold: 0.0 }, Provenance::by("turbo:gliner")).unwrap();
    doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("test:pii")).unwrap();
    Fixture { doc, dictionary, gold }
}

fn provider() -> CatalogProvider {
    let entities = vec![
        KbEntity::new("loc:tokyo", "東京").with_alias("東京", 1.0).with_types(["LOC"]),
        KbEntity::new("person:jobs", "Jobs").with_alias("Jobs", 1.0).with_types(["PERSON"]),
        KbEntity::new("org:apple", "Apple").with_alias("Apple", 1.0).with_types(["ORG"]),
    ];
    CatalogProvider::new(KnowledgeBase::from_entities("test:kb", entities).unwrap())
}

fn stage() -> HybridPrecision {
    HybridPrecision::new(0.2)
        .unwrap()
        .with_provider(Arc::new(provider()))
        .require_link("LOC")
        .require_link("PERSON")
        .require_link("ORG")
        .with_negative_rule("PERSON", PiiKind::Email)
}

#[test]
fn independent_arms_report_held_out_type_and_false_positives() {
    let fixture = fixture(1);
    assert!(provider().generate_candidates("Museum", None, 4, 1.0).is_empty());
    let filtered = stage()
        .filter(
            fixture.doc.get().unwrap(),
            fixture.doc.require(&Key::<Tokens>::DEFAULT).unwrap(),
            fixture.doc.require(&Key::<Entities>::DEFAULT).unwrap(),
            fixture.doc.require(&Key::<Decisions>::DEFAULT).unwrap(),
            fixture.doc.get::<Pii>(),
        )
        .unwrap();
    let evaluation = PrecisionEvaluation::new(
        fixture.doc.require(&Key::<Entities>::DEFAULT).unwrap(),
        &fixture.dictionary,
        &filtered,
        &fixture.gold,
    )
    .unwrap();
    assert_eq!((evaluation.model_only.candidates, evaluation.model_only.accepted), (7, 7));
    assert_eq!((evaluation.model_only.true_positives, evaluation.model_only.false_positives.len()), (3, 4));
    assert_eq!((evaluation.model_only.false_negatives, evaluation.dictionary_only.false_negatives), (1, 1));
    assert_eq!((evaluation.dictionary_only.true_positives, evaluation.dictionary_only.false_positives.len()), (3, 0));
    assert_eq!((evaluation.hybrid.candidates, evaluation.hybrid.accepted, evaluation.hybrid.rejected), (7, 4, 3));
    assert_eq!((evaluation.hybrid.true_positives, evaluation.hybrid.false_negatives), (4, 0));
    assert_eq!(evaluation.hybrid.precision, 1.0);
    assert_eq!(evaluation.hybrid.recall, 1.0);
    assert!(evaluation.hybrid.precision > evaluation.model_only.precision);
    assert!(evaluation.hybrid.recall > evaluation.model_only.recall);
    assert!(
        filtered.accepted_entities().spans.iter().any(|(_, span)| span == Span::new(45, 51)),
        "the dictionary-free held-out FAC candidate survives"
    );
    println!("EVALUATION_JSON={}", serde_json::to_string_pretty(&evaluation).unwrap());
}

#[test]
fn long_document_reduction_latency_and_graph_semantics() {
    let repeats = 256;
    let mut fixture = fixture(repeats);
    let started = Instant::now();
    let filtered = stage()
        .filter(
            fixture.doc.get().unwrap(),
            fixture.doc.require(&Key::<Tokens>::DEFAULT).unwrap(),
            fixture.doc.require(&Key::<Entities>::DEFAULT).unwrap(),
            fixture.doc.require(&Key::<Decisions>::DEFAULT).unwrap(),
            fixture.doc.get::<Pii>(),
        )
        .unwrap();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(filtered.spans.len(), repeats * MODEL.len());
    assert_eq!(filtered.accepted().count(), repeats * GOLD.len());
    assert_eq!(filtered.decision.iter().filter(|d| matches!(d, PrecisionDecision::Rejected(_))).count(), repeats * 3);
    let reads = vec![Key::<Entities>::DEFAULT.id(), Key::<Decisions>::DEFAULT.id(), Key::<Tokens>::DEFAULT.id()];
    let provenance = Provenance { version: "1", reads, ..Provenance::by("turbo:gliner-precision") };
    fixture.doc.add(&Key::<PrecisionCandidates>::DEFAULT, filtered, provenance).unwrap();
    let evidence = precision_evidence(&fixture.doc, Key::<PrecisionCandidates>::DEFAULT).unwrap();
    evidence.verify(&fixture.doc).unwrap();
    fixture.doc.add(&Key::DEFAULT, evidence, Provenance::by("turbo:evidence")).unwrap();
    let bytes = export(&fixture.doc, &codecs()).unwrap().to_bytes().unwrap();
    let graph = turbo_analysis::export::DocumentGraph::from_bytes(&bytes).unwrap();
    let restored = import(&graph, &codecs()).unwrap();
    let round_trip = restored.require(&Key::<PrecisionCandidates>::DEFAULT).unwrap();
    round_trip.check().unwrap();
    assert_eq!(round_trip.spans.len(), repeats * MODEL.len());
    for (i, row) in round_trip.accepted().enumerate() {
        let text = &restored.source()[row.span.range()];
        assert!(matches!(text, "東京" | "Jobs" | "Apple" | "Museum"), "row {i}: {text}");
    }
    println!(
        "LONG_DOCUMENT_JSON={}",
        serde_json::json!({
            "utf8_bytes": restored.source().len(),
            "candidates": round_trip.spans.len(),
            "accepted": round_trip.accepted().count(),
            "rejected": round_trip.spans.len() - round_trip.accepted().count(),
            "precision_ms": elapsed_ms,
            "graph_round_trip": true
        })
    );
}
