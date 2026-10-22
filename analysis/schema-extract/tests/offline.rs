//! The extraction annotator end to end on the reduced model, with no
//! checkpoint: the layers it writes and their byte offsets (nested and
//! overlapping values, multi-byte text), empty and zero-width documents,
//! the deterministic layers that check candidates, evidence, and the
//! record layer across the document graph.

mod common;

use turbo_analysis::export::{export, import};
use turbo_analysis::{Analyzer, Document, Evidence, EvidenceStep, Key, Layer, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Categories, Entities, Numbers, Pii, Tokens};
use turbo_numeric::{NumberNotation, NumberScanner};
use turbo_pii::PiiDetector;
use turbo_schema_extract::export::codecs;
use turbo_schema_extract::layer::{Records, source_code};
use turbo_schema_extract::{
    Cardinality, Check, ClassificationTask, EntityType, ExtractionSchema, FieldSpec, Label, Limits, SchemaAnnotator,
    Source, StructureSpec,
};

fn field(name: &str, cardinality: Cardinality, checks: Vec<Check>) -> FieldSpec {
    FieldSpec { name: name.into(), description: None, threshold: None, cardinality, choices: None, checks }
}

fn schema(checks: Vec<Check>) -> ExtractionSchema {
    ExtractionSchema {
        entities: vec![
            EntityType {
                name: "person".into(),
                description: Some("a human".into()),
                threshold: None,
                cardinality: Cardinality::Repeated,
                checks: Vec::new(),
            },
            EntityType {
                name: "place".into(),
                description: None,
                threshold: None,
                cardinality: Cardinality::Single,
                checks: Vec::new(),
            },
        ],
        classifications: vec![ClassificationTask {
            name: "tone".into(),
            labels: vec![
                Label { name: "calm".into(), description: None },
                Label { name: "urgent".into(), description: None },
            ],
            multi_label: true,
            threshold: 0.0,
            top_k: None,
        }],
        structures: vec![StructureSpec {
            name: "order".into(),
            fields: vec![
                field("buyer", Cardinality::Repeated, Vec::new()),
                field("item", Cardinality::Repeated, Vec::new()),
                field("amount", Cardinality::Repeated, checks),
                FieldSpec {
                    choices: Some(vec!["paid".into(), "open".into()]),
                    ..field("status", Cardinality::Single, Vec::new())
                },
            ],
        }],
    }
}

/// A document with whitespace tokens and the layers `with` adds.
fn document(text: &str) -> Document {
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut at = 0;
    for w in text.split_whitespace() {
        let start = text[at..].find(w).unwrap() + at;
        spans.push(Span::new(start as u32, (start + w.len()) as u32));
        at = start + w.len();
    }
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    doc
}

fn run(doc: Document, schema: ExtractionSchema, threshold: f32, given: &[&str]) -> Document {
    let annotator = SchemaAnnotator::new(common::offline(Limits::default()), schema, threshold).unwrap();
    let mut b = Analyzer::builder().given(Key::<Tokens>::DEFAULT);
    if given.contains(&"numbers") {
        b = b.add(NumberScanner::new(NumberNotation::LatinUs));
    }
    if given.contains(&"pii") {
        b = b.add(PiiDetector::cursor());
    }
    let analyzer = b.add(annotator).step(EvidenceStep::new(Key::<Records>::DEFAULT)).build().unwrap();
    let mut docs = vec![doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs.pop().unwrap()
}

fn on_boundaries(text: &str, s: Span) -> bool {
    let (a, b) = (s.start as usize, s.end as usize);
    a < b && b <= text.len() && text.is_char_boundary(a) && text.is_char_boundary(b)
}

#[test]
fn values_are_byte_spans_of_the_text_and_may_nest() {
    let text = "Zoë ordered two crêpes from Café São Paulo for €12.50; Ana paid.";
    let doc = run(document(text), schema(Vec::new()), 0.0, &[]);
    let r = doc.get::<Records>().unwrap();
    assert_eq!(r.check(), Ok(()));
    assert!(!r.spans.is_empty(), "a threshold of 0 keeps values");
    for (_, s) in r.spans.iter() {
        assert!(on_boundaries(text, s), "{s:?} is not a span of the text");
    }
    // Values of one record overlap across fields: the layer keeps both.
    let overlapping = (0..r.spans.len()).any(|i| {
        (0..r.spans.len()).any(|j| {
            let (a, b) =
                (r.spans.get(turbo_analysis::Ix::new(i as u32)), r.spans.get(turbo_analysis::Ix::new(j as u32)));
            i != j && r.record[i] == r.record[j] && r.path[i] != r.path[j] && a.start < b.end && b.start < a.end
        })
    });
    assert!(overlapping, "no two fields of a record overlap");
    // The closed choice is a choice value, not a span.
    assert!(r.choice.iter().all(|c| ["paid", "open"].contains(&r.choices.label(*c))));
    let e = doc.get::<Entities>().unwrap();
    assert_eq!(e.check(), Ok(()));
    assert!(e.spans.iter().all(|(_, s)| on_boundaries(text, s)));
    assert_eq!(e.spans.iter().filter(|(i, _)| e.kinds.label(e.kind[i.get()]) == "place").count(), 1, "place is single");
    let c = doc.get::<Categories>().unwrap();
    assert_eq!(c.check(), Ok(()));
    assert_eq!(c.categories.len(), 2);
    let ev = doc.get::<Evidence<Records>>().unwrap();
    assert_eq!(ev.verify(&doc), Ok(()));
    assert_eq!(ev.producer_of(0).model.as_ref().unwrap().id, "reduced");
}

#[test]
fn empty_and_zero_width_documents_extract_nothing_outside_the_text() {
    for text in ["", "   ", "\u{200B}", "\u{FEFF}\u{200B} \u{FEFF}", "a\u{200B}b"] {
        let doc = run(document(text), schema(Vec::new()), 0.0, &[]);
        let r = doc.get::<Records>().unwrap();
        assert_eq!(r.check(), Ok(()), "{text:?}");
        assert!(r.spans.iter().all(|(_, s)| on_boundaries(text, s)), "{text:?}");
        let e = doc.get::<Entities>().unwrap();
        assert!(e.spans.iter().all(|(_, s)| on_boundaries(text, s)), "{text:?}");
    }
}

#[test]
fn deterministic_layers_require_and_record_support() {
    let text = "Send 42 units to bob@example.com before 7 May; total 1,250 dollars.";
    let numbers = run(document(text), schema(vec![Check::Require(Source::Numbers)]), 0.0, &["numbers"]);
    let n = numbers.get::<Numbers>().unwrap();
    let r = numbers.get::<Records>().unwrap();
    let amount = r.paths.tag("order.amount").unwrap();
    let rows: Vec<usize> = (0..r.spans.len()).filter(|&i| r.path[i] == amount).collect();
    assert!(!rows.is_empty(), "amounts survive a number check at threshold 0");
    for i in rows {
        let s = r.spans.get(turbo_analysis::Ix::new(i as u32));
        let support: Vec<_> = r.support(i).collect();
        assert!(!support.is_empty(), "amount {s:?} kept without a number");
        for (source, row) in support {
            assert_eq!(source, Source::Numbers);
            let num = n.spans.get(turbo_analysis::Ix::new(row));
            assert!(num.start < s.end && s.start < num.end, "{num:?} does not overlap {s:?}");
        }
    }
    let ev = numbers.get::<Evidence<Records>>().unwrap();
    assert_eq!(ev.verify(&numbers), Ok(()));
    // Without the layer, a required check keeps nothing.
    let none = run(document(text), schema(vec![Check::Require(Source::Numbers)]), 0.0, &[]);
    let r = none.get::<Records>().unwrap();
    assert!(r.path.iter().all(|&p| p != amount), "an amount kept with no numbers layer");
    // A boost only adds: every value it lifts names its PII row.
    let pii = run(document(text), schema(vec![Check::Boost(Source::Pii, 1.0)]), 0.99, &["pii"]);
    let p = pii.get::<Pii>().unwrap();
    let r = pii.get::<Records>().unwrap();
    for i in (0..r.spans.len()).filter(|&i| r.path[i] == amount) {
        assert!(r.support(i).all(|(s, row)| s == Source::Pii && (row as usize) < p.spans.len()));
    }
    assert!(r.support_source.iter().all(|&s| s == source_code(Source::Pii)));
}

#[test]
fn records_cross_the_graph_and_corrupted_payloads_are_refused() {
    let text = "Zoë ordered two crêpes from Café São Paulo for 12 dollars; Ana paid.";
    let doc = run(document(text), schema(vec![Check::Require(Source::Numbers)]), 0.0, &["numbers"]);
    let registry = codecs();
    let graph = export(&doc, &registry).unwrap();
    let back = import(&graph, &registry).unwrap();
    let (a, b) = (doc.get::<Records>().unwrap(), back.get::<Records>().unwrap());
    assert_eq!(
        (a.spans.len(), &a.record, &a.score, &a.support_row),
        (b.spans.len(), &b.record, &b.score, &b.support_row)
    );
    assert_eq!(export(&back, &registry).unwrap().to_bytes(), graph.to_bytes(), "re-export is byte-identical");
    let at = graph.layers.iter().position(|l| l.layer == "turbo:records").unwrap();
    let body = graph.layers[at].body.clone().unwrap();
    // Every truncation, and every byte flipped, is an error or a checked
    // layer; none panics.
    for cut in 0..body.len() {
        let mut g = graph.clone();
        g.layers[at].body = Some(body[..cut].to_vec());
        assert!(import(&g, &registry).is_err(), "a payload cut at {cut} imported");
    }
    for i in 0..body.len() {
        for flip in [0x01u8, 0x80, 0xFF] {
            let mut g = graph.clone();
            let mut bad = body.clone();
            bad[i] ^= flip;
            g.layers[at].body = Some(bad);
            if let Ok(d) = import(&g, &registry) {
                assert_eq!(d.get::<Records>().unwrap().check(), Ok(()), "byte {i} ^ {flip:#x}");
            }
        }
    }
}

#[test]
fn glossary_and_location_layers_check_candidates() {
    use std::sync::Arc;
    use turbo_analysis::IxRange;
    use turbo_geo::{EntryParts, GazetteerEntry, GeoPoint, GeoResolution, Locations};
    use turbo_glossary_layers::{Glossary, GlossaryTerms};
    let text = "Ship the blue widget to Lyon on Friday.";
    let mut doc = document(text);
    let by = Provenance::by("test");
    let mut glossary = Glossary::new();
    let widget = glossary.push("w1", "blue widget").unwrap();
    let mut terms = GlossaryTerms::new(ViewId::SOURCE, Arc::new(glossary));
    terms.push(Span::new(9, 20), IxRange::new(2, 4), widget);
    doc.add(&Key::DEFAULT, terms, by.clone()).unwrap();
    let entry = GazetteerEntry::new(EntryParts {
        source: "test".into(),
        record_id: "1".into(),
        name: "Lyon".into(),
        alternate_names: Vec::new(),
        location: GeoPoint::new(45.76, 4.83).unwrap(),
        bounding_box: None,
        country_code: Some("FR".into()),
        containment: Vec::new(),
        population: 0,
        feature_class: None,
        attributes: Vec::new(),
    })
    .unwrap();
    let mut places = Locations::new(ViewId::SOURCE);
    places.push(GeoResolution::new(Span::new(24, 28), entry, 0.9).unwrap());
    doc.add(&Key::DEFAULT, places, by).unwrap();
    let schema = ExtractionSchema {
        entities: Vec::new(),
        classifications: Vec::new(),
        structures: vec![StructureSpec {
            name: "shipment".into(),
            fields: vec![
                field("item", Cardinality::Repeated, vec![Check::Require(Source::Glossary)]),
                field("destination", Cardinality::Repeated, vec![Check::Require(Source::Locations)]),
            ],
        }],
    };
    let annotator = SchemaAnnotator::new(common::offline(Limits::default()), schema, 0.0).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<GlossaryTerms>::DEFAULT)
        .given(Key::<Locations>::DEFAULT)
        .add(annotator)
        .step(EvidenceStep::new(Key::<Records>::DEFAULT))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let doc = docs.pop().unwrap();
    let r = doc.get::<Records>().unwrap();
    assert!(!r.spans.is_empty(), "candidates over the glossary term and the location survive");
    for i in 0..r.spans.len() {
        let s = r.spans.get(turbo_analysis::Ix::new(i as u32));
        let want = if r.paths.label(r.path[i]) == "shipment.item" {
            (Source::Glossary, 9, 20)
        } else {
            (Source::Locations, 24, 28)
        };
        assert!(s.start < want.2 && want.1 < s.end, "{s:?} kept without overlapping its layer");
        assert_eq!(r.support(i).collect::<Vec<_>>(), vec![(want.0, 0)]);
    }
    assert_eq!(doc.get::<Evidence<Records>>().unwrap().verify(&doc), Ok(()));
}
