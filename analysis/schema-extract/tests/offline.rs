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
    // Within one field of one record, a kept value overlaps no other: the
    // decoder suppressed the rest of the candidates a threshold of 0 keeps.
    let ix = |i: usize| r.spans.get(turbo_analysis::Ix::new(i as u32));
    for i in 0..r.spans.len() {
        for j in i + 1..r.spans.len() {
            let (a, b) = (ix(i), ix(j));
            let same = r.record[i] == r.record[j] && r.path[i] == r.path[j];
            assert!(!(same && a.start < b.end && b.start < a.end), "{a:?} and {b:?} in one field");
        }
    }
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

#[test]
fn a_choice_reads_its_own_fields_token() {
    use turbo_schema_extract::decode::{FieldValue, Nothing, decode};
    use turbo_schema_extract::model::Scores;
    let choice = |name: &str, c: &[&str]| FieldSpec {
        choices: Some(c.iter().map(|s| s.to_string()).collect()),
        ..field(name, Cardinality::Repeated, Vec::new())
    };
    let s = ExtractionSchema {
        structures: vec![StructureSpec {
            name: "r".into(),
            fields: vec![choice("first", &["yes", "no"]), choice("second", &["no", "yes", "first", "("])],
        }],
        ..Default::default()
    };
    let pass = common::offline(Limits::default()).pass("some words here", &s).unwrap();
    let out = decode(&pass, &s, 0.0, &Nothing).unwrap();
    let Scores::Spans { logits, .. } = &pass.scores()[0] else { panic!("a record task") };
    let size = pass.prompt().token_first.len() * pass.width();
    let second = pass.prompt().prefix.iter().position(|t| t == "second").unwrap();
    for v in &out.records[0].fields[1] {
        let FieldValue::Choice { choice, logit, .. } = v else { panic!("a choice") };
        let own = second + 2 + 2 * choice;
        assert_eq!(pass.prompt().prefix[own], s.structures[0].fields[1].choices.as_ref().unwrap()[*choice]);
        assert_eq!(logit.to_bits(), logits[size + own * pass.width()].to_bits(), "choice {choice}");
    }
    assert_eq!(out.records[0].fields[1].len(), 4);
}

#[test]
fn evidence_cites_the_numbers_instance_the_annotator_read() {
    use turbo_analysis::Bound;
    let text = "Send 42 units to Ana before 7 May; total 1,250 dollars.";
    let mut doc = document(text);
    // A default numbers layer the annotator does not read, as a decoy.
    let mut decoy = Numbers::new(ViewId::SOURCE);
    NumberScanner::new(NumberNotation::LatinUs).scan("Nothing here at all, really, 9.", &mut decoy);
    doc.add(&Key::DEFAULT, decoy, Provenance::by("decoy")).unwrap();
    let annotator =
        SchemaAnnotator::new(common::offline(Limits::default()), schema(vec![Check::Require(Source::Numbers)]), 0.0)
            .unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Numbers>::DEFAULT)
        .step(Bound::new(NumberScanner::new(NumberNotation::LatinUs)).write(0, "named"))
        .step(Bound::new(annotator).read(4, "named"))
        .step(EvidenceStep::new(Key::<Records>::DEFAULT))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let doc = docs.pop().unwrap();
    let named = doc.get_key(&Key::<Numbers>::named("named")).unwrap();
    let r = doc.get::<Records>().unwrap();
    assert!(!r.support_row.is_empty(), "amounts kept with the named numbers");
    for i in 0..r.spans.len() {
        let s = r.spans.get(turbo_analysis::Ix::new(i as u32));
        for (_, row) in r.support(i) {
            let n = named.spans.get(turbo_analysis::Ix::new(row));
            assert!(n.start < s.end && s.start < n.end, "{n:?} of the named layer does not overlap {s:?}");
        }
    }
    assert_eq!(doc.get::<Evidence<Records>>().unwrap().verify(&doc), Ok(()));
    let registry = codecs();
    let back = import(&export(&doc, &registry).unwrap(), &registry).unwrap();
    assert_eq!(back.get::<Records>().unwrap().support_row, r.support_row);
}

#[test]
fn a_pass_that_does_not_fit_its_schema_is_an_error() {
    use turbo_schema_extract::decode::{Nothing, decode};
    use turbo_schema_extract::model::{Pass, Scores};
    let m = common::offline(Limits::default());
    let s = schema(Vec::new());
    let (prompt, width, scores) = m.pass("Send 42 units.", &s).unwrap().into_parts();
    let first = scores.iter().position(|x| matches!(x, Scores::Spans { .. })).unwrap();
    // Truncated scores: an error, not a panic.
    let mut short = scores.clone();
    short[first] = Scores::Spans { count: 1, instances: 1, logits: Vec::new() };
    let pass = Pass::new(prompt, width, short);
    assert!(decode(&pass, &s, 0.5, &Nothing).is_err());
    // NaN scores: an error, never a confident value.
    let (prompt, width, _) = pass.into_parts();
    let mut nan = scores.clone();
    if let Scores::Spans { logits, .. } = &mut nan[first] {
        logits.iter_mut().for_each(|x| *x = f32::NAN);
    }
    let pass = Pass::new(prompt, width, nan);
    assert!(decode(&pass, &s, 0.99, &Nothing).is_err());
    // Another schema's pass, and a threshold that is not a probability.
    let (prompt, width, _) = pass.into_parts();
    let pass = Pass::new(prompt, width, scores);
    assert!(decode(&pass, &s, 0.5, &Nothing).is_ok());
    let mut other = s.clone();
    other.entities.pop();
    assert!(decode(&pass, &other, 0.5, &Nothing).is_err());
    assert!(decode(&pass, &s, f32::NAN, &Nothing).is_err());
    // A schema built by hand with a threshold, boost or top_k out of range.
    let mut bad = s.clone();
    bad.entities[0].threshold = Some(f32::NAN);
    assert!(decode(&pass, &bad, 0.5, &Nothing).is_err());
    for c in 0..s.classifications.len() {
        let mut bad = s.clone();
        bad.classifications[c].threshold = 2.0;
        assert!(decode(&pass, &bad, 0.5, &Nothing).is_err());
        let mut bad = s.clone();
        bad.classifications[c].top_k = Some(0);
        assert!(decode(&pass, &bad, 0.5, &Nothing).is_err());
    }
    for r in 0..s.structures.len() {
        let mut bad = s.clone();
        bad.structures[r].fields[0].threshold = Some(-0.1);
        assert!(decode(&pass, &bad, 0.5, &Nothing).is_err());
    }
    assert!(!s.classifications.is_empty() && !s.structures.is_empty());
    // A width of zero, words past the text tokens, or a prefix longer than
    // them: errors, not an index past the scores.
    let (prompt, width, scores) = pass.into_parts();
    let zero: Vec<Scores> = scores
        .iter()
        .map(|x| match x {
            Scores::Spans { count, .. } => Scores::Spans { count: *count.max(&1), instances: 1, logits: Vec::new() },
            o => o.clone(),
        })
        .collect();
    let pass = Pass::new(prompt, 0, zero);
    assert!(decode(&pass, &s, 0.5, &Nothing).is_err());
    let (mut prompt, _, _) = pass.into_parts();
    let word = prompt.words[0].clone();
    prompt.words.extend(std::iter::repeat_n(word, 20));
    let pass = Pass::new(prompt, width, scores.clone());
    assert!(decode(&pass, &s, 0.5, &Nothing).is_err());
    let (mut prompt, _, _) = pass.into_parts();
    prompt.words.truncate(prompt.words.len() - 20);
    let extra = prompt.token_first.len() + 50;
    prompt.prefix.extend(std::iter::repeat_n(String::from("x"), extra));
    let pass = Pass::new(prompt, width, scores.clone());
    assert!(decode(&pass, &s, 0.5, &Nothing).is_err());
    // A width or a choice position whose products overflow: errors too.
    let (mut prompt, _, _) = pass.into_parts();
    prompt.prefix.truncate(prompt.prefix.len() - extra);
    let pass = Pass::new(prompt, usize::MAX, scores.clone());
    assert!(decode(&pass, &s, 0.5, &Nothing).is_err());
    let (mut prompt, _, _) = pass.into_parts();
    let at = prompt.choice_at.iter_mut().flatten().find(|a| a.is_some()).expect("a choice field");
    *at = Some(usize::MAX - 1);
    let pass = Pass::new(prompt, width, scores);
    assert!(decode(&pass, &s, 0.5, &Nothing).is_err());
    // A classification task with no labels, single- and multi-label.
    use turbo_schema_extract::prompt::{Kind, Task};
    for multi_label in [false, true] {
        let (mut prompt, _, _) = pass_parts(&m, &s);
        let empty = ExtractionSchema {
            classifications: vec![ClassificationTask {
                name: "t".into(),
                labels: Vec::new(),
                multi_label,
                threshold: 0.5,
                top_k: None,
            }],
            ..Default::default()
        };
        prompt.tasks = vec![Task { kind: Kind::Classification(0), p: 0, markers: Vec::new() }];
        prompt.choice_at.clear();
        let pass = Pass::new(prompt, 1, vec![Scores::Labels(Vec::new())]);
        assert!(decode(&pass, &empty, 0.5, &Nothing).is_err(), "multi_label {multi_label}");
    }
}

fn pass_parts(
    m: &turbo_schema_extract::Gliner2Extractor,
    s: &ExtractionSchema,
) -> (turbo_schema_extract::prompt::Prompt, usize, Vec<turbo_schema_extract::model::Scores>) {
    m.pass("Send 42 units.", s).unwrap().into_parts()
}
