//! Probabilities and scores in the shared layers keep double precision.
//! OpenNLP computes them as Java doubles and parity with it is checked to
//! within 1e-9, which a 32-bit float (about 1e-7 relative error) cannot meet.

use turbo_analysis::*;
use turbo_analysis_layers::{Confidence, Entities, Language, PosTags, Tag, TagSet, Tokens};

/// A value f64 holds to the last digit and f32 rounds visibly.
const PRECISE: f64 = 0.123456789012345;

fn as_f32(v: f64) -> f64 {
    (v as f32) as f64
}

fn document_with_two_tokens() -> Document {
    let mut doc = Document::new("Dogs bark");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 4));
    spans.push(Span::new(5, 9));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    doc
}

#[test]
fn the_test_value_is_not_representable_as_f32() {
    assert_ne!(PRECISE, as_f32(PRECISE));
    assert!((PRECISE - as_f32(PRECISE)).abs() > 1e-9);
}

#[test]
fn pos_tag_probabilities_round_trip_exactly() {
    let mut doc = document_with_two_tokens();
    let tagset = TagSet::new("toy", &["X", "VBZ"]);
    let tags = PosTags { tagset, tags: vec![Tag(0), Tag(1)], prob: Some(vec![PRECISE, 1.0 - PRECISE]) };
    doc.add(&Key::DEFAULT, tags, Provenance::by("test")).unwrap();

    let prob = doc.get::<PosTags>().unwrap().prob.as_ref().unwrap();
    assert_eq!(prob[0], PRECISE);
    assert_eq!(prob[1], 1.0 - PRECISE);
    assert_ne!(prob[0], as_f32(PRECISE));
}

#[test]
fn token_confidences_round_trip_exactly() {
    let mut doc = document_with_two_tokens();
    let key = Key::<Confidence<Tokens>>::DEFAULT;
    doc.add(&key, Confidence::new(vec![PRECISE, 0.987654321098765]), Provenance::by("test")).unwrap();

    let value = &doc.get_key(&key).unwrap().value;
    assert_eq!(value[0], PRECISE);
    assert_eq!(value[1], 0.987654321098765);
    assert_ne!(value[0], as_f32(PRECISE));
    assert_ne!(value[1], as_f32(0.987654321098765));
}

#[test]
fn entity_scores_round_trip_exactly() {
    let mut doc = document_with_two_tokens();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 4));
    let entities = Entities {
        spans,
        tokens: vec![IxRange::new(0, 1)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("kinds", &["animal"]),
        score: vec![PRECISE],
    };
    doc.add(&Key::DEFAULT, entities, Provenance::by("test")).unwrap();

    let score = doc.get::<Entities>().unwrap().score[0];
    assert_eq!(score, PRECISE);
    assert_ne!(score, as_f32(PRECISE));
}

#[test]
fn language_scores_round_trip_exactly() {
    let mut doc = Document::new("Dogs bark");
    doc.add(&Key::DEFAULT, Language { code: "en".into(), score: PRECISE }, Provenance::by("test")).unwrap();

    let score = doc.get::<Language>().unwrap().score;
    assert_eq!(score, PRECISE);
    assert_ne!(score, as_f32(PRECISE));
}

#[test]
fn probabilities_one_nanounit_apart_stay_distinct() {
    let (a, b) = (0.5, 0.5 + 1e-9);
    assert_eq!(as_f32(a), as_f32(b));

    let mut doc = document_with_two_tokens();
    let key = Key::<Confidence<Tokens>>::DEFAULT;
    doc.add(&key, Confidence::new(vec![a, b]), Provenance::by("test")).unwrap();

    let value = &doc.get_key(&key).unwrap().value;
    assert_ne!(value[0], value[1]);
    assert!((value[1] - value[0] - 1e-9).abs() < 1e-15);
}
