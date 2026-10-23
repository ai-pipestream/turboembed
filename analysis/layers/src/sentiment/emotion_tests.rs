use super::*;
use crate::{Chunks, Sentences, Tokens};
use turbo_analysis::{Ix, IxRange, Layer, Span, Spans, ViewId};

fn sentences_and_tokens() -> (Sentences, Tokens) {
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 9));
    s.push(Span::new(10, 15));
    let mut t = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 1), (2, 4), (5, 9), (10, 15)] {
        t.push(Span::new(a, b));
    }
    (Sentences { spans: s }, Tokens { spans: t, sentence: None })
}

fn scores(d: EmotionDimension, v: f64) -> EmotionScores {
    let mut s = EmotionScores::default();
    s.set(d, v);
    s
}

#[test]
fn the_taxonomy_is_goemotions_in_its_order() {
    assert_eq!(EmotionDimension::ALL.len(), EMOTION_DIMENSIONS);
    for (i, d) in EmotionDimension::ALL.into_iter().enumerate() {
        assert_eq!(d.index(), i);
        assert_eq!(EmotionDimension::from_name(d.name()), Some(d));
        assert_eq!(EmotionDimension::from_index(i), Some(d));
    }
    assert_eq!(EmotionDimension::Admiration.name(), "admiration");
    assert_eq!(EmotionDimension::Neutral.index(), 27);
    assert_eq!(EmotionDimension::from_index(28), None);
}

#[test]
fn items_are_checked_as_they_are_pushed() {
    let (s, t) = sentences_and_tokens();
    let mut l = EmotionLayer::new();
    let joy = [(EmotionDimension::Joy, IxRange::new(2, 3))];
    let item = |i: u32, sc, ev| EmotionItem { target: EmotionTarget::Sentence(Ix::new(i)), scores: sc, evidence: ev };
    l.try_push(&t, Some(&s), None, item(0, scores(EmotionDimension::Joy, 0.9), &joy)).unwrap();
    // Out of order, out of bounds, no layer, a bad score, a bad span.
    assert!(l.try_push(&t, Some(&s), None, item(0, EmotionScores::default(), &[])).is_err());
    assert!(l.try_push(&t, Some(&s), None, item(2, EmotionScores::default(), &[])).is_err());
    assert!(l.try_push(&t, None, None, item(1, EmotionScores::default(), &[])).is_err());
    assert!(l.try_push(&t, Some(&s), None, item(1, scores(EmotionDimension::Fear, 1.5), &[])).is_err());
    let past = [(EmotionDimension::Fear, IxRange::new(3, 9))];
    assert!(l.try_push(&t, Some(&s), None, item(1, EmotionScores::default(), &past)).is_err());
    let chunk =
        EmotionItem { target: EmotionTarget::Chunk(Ix::new(0)), scores: EmotionScores::default(), evidence: &[] };
    assert!(l.try_push(&t, Some(&s), None::<&Chunks>, chunk).is_err());
    l.check().unwrap();
    assert_eq!(l.scores_of(Ix::new(0)).get(EmotionDimension::Joy), 0.9);
    assert_eq!(l.evidence_for_label(Ix::new(0), EmotionDimension::Joy), &[IxRange::new(2, 3)]);
    assert!(l.evidence_for_label(Ix::new(0), EmotionDimension::Fear).is_empty());
    assert_eq!(l.target_span(Ix::new(0), Some(&s), None), Some(Span::new(0, 9)));
}

#[test]
fn check_finds_a_broken_layer() {
    let (s, t) = sentences_and_tokens();
    let mut good = EmotionLayer::new();
    let ev = [(EmotionDimension::Joy, IxRange::new(0, 1)), (EmotionDimension::Anger, IxRange::new(1, 2))];
    let item =
        EmotionItem { target: EmotionTarget::Sentence(Ix::new(0)), scores: EmotionScores::default(), evidence: &ev };
    good.try_push(&t, Some(&s), None, item).unwrap();
    good.check().unwrap();
    // Evidence sorted by label: anger before joy.
    assert_eq!(good.evidence_label, [EmotionDimension::Anger, EmotionDimension::Joy]);
    let broken: [fn(&mut EmotionLayer); 5] = [
        |l| l.scores.pop().map(drop).unwrap_or(()),
        |l| l.scores[3] = f64::NAN,
        |l| l.evidence_end[0] = 3,
        |l| l.evidence_tokens[0] = IxRange::new(1, 1),
        |l| l.evidence_label.swap(0, 1),
    ];
    for (k, f) in broken.into_iter().enumerate() {
        let mut l = good.clone();
        f(&mut l);
        assert!(l.check().is_err(), "case {k}");
    }
}
