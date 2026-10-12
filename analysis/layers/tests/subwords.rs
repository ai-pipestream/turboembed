//! The subword pieces of a word token are found by binary search over
//! the token column, and the document checks that column like any other
//! index into another layer.

use turbo_analysis::*;
use turbo_analysis_layers::{Subwords, Tokens};

fn model() -> ModelRef {
    ModelRef { id: "toy".into(), revision: "1".into(), sha256: String::new() }
}

fn two_tokens() -> (Document, Spans<Subwords>) {
    let mut doc = Document::new("playing cards");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 7));
    spans.push(Span::new(8, 13));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test")).unwrap();
    let mut pieces = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 4), (4, 7), (8, 13)] {
        pieces.push(Span::new(a, b));
    }
    (doc, pieces)
}

#[test]
fn the_pieces_of_a_token_are_a_range() {
    let (mut doc, pieces) = two_tokens();
    let token = Some(vec![Ix::new(0), Ix::new(0), Ix::new(1)]);
    doc.add(
        &Key::DEFAULT,
        Subwords { spans: pieces, ids: vec![1, 2, 3], model: model(), token },
        Provenance::by("test"),
    )
    .unwrap();
    let sub = doc.get::<Subwords>().unwrap();
    assert_eq!(sub.of(Ix::new(0)), IxRange::new(0, 2));
    assert_eq!(sub.of(Ix::new(1)), IxRange::new(2, 3));
}

#[test]
fn a_token_index_past_the_tokens_is_refused() {
    let (mut doc, pieces) = two_tokens();
    let token = Some(vec![Ix::new(0), Ix::new(0), Ix::new(2)]);
    let sub = Subwords { spans: pieces, ids: vec![1, 2, 3], model: model(), token };
    assert!(doc.add(&Key::DEFAULT, sub, Provenance::by("test")).is_err());
}

#[test]
fn pieces_out_of_token_order_are_refused() {
    let (mut doc, pieces) = two_tokens();
    let token = Some(vec![Ix::new(1), Ix::new(0), Ix::new(1)]);
    let sub = Subwords { spans: pieces, ids: vec![1, 2, 3], model: model(), token };
    let e = doc.add(&Key::DEFAULT, sub, Provenance::by("test")).unwrap_err();
    assert!(e.to_string().contains("token order"), "{e}");
}
