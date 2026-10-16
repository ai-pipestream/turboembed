//! The `masked` view: its text is what `mask` writes, and its alignment
//! maps every offset both ways, exactly, in UTF-8 bytes.

mod support;

use support::*;
use turbo_analysis::text::{Alignment, Side};
use turbo_analysis::{Analyzer, Annotator, Document, Span, View, ViewId};
use turbo_pii::rewrite::{MASKED, MaskError, MaskPolicy, MaskedView, mask, mask_by_kind, mask_view};
use turbo_pii::{Pii, PiiDetector, PiiKind};

fn whole(c: char) -> MaskPolicy {
    MaskPolicy::new(c).unwrap()
}

fn source(text: &str) -> Document {
    Document::new(text)
}

fn view_of(text: &str, spans: &[(Span, MaskPolicy)]) -> View {
    let doc = source(text);
    mask_view(doc.get::<View>().unwrap(), spans.iter().copied()).unwrap()
}

fn boundaries(text: &str) -> Vec<u32> {
    (0..=text.len()).filter(|&i| text.is_char_boundary(i)).map(|i| i as u32).collect()
}

/// Checks the alignment of a view of `text` that has its byte length:
/// one edit per changed character, each the same span on both sides;
/// every character boundary of the source maps to itself both ways; a
/// byte inside a character's mask maps back to the whole character.
#[track_caller]
fn assert_exact(text: &str, masked: &str, a: &Alignment) {
    assert_eq!(masked.len(), text.len());
    let mut previous_end = 0;
    for e in a.edits() {
        assert_eq!(e.parent, e.child, "{text:?}");
        let piece = &text[e.parent.range()];
        assert_eq!(piece.chars().count(), 1, "{text:?}: an edit over {piece:?}");
        assert_ne!(piece, &masked[e.child.range()], "{text:?}: an edit that changes nothing");
        assert!(e.parent.start >= previous_end);
        previous_end = e.parent.end;
    }
    let bounds = boundaries(text);
    for &x in &bounds {
        for side in [Side::Start, Side::End] {
            assert_eq!(a.to_child(x, side), x, "{text:?}: source offset {x}");
            assert_eq!(a.to_parent(x, side), x, "{text:?}: masked offset {x}");
        }
    }
    for w in bounds.windows(2) {
        let c = Span::new(w[0], w[1]);
        assert_eq!(a.span_to_child(c), c, "{text:?}: character {c:?}");
        assert_eq!(a.span_to_parent(c), c, "{text:?}: character {c:?}");
        // Inside a masked character, each mask byte stands for the whole
        // character, and only for it.
        let changed = a.edits().iter().any(|e| e.parent == c);
        for y in c.start + 1..c.end {
            if changed {
                assert_eq!(a.to_parent(y, Side::Start), c.start, "{text:?}: masked offset {y}");
                assert_eq!(a.to_parent(y, Side::End), c.end, "{text:?}: masked offset {y}");
                assert_eq!(a.span_to_parent(Span::new(y, y + 1)), c);
            }
        }
    }
}

#[test]
fn every_pair_of_spans_masks_as_mask_does_and_maps_exactly() {
    let mut views = 0;
    for text in ["123456", "12-34", "a𝕒b3", "𝕒-𝕓", "*12X", "Zoë ß-9"] {
        let bounds = boundaries(text);
        let spans: Vec<Span> = bounds
            .iter()
            .flat_map(|&s| bounds.iter().filter(move |&&e| e >= s).map(move |&e| Span::new(s, e)))
            .collect();
        for c in ['*', 'X', '7'] {
            for keep_format in [false, true] {
                for trailing in [0, 1, 2, 6] {
                    let mut p = whole(c).keeping_trailing(trailing);
                    if keep_format {
                        p = p.keeping_format();
                    }
                    let q = whole('#').keeping_trailing(1);
                    for &a in &spans {
                        for &b in &spans {
                            for pair in [[(a, p), (b, p)], [(a, p), (b, q)]] {
                                let v = view_of(text, &pair);
                                assert_eq!(v.text(), mask(text, pair).unwrap(), "{text:?} {pair:?}");
                                assert_exact(text, v.text(), v.alignment());
                                views += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(views > 100_000, "{views}");
}

#[test]
fn multi_byte_characters_map_to_their_own_bytes() {
    // é (2 bytes), 𝕒 (4), ß (2) and 中 (3) masked whole, digits kept.
    let text = "id é𝕒ß中 4111 end";
    let span = Span::new(3, 19);
    let v = view_of(text, &[(span, whole('*').keeping_format().keeping_trailing(4))]);
    assert_eq!(v.text(), "id *********** 4111 end");
    let edits: Vec<(u32, u32)> = v.alignment().edits().iter().map(|e| (e.parent.start, e.parent.end)).collect();
    assert_eq!(edits, [(3, 5), (5, 9), (9, 11), (11, 14)]);
    let a = v.alignment();
    // The four mask bytes of 𝕒 map back to 𝕒, and 𝕒 forward to them.
    for y in 5..9 {
        assert_eq!(&text[a.span_to_parent(Span::new(y, y + 1)).range()], "𝕒");
    }
    assert_eq!(a.span_to_child(Span::new(5, 9)), Span::new(5, 9));
    // A span over a masked and a kept stretch maps to the same bytes.
    assert_eq!(a.span_to_parent(Span::new(9, 20)), Span::new(9, 20));
    assert_eq!(&text[a.span_to_parent(Span::new(15, 19)).range()], "4111");
    assert_exact(text, v.text(), a);
}

#[test]
fn adjacent_mentions_keep_their_boundary() {
    let text = "ab𝕒cdé";
    let pii = manual(&[(0, 2, PiiKind::Email, "ab"), (2, 6, PiiKind::Phone, "x"), (6, 10, PiiKind::Card, "cd")]);
    let doc = source(text);
    let v = MaskedView::all(whole('*')).view(doc.get::<View>().unwrap(), &pii).unwrap();
    assert_eq!(v.text(), "*".repeat(text.len()));
    let a = v.alignment();
    for x in [0, 2, 6, 10] {
        assert_eq!((a.to_parent(x, Side::Start), a.to_parent(x, Side::End)), (x, x), "{x}");
        assert_eq!((a.to_child(x, Side::Start), a.to_child(x, Side::End)), (x, x), "{x}");
    }
    for m in pii.mentions() {
        assert_eq!(a.span_to_parent(m.span), m.span);
        assert_eq!(a.span_to_child(m.span), m.span);
    }
    assert_eq!(a.edits().len(), text.chars().count());
    assert_exact(text, v.text(), a);
    // Two detected mentions with only a space between them.
    let text = "jane@example.com bob@example.com";
    let doc = source(text);
    let v = MaskedView::by_kind('*').unwrap().view(doc.get::<View>().unwrap(), &layer(&PiiDetector::cursor(), text));
    let v = v.unwrap();
    assert_eq!(v.text(), "****@*******.*** ***@*******.***");
    assert_eq!(v.alignment().span_to_parent(Span::new(17, 32)), Span::new(17, 32));
    assert_exact(text, v.text(), v.alignment());
}

#[test]
fn a_mask_byte_over_itself_is_not_an_edit() {
    let v = view_of("a*b", &[(Span::new(0, 3), whole('*'))]);
    assert_eq!(v.text(), "***");
    let edits: Vec<Span> = v.alignment().edits().iter().map(|e| e.parent).collect();
    assert_eq!(edits, [Span::new(0, 1), Span::new(2, 3)]);
}

#[test]
fn nothing_to_mask_is_the_identity() {
    for spans in [vec![], vec![(Span::new(2, 2), whole('*'))], vec![(Span::new(0, 3), whole('*').keeping_trailing(9))]]
    {
        let v = view_of("abc", &spans);
        assert_eq!(v.text(), "abc");
        assert!(v.alignment().edits().is_empty());
    }
}

#[test]
fn bad_spans_are_refused() {
    let doc = source("a𝕒b");
    let src = doc.get::<View>().unwrap();
    assert_eq!(
        mask_view(src, [(Span::new(2, 5), whole('*'))]).map(|v| v.text().to_string()),
        Err(MaskError::Span(Span::new(2, 5)))
    );
    assert_eq!(
        mask_view(src, [(Span::new(1, 9), whole('*'))]).map(|v| v.text().to_string()),
        Err(MaskError::Span(Span::new(1, 9)))
    );
}

#[test]
fn policies_by_kind_and_overrides() {
    let v = MaskedView::by_kind('*').unwrap();
    assert_eq!(v.policy(PiiKind::Card), whole('*').keeping_format().keeping_trailing(4));
    assert_eq!(v.policy(PiiKind::Jwt), whole('*'));
    let v = v.with(PiiKind::Email, whole('#'));
    assert_eq!(v.policy(PiiKind::Email), whole('#'));
    assert_eq!(MaskedView::by_kind('é'), Err(MaskError::NotAscii('é')));
    assert_ne!(MaskedView::by_kind('*').unwrap().config_hash(), v.config_hash());
    assert_eq!(MaskedView::by_kind('*').unwrap().config_hash(), MaskedView::by_kind('*').unwrap().config_hash());
}

const TEXT: &str = "Für Zoë: jane@example.com, (555) 123-4567, Karte 4111 1111 1111 1111 😀.";

#[test]
fn the_annotator_writes_the_masked_view_in_a_pipeline() {
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(MaskedView::by_kind('*').unwrap().bound()).build().unwrap();
    let mut docs = vec![Document::new(TEXT)];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let doc = &docs[0];
    let masked = doc.get_key(&MASKED).expect("the masked view");
    let pii = doc.get::<Pii>().unwrap();
    assert_eq!(masked.text(), mask_by_kind(TEXT, pii, '*').unwrap());
    assert_eq!(masked.text(), "Für Zoë: ****@*******.***, (***) ***-****, Karte **** **** **** 1111 😀.");
    assert_eq!(masked.parent(), Some(ViewId::SOURCE));
    for m in pii.mentions() {
        let in_view = doc.from_source(masked.id(), m.span);
        assert_eq!(in_view, m.span);
        assert_eq!(doc.to_source(masked.id(), in_view), m.span);
    }
    // "Zoë" and the emoji are outside every mention and map to themselves.
    for word in ["Zoë", "😀"] {
        let at = TEXT.find(word).unwrap() as u32;
        let s = Span::new(at, at + word.len() as u32);
        assert_eq!(&masked.text()[doc.from_source(masked.id(), s).range()], word);
    }
    assert_exact(TEXT, masked.text(), masked.alignment());
}

#[test]
fn a_layer_over_another_view_is_refused() {
    let doc = source("x");
    let src = doc.get::<View>().unwrap();
    let pii = Pii::new(ViewId(3));
    let err = MaskedView::all(whole('*')).annotate((src, &pii)).map(|_| ()).unwrap_err();
    assert!(err.to_string().contains("view 3"), "{err}");
}
