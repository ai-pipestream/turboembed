use super::*;
use turbo_analysis::{Alignment, IxRange, Provenance, Spans, ViewBuilder};
use turbo_analysis_layers::{Tag, TagSet};

fn tokens(view: ViewId, spans: &[(u32, u32)]) -> Tokens {
    let mut s = Spans::new(view);
    for &(a, b) in spans {
        s.push(Span::new(a, b));
    }
    Tokens { spans: s, sentence: None }
}

#[test]
fn well_formed_spans_pass_in_text_of_every_width() {
    let text = "a é 中 😀 b";
    let mut doc = Document::new(text);
    let spans: Vec<(u32, u32)> = text
        .split(' ')
        .scan(0u32, |at, w| {
            let s = (*at, *at + w.len() as u32);
            *at += w.len() as u32 + 1;
            Some(s)
        })
        .collect();
    doc.add(&Key::DEFAULT, tokens(ViewId::SOURCE, &spans), Provenance::by("test")).unwrap();
    assert_eq!(assert_invariants(&doc, "plain"), 1);
}

#[test]
fn a_layer_out_of_its_contract_is_reported() {
    // Sentences must be in text order and not overlap; tokens with a
    // sentence index must have one per token, in order.
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(3, 5));
    s.push(Span::new(0, 2));
    let v = check_contract(&Sentences { spans: s }, "opennlp:sentences");
    assert_eq!(v.len(), 1);
    assert_eq!(
        v[0].to_string(),
        "opennlp:sentences item 0 at 0..0 in view 0: the layer's contract: sentences in text order, not overlapping"
    );
    let mut t = tokens(ViewId::SOURCE, &[(0, 1), (2, 3)]);
    t.sentence = Some(vec![turbo_analysis::Ix::new(1), turbo_analysis::Ix::new(0)]);
    let v = check_contract(&t, "opennlp:tokens");
    assert_eq!(v.len(), 1);
    assert!(v[0].what.ends_with("tokens in sentence order"), "{}", v[0]);
    assert!(check_contract(&tokens(ViewId::SOURCE, &[(0, 1), (2, 3)]), "opennlp:tokens").is_empty());
    // A document refuses such a layer, so the check on a document's
    // layers is a second line.
    let mut doc = Document::new("ab cd");
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(3, 5));
    s.push(Span::new(0, 2));
    assert!(doc.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("test")).is_err());
}

#[test]
fn a_span_cutting_a_character_or_leaving_the_text_is_reported() {
    let doc = Document::new("é😀");
    let v = check_spans(&doc, "opennlp:tokens", ViewId::SOURCE, &[1, 0, 3], &[2, 7, 2]);
    assert_eq!(v.len(), 3);
    assert!(v[0].what.contains("cuts a character"), "{}", v[0]);
    assert!(v[1].what.contains("not inside"), "{}", v[1]);
    assert!(v[2].what.contains("ends before"), "{}", v[2]);
    assert_eq!(v[0].to_string(), "opennlp:tokens item 0 at 1..2 in view 0: cuts a character");
}

#[test]
fn spans_in_a_normalized_view_map_back_exactly_outside_the_edits() {
    // "Hello  WORLD" with the double space collapsed and WORLD lowered.
    let source = "Hello  WORLD!";
    let mut doc = Document::new(source);
    let mut b = ViewBuilder::new(source);
    b.keep(5);
    b.replace(7, " ");
    b.replace(12, "world");
    let (text, alignment) = b.finish();
    assert_eq!(text, "Hello world!");
    let view = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::<View>::named("norm"), view, Provenance::by("test")).unwrap();
    let id = doc.get_key(&Key::<View>::named("norm")).unwrap().id();
    // "Hello", "world" and "!" as tokens of the view.
    doc.add(&Key::DEFAULT, tokens(id, &[(0, 5), (6, 11), (11, 12)]), Provenance::by("test")).unwrap();
    assert_eq!(assert_invariants(&doc, "normalized"), 1);
    assert_eq!(doc.to_source(id, Span::new(6, 11)), Span::new(7, 12));
    assert_eq!(doc.to_source(id, Span::new(11, 12)), Span::new(12, 13));
}

#[test]
fn a_view_whose_alignment_lies_is_caught() {
    // An alignment claiming the view is the source byte for byte,
    // while the view's text differs: the mapped source bytes are not
    // the span's own.
    let source = "abc";
    let mut doc = Document::new(source);
    let view = View::derived(doc.get::<View>().unwrap(), "xyz".to_string(), Alignment::identity());
    doc.add(&Key::<View>::named("lie"), view, Provenance::by("test")).unwrap();
    let id = doc.get_key(&Key::<View>::named("lie")).unwrap().id();
    doc.add(&Key::DEFAULT, tokens(id, &[(0, 3)]), Provenance::by("test")).unwrap();
    let (checked, v) = check_document(&doc);
    assert_eq!(checked, 1);
    assert_eq!(v.len(), 1);
    assert!(v[0].what.contains("touches no edit"), "{}", v[0]);
}

/// A view derived from a normalized view, not from the source: spans
/// outside its own edits are checked against its parent's text, which
/// the earlier link changed, and a lying second link is still caught.
#[test]
fn a_view_derived_from_another_view_is_checked_against_its_parent() {
    let source = "Hello  WORLD!";
    let mut doc = Document::new(source);
    let mut b = ViewBuilder::new(source);
    b.keep(5);
    b.replace(7, " ");
    b.replace(12, "world");
    let (text, alignment) = b.finish();
    let first = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::<View>::named("norm"), first, Provenance::by("test")).unwrap();
    // The second link drops the exclamation mark; "world" touches none
    // of its edits and reads "world" in its parent, "WORLD" in the source.
    let mut b = ViewBuilder::new("Hello world!");
    b.keep(11);
    b.replace(12, "");
    let (text, alignment) = b.finish();
    assert_eq!(text, "Hello world");
    let second = View::derived(doc.get_key(&Key::<View>::named("norm")).unwrap(), text, alignment);
    doc.add(&Key::<View>::named("trimmed"), second, Provenance::by("test")).unwrap();
    let second_id = doc.get_key(&Key::<View>::named("trimmed")).unwrap().id();
    doc.add(&Key::DEFAULT, tokens(second_id, &[(0, 5), (6, 11)]), Provenance::by("test")).unwrap();
    assert_eq!(check_document(&doc), (1, Vec::new()));
    assert_eq!(doc.to_source(second_id, Span::new(6, 11)), Span::new(7, 12));
    // The same chain with a second link whose text lies about its parent.
    let lie = View::derived(
        doc.get_key(&Key::<View>::named("norm")).unwrap(),
        "Hello wurld".to_string(),
        Alignment::identity(),
    );
    doc.add(&Key::<View>::named("lie"), lie, Provenance::by("test")).unwrap();
    let lie_id = doc.get_key(&Key::<View>::named("lie")).unwrap().id();
    doc.add(&Key::<Tokens>::named("lied"), tokens(lie_id, &[(6, 11)]), Provenance::by("test")).unwrap();
    let (checked, v) = check_document(&doc);
    assert_eq!(checked, 2);
    assert_eq!(v.len(), 1);
    assert!(v[0].what.contains("of its parent view"), "{}", v[0]);
}

#[test]
fn named_instances_of_several_types_are_all_checked() {
    let mut doc = Document::new("one two");
    doc.add(&Key::DEFAULT, tokens(ViewId::SOURCE, &[(0, 3), (4, 7)]), Provenance::by("test")).unwrap();
    doc.add(&Key::<Tokens>::named("other"), tokens(ViewId::SOURCE, &[(0, 7)]), Provenance::by("test")).unwrap();
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 7));
    doc.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("test")).unwrap();

    let mut m_spans = Spans::new(ViewId::SOURCE);
    m_spans.push(Span::new(0, 3));
    let matches = Matches {
        spans: m_spans,
        tokens: vec![IxRange::new(0, 1)],
        label: vec![Tag(0)],
        labels: TagSet::new("dict", &["term"]),
        rule: vec![0],
    };
    doc.add(&Key::<Matches>::named("dict"), matches, Provenance::by("test")).unwrap();
    assert_eq!(check_document(&doc), (4, Vec::new()));
}

#[test]
fn discontinuous_fragments_are_checked_in_derived_views() {
    // Includes 2-byte ('é'), 3-byte ('中'), and 4-byte non-BMP ('😀') characters
    // so per-fragment UTF-8 <-> UTF-16 and code-point round-trips are exercised.
    let source = "take  é中😀  off  soon";
    let mut doc = Document::new(source);
    let mut b = ViewBuilder::new(source);
    b.keep(4);
    b.replace(6, " ");
    b.keep(15);
    b.replace(17, " ");
    b.keep(20);
    b.replace(22, " ");
    b.keep(26);
    let (text, alignment) = b.finish();
    assert_eq!(text, "take é中😀 off soon");
    let view = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::<View>::named("norm"), view, Provenance::by("test")).unwrap();
    let id = doc.get_key(&Key::<View>::named("norm")).unwrap().id();
    doc.add(&Key::DEFAULT, tokens(id, &[(0, 4), (5, 14), (15, 18), (19, 23)]), Provenance::by("test")).unwrap();

    let kinds = TagSet::new("phrasal", &["verb", "adv"]);
    let mut spans = Spans::new(id);
    spans.push(Span::new(0, 4));
    spans.push_fragments(&[Span::new(5, 14), Span::new(15, 18)]);
    let entities = Entities {
        spans,
        tokens: vec![IxRange::new(0, 1), IxRange::new(1, 3)],
        kind: vec![Tag(0), Tag(1)],
        kinds: kinds.clone(),
        score: vec![0.8, 0.9],
    };
    doc.add(&Key::DEFAULT, entities, Provenance::by("test")).unwrap();
    assert_eq!(assert_invariants(&doc, "discontinuous"), 2);

    // A lying view whose second entity's second fragment (flat fragment 2,
    // belonging to item 1) mismatches parent text reports index == 1 (the
    // owning item index, not flat fragment index 2).
    let lie = View::derived(
        doc.get_key(&Key::<View>::named("norm")).unwrap(),
        "take é中😀 Xff soon".to_string(),
        Alignment::identity(),
    );
    doc.add(&Key::<View>::named("lie"), lie, Provenance::by("test")).unwrap();
    let lie_id = doc.get_key(&Key::<View>::named("lie")).unwrap().id();
    doc.add(
        &Key::<Tokens>::named("lied"),
        tokens(lie_id, &[(0, 4), (5, 14), (15, 18), (19, 23)]),
        Provenance::by("test"),
    )
    .unwrap();
    let mut lie_spans = Spans::new(lie_id);
    lie_spans.push(Span::new(0, 4));
    lie_spans.push_fragments(&[Span::new(5, 14), Span::new(15, 18)]);
    let lie_entities = Entities {
        spans: lie_spans,
        tokens: vec![IxRange::new(0, 1), IxRange::new(1, 3)],
        kind: vec![Tag(0), Tag(1)],
        kinds,
        score: vec![0.8, 0.9],
    };
    doc.add(&Key::<Entities>::named("lied"), lie_entities, Provenance::by("test")).unwrap();
    let v = check_layer(&doc, &Key::<Entities>::named("lied"));
    let frag_v = v.iter().find(|v| v.layer.contains("(fragments)")).expect("fragment violation reported");
    assert_eq!(frag_v.index, 1, "expected item index 1, got {frag_v}");
}
