use turbo_analysis::{Document, DocumentGraph, Ix, Key, Layer, Provenance, Span, Spans, ViewId, export, import};
use turbo_analysis_layers::{Edus, Sentences};

fn document() -> Document {
    let mut d = Document::new("ab cd");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 2));
    spans.push(Span::new(3, 5));
    d.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test:sentences")).unwrap();
    d
}

fn units(span: Span, sentence: u32) -> Edus {
    let mut units = Edus::new(ViewId::SOURCE);
    units.push(span, Ix::new(sentence));
    units
}

#[test]
fn an_in_range_sentence_index_must_contain_its_unit() {
    for span in [Span::new(0, 5), Span::new(3, 5)] {
        let mut d = document();
        assert!(d.add(&Key::DEFAULT, units(span, 0), Provenance::by("test:edus")).is_err());
    }
    document().add(&Key::DEFAULT, units(Span::new(3, 5), 1), Provenance::by("test:edus")).unwrap();
}

#[test]
fn wire_sentence_reference_cannot_point_at_another_existing_sentence() {
    let mut d = document();
    d.add(&Key::DEFAULT, units(Span::new(3, 5), 1), Provenance::by("test:edus")).unwrap();
    let codecs = turbo_analysis_layers::export::codecs();
    let mut graph = export(&d, &codecs).unwrap();
    let bytes = graph.layers.iter_mut().find(|l| l.layer == Edus::NAME.to_string()).unwrap().body.as_mut().unwrap();
    let end = bytes.len();
    bytes[end - 4..].copy_from_slice(&0u32.to_le_bytes());
    let graph = DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap();
    assert!(import(&graph, &codecs).is_err());
}

#[test]
fn containment_uses_the_named_sentence_instance() {
    let mut d = Document::new("ab cd");
    let mut short = Spans::new(ViewId::SOURCE);
    short.push(Span::new(0, 2));
    d.add(&Key::DEFAULT, Sentences { spans: short }, Provenance::by("test:default")).unwrap();
    let key = Key::<Sentences>::named("gold");
    let mut long = Spans::new(ViewId::SOURCE);
    long.push(Span::new(0, 5));
    d.add(&key, Sentences { spans: long }, Provenance::by("test:gold")).unwrap();
    let mut p = Provenance::by("test:edus");
    p.reads.push(key.id());
    d.add(&Key::DEFAULT, units(Span::new(3, 5), 0), p).unwrap();
    let codecs = turbo_analysis_layers::export::codecs();
    let graph = DocumentGraph::from_bytes(&export(&d, &codecs).unwrap().to_bytes().unwrap()).unwrap();
    let back = import(&graph, &codecs).unwrap();
    assert_eq!(back.made_on(Key::<Edus>::DEFAULT.id()), &[key.id()]);
}

#[test]
fn equal_byte_offsets_in_different_views_are_not_a_sentence_match() {
    use turbo_analysis::{View, ViewBuilder};
    let mut d = Document::new("ab cd");
    let source = d.get::<View>().unwrap().clone();
    let (text, edits) = ViewBuilder::new(source.text()).finish();
    let key = Key::named("other");
    d.add(&key, View::derived(&source, text, edits), Provenance::by("test:view")).unwrap();
    let view = d.get_key(&key).unwrap().id();
    let mut spans = Spans::new(view);
    spans.push(Span::new(0, 5));
    d.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test:sentences")).unwrap();
    assert!(d.add(&Key::DEFAULT, units(Span::new(0, 2), 0), Provenance::by("test:edus")).is_err());
}
