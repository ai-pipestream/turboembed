mod discourse_support;

use turbo_analysis::{Analyzer, Document, DocumentGraph, Key, Layer, export, import};
use turbo_analysis_layers::{Dependencies, Edus, Sentences, Tokens};
use turbo_chunker::DiscourseAnnotator;

fn graph() -> DocumentGraph {
    let d = discourse_support::doc(
        &[("Acme closes because it rains .", "2:nsubj 0:root 5:mark 5:nsubj 2:advcl 2:punct")],
        true,
    );
    let a = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .add(DiscourseAnnotator::new())
        .build()
        .unwrap();
    let mut docs = vec![d];
    for result in a.run(&mut docs) {
        result.unwrap();
    }
    export(&docs[0], &turbo_analysis_layers::export::codecs()).unwrap()
}

fn restore(g: &DocumentGraph) -> Result<Document, turbo_analysis::export::ExportError> {
    import(&DocumentGraph::from_bytes(&g.to_bytes().unwrap()).unwrap(), &turbo_analysis_layers::export::codecs())
}

#[test]
fn valid_discourse_pipeline_roundtrips() {
    let back = restore(&graph()).unwrap();
    let sentences = back.require(&Key::<Sentences>::DEFAULT).unwrap();
    let edus = back.require(&Key::<Edus>::DEFAULT).unwrap();
    assert!(!edus.sentence.is_empty());
    assert!(edus.sentence.iter().all(|s| s.get() < sentences.spans.len()));
}

#[test]
fn out_of_bounds_edu_sentence_is_refused() {
    let mut g = graph();
    let layer = g.layers.iter_mut().find(|l| l.layer == Edus::NAME.to_string()).unwrap();
    let body = layer.body.as_mut().unwrap();
    let end = body.len();
    body[end - 4..].copy_from_slice(&999_u32.to_le_bytes());
    let result = restore(&g);
    if let Ok(ref back) = result {
        let edus = back.require(&Key::<Edus>::DEFAULT).unwrap();
        eprintln!(
            "accepted sentence {} with {} sentences",
            edus.sentence.last().unwrap().get(),
            back.require(&Key::<Sentences>::DEFAULT).unwrap().spans.len()
        );
    }
    assert!(result.is_err(), "a dangling Edus sentence index imported");
}

#[test]
fn storage_requires_the_referenced_sentence_instance_and_bounds() {
    use turbo_analysis::{Ix, Provenance, Span, Spans, ViewId};
    let units = |index| {
        let mut layer = Edus::new(ViewId::SOURCE);
        layer.push(Span::new(0, 2), Ix::new(index));
        layer
    };
    let mut d = Document::new("é");
    assert!(d.add(&Key::DEFAULT, units(0), Provenance::by("test:edus")).is_err());
    d.add(&Key::DEFAULT, Sentences { spans: Spans::new(ViewId::SOURCE) }, Provenance::by("test:sentences")).unwrap();
    assert!(d.add(&Key::DEFAULT, units(0), Provenance::by("test:edus")).is_err());
    let mut d = Document::new("é");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 2));
    d.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test:sentences")).unwrap();
    for index in [1, 999, u32::MAX - 1] {
        assert!(d.add(&Key::DEFAULT, units(index), Provenance::by("test:edus")).is_err());
    }
    d.add(&Key::DEFAULT, units(0), Provenance::by("test:edus")).unwrap();
}

#[test]
fn named_sentence_reference_survives_graph_bytes_and_refuses_missing_target() {
    use turbo_analysis::{Ix, Provenance, Span, Spans, ViewId};
    let mut d = Document::new("é");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 2));
    let sentence_key = Key::<Sentences>::named("gold");
    d.add(&sentence_key, Sentences { spans }, Provenance::by("test:sentences")).unwrap();
    let mut units = Edus::new(ViewId::SOURCE);
    units.push(Span::new(0, 2), Ix::new(0));
    let mut p = Provenance::by("test:edus");
    p.reads.push(sentence_key.id());
    d.add(&Key::DEFAULT, units, p).unwrap();
    assert_eq!(d.made_on(Key::<Edus>::DEFAULT.id()), &[sentence_key.id()]);
    let codecs = turbo_analysis_layers::export::codecs();
    let mut graph = export(&d, &codecs).unwrap();
    let restored = restore(&graph).unwrap();
    assert_eq!(restored.made_on(Key::<Edus>::DEFAULT.id()), &[sentence_key.id()]);
    assert_eq!(restored.get::<Edus>().unwrap().sentence, [Ix::new(0)]);
    graph.layers.retain(|l| l.layer != Sentences::NAME.to_string());
    assert!(restore(&graph).is_err());
}

#[test]
fn empty_edus_need_no_sentence_reference() {
    use turbo_analysis::{Provenance, ViewId};
    let mut d = Document::new("");
    d.add(&Key::DEFAULT, Edus::new(ViewId::SOURCE), Provenance::by("test:edus")).unwrap();
    let codecs = turbo_analysis_layers::export::codecs();
    assert!(restore(&export(&d, &codecs).unwrap()).is_ok());
}

#[test]
fn reserved_wire_sentence_index_returns_an_error_without_panicking() {
    let mut graph = graph();
    let layer = graph.layers.iter_mut().find(|l| l.layer == Edus::NAME.to_string()).unwrap();
    let body = layer.body.as_mut().unwrap();
    let end = body.len();
    body[end - 4..].copy_from_slice(&u32::MAX.to_le_bytes());
    let result = std::panic::catch_unwind(|| restore(&graph));
    assert!(result.is_ok(), "reserved wire index panicked");
    assert!(result.unwrap().is_err());
}
