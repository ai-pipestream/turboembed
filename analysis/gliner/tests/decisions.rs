use turbo_analysis::{Document, IxRange, Key, Layer, Provenance, Span, Spans, Tag, TagSet, View};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_gliner::{Decisions, GlinerAnnotator};

#[test]
fn named_decisions_evidence_and_wire_boundaries() {
    let mut doc = Document::new("Zoë");
    let view = doc.get::<View>().unwrap().id();
    let mut spans = Spans::new(view);
    spans.push(Span::new(0, 4));
    doc.add(&Key::DEFAULT, Tokens { spans: spans.clone(), sentence: None }, Provenance::by("tokens")).unwrap();
    let mut spans = Spans::new(view);
    spans.push(Span::new(0, 4));
    let entities = Key::<Entities>::named("ner");
    let decisions = Key::<Decisions>::named("ner");
    let mut p = Provenance::by("fixture:ner");
    p.reads = vec![Key::<Tokens>::DEFAULT.id()];
    doc.add(
        &entities,
        Entities {
            spans,
            tokens: vec![IxRange::new(0, 1)],
            kind: vec![Tag(0)],
            kinds: TagSet::new("fixture:labels", &["person"]),
            score: vec![0.8],
        },
        p,
    )
    .unwrap();
    let mut p = Provenance::by("fixture:ner");
    p.on = vec![entities.id()];
    doc.add(&decisions, Decisions { logits: vec![1.3862944], threshold: 0.5 }, p).unwrap();
    let evidence = GlinerAnnotator::evidence(&doc, entities, decisions).unwrap();
    evidence.verify(&doc).unwrap();
    assert_eq!(evidence.raw, vec![(0, f64::from(1.3862944f32))]);
    let mut codecs = turbo_analysis_layers::export::codecs();
    codecs.register::<Decisions>();
    let graph = turbo_analysis::export::export(&doc, &codecs).unwrap();
    let restored = turbo_analysis::export::import(&graph, &codecs).unwrap();
    assert_eq!(restored.require(&decisions).unwrap().threshold, 0.5);
    assert!(GlinerAnnotator::evidence(&doc, Key::<Entities>::named("other"), decisions).is_err());
    // Corrupt the raw scalar payload without changing its layer metadata.
    for payload in [
        vec![],
        vec![0; 5],
        f32::NAN.to_le_bytes().to_vec(),
        [0.5f32.to_le_bytes(), f32::INFINITY.to_le_bytes()].concat(),
    ] {
        let mut bad = graph.clone();
        let row = bad.layers.iter_mut().find(|r| r.layer == Decisions::ID).unwrap();
        row.body = Some(payload);
        assert!(turbo_analysis::export::import(&bad, &codecs).is_err());
    }
}

fn aligned_doc(threshold: f32) -> Document {
    let mut doc = Document::new("Zoë Ann");
    let view = doc.get::<View>().unwrap().id();
    let mut spans = Spans::new(view);
    spans.push(Span::new(0, 4));
    spans.push(Span::new(5, 8));
    doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("tokens")).unwrap();
    let mut p = Provenance::by("fixture:ner");
    p.reads = vec![Key::<Tokens>::DEFAULT.id()];
    let mut spans = Spans::new(view);
    spans.push(Span::new(0, 4));
    spans.push(Span::new(5, 8));
    let entities = Entities {
        spans,
        tokens: vec![IxRange::new(0, 1), IxRange::new(1, 2)],
        kind: vec![Tag(0), Tag(0)],
        kinds: TagSet::new("fixture:labels", &["person"]),
        score: vec![0.8, 0.7],
    };
    doc.add(&Key::DEFAULT, entities, p).unwrap();
    let mut p = Provenance::by("fixture:ner");
    p.on = vec![Key::<Entities>::DEFAULT.id()];
    doc.add(&Key::DEFAULT, Decisions { logits: vec![1.0, 0.5], threshold }, p).unwrap();
    doc
}

#[test]
fn decisions_whose_length_is_not_the_aligned_entities_length_are_refused() {
    let doc = aligned_doc(0.5);
    let codecs = turbo_gliner::codecs();
    let graph = turbo_analysis::export::export(&doc, &codecs).unwrap();
    let back = turbo_analysis::export::import(&graph, &codecs).unwrap();
    assert_eq!(back.get::<Decisions>().unwrap().logits, vec![1.0, 0.5]);
    for logits in [vec![1.0f32], vec![1.0, 0.5, 0.25], vec![]] {
        let mut bad = graph.clone();
        let row = bad.layers.iter_mut().find(|r| r.layer == Decisions::ID).unwrap();
        let mut payload = 0.5f32.to_le_bytes().to_vec();
        payload.extend(logits.iter().flat_map(|x| x.to_le_bytes()));
        row.body = Some(payload);
        let err = turbo_analysis::export::import(&bad, &codecs).err();
        assert!(
            matches!(err, Some(turbo_analysis::export::ExportError::InvalidLayer { .. })),
            "{} logits: {err:?}",
            logits.len()
        );
    }
}
