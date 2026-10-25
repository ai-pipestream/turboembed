use turbo_analysis::{Document, DocumentGraph, Ix, Key, Layer, Provenance, Span, Spans, ViewId, export, import};
use turbo_analysis_layers::{
    AnchoredLayerKind, ArgumentRole, DeonticKind, Edus, EpistemicStatus, EventArgument, EventFrame, Events, Modality,
    ModalityPolarity, ModalityTag, Sentences,
};

const TEXT: &str = "éab😀cd";
const A: &[Span] = &[Span::new(0, 2), Span::new(8, 10)];
const B: &[Span] = &[Span::new(0, 3), Span::new(8, 10)];

fn spans<L>(fragments: &[Span]) -> Spans<L> {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.try_push_fragments(fragments).unwrap();
    spans
}

fn modality(fragments: &[Span]) -> Modality {
    let mut layer = Modality::new(ViewId::SOURCE);
    layer.push(
        Span::new(0, 10),
        ModalityTag {
            polarity: ModalityPolarity::Negated,
            epistemic: EpistemicStatus::Hedged,
            deontic: DeonticKind::Prohibition,
            cue: Some(Span::new(0, 2)),
            confidence: 0.75,
        },
    );
    // Keep the same envelope, so the derived reach column remains valid.
    layer.spans = spans(fragments);
    layer
}

fn events(fragments: &[Span]) -> Events {
    let mut layer = Events::new(ViewId::SOURCE);
    layer.push(EventFrame {
        trigger: Span::new(0, 10),
        trigger_lemma: Some("act"),
        sentence_ix: 0,
        arguments: &[EventArgument {
            span: Span::new(0, 2),
            role: ArgumentRole::Agent,
            anchored_layer: AnchoredLayerKind::None,
            anchored_index: None,
        }],
        confidence: 0.75,
    });
    layer.spans = spans(fragments);
    layer
}

fn edus(fragments: &[Span]) -> Edus {
    Edus { spans: spans(fragments), sentence: vec![Ix::new(0)] }
}

fn insert<L: Layer + Clone>(layer: &L, name: &'static str) -> Document {
    layer.check().expect("the discontinuous layer is natively valid");
    let mut doc = Document::new(TEXT);
    let mut sentences = Spans::new(ViewId::SOURCE);
    sentences.push(Span::new(0, 10));
    for key in [Key::DEFAULT, Key::named("gold")] {
        doc.add(&key, Sentences { spans: sentences.clone() }, Provenance::by("test:sentences")).unwrap();
    }
    let provenance =
        Provenance { reads: vec![Key::<Sentences>::named("gold").id()], ..Provenance::by("test:fragment-equality") };
    doc.add(&Key::named(name), layer.clone(), provenance).expect("valid document insertion");
    doc
}

fn fragment_rows<L: Layer>(layer: &L) -> Vec<Vec<Span>> {
    let spans = layer.text_spans().unwrap();
    (0..layer.len())
        .map(|i| {
            if spans.frag_item_end.is_empty() {
                vec![Span::new(spans.start[i], spans.end[i])]
            } else {
                spans.item_frag_range(i).map(|f| Span::new(spans.frag_start[f], spans.frag_end[f])).collect()
            }
        })
        .collect()
}

fn distinct<L: Layer + Clone + PartialEq + std::fmt::Debug>(a: L, b: L) {
    for name in ["", "named"] {
        let a_doc = insert(&a, name);
        let b_doc = insert(&b, name);
        let a = a_doc.get_key(&Key::<L>::named(name)).unwrap();
        let b = b_doc.get_key(&Key::<L>::named(name)).unwrap();
        assert_ne!(fragment_rows(a), fragment_rows(b));
        assert_ne!(a, b, "valid native records with different covered fragments must differ");
    }
}

fn roundtrip<L: Layer + Clone + PartialEq + std::fmt::Debug>(layer: L) {
    let codecs = turbo_analysis_layers::export::codecs();
    for name in ["", "named"] {
        let doc = insert(&layer, name);
        let bytes = export(&doc, &codecs).unwrap().to_bytes().unwrap();
        let restored = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
        let key = Key::<L>::named(name);
        let back = restored.get_key(&key).unwrap();
        back.check().unwrap();
        assert_eq!(fragment_rows(back), fragment_rows(&layer), "binary import must retain actual fragments");
        assert_eq!(back, &layer);
        assert_eq!(restored.made_on(key.id()), doc.made_on(key.id()));
        assert_eq!(export(&restored, &codecs).unwrap().to_bytes().unwrap(), bytes);
    }
}

#[test]
fn modality_equality_distinguishes_valid_discontinuous_scope_coverage() {
    distinct(modality(A), modality(B));
    distinct(modality(A), modality(&[Span::new(0, 10)]));
}

#[test]
fn events_equality_distinguishes_valid_discontinuous_trigger_coverage() {
    distinct(events(A), events(B));
    distinct(events(A), events(&[Span::new(0, 10)]));
}

#[test]
fn edus_discontinuity_is_refused_by_native_storage_and_binary_import() {
    use turbo_analysis::export::{Writer, put_spans_raw};
    let codecs = turbo_analysis_layers::export::codecs();
    for fragments in [A, B] {
        let invalid = edus(fragments);
        assert!(invalid.check().unwrap_err().contains("discontinuous spans are not allowed"));
        let mut doc = insert(&edus(&[Span::new(0, 10)]), "baseline");
        let provenance =
            Provenance { reads: vec![Key::<Sentences>::named("gold").id()], ..Provenance::by("test:invalid-edus") };
        assert!(doc.add(&Key::named("invalid"), invalid, provenance).is_err());
        let mut graph = export(&doc, &codecs).unwrap();
        let mut writer = Writer::new();
        put_spans_raw(&mut writer, "", &[fragments.to_vec()]).unwrap();
        writer.u32(0);
        graph.layers.iter_mut().find(|record| record.layer == Edus::NAME.to_string()).unwrap().body =
            Some(writer.finish());
        let bytes = graph.to_bytes().unwrap();
        let error = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs)
            .err()
            .expect("invalid discontinuous Edus must be refused");
        assert!(error.to_string().contains("discontinuous spans are not allowed"), "{error}");
    }
}

#[test]
fn modality_binary_graph_preserves_valid_discontinuous_scopes() {
    roundtrip(modality(A));
    roundtrip(modality(B));
}

#[test]
fn events_binary_graph_preserves_valid_discontinuous_triggers() {
    roundtrip(events(A));
    roundtrip(events(B));
}

#[test]
fn edus_contiguous_units_roundtrip_without_widening_the_valid_domain() {
    roundtrip(edus(&[Span::new(0, 10)]));
}

#[test]
fn identical_fragment_layouts_and_contiguous_records_stay_equal() {
    for fragments in [A, B, &[Span::new(0, 10)]] {
        assert_eq!(modality(fragments), modality(fragments));
        assert_eq!(events(fragments), events(fragments));
    }
    assert_eq!(edus(&[Span::new(0, 10)]), edus(&[Span::new(0, 10)]));
    let mut changed = modality(A);
    changed.confidence[0] = 0.5;
    assert_ne!(modality(A), changed);
    let mut changed = events(A);
    changed.arguments[0].role = ArgumentRole::Patient;
    assert_ne!(events(A), changed);
    let mut positive = modality(A);
    positive.confidence[0] = 0.0;
    let mut negative = positive.clone();
    negative.confidence[0] = -0.0;
    assert_eq!(positive, negative);
    let mut positive = events(A);
    positive.confidence[0] = 0.0;
    let mut negative = positive.clone();
    negative.confidence[0] = -0.0;
    assert_eq!(positive, negative);
}
