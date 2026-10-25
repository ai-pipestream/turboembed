use turbo_analysis::{Document, DocumentGraph, Key, Provenance, Span, Spans, ViewId, export, import};
use turbo_analysis_layers::{AnchoredLayerKind, ArgumentRole, EventArgument, EventFrame, Events, Money, Sentences};

fn doc() -> Document {
    let mut doc = Document::new("éx");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 3));
    doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans }, Provenance::by("test:sentences")).unwrap();
    doc
}

fn event(span: Span, sentence: u32, anchor: Option<u32>) -> Events {
    let mut events = Events::new(ViewId::SOURCE);
    let arg = EventArgument {
        span,
        role: ArgumentRole::Patient,
        anchored_layer: if anchor.is_some() { AnchoredLayerKind::Money } else { AnchoredLayerKind::None },
        anchored_index: anchor,
    };
    events.push(EventFrame {
        trigger: Span::new(0, 2),
        trigger_lemma: None,
        sentence_ix: sentence,
        arguments: &[arg],
        confidence: 0.8,
    });
    events
}

#[test]
fn storage_checks_argument_bounds_and_both_utf8_endpoints() {
    for span in [Span::new(0, 99), Span::new(1, 2), Span::new(0, 1)] {
        let err = doc().add(&Key::DEFAULT, event(span, 0, None), Provenance::by("test:events")).unwrap_err();
        assert!(err.to_string().contains("argument"), "{err}");
    }
    doc().add(&Key::DEFAULT, event(Span::new(0, 2), 0, None), Provenance::by("test:events")).unwrap();
}

#[test]
fn graph_import_checks_argument_utf8_boundaries() {
    let mut doc = doc();
    doc.add(&Key::DEFAULT, event(Span::new(0, 2), 0, None), Provenance::by("test:events")).unwrap();
    let codecs = turbo_relation::export::codecs();
    let mut graph = export(&doc, &codecs).unwrap();
    let body = graph.layers.iter_mut().find(|l| l.layer == "turbo:events").unwrap().body.as_mut().unwrap();
    // One unanchored argument ends the payload: start, end, role, kind.
    let at = body.len() - 10;
    assert_eq!(&body[at..at + 4], &0u32.to_le_bytes());
    body[at..at + 4].copy_from_slice(&1u32.to_le_bytes());
    let bytes = graph.to_bytes().unwrap();
    let graph = DocumentGraph::from_bytes(&bytes).unwrap();
    let err = match import(&graph, &codecs) {
        Err(e) => e,
        Ok(_) => panic!("import accepted an argument inside é"),
    };
    assert!(err.to_string().contains("argument"), "{err}");
}

#[test]
fn sentence_and_anchor_references_must_exist() {
    assert!(doc().add(&Key::DEFAULT, event(Span::new(0, 2), 99, None), Provenance::by("test:events")).is_err());
    assert!(doc().add(&Key::DEFAULT, event(Span::new(0, 2), 0, Some(0)), Provenance::by("test:events")).is_err());
    let mut d = doc();
    d.add(&Key::DEFAULT, Money::new(ViewId::SOURCE), Provenance::by("test:money")).unwrap();
    assert!(d.add(&Key::DEFAULT, event(Span::new(0, 2), 0, Some(0)), Provenance::by("test:events")).is_err());
}

#[test]
fn named_sentence_instances_keep_their_resolved_reference() {
    let mut d = Document::new("éx");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 3));
    let key = Key::<Sentences>::named("gold");
    d.add(&key, Sentences { spans }, Provenance::by("test:sentences")).unwrap();
    let mut p = Provenance::by("test:events");
    p.reads.push(key.id());
    d.add(&Key::DEFAULT, event(Span::new(0, 2), 0, None), p).unwrap();
    assert_eq!(d.made_on(Key::<Events>::DEFAULT.id()), &[key.id()]);
    let codecs = turbo_relation::export::codecs();
    let graph = export(&d, &codecs).unwrap();
    let bytes = graph.to_bytes().unwrap();
    let restored = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    assert_eq!(restored.made_on(Key::<Events>::DEFAULT.id()), &[key.id()]);
}

#[test]
fn convenience_extraction_refuses_layers_from_a_different_view() {
    use turbo_analysis::{Ix, TagSet, View, ViewBuilder};
    use turbo_analysis_layers::{Dependencies, Tokens};
    let mut d = Document::new("a");
    let source = d.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(source.text());
    b.replace(1, "Acme raised");
    let (text, edits) = b.finish();
    let key = Key::named("expanded");
    d.add(&key, View::derived(&source, text, edits), Provenance::by("test:view")).unwrap();
    let view = d.get_key(&key).unwrap().id();
    let mut sentence = Spans::new(view);
    sentence.push(Span::new(0, 11));
    d.add(&Key::DEFAULT, Sentences { spans: sentence }, Provenance::by("test:sentences")).unwrap();
    let mut tokens = Spans::new(view);
    tokens.push(Span::new(0, 4));
    tokens.push(Span::new(5, 11));
    d.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: Some(vec![Ix::new(0); 2]) }, Provenance::by("test:tokens"))
        .unwrap();
    let relations = TagSet::new("ud", &["nsubj", "root"]);
    let deps: Dependencies = Dependencies {
        head: vec![Some(Ix::new(1)), None],
        relation: vec![relations.tag("nsubj").unwrap(), relations.tag("root").unwrap()],
        relations,
    };
    d.add(&Key::DEFAULT, deps, Provenance::by("test:deps")).unwrap();
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| turbo_relation::EventExtractor::new().events(&d)));
    assert!(result.is_ok(), "extraction panicked instead of refusing incompatible views");
    assert!(result.unwrap().is_err());
}

#[test]
fn foreign_anchor_views_are_refused_only_when_anchoring_is_enabled() {
    use turbo_analysis::{View, ViewBuilder};
    use turbo_analysis_layers::{Currency, MoneyMention};
    use turbo_relation::{EventConfig, EventExtractor};
    let text = "Revenue rose $1 .";
    let mut d = Document::new(text);
    let mut sentence = Spans::new(ViewId::SOURCE);
    sentence.push(Span::new(0, text.len() as u32));
    d.add(&Key::DEFAULT, Sentences { spans: sentence }, Provenance::by("test:sentence")).unwrap();
    let source = d.get::<View>().unwrap().clone();
    let (text, alignment) = ViewBuilder::new(source.text()).finish();
    let key = Key::named("another");
    d.add(&key, View::derived(&source, text, alignment), Provenance::by("test:view")).unwrap();
    let view = d.get_key(&key).unwrap().id();
    let mut money = Money::new(view);
    money.push(MoneyMention {
        span: Span::new(13, 15),
        negative: false,
        coefficient: "1",
        scale: 0,
        currency: Currency::parse("USD").unwrap(),
    });
    d.add(&Key::DEFAULT, money, Provenance::by("test:money")).unwrap();
    // Exercise public extraction with actual checked layers.
    let mut tokens = Spans::new(ViewId::SOURCE);
    for (a, z) in [(0, 7), (8, 12), (13, 15), (16, 17)] {
        tokens.push(Span::new(a, z));
    }
    let tokens = turbo_analysis_layers::Tokens { spans: tokens, sentence: None };
    let relations = turbo_analysis::TagSet::new("ud", &["nsubj", "obj", "punct", "root"]);
    let deps: turbo_analysis_layers::Dependencies = turbo_analysis_layers::Dependencies {
        head: vec![
            Some(turbo_analysis::Ix::new(1)),
            None,
            Some(turbo_analysis::Ix::new(1)),
            Some(turbo_analysis::Ix::new(1)),
        ],
        relation: ["nsubj", "root", "obj", "punct"].map(|r| relations.tag(r).unwrap()).to_vec(),
        relations,
    };
    d.add(&Key::DEFAULT, tokens, Provenance::by("test:tokens")).unwrap();
    d.add(&Key::DEFAULT, deps, Provenance::by("test:deps")).unwrap();
    assert!(EventExtractor::new().events(&d).is_err());
    let off = EventExtractor::with_config(EventConfig { anchor: false, ..Default::default() });
    assert!(off.events(&d).is_ok());
}
