//! Event arguments retain valid projected emptiness and strict wire boundaries.
mod common;
use turbo_analysis::{Document, Key, Provenance, Span, Spans, View, ViewBuilder, ViewId, export::export};
use turbo_analysis_layers::{AnchoredLayerKind, ArgumentRole, EventArgument, EventFrame, Events, Sentences};
use turbo_kserve::{
    proto::result::{
        DocumentGraphRequest, EventAnnotation, InspectDocumentGraphRequest, OffsetMode, SpanFragment,
        TypedDocumentResult, typed_annotation::Value,
    },
    results::document::codecs,
};

fn document() -> Document {
    let mut d = Document::new("é-x-😀");
    let source = d.get::<View>().unwrap().clone();
    for (name, partial, replacement) in [("partial", true, ""), ("overlap", false, "Q"), ("deleted", false, "")] {
        let mut b = ViewBuilder::new(source.text());
        if partial {
            b.replace(2, "");
            b.keep(9);
        } else {
            b.replace(9, replacement);
        }
        let (text, align) = b.finish();
        d.add(&Key::named(name), View::derived(&source, text, align), Provenance::by("test:view")).unwrap();
    }
    for name in ["", "gold"] {
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(Span::new(0, 9));
        d.add(&Key::named(name), Sentences { spans }, Provenance::by("test:sentences")).unwrap();
        let arguments = [(0, 2, ArgumentRole::Agent), (3, 4, ArgumentRole::Patient), (5, 9, ArgumentRole::Location)]
            .map(|(start, end, role)| EventArgument {
                span: Span::new(start, end),
                role,
                anchored_layer: AnchoredLayerKind::None,
                anchored_index: None,
            });
        let mut events = Events::new(ViewId::SOURCE);
        events.push(EventFrame {
            trigger: Span::new(0, 9),
            trigger_lemma: Some("act"),
            sentence_ix: 0,
            arguments: &arguments,
            confidence: 0.75,
        });
        let provenance =
            Provenance { reads: vec![Key::<Sentences>::named(name).id()], ..Provenance::by("test:event-projection") };
        d.add(&Key::named(name), events, provenance).unwrap();
    }
    d
}

fn event<'a>(msg: &'a mut TypedDocumentResult, name: &str) -> &'a mut EventAnnotation {
    let layer = msg.layers.iter_mut().find(|l| l.id == "turbo:events" && l.instance == name).unwrap();
    assert_eq!(layer.annotations.len(), 1);
    let Some(Value::Event(e)) = &mut layer.annotations[0].value else { panic!("missing event") };
    e
}

fn expected(view: u32, mode: OffsetMode) -> [(u32, u32); 3] {
    match (view, mode) {
        (0, OffsetMode::Utf8) => [(0, 2), (3, 4), (5, 9)],
        (0, OffsetMode::Utf16) => [(0, 1), (2, 3), (4, 6)],
        (0, OffsetMode::CodePoint) => [(0, 1), (2, 3), (4, 5)],
        (1, OffsetMode::Utf8) => [(0, 0), (1, 2), (3, 7)],
        (1, OffsetMode::Utf16) => [(0, 0), (1, 2), (3, 5)],
        (1, OffsetMode::CodePoint) => [(0, 0), (1, 2), (3, 4)],
        (2, _) => [(0, 1); 3],
        (3, _) => [(0, 0); 3],
        _ => unreachable!(),
    }
}

#[tokio::test]
async fn projected_event_arguments_survive_text_deletion() {
    let bytes = export(&document(), codecs()).unwrap().to_bytes().unwrap();
    let (server, mut c) = common::serve_no_admin().await;
    let graph =
        c.round_trip_document_graph(DocumentGraphRequest { graph: bytes.into() }).await.unwrap().into_inner().graph;
    let mut failures = Vec::new();
    let mut passed = 0;
    for view in [0, 1, 2, 3] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let mut out = c
                .inspect_document_graph(InspectDocumentGraphRequest {
                    doc_id: "event-deletion".into(),
                    graph: graph.to_vec(),
                    view,
                    offsets: mode as i32,
                })
                .await
                .unwrap()
                .into_inner();
            for name in ["", "gold"] {
                let e = event(&mut out, name);
                assert_eq!(e.arguments.len(), 3, "all arguments retained for {name}/{view}/{mode:?}");
                assert_eq!(e.trigger_lemma.as_deref(), Some("act"));
                assert_eq!(e.sentence_index, 0);
                assert_eq!(e.confidence.to_bits(), 0.75_f32.to_bits());
                for ((a, span), role) in
                    e.arguments.iter().zip(expected(view, mode)).zip(["agent", "patient", "location"])
                {
                    let actual = a.span.as_ref().unwrap();
                    assert_eq!((actual.start, actual.end), span, "{name}/{view}/{mode:?}");
                    assert_eq!(a.role, role);
                    assert_eq!(a.anchored_layer, "none");
                    assert_eq!(a.anchored_index, None);
                }
            }
            match c.round_trip_typed_document(out.clone()).await {
                Ok(echoed) => {
                    assert_eq!(echoed.into_inner(), out);
                    passed += 1;
                }
                Err(e) => failures.push(format!("{view}/{mode:?}: {e}")),
            }
        }
    }
    server.stop().await;
    eprintln!("event argument projection: {passed}/12 view/mode cases passed, each with default/named instances");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[tokio::test]
async fn malformed_projected_arguments_and_anchors_are_refused() {
    let graph = export(&document(), codecs()).unwrap().to_bytes().unwrap();
    let (server, mut c) = common::serve_no_admin().await;
    let mut refused = 0;
    for view in [0, 1, 3] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let out = c
                .inspect_document_graph(InspectDocumentGraphRequest {
                    doc_id: "event-malformed".into(),
                    graph: graph.clone(),
                    view,
                    offsets: mode as i32,
                })
                .await
                .unwrap()
                .into_inner();
            for name in ["", "gold"] {
                for variant in 0..9 {
                    let mut bad = out.clone();
                    let len = match mode {
                        OffsetMode::Utf8 => bad.raw_text.len(),
                        OffsetMode::Utf16 => bad.raw_text.encode_utf16().count(),
                        OffsetMode::CodePoint => bad.raw_text.chars().count(),
                    } as u32;
                    let interior = match mode {
                        OffsetMode::Utf8 if view == 0 => 1,
                        OffsetMode::Utf8 if view == 1 => 4,
                        OffsetMode::Utf16 if view != 3 => len - 1,
                        _ => len + 1,
                    };
                    let a = &mut event(&mut bad, name).arguments[0];
                    match variant {
                        0 => a.span = None,
                        1 => a.span = Some(SpanFragment { start: 1, end: 0 }),
                        2 => a.span = Some(SpanFragment { start: 0, end: len + 1 }),
                        3 => a.span = Some(SpanFragment { start: interior, end: interior }),
                        4 => a.anchored_index = Some(0),
                        5 => a.anchored_layer = "opennlp:entities".into(),
                        6 => {
                            a.anchored_layer = "test:unknown".into();
                            a.anchored_index = Some(0);
                        }
                        7 => {
                            a.anchored_layer = "opennlp:entities".into();
                            a.anchored_index = Some(u32::MAX);
                        }
                        8 => a.anchored_layer.clear(),
                        _ => unreachable!(),
                    }
                    let error = c.round_trip_typed_document(bad).await.unwrap_err();
                    assert_eq!(error.code(), tonic::Code::InvalidArgument);
                    assert_eq!(error.metadata().get("turbo-code").unwrap(), "256");
                    assert_eq!(error.metadata().get("turbo-field").unwrap(), "0");
                    refused += 1;
                }
            }
        }
    }
    server.stop().await;
    assert_eq!(refused, 162);
}

#[test]
fn native_event_arguments_still_require_nonempty_source_spans() {
    let mut d = document();
    let mut events = d.get::<Events>().unwrap().clone();
    events.arguments[0].span = Span::new(0, 0);
    let error = d.add(&Key::named("invalid"), events, Provenance::by("test:invalid-native")).unwrap_err();
    assert!(error.to_string().contains("empty argument"), "{error}");
}
