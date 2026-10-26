//! Asset payloads keep presence, containment and metadata through text deletion.
mod common;
use turbo_analysis::{Document, Key, Provenance, Span, View, ViewBuilder, ViewId, export::export};
use turbo_kserve::{
    proto::result::{
        DocumentGraphRequest, InspectDocumentGraphRequest, OffsetMode, SpanFragment, TypedAnnotation,
        TypedDocumentResult, typed_annotation::Value,
    },
    results::document::codecs,
};
use turbo_normalize_layers::{AssetEncoding, AssetFormat, Assets};

const TEXT: &str = "édata:application/octet-stream;base64,TQ==😀";

fn document() -> Document {
    let start = TEXT.find("TQ==").unwrap() as u32;
    let end = start + 4;
    let mut d = Document::new(TEXT);
    let source = d.get::<View>().unwrap().clone();
    for (name, kind) in [("partial", 1), ("overlap", 2), ("deleted", 3)] {
        let mut b = ViewBuilder::new(TEXT);
        if kind == 1 {
            b.keep(start as usize);
            b.replace(end as usize, "");
            b.keep(TEXT.len());
        } else {
            b.replace(TEXT.len(), if kind == 2 { "Q" } else { "" });
        }
        let (text, alignment) = b.finish();
        d.add(&Key::named(name), View::derived(&source, text, alignment), Provenance::by("test:view")).unwrap();
    }
    for name in ["", "gold"] {
        d.add(&Key::named(name), assets(), Provenance::by("test:asset")).unwrap();
    }
    d
}

fn assets() -> Assets {
    let start = TEXT.find("TQ==").unwrap() as u32;
    let mut a = Assets::new(ViewId::SOURCE);
    a.spans.push(Span::new(2, start + 4));
    a.payload.push(Span::new(start, start + 4));
    a.format.push(AssetFormat::Declared);
    a.declared_type.push("application/octet-stream");
    a.encoding.push(AssetEncoding::Base64);
    a.decoded_len.push(1);
    a.width.push(None);
    a.height.push(None);
    a
}

fn annotation<'a>(msg: &'a mut TypedDocumentResult, name: &str) -> &'a mut TypedAnnotation {
    let layer = msg.layers.iter_mut().find(|l| l.id == "turbo:assets" && l.instance == name).unwrap();
    assert_eq!(layer.annotations.len(), 1);
    &mut layer.annotations[0]
}

#[tokio::test]
async fn assets_keep_payload_presence_when_its_text_is_deleted() {
    let graph = export(&document(), codecs()).unwrap().to_bytes().unwrap();
    let (server, mut c) = common::serve_no_admin().await;
    let graph =
        c.round_trip_document_graph(DocumentGraphRequest { graph: graph.into() }).await.unwrap().into_inner().graph;
    let mut failures = Vec::new();
    let mut passed = 0;
    for view in [0, 1, 2, 3] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let mut out = c
                .inspect_document_graph(InspectDocumentGraphRequest {
                    doc_id: "asset-deletion".into(),
                    graph: graph.to_vec(),
                    view,
                    offsets: mode as i32,
                })
                .await
                .unwrap()
                .into_inner();
            let delta = u32::from(mode != OffsetMode::Utf8);
            let start = TEXT.find("TQ==").unwrap() as u32 - delta;
            let (payload, outer) = match view {
                0 => ((start, start + 4), (2 - delta, start + 4)),
                1 => ((start, start), (2 - delta, start)),
                2 => ((0, 1), (0, 1)),
                3 => ((0, 0), (0, 0)),
                _ => unreachable!(),
            };
            for name in ["", "gold"] {
                let row = annotation(&mut out, name);
                assert_eq!(row.spans.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(), [outer]);
                let Some(Value::Asset(a)) = &row.value else { panic!("missing asset") };
                let span = a.payload_span.as_ref().expect("payload presence preserved");
                assert_eq!((span.start, span.end), payload, "{name}/{view}/{mode:?}");
                assert_eq!(a.decoded_bytes, 1);
                assert_eq!(a.format, "octet-stream");
                assert_eq!(a.media_type, "application/octet-stream");
                assert_eq!(a.declared_type, "application/octet-stream");
                assert_eq!(a.encoding, "base64");
                assert_eq!((a.width, a.height), (None, None));
                assert!(row.model_ref.is_some());
            }
            match c.round_trip_typed_document(out.clone()).await {
                Ok(echo) => {
                    assert_eq!(echo.into_inner(), out);
                    passed += 1;
                }
                Err(e) => failures.push(format!("{view}/{mode:?}: {e}")),
            }
        }
    }
    server.stop().await;
    eprintln!("asset projection: {passed}/12 view/mode cases passed, each with default/named instances");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[tokio::test]
async fn missing_malformed_or_uncontained_asset_payloads_are_refused() {
    let graph = export(&document(), codecs()).unwrap().to_bytes().unwrap();
    let (server, mut c) = common::serve_no_admin().await;
    let mut accepted = Vec::new();
    let mut refused = 0;
    for view in [0, 1, 3] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let out = c
                .inspect_document_graph(InspectDocumentGraphRequest {
                    doc_id: "asset-malformed".into(),
                    graph: graph.clone(),
                    view,
                    offsets: mode as i32,
                })
                .await
                .unwrap()
                .into_inner();
            for name in ["", "gold"] {
                for variant in 0..6 {
                    let mut bad = out.clone();
                    let len = match mode {
                        OffsetMode::Utf8 => bad.raw_text.len(),
                        OffsetMode::Utf16 => bad.raw_text.encode_utf16().count(),
                        OffsetMode::CodePoint => bad.raw_text.chars().count(),
                    } as u32;
                    let interior = match mode {
                        OffsetMode::Utf8 if view != 3 => 1,
                        OffsetMode::Utf16 if view != 3 => len - 1,
                        _ => len + 1,
                    };
                    let row = annotation(&mut bad, name);
                    let Some(Value::Asset(a)) = &mut row.value else { panic!("missing asset") };
                    match variant {
                        0 => a.payload_span = None,
                        1 => a.payload_span = Some(SpanFragment { start: 1, end: 0 }),
                        2 => a.payload_span = Some(SpanFragment { start: 0, end: len + 1 }),
                        3 => a.payload_span = Some(SpanFragment { start: interior, end: interior }),
                        // In nonempty views this is an in-bounds empty span
                        // outside the asset, so emptiness alone cannot excuse it.
                        4 => a.payload_span = Some(SpanFragment { start: 0, end: u32::from(view == 3) }),
                        5 => row.spans.clear(),
                        _ => unreachable!(),
                    }
                    match c.round_trip_typed_document(bad).await {
                        Ok(_) => accepted.push((view, mode, name, variant)),
                        Err(error) => {
                            assert_eq!(error.code(), tonic::Code::InvalidArgument);
                            assert_eq!(error.metadata().get("turbo-code").unwrap(), "256");
                            assert_eq!(error.metadata().get("turbo-field").unwrap(), "0");
                            refused += 1;
                        }
                    }
                }
            }
        }
    }
    server.stop().await;
    assert!(accepted.is_empty(), "accepted {} malformed asset payloads: {accepted:?}", accepted.len());
    assert_eq!(refused, 108);
}

#[test]
fn native_assets_still_require_nonempty_payloads_inside_the_source_span() {
    for payload in [Span::new(2, 2), Span::new(0, 2)] {
        let mut d = Document::new(TEXT);
        let mut a = assets();
        a.payload[0] = payload;
        let error = d.add(&Key::named("invalid"), a, Provenance::by("test:invalid-native")).unwrap_err();
        assert!(error.to_string().contains("non-empty payload"), "{error}");
    }
}
