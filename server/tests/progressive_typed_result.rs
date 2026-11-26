//! The progressive Complete event carries the typed projection of its
//! graph when the request asks for it (issue #2517 F15). Before, the
//! answer was only DocumentGraph bytes, which a proto client could read
//! only by sending them back through InspectDocumentGraph.

mod common;

use tokio_stream::StreamExt as _;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::{
    AnalyzePipelineRequest, DocumentGraphResponse, InputBinding, InspectDocumentGraphRequest, OffsetMode,
    PiiOutputOptions, TypedDocumentResult, TypedResultOptions,
};

use common::*;

const TEXT: &str = "# Heading\n\nCafé typed answers. A second sentence follows.";

fn request(typed_result: Option<TypedResultOptions>) -> AnalyzePipelineRequest {
    AnalyzePipelineRequest {
        doc_id: "doc-typed".into(),
        text: TEXT.into(),
        offsets: OffsetMode::Utf8 as i32,
        profile: "minimal".into(),
        binding: Some(InputBinding {
            source_revision: "rev-1".into(),
            parse_artifact_id: "parse/1".into(),
            parse_artifact_digest: vec![7; 32],
        }),
        typed_result,
        operation_key: "op-typed".into(),
        ..Default::default()
    }
}

/// `t` with its receipt's attempt cleared: each projection's receipt is
/// its own attempt, and all else is equal.
fn attempt(mut t: TypedDocumentResult) -> TypedDocumentResult {
    t.receipt.as_mut().expect("a receipt").attempt_id.clear();
    t
}

async fn complete(client: &mut Analysis, req: AnalyzePipelineRequest) -> DocumentGraphResponse {
    let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut done = None;
    while let Some(item) = stream.next().await {
        if let Some(Event::Complete(c)) = item.unwrap().event {
            done = Some(c);
        }
    }
    done.expect("a complete event")
}

#[tokio::test]
async fn the_complete_event_carries_the_typed_projection_it_was_asked_for() {
    let (s, mut c) = serve().await;
    let opts = TypedResultOptions { view: 0, offsets: OffsetMode::Utf16 as i32 };
    let done = complete(&mut c, request(Some(opts))).await;
    let typed = done.typed.clone().expect("the typed projection");
    assert_eq!((typed.doc_id.as_str(), typed.offsets), ("doc-typed", OffsetMode::Utf16 as i32));
    assert!(!typed.layers.is_empty(), "{typed:?}");
    // The same projection InspectDocumentGraph gives for the same graph.
    let inspected = c
        .inspect_document_graph(InspectDocumentGraphRequest {
            doc_id: "doc-typed".into(),
            graph: done.graph.to_vec(),
            view: 0,
            offsets: OffsetMode::Utf16 as i32,
            binding: request(None).binding,
            operation_key: "op-typed".into(),
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(typed.receipt.as_ref().map(|r| r.operation_key.as_str()), Some("op-typed"));
    assert_eq!(attempt(typed), attempt(inspected));
    // Unasked, there is none, as before.
    assert_eq!(complete(&mut c, request(None)).await.typed, None);
    s.stop().await;
}

#[tokio::test]
async fn a_view_past_u16_is_refused_before_anything_runs() {
    let (s, mut c) = serve().await;
    let opts = TypedResultOptions { view: 70_000, offsets: OffsetMode::Utf8 as i32 };
    let e = c.analyze_document_progressive(request(Some(opts))).await.unwrap_err();
    assert_eq!(e.code(), tonic::Code::InvalidArgument, "{e:?}");
    assert!(e.message().contains("typed_result.view 70000"), "{e:?}");
    s.stop().await;
}

/// The masked view the `turbo-pii:masked` step writes beside the source.
const MASKED: u32 = 1;

#[tokio::test]
async fn the_projection_is_of_the_view_asked_for() {
    let (s, mut c) = serve().await;
    // Both texts disclosed, so each view's projection carries its own.
    let pii_output = Some(PiiOutputOptions { full_text: true, rewritten_text: true, ..Default::default() });
    let mut texts = Vec::new();
    for view in [0, MASKED] {
        let req = AnalyzePipelineRequest {
            doc_id: "doc-view".into(),
            text: "Write to alice.view@example.com before noon.".into(),
            offsets: OffsetMode::Utf8 as i32,
            steps: vec!["turbo-pii:pii".into(), "turbo-pii:masked".into()],
            pii_output,
            typed_result: Some(TypedResultOptions { view, offsets: OffsetMode::Utf8 as i32 }),
            ..Default::default()
        };
        let done = complete(&mut c, req).await;
        let typed = done.typed.clone().expect("the typed projection");
        assert_eq!(typed.view, view, "{typed:?}");
        let inspected = c
            .inspect_document_graph(InspectDocumentGraphRequest {
                doc_id: "doc-view".into(),
                graph: done.graph.to_vec(),
                view,
                offsets: OffsetMode::Utf8 as i32,
                pii_output,
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        texts.push(typed.raw_text.clone());
        assert_eq!(attempt(typed), attempt(inspected), "view {view}");
    }
    assert!(texts[0].contains("alice.view@example.com") && !texts[1].contains("alice.view"), "{texts:?}");
    s.stop().await;
}

#[tokio::test]
async fn a_view_the_graph_lacks_fails_the_stream_at_complete() {
    let (s, mut c) = serve().await;
    let opts = TypedResultOptions { view: 3, offsets: OffsetMode::Utf8 as i32 };
    let mut stream = c.analyze_document_progressive(request(Some(opts))).await.unwrap().into_inner();
    let mut failed = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(ev) => assert!(!matches!(ev.event, Some(Event::Complete(_))), "a complete event despite the view"),
            Err(e) => failed = Some(e),
        }
    }
    let e = failed.expect("the stream fails");
    assert_eq!(e.code(), tonic::Code::InvalidArgument, "{e:?}");
    assert!(e.message().contains("target view 3 is outside"), "{e:?}");
    s.stop().await;
}
