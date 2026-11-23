//! The progressive Complete event carries the typed projection of its
//! graph when the request asks for it (issue #2517 F15). Before, the
//! answer was only DocumentGraph bytes, which a proto client could read
//! only by sending them back through InspectDocumentGraph.

mod common;

use tokio_stream::StreamExt as _;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::{
    AnalyzePipelineRequest, DocumentGraphResponse, InputBinding, InspectDocumentGraphRequest, OffsetMode,
    TypedResultOptions,
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
    // Each projection's receipt is its own attempt; all else is equal.
    let attempt = |mut t: turbo_kserve::proto::result::TypedDocumentResult| {
        t.receipt.as_mut().expect("a receipt").attempt_id.clear();
        t
    };
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
