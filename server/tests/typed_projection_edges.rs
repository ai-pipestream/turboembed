//! Edge cases of projecting layers onto another view (#1219).
mod common;

use turbo_analysis::export::export;
use turbo_analysis::{Document, Key, Layer, Provenance, Span, Spans, View, ViewBuilder, ViewId};
use turbo_analysis_layers::Tokens;
use turbo_kserve::proto::result::{InspectDocumentGraphRequest, OffsetMode};
use turbo_kserve::results::document::codecs;

/// "a-b-c" with a child view "abc" that drops both hyphens.
fn doc_with_dropped_hyphens() -> Document {
    let mut doc = Document::new("a-b-c");
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    builder.keep(1);
    builder.replace(2, "");
    builder.keep(3);
    builder.replace(4, "");
    builder.keep(5);
    let (text, alignment) = builder.finish();
    doc.add(&Key::<View>::named("plain"), View::derived(&source, text, alignment), Provenance::by("test:view"))
        .unwrap();
    doc
}

/// Projects one discontinuous token onto the plain view in every offset
/// mode, checks its fragments, and sends the result back for validation.
async fn project_and_round_trip(fragments: &[Span], expected: (u32, u32)) {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = doc_with_dropped_hyphens();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push_fragments(fragments);
    doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test:tokens")).unwrap();
    let graph = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let out = client
            .inspect_document_graph(InspectDocumentGraphRequest {
                doc_id: "edges".into(),
                graph: graph.clone(),
                view: 1,
                offsets: mode as i32,
            })
            .await
            .unwrap()
            .into_inner();
        let tok = out.layers.iter().find(|l| l.id == Tokens::NAME.to_string()).unwrap();
        let got: Vec<(u32, u32)> = tok.annotations[0].spans.iter().map(|f| (f.start, f.end)).collect();
        assert_eq!(got, [expected], "{mode:?}");
        let back = client.round_trip_typed_document(out.clone()).await;
        assert!(back.is_ok(), "{mode:?}: the server refused its own projection: {:?}", back.err());
    }
    server.stop().await;
}

#[tokio::test]
async fn a_fragment_on_deleted_text_is_dropped_from_the_projection() {
    // "a" and the hyphen after "b": the hyphen has no text in the plain view.
    project_and_round_trip(&[Span::new(0, 1), Span::new(3, 4)], (0, 1)).await;
}

#[tokio::test]
async fn an_item_wholly_on_deleted_text_keeps_one_empty_fragment() {
    // Both hyphens: nothing of the token is left in the plain view.
    project_and_round_trip(&[Span::new(1, 2), Span::new(3, 4)], (1, 1)).await;
}
