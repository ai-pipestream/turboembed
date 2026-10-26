//! The progressive analysis and the routes that read its graph back:
//! window, typed annotations, output formats and chunk projection.

use super::*;

const TEXT: &str = "# Heading\n\nWorld Health Organization (WHO) streams layers. Fast stages arrive first!";

fn pipeline(profile: &str) -> Value {
    json!({ "doc_id": "doc-1", "text": TEXT, "profile": profile })
}

#[tokio::test]
async fn a_progressive_analysis_streams_and_its_graph_reads_back() {
    let (s, d) = start(false).await;
    let events = lines(&d, "/api/analyze/progressive", pipeline("default")).await;
    let sequence: Vec<u64> = events.iter().map(|e| e["sequence"].as_u64().unwrap()).collect();
    assert_eq!(sequence, (1..=events.len() as u64).collect::<Vec<_>>());
    assert_eq!(events[0]["event"]["started"]["doc_id"], "doc-1", "{}", events[0]);
    assert!(events[1]["event"]["layers_ready"]["completed_step"].is_string(), "{}", events[1]);
    let graph = events.last().unwrap()["event"]["complete"]["graph"].as_str().unwrap().to_string();
    assert!(!unbase64(&graph).is_empty());

    // The same events as server-sent events.
    let (status, kind, body) =
        http(&d, "POST", "/api/analyze/progressive", "text/event-stream", &pipeline("default").to_string()).await;
    assert_eq!((status, kind.as_str()), (200, "text/event-stream"), "{body}");
    assert_eq!(body.split("\n\n").filter(|e| !e.is_empty()).count(), events.len());

    let window = ok(&d, "POST", "/api/analyze/window", json!({ "graph": graph, "page_size": 16000 })).await;
    assert!(window["segments"].as_array().is_some_and(|s| !s.is_empty()), "{window}");
    let typed = ok(&d, "POST", "/api/analyze/typed", json!({ "doc_id": "doc-1", "graph": graph })).await;
    assert!(typed.is_object(), "{typed}");

    let formats = ok(&d, "GET", "/api/analyze/formats", Value::Null).await;
    let ids: Vec<&str> =
        formats["formats"].as_array().unwrap().iter().map(|f| f["format_id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"markdown"), "{formats}");
    let md = ok(&d, "POST", "/api/analyze/format", json!({ "graph": graph, "format_id": "markdown" })).await;
    let content = String::from_utf8(unbase64(md["content"].as_str().unwrap())).unwrap();
    assert!(content.contains("World Health Organization"), "{content}");

    let projected = ok(&d, "POST", "/api/chunks/project", json!({ "graph": graph })).await;
    assert!(projected["document_term_stack"]["segments"].as_array().is_some_and(|s| !s.is_empty()), "{projected}");
    d.stop().await;
    s.stop().await;
}

#[tokio::test]
async fn analysis_refusals_carry_the_turbo_trailers() {
    let (s, d) = start(false).await;
    let bad_step = json!({ "doc_id": "x", "text": "Hello.", "steps": ["turbo:nonexistent"] });
    let e = stream_refused(&d, "/api/analyze/progressive", bad_step, "InvalidArgument").await;
    assert!(e["error"]["message"].as_str().unwrap().contains("unknown pipeline step"), "{e}");
    refused(&d, "POST", "/api/analyze/window", json!({ "graph": base64(b"not a graph") }), "InvalidArgument").await;
    let e = refused(&d, "POST", "/api/analyze/format", json!({ "graph": "", "format_id": "nope" }), "InvalidArgument")
        .await;
    assert!(e["error"]["turbo_field"].is_string(), "{e}");
    // A graph that is not base64 is refused by the demo itself.
    let (status, e) = call(&d, "POST", "/api/analyze/typed", json!({ "graph": "!!" })).await;
    assert_eq!((status, &e["error"]["grpc"]), (400, &Value::Null), "{e}");
    d.stop().await;
    s.stop().await;
}
