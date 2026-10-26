//! Body limits: the client-stream routes take bodies past axum's 2 MB
//! default, every other route keeps it and answers 413 over it.

use super::*;

/// Past the 2 MB default.
const BIG: usize = 3 << 20;

#[tokio::test]
async fn client_stream_routes_take_large_bodies_and_others_keep_the_default() {
    let store = scratch("limits");
    let (s, d) = start_with(Serve { admin: true, vocabulary: Some(store.clone()), ..Serve::default() }).await;

    let (status, e) = call(&d, "POST", "/api/analyze/window", json!({ "graph": "A".repeat(BIG) })).await;
    assert_eq!((status, &e["error"]["grpc"]), (413, &Value::Null), "{e}");

    // A dictionary upload over 2 MB as base64 is read and stored.
    let words: Vec<u8> = (0..300_000).flat_map(|i| format!("w{i:08}\n").into_bytes()).collect();
    assert!(base64(&words).len() > BIG);
    let body = json!({
        "header": { "format_id": "plain-word-list", "name": "big", "byte_size": words.len(),
            "sha256": turbo_maxent::sha256(&words).to_hex() },
        "data": base64(&words)
    });
    let dict = ok(&d, "POST", "/api/vocabulary/import", body).await;
    assert_eq!(dict["entry_count"], 300_000, "{dict}");

    // A learning text over 2 MB is learned from.
    let text = "word ".repeat(BIG / 5 + 1);
    let learned = ok(&d, "POST", "/api/vocabulary/learn", json!({ "documents": [{ "text": text }] })).await;
    assert_eq!(learned["tokens"], BIG / 5 + 1, "{learned}");
    // An index batch over 2 MB passes the demo and reaches the server,
    // which has no such index.
    let docs = json!({ "documents": [{ "index_name": "missing", "doc_id": "d", "text": text }] });
    refused(&d, "POST", "/api/search/index", docs, "NotFound").await;
    d.stop().await;
    s.stop().await;
    let _ = std::fs::remove_dir_all(&store);
}
