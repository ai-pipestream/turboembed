//! The TurboSearchService routes: an index created, filled, queried,
//! aliased, sealed, persisted, reindexed and deleted, and a collection
//! watched while documents arrive.

use std::time::Duration;

use super::*;

fn space() -> Value {
    json!({
        "model": { "id": "turbo/test-axis-16", "revision": "v1", "sha256": "0123456789abcdef".repeat(4) },
        "dim": 16
    })
}

fn vec16(axis: usize) -> Vec<f32> {
    let mut v = vec![0.1f32; 16];
    v[axis % 16] = 1.0;
    v
}

fn spec(name: &str) -> Value {
    json!({ "index_name": name, "space": space(), "collection_name": "corpus", "chunk_budget_tokens": 32 })
}

fn documents(index: &str) -> Value {
    let doc = |id: &str, text: &str, axis: usize| {
        json!({
            "index_name": index, "doc_id": id, "text": text,
            "chunks": [{
                "chunk_id": format!("{id}#0"), "level_kind": "sentence",
                "span_start": 0, "span_end": text.len(), "dense_vector": vec16(axis)
            }]
        })
    };
    json!({ "documents": [
        doc("doc-1", "Neural retrieval pipeline in Paris.", 0),
        doc("doc-2", "Sparse vectors compress in Berlin.", 3),
    ] })
}

/// Waits for the first NDJSON line of an open stream, then drops it.
async fn first_line(mut c: TcpStream) -> Value {
    let mut seen = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        let n = tokio::time::timeout(Duration::from_secs(10), c.read(&mut buf)).await.unwrap().unwrap();
        assert!(n > 0, "stream closed: {}", String::from_utf8_lossy(&seen));
        seen.extend_from_slice(&buf[..n]);
        let text = String::from_utf8_lossy(&seen);
        let Some((head, body)) = text.split_once("\r\n\r\n") else { continue };
        assert!(head.starts_with("HTTP/1.1 200"), "{head}");
        if let Some(line) = body.lines().find(|l| l.starts_with('{')) {
            return serde_json::from_str(line).unwrap();
        }
    }
}

#[tokio::test]
async fn an_index_is_filled_queried_and_managed_over_http() {
    let snapshots = scratch("snapshots");
    let (s, d) = start(true).await;
    s.set_search_snapshot_dir(&snapshots);

    let providers = ok(&d, "GET", "/api/search/providers", Value::Null).await;
    assert!(providers["providers"].as_array().unwrap().iter().any(|p| p["id"] == "turbo-chunker:bm25-posting"));
    let col = ok(&d, "POST", "/api/search/collection", json!({ "collection_name": "corpus" })).await;
    assert_eq!(col["collection"]["collection_name"], "corpus", "{col}");
    let cols = ok(&d, "GET", "/api/search/collections", Value::Null).await;
    assert_eq!(cols["collections"][0]["collection_name"], "corpus", "{cols}");
    let got = ok(&d, "POST", "/api/search/collection/get", json!({ "collection_name": "corpus" })).await;
    assert_eq!(got["collection"], col["collection"]);

    // A watch opened before the documents arrive sees their drift report.
    let mut watch = TcpStream::connect(d.local_addr()).await.unwrap();
    let body = json!({ "collection_name": "corpus" }).to_string();
    let req = format!(
        "POST /api/search/watch HTTP/1.1\r\nHost: demo\r\nAccept: application/x-ndjson\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    watch.write_all(req.as_bytes()).await.unwrap();
    let watched = tokio::spawn(first_line(watch));

    ok(&d, "POST", "/api/search/create", json!({ "spec": spec("idx-a") })).await;
    ok(&d, "POST", "/api/search/alias", json!({ "alias": "active", "index_name": "idx-a" })).await;
    let aliases = ok(&d, "GET", "/api/search/aliases", Value::Null).await;
    assert_eq!(aliases["aliases"][0], json!({ "alias": "active", "index_name": "idx-a" }));

    let indexed = ok(&d, "POST", "/api/search/index", documents("active")).await;
    assert_eq!((indexed["index_name"].as_str(), indexed["indexed_chunks"].as_u64()), (Some("idx-a"), Some(2)));
    let event = watched.await.unwrap();
    assert_eq!(event["collection_name"], "corpus", "{event}");

    let hits = ok(
        &d,
        "POST",
        "/api/search/query",
        json!({ "index_name": "active", "query_vector": vec16(0), "query_space": space(), "top_k": 1 }),
    )
    .await;
    assert_eq!(hits["hits"][0]["chunk_id"], "doc-1#0", "{hits}");
    let listed = ok(&d, "GET", "/api/search/indexes", Value::Null).await;
    assert_eq!(listed["indexes"][0]["document_count"], 2, "{listed}");

    let persisted = ok(&d, "POST", "/api/search/persist", json!({ "index_name": "idx-a", "path": "a.snap" })).await;
    assert!(persisted["bytes_written"].as_u64().unwrap() > 0, "{persisted}");
    let sealed = ok(&d, "POST", "/api/search/seal", json!({ "index_name": "idx-a" })).await;
    assert_eq!(sealed["summary"]["sealed"], true, "{sealed}");
    refused(&d, "POST", "/api/search/index", documents("idx-a"), "FailedPrecondition").await;

    let body = json!({ "source_index": "idx-a", "target_spec": spec("idx-b"), "swap_alias": "active" });
    let reindexed = ok(&d, "POST", "/api/search/reindex", body).await;
    assert_eq!(reindexed["swapped_alias"], "active", "{reindexed}");
    ok(&d, "DELETE", "/api/search/alias", json!({ "alias": "active" })).await;
    ok(&d, "POST", "/api/search/delete", json!({ "index_name": "idx-a" })).await;
    let listed = ok(&d, "GET", "/api/search/indexes", Value::Null).await;
    assert_eq!(listed["indexes"].as_array().unwrap().len(), 1, "{listed}");

    ok(&d, "POST", "/api/search/collection/delete", json!({ "collection_name": "corpus" })).await;
    let cols = ok(&d, "GET", "/api/search/collections", Value::Null).await;
    assert!(cols["collections"].as_array().is_none_or(|c| c.is_empty()), "{cols}");
    refused(&d, "POST", "/api/search/collection/get", json!({ "collection_name": "corpus" }), "NotFound").await;
    let e = refused(&d, "POST", "/api/search/query", json!({ "index_name": "missing" }), "NotFound").await;
    assert!(e["error"]["message"].as_str().unwrap().contains("missing"), "{e}");
    d.stop().await;
    s.stop().await;
    let _ = std::fs::remove_dir_all(&snapshots);
}
