//! The TurboVocabularyService routes: a dictionary imported, a vocabulary
//! learned, both listed and downloaded, and the admin refusal.

use super::*;

const DOCS: [&str; 3] = [
    "The New York office opened. The office in New York is large.",
    "A new office opened in Boston, and the Boston office is small.",
    "New York and Boston each have an office.",
];

fn import(words: &[u8]) -> Value {
    json!({
        "header": {
            "format_id": "plain-word-list", "name": "words", "byte_size": words.len(),
            "sha256": turbo_maxent::sha256(words).to_hex()
        },
        "data": base64(words)
    })
}

/// A download's descriptor and its data frames rejoined.
async fn download(d: &Demo, id: &str) -> (Value, Vec<u8>) {
    let frames = lines(d, "/api/vocabulary/download", json!({ "artifact_id": id })).await;
    let data = frames[1..].iter().flat_map(|f| unbase64(f["frame"]["data"].as_str().unwrap())).collect();
    (frames[0]["frame"].clone(), data)
}

#[tokio::test]
async fn dictionaries_and_vocabularies_round_trip_over_http() {
    let store = scratch("vocabulary");
    let (s, d) = start_with(Serve { admin: true, vocabulary: Some(store.clone()), ..Serve::default() }).await;
    let formats = ok(&d, "GET", "/api/vocabulary/formats", Value::Null).await;
    assert!(formats["formats"].as_array().unwrap().iter().any(|f| f["format_id"] == "plain-word-list"), "{formats}");
    assert_eq!(formats["limits"]["store"], true, "{formats}");

    // More than one 64 KiB frame, so the upload is split.
    let words: Vec<u8> = (0..20_000).flat_map(|i| format!("word{i}\n").into_bytes()).collect();
    let dict = ok(&d, "POST", "/api/vocabulary/import", import(&words)).await;
    assert_eq!(dict["entry_count"], 20_000, "{dict}");
    let id = dict["artifact_id"].as_str().unwrap();
    let listed = ok(&d, "GET", "/api/vocabulary/dictionaries", Value::Null).await;
    assert_eq!(listed["dictionaries"][0]["artifact_id"], id, "{listed}");
    let (head, data) = download(&d, id).await;
    assert_eq!((&head["dictionary"], data == words), (&dict, true));

    let docs: Vec<Value> = DOCS.iter().map(|t| json!({ "text": t })).collect();
    let header = json!({ "min_term_frequency": 2, "lowercase": true, "include_ngrams": true,
        "min_ngram_size": 2, "max_ngram_size": 3, "min_pmi": 1.0, "language": "en" });
    let vocab = ok(&d, "POST", "/api/vocabulary/learn", json!({ "header": header, "documents": docs })).await;
    assert_eq!(vocab["documents"], 3, "{vocab}");
    let id = vocab["artifact_id"].as_str().unwrap();
    let listed = ok(&d, "GET", "/api/vocabulary/vocabularies", Value::Null).await;
    assert_eq!(listed["vocabularies"][0], vocab, "{listed}");
    let (head, data) = download(&d, id).await;
    assert_eq!(head["vocabulary"], vocab);
    assert_eq!(turbo_maxent::sha256(&data).to_hex(), vocab["sha256"].as_str().unwrap());

    ok(&d, "POST", "/api/vocabulary/delete", json!({ "artifact_id": id })).await;
    stream_refused(&d, "/api/vocabulary/download", json!({ "artifact_id": id }), "NotFound").await;
    // A header whose digest does not match the upload names the field.
    let mut bad = import(b"cat\n");
    bad["header"]["sha256"] = json!("0".repeat(64));
    refused(&d, "POST", "/api/vocabulary/import", bad, "InvalidArgument").await;
    d.stop().await;
    s.stop().await;
    let _ = std::fs::remove_dir_all(&store);
}

#[tokio::test]
async fn vocabulary_writes_without_admin_are_refused() {
    let store = scratch("vocabulary-no-admin");
    let (s, d) = start_with(Serve { vocabulary: Some(store.clone()), ..Serve::default() }).await;
    refused(&d, "POST", "/api/vocabulary/import", import(b"cat\n"), "Unimplemented").await;
    refused(&d, "POST", "/api/vocabulary/learn", json!({ "documents": [{ "text": "a cat" }] }), "Unimplemented").await;
    d.stop().await;
    s.stop().await;
    let _ = std::fs::remove_dir_all(&store);
}
