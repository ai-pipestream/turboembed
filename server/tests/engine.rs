//! Engine behavior: bucket chunking, the session pool, the generation
//! bound, a client that hangs up mid-stream, and the model loads that must
//! fail rather than leave the server half ready.

mod common;

use std::time::Duration;

use common::{bundle, engine_of, spec};
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use turbo::provider::{EmbedOptions, GenerateDesc};
use turbo_inferstream::config::Config;
use turbo_inferstream::engine::{self, Engine};

const TEXTS: [&str; 3] = ["alpha one", "beta two", "gamma three"];

fn narrow_embedder() -> Vec<String> {
    vec![format!("name=embed,bundle={},provider=mock,ordinal=1,buckets=1x16", bundle("embedding"))]
}

#[tokio::test]
async fn a_request_wider_than_every_bucket_is_served_in_chunks_in_order() {
    let flags = narrow_embedder();
    let e = engine_of(&flags.iter().map(String::as_str).collect::<Vec<_>>());
    let served = e.model("embed").expect("the embed model");
    assert_eq!(served.max_batch(), 1, "the test needs a one-row bucket, got {}", served.max_batch());

    let texts: Vec<String> = TEXTS.iter().map(|s| s.to_string()).collect();
    let all = engine::embed(served.clone(), texts.clone(), EmbedOptions::default())
        .await
        .unwrap_or_else(|err| panic!("three texts through a one-row bucket: {err}"));
    assert_eq!(all.vectors.len(), 3, "a chunked request lost rows");

    for (i, text) in TEXTS.iter().enumerate() {
        let one = engine::embed(served.clone(), vec![text.to_string()], EmbedOptions::default())
            .await
            .unwrap_or_else(|err| panic!("`{text}` alone: {err}"));
        assert_eq!(all.vectors[i], one.vectors[0], "row {i} of the chunked request is not the embedding of `{text}`");
    }
}

#[tokio::test]
async fn concurrent_requests_queue_for_the_one_session_instead_of_colliding() {
    let flags = narrow_embedder();
    let e = engine_of(&flags.iter().map(String::as_str).collect::<Vec<_>>());
    let served = e.model("embed").expect("the embed model");
    // One bucket, one session: a second caller must wait for the pool rather
    // than reach the session and get TURBO_E_BUSY.
    let wanted: Vec<Vec<f32>> = {
        let mut v = Vec::new();
        for text in TEXTS {
            let r = engine::embed(served.clone(), vec![text.to_string()], EmbedOptions::default())
                .await
                .unwrap_or_else(|err| panic!("`{text}`: {err}"));
            v.push(r.vectors[0].clone());
        }
        v
    };

    let mut tasks = Vec::new();
    for round in 0..8 {
        let text = TEXTS[round % TEXTS.len()].to_string();
        let served = served.clone();
        tasks.push(tokio::spawn(
            async move { (round, engine::embed(served, vec![text], EmbedOptions::default()).await) },
        ));
    }
    for task in tasks {
        let (round, result) = task.await.expect("the embed task");
        // A session is single-owner: had the pool let two callers reach the
        // same one, the core would have answered TURBO_E_BUSY.
        let e = result.unwrap_or_else(|err| panic!("concurrent request {round} was not queued for the session: {err}"));
        assert_eq!(
            e.vectors[0],
            wanted[round % TEXTS.len()],
            "concurrent request {round} returned another request's row"
        );
    }

    let after = engine::embed(served, vec!["alpha one".to_string()], EmbedOptions::default())
        .await
        .expect("the pool is usable after the concurrent round");
    assert_eq!(after.vectors[0], wanted[0], "the session was not returned to the pool in a usable state");
}

#[tokio::test]
async fn a_generation_past_the_configured_bound_waits_for_a_free_slot() {
    let flag = format!("name=chat,bundle={},provider=mock,ordinal=1,generations=1", bundle("generative"));
    let e = engine_of(&[&flag]);
    let served = e.model("chat").expect("the chat model");
    let messages = vec![("user".to_string(), "hello".to_string())];
    // More new tokens than the piece channel holds, so the first generation
    // is still running (and still holding its slot) while nothing reads it.
    let desc = GenerateDesc { max_new_tokens: 4096, ..Default::default() };

    let first = engine::generate(served.clone(), messages.clone(), desc.clone()).await.expect("the first generation");

    let waiting = tokio::spawn({
        let served = served.clone();
        let messages = messages.clone();
        async move { engine::generate(served, messages, GenerateDesc { max_new_tokens: 1, ..Default::default() }).await }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(!waiting.is_finished(), "a second generation started although the model is configured for one at a time");

    drop(first);
    let second = tokio::time::timeout(Duration::from_secs(5), waiting)
        .await
        .expect("the waiting generation never got a slot after the first was dropped")
        .expect("the waiting task");
    let mut rx = second.expect("the second generation");
    let mut pieces = 0;
    while let Some(piece) = rx.recv().await {
        piece.expect("a generated piece");
        pieces += 1;
    }
    assert!(pieces > 0, "the second generation produced nothing after it got the slot");
}

#[tokio::test]
async fn a_client_that_hangs_up_mid_stream_leaves_the_server_serving() {
    let flag = format!("name=chat,bundle={},provider=mock,ordinal=1,generations=1", bundle("generative"));
    let e = engine_of(&[&flag]);
    let addr = common::serve_http(e).await;

    let body = json!({"model": "chat", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4096, "stream": true});
    let mut sock = tokio::net::TcpStream::connect(addr).await.expect("connect for the streamed request");
    sock.write_all(&common::http_post_bytes(addr, "/v1/chat/completions", &body))
        .await
        .expect("send the streamed request");
    let mut seen = Vec::new();
    while !String::from_utf8_lossy(&seen).contains("chat.completion.chunk") {
        let mut buf = [0u8; 1024];
        let n = tokio::time::timeout(Duration::from_secs(5), sock.read(&mut buf))
            .await
            .expect("the stream started within five seconds")
            .expect("read the stream");
        assert_ne!(n, 0, "the server closed the stream before sending a chunk: {}", String::from_utf8_lossy(&seen));
        seen.extend_from_slice(&buf[..n]);
    }
    // Hang up in the middle of the generation.
    drop(sock);

    // The model allows one generation at a time, so this only completes once
    // the abandoned one has let its slot go.
    let after = tokio::time::timeout(Duration::from_secs(10), async {
        let mut sock = tokio::net::TcpStream::connect(addr).await.expect("connect for the second request");
        let body = json!({"model": "chat", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 2});
        sock.write_all(&common::http_post_bytes(addr, "/v1/chat/completions", &body))
            .await
            .expect("send the second request");
        let mut text = String::new();
        sock.read_to_string(&mut text).await.expect("read the second response");
        text
    })
    .await
    .expect("the server did not answer within ten seconds of the streaming client hanging up");
    assert!(after.starts_with("HTTP/1.1 200 "), "the request after the hang-up did not succeed: {after}");
    assert!(after.contains("chat.completion"), "the request after the hang-up returned no completion: {after}");
}

#[tokio::test]
async fn two_models_of_the_same_name_fail_the_load() {
    let flag = format!("name=twice,bundle={},provider=mock,ordinal=1", bundle("embedding"));
    let config = Config { provider_libs: Vec::new(), models: vec![spec(&flag), spec(&flag)] };
    let err = match Engine::load(&config) {
        Ok(e) => panic!("two models named `twice` loaded; served: {:?}", e.names()),
        Err(e) => e,
    };
    assert!(err.message.contains("twice"), "the error names the clashing name: {err}");
    assert!(err.message.contains("name="), "the error says how to fix it: {err}");
}

#[tokio::test]
async fn a_bucket_past_the_models_limits_fails_the_load() {
    for (buckets, what) in
        [("16x16", "batch 16 over the model's max_batch 8"), ("1x64", "sequence 64 over the model's max_seq 16")]
    {
        let flag = format!("name=embed,bundle={},provider=mock,ordinal=1,buckets={buckets}", bundle("embedding"));
        let config = Config { provider_libs: Vec::new(), models: vec![spec(&flag)] };
        let err = match Engine::load(&config) {
            Ok(e) => panic!("bucket {buckets} loaded although it is {what}; served: {:?}", e.names()),
            Err(e) => e,
        };
        assert!(err.message.contains(buckets), "the error does not name the bucket {buckets}: {err}");
        assert!(
            err.message.contains("exceeds the model's limits"),
            "the error does not say why {buckets} is too big: {err}"
        );
    }
}

#[tokio::test]
async fn a_model_the_runtime_cannot_serve_fails_the_load() {
    let cases = [
        (format!("name=embed,bundle={},provider=nope,ordinal=1", bundle("embedding")), "a provider that is not loaded"),
        (
            format!("name=embed,bundle={},provider=mock,ordinal=7", bundle("embedding")),
            "a device the mock does not have",
        ),
        (
            format!("name=embed,bundle={}-missing,provider=mock,ordinal=1", bundle("embedding")),
            "a bundle directory that is not there",
        ),
    ];
    for (flag, what) in cases {
        let config = Config { provider_libs: Vec::new(), models: vec![spec(&flag)] };
        match Engine::load(&config) {
            Ok(e) => panic!("{what} loaded anyway; served: {:?}", e.names()),
            Err(e) => assert!(e.code != 0, "{what} failed as a server condition rather than a Turbo status: {e}"),
        }
    }
}

#[tokio::test]
async fn an_empty_configuration_fails_the_load() {
    let err = match Engine::load(&Config::default()) {
        Ok(_) => panic!("a server with no models loaded"),
        Err(e) => e,
    };
    assert!(err.message.contains("no models configured"), "the error says what is missing: {err}");
}

/// A mock embedding bundle that also carries the MiniLM `tokenizer.json`,
/// so the server can count tokens: the capacity path of `Served::lease`
/// is reachable only with a tokenizer the core can load.
fn tokenized_mock_bundle() -> std::path::PathBuf {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/bundles/minilm-tokenizer");
    let dir = std::env::temp_dir().join(format!(
        "inferstream-tokenized-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).expect("temp bundle dir");
    turbo::mock::write_mock_bundle(&dir, turbo::mock::MockBundleKind::Embedding).expect("mock bundle");
    std::fs::copy(root.join("tokenizer.json"), dir.join("tokenizer.json")).expect("copy tokenizer.json");
    let source: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("bundle.json")).unwrap()).unwrap();
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("bundle.json")).unwrap()).unwrap();
    manifest["tokenizer"] = source["tokenizer"].clone();
    std::fs::write(dir.join("bundle.json"), serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    dir
}

#[tokio::test]
async fn a_text_longer_than_the_longest_bucket_is_a_capacity_error_unless_truncation_is_asked_for() {
    let dir = tokenized_mock_bundle();
    let flag = format!("name=tok,bundle={},provider=mock,ordinal=1,buckets=1x16", dir.display());
    let engine = engine_of(&[&flag]);
    let served = engine.model("tok").unwrap();
    assert!(served.tokenizer.is_some(), "the fixture must load a core tokenizer");
    let long =
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen".to_string();
    let e = engine::embed(served.clone(), vec![long.clone()], EmbedOptions::default()).await.unwrap_err();
    assert_eq!(e.code, turbo::abi::TURBO_E_CAPACITY, "{e}");
    assert_eq!(e.field, EmbedOptions::FIELD_TRUNCATE, "the error names the truncate field: {e}");
    assert!(e.message.contains("holds 16"), "the error names the limit: {e}");
    let ok = engine::embed(
        served.clone(),
        vec![long],
        EmbedOptions { truncate: turbo::types::Truncate::Right, ..Default::default() },
    )
    .await
    .expect("an explicit right truncation cuts the text");
    assert_eq!(ok.vectors.len(), 1);
    assert!(ok.tokens.unwrap() > 16, "the count is of the untruncated text");
    let short = engine::embed(served, vec!["one two".into()], EmbedOptions::default()).await.unwrap();
    assert_eq!(short.tokens, Some(4), "[CLS] one two [SEP]");
    std::fs::remove_dir_all(&dir).ok();
}
