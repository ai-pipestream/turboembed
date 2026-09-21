//! Edge-case integration tests for the `inferstream.v1.InferstreamService`
//! extension over the mock backend: empty strings, byte-level unicode
//! offsets, truncation and padding semantics, detokenize failure modes,
//! batch-of-four embeds, EmbedStream chunk/final-flag contracts, the TEI
//! `max_client_batch_size` rerank boundary, and ListModels on multi-model /
//! empty registries. Complements `extension.rs`, which covers the happy
//! paths; every test here boots the in-process server on 127.0.0.1:0.

use tonic::transport::Channel;

use inferstream_backend_mock::{MockBackend, MOCK_BOS_ID, MOCK_EOS_ID, MOCK_PAD_ID};
use inferstream_protocol::extension::inferstream_service_client::InferstreamServiceClient;
use inferstream_protocol::extension::{
    DetokenizeRequest, EmbedRequest, ListModelsRequest, RerankRequest, TokenIds, TokenizeRequest,
};
use inferstream_server::config::Config;

const BASE_CONFIG: &str = r#"
listen = "127.0.0.1:0"

[[models]]
name = "mock-embed"
backend = "mock"
"#;

const MULTI_CONFIG: &str = r#"
listen = "127.0.0.1:0"

[[models]]
name = "zeta"
backend = "mock"

[[models]]
name = "alpha"
backend = "mock"
"#;

const EMPTY_CONFIG: &str = r#"listen = "127.0.0.1:0""#;

async fn start_server(config_text: &str) -> (Channel, ServerGuard) {
    let config = Config::from_toml(config_text).expect("test config parses");
    let registry = inferstream_server::build_registry(&config, &inferstream_server::mock_factory())
        .expect("registry builds");
    let (bound_tx, bound_rx) = tokio::sync::oneshot::channel();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let handle = tokio::spawn(inferstream_server::serve(
        config,
        registry,
        bound_tx,
        async move {
            let _ = shutdown_rx.await;
        },
    ));
    let addr = bound_rx.await.expect("server reports bound address");
    let channel = Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect()
        .await
        .expect("client connects");
    (
        channel,
        ServerGuard {
            shutdown: Some(shutdown_tx),
            handle,
        },
    )
}

struct ServerGuard {
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    handle: tokio::task::JoinHandle<Result<(), inferstream_server::ServerError>>,
}

impl ServerGuard {
    async fn stop(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        let _ = self.handle.await;
    }
}

/// Empty text still frames BOS/EOS, and decodes back to the empty string.
#[tokio::test]
async fn tokenize_empty_text_frames_only_specials() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let response = client
        .tokenize(TokenizeRequest {
            model_name: "mock-embed".into(),
            texts: vec![String::new()],
            with_offsets: true,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.encodings.len(), 1);
    let encoding = &response.encodings[0];
    assert_eq!(encoding.input_ids, vec![MOCK_BOS_ID, MOCK_EOS_ID]);
    assert_eq!(encoding.tokens, vec!["<s>", "</s>"]);
    assert_eq!(encoding.attention_mask, vec![1, 1]);
    // Special tokens carry (0, 0) offsets per the proto contract.
    assert_eq!(encoding.offsets.len(), 2);
    assert_eq!((encoding.offsets[0].start, encoding.offsets[0].end), (0, 0));

    let decoded = client
        .detokenize(DetokenizeRequest {
            model_name: "mock-embed".into(),
            sequences: vec![TokenIds {
                ids: encoding.input_ids.clone(),
            }],
            skip_special_tokens: true,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(decoded.texts, vec![""]);

    guard.stop().await;
}

/// With specials disabled, empty text produces a genuinely empty encoding.
#[tokio::test]
async fn tokenize_empty_text_without_specials_is_empty_encoding() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let response = client
        .tokenize(TokenizeRequest {
            model_name: "mock-embed".into(),
            texts: vec![String::new()],
            no_special_tokens: true,
            with_offsets: true,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    let encoding = &response.encodings[0];
    assert!(encoding.input_ids.is_empty());
    assert!(encoding.tokens.is_empty());
    assert!(encoding.attention_mask.is_empty());
    assert!(encoding.offsets.is_empty());

    let decoded = client
        .detokenize(DetokenizeRequest {
            model_name: "mock-embed".into(),
            sequences: vec![TokenIds { ids: vec![] }],
            skip_special_tokens: false,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(decoded.texts, vec![""]);

    guard.stop().await;
}

/// The mock tokenizer is byte-level: multibyte characters become one token
/// per byte, and content offsets are byte ranges into the original text
/// (not char boundaries), so reassembly must slice the byte buffer.
#[tokio::test]
async fn tokenize_unicode_offsets_address_original_bytes() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);
    let text = "é✓"; // 5 bytes: C3 A9 E2 9C 93
    let content_ids: Vec<u32> = text.as_bytes().iter().map(|&b| u32::from(b) + 3).collect();

    let response = client
        .tokenize(TokenizeRequest {
            model_name: "mock-embed".into(),
            texts: vec![text.into()],
            with_offsets: true,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    let encoding = &response.encodings[0];
    let expected_len = content_ids.len() + 2;
    assert_eq!(encoding.input_ids.len(), expected_len);
    assert_eq!(encoding.input_ids.first(), Some(&MOCK_BOS_ID));
    assert_eq!(encoding.input_ids.last(), Some(&MOCK_EOS_ID));
    assert_eq!(&encoding.input_ids[1..expected_len - 1], &content_ids[..]);
    assert_eq!(encoding.offsets.len(), expected_len);
    // Non-graphic bytes get the <0xXX> display form.
    assert_eq!(encoding.tokens[1], "<0xC3>");
    // Content offsets walk the source one byte at a time; specials are (0, 0).
    for (i, offset) in encoding.offsets[1..expected_len - 1].iter().enumerate() {
        assert_eq!(
            (offset.start, offset.end),
            (i as u32, i as u32 + 1),
            "content token {i} covers byte {i}"
        );
    }
    let reassembled: Vec<u8> = encoding.offsets[1..expected_len - 1]
        .iter()
        .flat_map(|o| {
            text.as_bytes()[o.start as usize..o.end as usize]
                .iter()
                .copied()
        })
        .collect();
    assert_eq!(reassembled, text.as_bytes());

    // Offsets are omitted entirely unless requested.
    let bare = client
        .tokenize(TokenizeRequest {
            model_name: "mock-embed".into(),
            texts: vec![text.into()],
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert!(bare.encodings[0].offsets.is_empty());

    // Byte-for-byte round-trip through Detokenize.
    let decoded = client
        .detokenize(DetokenizeRequest {
            model_name: "mock-embed".into(),
            sequences: vec![TokenIds {
                ids: encoding.input_ids.clone(),
            }],
            skip_special_tokens: true,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(decoded.texts, vec![text]);

    guard.stop().await;
}

/// truncate_to counts the special-token frame: limit 4 on a 6-byte text
/// yields BOS + 2 content + EOS. A limit below the frame still yields both
/// specials (current mock behavior: content truncates to zero, specials are
/// always added), which callers should not read as "no specials".
#[tokio::test]
async fn tokenize_truncate_to_counts_special_tokens() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let truncated = client
        .tokenize(TokenizeRequest {
            model_name: "mock-embed".into(),
            texts: vec!["abcdef".into()],
            truncate_to: 4,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    let encoding = &truncated.encodings[0];
    assert_eq!(encoding.input_ids.len(), 4, "limit 4 = BOS + 2 + EOS");
    assert_eq!(encoding.tokens, vec!["<s>", "a", "b", "</s>"]);
    assert_eq!(encoding.attention_mask, vec![1, 1, 1, 1]);

    // truncate_to = 0 means "tokenizer default": no truncation.
    let untruncated = client
        .tokenize(TokenizeRequest {
            model_name: "mock-embed".into(),
            texts: vec!["abcdef".into()],
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(untruncated.encodings[0].input_ids.len(), 8);

    // Limit smaller than the special frame: content drops out, specials stay.
    let tiny = client
        .tokenize(TokenizeRequest {
            model_name: "mock-embed".into(),
            texts: vec!["abcdef".into()],
            truncate_to: 1,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(
        tiny.encodings[0].input_ids,
        vec![MOCK_BOS_ID, MOCK_EOS_ID],
        "specials frame even a below-frame limit"
    );

    guard.stop().await;
}

/// pad_to_longest pads with the mock pad id, marks padding in the mask, and
/// gives pad tokens (0, 0) offsets; without the flag the batch stays ragged.
#[tokio::test]
async fn tokenize_pad_to_longest_marks_pad_rows() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);
    let texts = vec!["hi".to_string(), "longer text".to_string()];

    let padded = client
        .tokenize(TokenizeRequest {
            model_name: "mock-embed".into(),
            texts: texts.clone(),
            pad_to_longest: true,
            with_offsets: true,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    let (short, long) = (&padded.encodings[0], &padded.encodings[1]);
    assert_eq!(short.input_ids.len(), long.input_ids.len());
    assert_eq!(
        long.attention_mask.iter().filter(|&&m| m == 1).count() as u32,
        13
    );
    // "hi" = BOS + h + i + EOS = 4 real tokens; the rest is padding.
    let real: u32 = short.attention_mask.iter().sum();
    assert_eq!(real, 4);
    assert!(short.attention_mask.ends_with(&[0]));
    let pads = short
        .input_ids
        .iter()
        .filter(|&&id| id == MOCK_PAD_ID)
        .count();
    assert_eq!(pads, short.input_ids.len() - 4);
    assert!(short.tokens.iter().any(|t| t == "<pad>"));
    // Pad tokens carry (0, 0) offsets, like specials.
    for (token, offset) in short.tokens.iter().zip(short.offsets.iter()) {
        if token == "<pad>" {
            assert_eq!((offset.start, offset.end), (0, 0));
        }
    }

    // Without the flag, encodings keep their natural (ragged) lengths.
    let ragged = client
        .tokenize(TokenizeRequest {
            model_name: "mock-embed".into(),
            texts,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_ne!(
        ragged.encodings[0].input_ids.len(),
        ragged.encodings[1].input_ids.len(),
        "no padding keeps per-text lengths"
    );
    assert!(ragged.encodings[0].attention_mask.iter().all(|&m| m == 1));

    guard.stop().await;
}

/// skip_special_tokens controls whether the BOS/EOS frame appears in the
/// decoded text; content always round-trips exactly.
#[tokio::test]
async fn detokenize_skip_special_tokens_controls_framing() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let tokenized = client
        .tokenize(TokenizeRequest {
            model_name: "mock-embed".into(),
            texts: vec!["edge case".into()],
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    let ids = tokenized.encodings[0].input_ids.clone();

    for (skip, expected) in [(true, "edge case"), (false, "<s>edge case</s>")] {
        let decoded = client
            .detokenize(DetokenizeRequest {
                model_name: "mock-embed".into(),
                sequences: vec![TokenIds { ids: ids.clone() }],
                skip_special_tokens: skip,
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(decoded.texts, vec![expected], "skip_special_tokens={skip}");
    }

    guard.stop().await;
}

/// Token ids beyond the mock range (byte + 3 tops out at 258) are an
/// InvalidArgument from the backend, not a silent best-effort decode.
#[tokio::test]
async fn detokenize_rejects_out_of_range_token_id() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let err = client
        .detokenize(DetokenizeRequest {
            model_name: "mock-embed".into(),
            sequences: vec![TokenIds { ids: vec![259] }],
            skip_special_tokens: false,
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::InvalidArgument);
    assert!(
        err.message().contains("out of the mock tokenizer's range"),
        "unexpected message: {}",
        err.message()
    );

    guard.stop().await;
}

/// Detokenize on an unconfigured model is NotFound, matching the other RPCs.
#[tokio::test]
async fn detokenize_unknown_model_is_not_found() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let err = client
        .detokenize(DetokenizeRequest {
            model_name: "no-such-model".into(),
            sequences: vec![TokenIds {
                ids: vec![MOCK_BOS_ID],
            }],
            skip_special_tokens: true,
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);

    guard.stop().await;
}

/// Embed accepts an empty string as a batch element (deterministic FNV of
/// the empty byte string) and handles a four-text batch: one row per text,
/// duplicate texts produce identical rows, distinct texts differ.
#[tokio::test]
async fn embed_handles_empty_string_and_batch_of_four() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let response = client
        .embed(EmbedRequest {
            model_name: "mock-embed".into(),
            texts: vec![String::new(), "alpha".into(), "beta".into(), String::new()],
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.model_name, "mock-embed");
    assert_eq!(response.embeddings.len(), 4);
    for (row, embedding) in response.embeddings.iter().enumerate() {
        assert_eq!(embedding.values.len(), 8, "row {row} dim");
        assert!(
            embedding.values.iter().all(|v| (-1.0..1.0).contains(v)),
            "row {row} values in [-1, 1)"
        );
    }
    assert_eq!(
        response.embeddings[0].values, response.embeddings[3].values,
        "same empty text embeds identically"
    );
    assert_ne!(response.embeddings[1].values, response.embeddings[2].values);
    assert_ne!(response.embeddings[0].values, response.embeddings[1].values);

    guard.stop().await;
}

/// EmbedStream yields exactly one typed chunk per input text — here one per
/// the mock's default stream_chunks (4) — with sequential indices, values
/// sized to the mock dim, and the final flag set only on the last chunk.
#[tokio::test]
async fn embed_stream_chunk_count_matches_text_batch() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);
    let mock = MockBackend::default();
    let n = mock.stream_chunks();
    let texts: Vec<String> = (0..n).map(|i| format!("stream text {i}")).collect();

    let mut stream = client
        .embed_stream(EmbedRequest {
            model_name: "mock-embed".into(),
            texts,
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.message().await.unwrap() {
        chunks.push(chunk);
    }

    assert_eq!(chunks.len(), n, "one chunk per input text");
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.index, i as u32, "sequential indices");
        assert_eq!(
            chunk.r#final,
            i + 1 == n,
            "final flag only on the last chunk"
        );
        let embedding = chunk
            .embedding
            .as_ref()
            .expect("typed chunk carries values");
        assert_eq!(embedding.values.len(), mock.embedding_dim());
        assert!(
            chunk.packed_row.is_empty(),
            "typed chunks carry no packed row"
        );
    }

    guard.stop().await;
}

/// A bad model fails EmbedStream before the first chunk with NotFound.
#[tokio::test]
async fn embed_stream_unknown_model_fails_before_first_chunk() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let err = client
        .embed_stream(EmbedRequest {
            model_name: "no-such-model".into(),
            texts: vec!["x".into()],
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);

    guard.stop().await;
}

/// The TEI max-client-batch-size contract: empty documents are rejected,
/// exactly 32 documents succeed (score ties keep input order), and 33 are
/// rejected with the max_client_batch_size message.
#[tokio::test]
async fn rerank_enforces_tei_max_batch_boundary() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let empty = client
        .rerank(RerankRequest {
            model_name: "mock-embed".into(),
            query: "q".into(),
            documents: vec![],
            top_n: 0,
            return_documents: false,
            raw_scores: false,
        })
        .await
        .unwrap_err();
    assert_eq!(empty.code(), tonic::Code::InvalidArgument);
    assert!(empty.message().contains("documents must not be empty"));

    let max = 32_usize;
    let docs: Vec<String> = (0..max).map(|i| format!("doc {i}")).collect();
    let at_limit = client
        .rerank(RerankRequest {
            model_name: "mock-embed".into(),
            query: "unmatched".into(),
            documents: docs,
            top_n: 0,
            return_documents: true,
            raw_scores: false,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(
        at_limit.results.len(),
        max,
        "exactly 32 documents are served"
    );
    // All scores are 0.0 (no query-word overlap); the stable sort keeps
    // input order, and every row echoes its document.
    let indexes: Vec<u32> = at_limit.results.iter().map(|r| r.index).collect();
    assert_eq!(indexes, (0..max as u32).collect::<Vec<_>>());
    assert!(at_limit.results.iter().all(|r| r.score == 0.0));
    assert!(at_limit
        .results
        .iter()
        .all(|r| r.document == format!("doc {}", r.index)));

    let too_many: Vec<String> = (0..=max).map(|i| format!("doc {i}")).collect();
    let over = client
        .rerank(RerankRequest {
            model_name: "mock-embed".into(),
            query: "q".into(),
            documents: too_many,
            top_n: 0,
            return_documents: false,
            raw_scores: false,
        })
        .await
        .unwrap_err();
    assert_eq!(over.code(), tonic::Code::InvalidArgument);
    assert!(
        over.message().contains("max_client_batch_size 32"),
        "unexpected message: {}",
        over.message()
    );

    guard.stop().await;
}

/// Rerank on an unconfigured model is NotFound once shape checks pass.
#[tokio::test]
async fn rerank_unknown_model_is_not_found() {
    let (channel, guard) = start_server(BASE_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let err = client
        .rerank(RerankRequest {
            model_name: "no-such-model".into(),
            query: "q".into(),
            documents: vec!["one document".into()],
            top_n: 0,
            return_documents: false,
            raw_scores: false,
        })
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::NotFound);

    guard.stop().await;
}

/// ListModels sorts names across a multi-model registry and reports the
/// full per-model contents for each entry.
#[tokio::test]
async fn list_models_sorts_multi_model_registry() {
    let (channel, guard) = start_server(MULTI_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let response = client
        .list_models(ListModelsRequest {})
        .await
        .unwrap()
        .into_inner();
    let names: Vec<&str> = response.models.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        names,
        ["alpha", "zeta"],
        "names sorted, config order ignored"
    );
    for model in &response.models {
        assert_eq!(model.backend, "mock");
        assert!(model.ready);
        assert_eq!(model.platform, "mock");
        assert_eq!(model.versions, ["1"]);
        assert_eq!(model.embedding_dim, 8);
        assert!(model.has_tokenizer);
    }

    guard.stop().await;
}

/// A config with no models serves ListModels as an empty listing.
#[tokio::test]
async fn list_models_on_empty_registry_returns_no_models() {
    let (channel, guard) = start_server(EMPTY_CONFIG).await;
    let mut client = InferstreamServiceClient::new(channel);

    let response = client
        .list_models(ListModelsRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(response.models.is_empty());

    guard.stop().await;
}
