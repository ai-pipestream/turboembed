//! Over-limit and malformed requests against the analysis result layer,
//! and LoadBundle concurrency, all over a real gRPC client with the
//! server in-process on the CPU. Nothing is mocked.

mod common;

use tonic::Code;
use turbo_analysis::View;
use turbo_analysis_layers::SparseTerms;
use turbo_kserve::proto::infer_parameter::ParameterChoice;
use turbo_kserve::proto::model_infer_request::InferInputTensor;
use turbo_kserve::proto::result::*;
use turbo_kserve::proto::{InferTensorContents, ModelInferRequest};
use turbo_kserve::results::Wire;
use turbo_kserve::results::tokens::TokenVectors;

use common::*;

/// Token rows [batch, seq] sent as ints, the direct way past truncation:
/// ids 0 (the pad row) under a full mask.
fn token_rows(batch: i64, seq: i64) -> ModelInferRequest {
    let n = (batch * seq) as usize;
    let int_contents = |v: Vec<i32>| Some(InferTensorContents { int_contents: v, ..Default::default() });
    ModelInferRequest {
        model_name: NAME.into(),
        model_version: String::new(),
        id: String::new(),
        parameters: Default::default(),
        inputs: vec![
            InferInputTensor {
                name: "ids".into(),
                datatype: "INT32".into(),
                shape: vec![batch, seq],
                parameters: Default::default(),
                contents: int_contents(vec![0; n]),
            },
            InferInputTensor {
                name: "mask".into(),
                datatype: "INT32".into(),
                shape: vec![batch, seq],
                parameters: Default::default(),
                contents: int_contents(vec![1; n]),
            },
        ],
        outputs: vec![],
        raw_input_contents: vec![],
    }
}

#[tokio::test]
async fn an_over_limit_request_names_each_listed_limit() {
    let (s, mut c) = serve().await;
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let b = the_one_bundle(&list);
    let limits = list.limits.unwrap();

    // The bundle's limits, exactly as listed, refused the way docs/kserve.md
    // maps TURBO_E_CAPACITY, and nothing is truncated.
    let mut k = kserve(&s).await;
    let over_seq = k.model_infer(token_rows(1, b.context_limit as i64 + 1)).await.expect_err("over context_limit");
    assert_eq!(refusal(&over_seq), (Code::OutOfRange, 771));
    assert!(over_seq.message().contains("max_seq"), "{}", over_seq.message());
    let over_batch = k.model_infer(token_rows(b.batch_limit as i64 + 1, 1)).await.expect_err("over batch_limit");
    assert_eq!(refusal(&over_batch), (Code::OutOfRange, 771));
    assert!(over_batch.message().contains("max_batch"), "{}", over_batch.message());
    // At the limit itself the request is answered.
    k.model_infer(token_rows(1, b.context_limit as i64)).await.unwrap();

    // The result-message limits, one refusal per listed limit.
    let sparse = SparseTermsResult {
        row: vec![0, limits.max_sparse_entries as u32 + 1],
        term: vec![1; limits.max_sparse_entries as usize + 1],
        weight: vec![1.0; limits.max_sparse_entries as usize + 1],
        ..Default::default()
    };
    let e = c.round_trip_sparse_terms(sparse).await.expect_err("over max_sparse_entries");
    assert_eq!(refusal(&e), (Code::OutOfRange, 771));
    assert!(e.message().contains("max_sparse_entries"), "{}", e.message());

    let vectors = TokenVectorsResult {
        dim: 1,
        data: vec![0u8; limits.max_token_vector_bytes as usize + 4].into(),
        ..Default::default()
    };
    let e = c.round_trip_token_vectors(vectors).await.expect_err("over max_token_vector_bytes");
    assert_eq!(refusal(&e), (Code::OutOfRange, 771));
    assert!(e.message().contains("max_token_vector_bytes"), "{}", e.message());

    let rerank = RerankScoresResult {
        item: vec![1; limits.max_rerank_items as usize + 1],
        score: vec![1.0; limits.max_rerank_items as usize + 1],
        ..Default::default()
    };
    let e = c.round_trip_rerank_scores(rerank).await.expect_err("over max_rerank_items");
    assert_eq!(refusal(&e), (Code::OutOfRange, 771));
    assert!(e.message().contains("max_rerank_items"), "{}", e.message());

    let extraction = ExtractionResult {
        null_probability: 1.0,
        span_start: vec![0; limits.max_answers as usize + 1],
        span_end: vec![1; limits.max_answers as usize + 1],
        ..Default::default()
    };
    let e = c.round_trip_extraction(extraction).await.expect_err("over max_answers");
    assert_eq!(refusal(&e), (Code::OutOfRange, 771));
    assert!(e.message().contains("max_answers"), "{}", e.message());
}

#[tokio::test]
async fn a_malformed_result_message_is_a_typed_refusal_not_a_panic() {
    let (s, mut c) = serve().await;
    // A short weight column.
    let mut msg = <SparseTerms<View> as Wire>::to_msg(&sparse_terms());
    msg.weight.pop();
    let e = c.round_trip_sparse_terms(msg).await.expect_err("garbage");
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256));
    assert!(e.message().contains("weight"), "{}", e.message());
    // A window that ends before it starts.
    let (v, w) = token_vectors();
    let mut msg = TokenVectors::new(&v, &w).to_msg();
    msg.window_start[0] = 30;
    let e = c.round_trip_token_vectors(msg).await.expect_err("garbage");
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256));
    // Nothing at all.
    let e = c.round_trip_extraction(ExtractionResult::default()).await.expect_err("garbage");
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256));
    // The listing's roles answer what the inference service honors: a
    // text request names the role the bundle lists.
    let mut k = kserve(&s).await;
    let mut req = texts_req(NAME, &[b"hello"]);
    req.parameters.insert(
        "prompt_role".into(),
        turbo_kserve::proto::InferParameter {
            parameter_choice: Some(ParameterChoice::StringParam("PROMPT_QUERY".into())),
        },
    );
    k.model_infer(req).await.unwrap();
}

#[tokio::test]
async fn concurrent_loads_fail_only_their_own_registration() {
    let (_s, mut c) = serve().await;
    let bundle = tiny().to_str().unwrap().to_string();
    // Rounds of concurrent loads: three valid names and one request for
    // a missing bundle. Every valid load succeeds even alongside the
    // failing one, and the failure names its own model, never another's.
    for round in 0..3 {
        let mut set = tokio::task::JoinSet::new();
        for k in 0..3 {
            let mut c = c.clone();
            let bundle = bundle.clone();
            let name = format!("valid-{round}-{k}");
            set.spawn(async move {
                c.load_bundle(load_req(&name, &bundle, 1, 0)).await.map_err(|s| s.message().to_string())
            });
        }
        let mut c2 = c.clone();
        set.spawn(async move {
            c2.load_bundle(load_req("missing", "/no/such/bundle", 1, 0)).await.map_err(|s| s.message().to_string())
        });
        let mut loaded = 0;
        while let Some(r) = set.join_next().await {
            match r.unwrap() {
                Ok(_) => loaded += 1,
                // The only failure is the missing bundle's own, naming it.
                Err(message) => assert!(message.contains("missing"), "a valid load failed: {message}"),
            }
        }
        assert_eq!(loaded, 3, "round {round}: the three valid loads survived the missing one");
        // Unload them, then reuse a freed name for a fresh load.
        for k in 0..3 {
            c.unload_bundle(UnloadBundleRequest { name: format!("valid-{round}-{k}") }).await.unwrap();
        }
        c.load_bundle(load_req("valid-0-0", &bundle, 1, 0)).await.unwrap();
        c.unload_bundle(UnloadBundleRequest { name: "valid-0-0".into() }).await.unwrap();
    }
    // The failed registration rolled back: only the startup bundle lists.
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(list.bundles.len(), 1);
}
