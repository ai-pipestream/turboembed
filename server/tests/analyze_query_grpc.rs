//! gRPC integration tests for `#2339`: `AnalysisService::AnalyzeQuery` and
//! `AnalysisService::RoundTripAnalyzeQuery` across `OFFSET_MODE_UTF8`,
//! `OFFSET_MODE_UTF16`, and `OFFSET_MODE_CODE_POINT`, verifying ordinal-exact
//! empty-normalized gaps, multi-token synonyms, structured filters + residual
//! splitting, and simultaneous query-prompted dense/Matryoshka + SPLADE
//! encodings.

mod common;

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::*;
use serde_json::{Value, json};
use turbo::TURBO_PROMPT_QUERY;
use turbo::bundle::sha256_hex;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_stream_request::Kind;
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{
    AnalyzeDocument, AnalyzeOptions, AnalyzeOutput, AnalyzeQueryRequest, AnalyzeStreamRequest,
    OffsetMode as ProtoOffsetMode, QueryDenseTargetProto, QueryFieldProfileProto, QueryFieldTokenizerProto,
    QuerySpladeTargetProto, QuerySynonymRuleProto, QueryTermFormProto,
};
use turbo_kserve::{Serve, Server};

struct TempBundle(PathBuf);

impl Drop for TempBundle {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn hybrid_query_bundle(tag: &str) -> TempBundle {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("turbo-kserve-2339-{}-{tag}-{id}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    let src_dir = tiny();
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(src_dir.join(f), dir.join(f)).unwrap();
    }
    let src = fs::read(src_dir.join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    let mut data = src[8 + n..].to_vec();
    let lin = |start: f32, end: f32, count: usize| {
        (0..count)
            .flat_map(|i| {
                let t = i as f32 / (count - 1) as f32;
                (start + (end - start) * t).to_le_bytes()
            })
            .collect::<Vec<u8>>()
    };
    for (name, shape, bytes) in [
        ("mlm.transform.weight", vec![32, 32], lin(-0.2, 0.3, 1024)),
        ("mlm.transform.bias", vec![32], lin(-0.1, 0.1, 32)),
        ("mlm.transform.LayerNorm.weight", vec![32], lin(0.8, 1.2, 32)),
        ("mlm.transform.LayerNorm.bias", vec![32], lin(-0.05, 0.05, 32)),
        ("mlm.decoder.bias", vec![30522], lin(-0.06, 0.06, 30522)),
    ] {
        let at = data.len();
        data.extend_from_slice(&bytes);
        header.insert(name.into(), json!({ "dtype": "F32", "shape": shape, "data_offsets": [at, data.len()] }));
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut weights = (h.len() as u64).to_le_bytes().to_vec();
    weights.extend(h);
    weights.extend(data);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights).unwrap();

    let mut m: Value = serde_json::from_slice(&fs::read(src_dir.join("manifest.json")).unwrap()).unwrap();
    m["embed"]["output_dims"] = json!([8, 16, 32]);
    m["embed"]["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
    let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    tn.insert("mlm_transform_weight".into(), json!("mlm.transform.weight"));
    tn.insert("mlm_transform_bias".into(), json!("mlm.transform.bias"));
    tn.insert("mlm_transform_ln_weight".into(), json!("mlm.transform.LayerNorm.weight"));
    tn.insert("mlm_transform_ln_bias".into(), json!("mlm.transform.LayerNorm.bias"));
    tn.insert("mlm_decoder_bias".into(), json!("mlm.decoder.bias"));
    let w = m["files"].as_array_mut().unwrap().iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(sha256_hex(&weights));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    TempBundle(dir)
}

#[tokio::test]
async fn analyze_query_end_to_end_offsets_synonyms_filters_and_query_prompt_parity() {
    let bundle = hybrid_query_bundle("grpc");
    let models = vec![ModelConfig {
        name: Some("hybrid".into()),
        bundle: bundle.0.to_str().unwrap().into(),
        device: Device::Index(cpu()),
        sessions: 2,
        precision: 0,
        max_batch: 4,
        max_seq: 64,
    }];
    let server = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, Serve::default()).await.unwrap();
    server.load().await.unwrap();
    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX);

    // Query includes café (2-byte UTF-8), soft-hyphen inside "new\u{00AD}york",
    // standalone zero-width space ("\u{200B}"), emoji ("\u{1F680}"), and structured filters.
    let query = "Caf\u{e9} \u{200B} new york city \u{1F680} startups over $5M in Berlin since 2024";

    for mode in [ProtoOffsetMode::Utf8, ProtoOffsetMode::Utf16, ProtoOffsetMode::CodePoint] {
        let req = AnalyzeQueryRequest {
            query: query.into(),
            offsets: mode as i32,
            default_language: "en".into(),
            reference_date_iso: "2026-07-14".into(),
            field_profiles: vec![
                QueryFieldProfileProto {
                    field: "body".into(),
                    tokenizer: QueryFieldTokenizerProto::QueryFieldTokenizerWord as i32,
                    term_form: QueryTermFormProto::QueryTermFormStem as i32,
                    language: String::new(),
                    keep_invisible: false,
                    include_stopwords_in_residual: false,
                    synonyms: Vec::new(),
                },
                QueryFieldProfileProto {
                    field: "raw_ws".into(),
                    tokenizer: QueryFieldTokenizerProto::QueryFieldTokenizerWhitespace as i32,
                    term_form: QueryTermFormProto::QueryTermFormFolded as i32,
                    language: String::new(),
                    keep_invisible: false,
                    include_stopwords_in_residual: false,
                    synonyms: Vec::new(),
                },
            ],
            global_synonyms: vec![QuerySynonymRuleProto {
                trigger: vec!["new".into(), "york".into(), "city".into()],
                expansion: vec!["nyc".into()],
                bidirectional: true,
            }],
            dense_targets: vec![QueryDenseTargetProto {
                target_id: "dense-primary".into(),
                model: "hybrid".into(),
                matryoshka_dims: vec![16, 8],
                use_residual_text: false,
                turboquant_bits: 4,
                turboquant_calibrate: false,
            }],
            splade_targets: vec![QuerySpladeTargetProto {
                target_id: "splade-primary".into(),
                model: "hybrid".into(),
                top_k: 16,
                floor: 0.01,
                impact_bits: 8,
                impact_clip_max: 5.0,
                emit_term_keys: true,
                use_residual_text: false,
            }],
        };

        let resp = client.analyze_query(req).await.unwrap().into_inner();
        assert_eq!(resp.fields.len(), 2);

        // Whitespace field counts both standalone "\u{200B}" (ord 1) and "\u{1F680}" (ord 5) as empty-normalized.
        let ws_field = &resp.fields[1];
        assert!(ws_field.empty_normalized_ordinals.contains(&1), "{:?}", ws_field.empty_normalized_ordinals);
        assert!(ws_field.empty_normalized_ordinals.contains(&5), "{:?}", ws_field.empty_normalized_ordinals);
        assert_eq!(ws_field.synonyms.len(), 1);
        assert_eq!(ws_field.synonyms[0].terms, vec!["nyc"]);

        // Structured filters: over $5M, in Berlin, since 2024 (plus New York City place).
        assert!(resp.filters.iter().any(|f| f.surface == "over $5M" && f.comparison == "gt"));
        assert!(resp.filters.iter().any(|f| f.surface == "since 2024" && f.comparison == "gte"));

        // Dense + Matryoshka + TurboQuant results.
        assert_eq!(resp.dense_results.len(), 1);
        let dense = &resp.dense_results[0];
        let full = dense.full.as_ref().unwrap();
        assert_eq!(full.dim, 32);
        assert_eq!(full.data.len(), 32 * 4);
        assert_eq!(full.sha256.len(), 32);
        assert!(!full.turboquant_codes.is_empty());
        assert_eq!(dense.matryoshka.len(), 2);
        assert_eq!((dense.matryoshka[0].dim, dense.matryoshka[1].dim), (8, 16));

        // SPLADE query sparse impacts.
        assert_eq!(resp.splade_results.len(), 1);
        let splade = &resp.splade_results[0];
        assert_eq!(splade.impact_bits, 8);
        assert_eq!(splade.impact.len(), splade.term_id.len());
        assert_eq!(splade.term_keys.len(), splade.term_id.len());

        // RoundTripAnalyzeQuery validates every offset, ordinal, surface, and vector invariant.
        let echoed = client.round_trip_analyze_query(resp.clone()).await.unwrap().into_inner();
        assert_eq!(echoed, resp);

        // Verify exact bit-parity with standalone AnalyzeStream TURBO_PROMPT_QUERY vector!
        if mode == ProtoOffsetMode::Utf8 {
            let mut stream = open_stream(&mut client, 0, 4, 2).await;
            stream
                .tx
                .send(AnalyzeStreamRequest {
                    kind: Some(Kind::Document(AnalyzeDocument {
                        id: "qvec".into(),
                        model: "hybrid".into(),
                        text: query.as_bytes().to_vec(),
                        options: Some(AnalyzeOptions { prompt_role: TURBO_PROMPT_QUERY, ..Default::default() }),
                        output: AnalyzeOutput::Vectors as i32,
                        offsets: 0,
                        binding: None,
                        vocabulary: None,
                    })),
                })
                .await
                .unwrap();
            stream.sent.push("qvec".into());
            let got = by_id(stream.half_close()).await;
            let Outcome::Vectors(v) = &got["qvec"] else { panic!("expected Vectors") };
            assert_eq!(full.data.as_slice(), v.data.as_ref(), "query-prompt vector parity");

            // Tampering with ordinal gap (duplicating an empty_normalized_ordinal into bm25_terms) is refused.
            let mut bad_resp = resp.clone();
            bad_resp.fields[1].bm25_terms[0].ordinal = 1;
            assert!(client.round_trip_analyze_query(bad_resp).await.is_err());
        }
    }

    server.stop().await;
}
