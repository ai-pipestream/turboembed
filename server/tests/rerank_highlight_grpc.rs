//! gRPC integration and latency benchmark tests for `#2342`:
//! `AnalysisService::RerankAndHighlight` and `RoundTripRerankAndHighlight`.

mod common;

use std::time::Instant;

use common::*;
use turbo::bundle::Bundle;
use turbo_analysis::ModelRef as CoreModelRef;
use turbo_analysis_embed::{RERANK_CALIBRATION_LABELS, RerankCalibrator};
use turbo_calibration::{Calibration as ProbCalibration, DataIdentity, Identity, Task, Transform};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::rerank_calibration_spec_proto::Kind as CalKind;
use turbo_kserve::proto::result::{
    CapabilitiesRequest, OffsetMode as ProtoOffsetMode, RerankAndHighlightRequest, RerankCalibrationSpecProto,
    RerankCandidateInputProto, RerankHighlightMode, RerankIsotonicSpecProto, RerankPlattSpecProto, UnloadBundleRequest,
};
use turbo_kserve::{Serve, Server};

fn slice_by_mode(text: &str, mode: ProtoOffsetMode, start: u32, end: u32) -> String {
    match mode {
        ProtoOffsetMode::Utf8 => {
            assert!(text.is_char_boundary(start as usize));
            assert!(text.is_char_boundary(end as usize));
            text[start as usize..end as usize].to_owned()
        }
        ProtoOffsetMode::Utf16 => {
            let units: Vec<u16> = text.encode_utf16().collect();
            String::from_utf16(&units[start as usize..end as usize]).expect("valid UTF-16 boundary")
        }
        ProtoOffsetMode::CodePoint => text.chars().skip(start as usize).take((end - start) as usize).collect(),
    }
}

#[tokio::test]
async fn rerank_and_highlight_modes_calibration_offsets_and_c50_benchmark() {
    let rerank_dir = make_rerank_bundle("PAIR_BERT");
    let qa_dir = make_qa_bundle();
    let embed_dir = tiny();

    let models = vec![
        ModelConfig {
            name: Some("rerank".into()),
            bundle: rerank_dir.0.to_str().unwrap().into(),
            device: Device::Index(cpu()),
            sessions: 2,
            precision: 0,
            max_batch: 0,
            max_seq: 0,
        },
        ModelConfig {
            name: Some("embed".into()),
            bundle: embed_dir.to_str().unwrap().into(),
            device: Device::Index(cpu()),
            sessions: 2,
            precision: 0,
            max_batch: 0,
            max_seq: 0,
        },
        ModelConfig {
            name: Some("qa".into()),
            bundle: qa_dir.0.to_str().unwrap().into(),
            device: Device::Index(cpu()),
            sessions: 2,
            precision: 0,
            max_batch: 0,
            max_seq: 0,
        },
    ];

    let server = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, Serve::default()).await.unwrap();
    server.load().await.unwrap();
    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX);

    let caps = client.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let rerank_cap = caps.bundles.iter().find(|b| b.name == "rerank").unwrap();
    let rerank_mref = rerank_cap.model.as_ref().unwrap();

    let tcal = ProbCalibration::new(
        Identity {
            model: rerank_mref.id.clone(),
            revision: rerank_mref.revision.clone(),
            sha256: rerank_mref.sha256.clone(),
            task: Task::Categorization,
            domain: "search".into(),
            labels: RERANK_CALIBRATION_LABELS.iter().map(|&s| s.to_owned()).collect(),
            inference_hash: 7,
        },
        Transform::Temperature(1.25),
        DataIdentity { split: "fit".into(), sha256: "1".repeat(64) },
        DataIdentity { split: "conf".into(), sha256: "2".repeat(64) },
        vec![0.05, 0.15, 0.30, 0.55, 0.80],
    )
    .unwrap();
    let tcal_bytes = tcal.to_bytes();

    let query = "password reset \u{1F680} settings";
    let candidates = vec![
        RerankCandidateInputProto {
            doc_id: "c0".into(),
            text: "Caf\u{e9} menu note. To re\u{00AD}set a pass\u{200B}word, open Settings \u{1F680} now! Footer."
                .into(),
            parent_offset: 1200,
            sentence_span_start: Vec::new(),
            sentence_span_end: Vec::new(),
        },
        RerankCandidateInputProto {
            doc_id: "c1".into(),
            text: "Unrelated gardening tips for spring tomatoes. Water plants daily.".into(),
            parent_offset: 400,
            sentence_span_start: Vec::new(),
            sentence_span_end: Vec::new(),
        },
        RerankCandidateInputProto {
            doc_id: "c2".into(),
            text: "Billing overview \u{1F680}. Password recovery lives under Security Settings.".into(),
            parent_offset: 950,
            sentence_span_start: Vec::new(),
            sentence_span_end: Vec::new(),
        },
    ];

    for proto_mode in [ProtoOffsetMode::Utf8, ProtoOffsetMode::Utf16, ProtoOffsetMode::CodePoint] {
        for (hl_mode, embed_m, qa_m, cal_kind) in [
            (RerankHighlightMode::RangePool, "embed", "", CalKind::TcalArtifact(tcal_bytes.clone())),
            (
                RerankHighlightMode::SubwordMaxsim,
                "",
                "",
                CalKind::Platt(RerankPlattSpecProto { slope: 1.2, intercept: -0.1 }),
            ),
            (
                RerankHighlightMode::SubwordMaxsim,
                "embed",
                "",
                CalKind::Isotonic(RerankIsotonicSpecProto {
                    upper: vec![0.25, 0.5, 0.75, 1.0],
                    value: vec![0.1, 0.4, 0.7, 0.95],
                }),
            ),
            (RerankHighlightMode::ExtractiveQa, "", "qa", CalKind::Temperature(1.1)),
        ] {
            let is_tcal = matches!(cal_kind, CalKind::TcalArtifact(_));
            let resp = client
                .rerank_and_highlight(RerankAndHighlightRequest {
                    query: query.into(),
                    rerank_model: "rerank".into(),
                    embed_model: embed_m.into(),
                    qa_model: qa_m.into(),
                    candidates: candidates.clone(),
                    offsets: proto_mode as i32,
                    calibration: Some(RerankCalibrationSpecProto {
                        kind: Some(cal_kind),
                        domain: "search".into(),
                        inference_hash: 7,
                    }),
                    highlight_mode: hl_mode as i32,
                    highlight_top_k: 3,
                    max_snippets_per_candidate: 2,
                    language: "en".into(),
                    query_tokens: 8,
                    max_tokens: 32,
                    stride: 0,
                    max_answer_tokens: 12,
                    qa_null_threshold: if hl_mode == RerankHighlightMode::ExtractiveQa { Some(100.0) } else { None },
                })
                .await
                .unwrap()
                .into_inner();

            assert_eq!(resp.results.len(), 3);
            let echoed = client.round_trip_rerank_and_highlight(resp.clone()).await.unwrap().into_inner();
            assert_eq!(echoed, resp);

            for r in &resp.results {
                assert!((0.0..=1.0).contains(&r.calibrated_probability));
                if is_tcal {
                    let pv = r.conformal_p_value.expect("TCAL conformal p-value");
                    assert!(pv > 0.0 && pv <= 1.0);
                }
                let orig = &candidates[r.input_index as usize];
                if hl_mode != RerankHighlightMode::ExtractiveQa || !r.qa_abstained {
                    assert!(!r.highlights.is_empty(), "expected highlights in mode {hl_mode:?}");
                }
                for h in &r.highlights {
                    let sliced = slice_by_mode(&orig.text, proto_mode, h.span_start, h.span_end);
                    assert_eq!(sliced, h.surface);
                    assert_eq!(h.parent_span_start, orig.parent_offset + u64::from(h.span_start));
                    assert_eq!(h.parent_span_end, orig.parent_offset + u64::from(h.span_end));
                }
            }
        }
    }

    // Benchmark C = 50 shortlist rerank vs rerank + top-10 semantic highlight attribution.
    let shortlist_50: Vec<RerankCandidateInputProto> = (0..50)
        .map(|i| RerankCandidateInputProto {
            doc_id: format!("doc-{i}"),
            text: format!(
                "Section {i} intro. Password reset and account recovery live in Security Settings panel {i}. Extra context line."
            ),
            parent_offset: (i as u64) * 512,
            sentence_span_start: Vec::new(),
            sentence_span_end: Vec::new(),
        })
        .collect();

    let make_bench_req = |hl_mode: RerankHighlightMode, embed_m: &str, qa_m: &str| RerankAndHighlightRequest {
        query: "how to reset account password in security settings".into(),
        rerank_model: "rerank".into(),
        embed_model: embed_m.into(),
        qa_model: qa_m.into(),
        candidates: shortlist_50.clone(),
        offsets: ProtoOffsetMode::Utf8 as i32,
        calibration: Some(RerankCalibrationSpecProto {
            kind: Some(CalKind::TcalArtifact(tcal_bytes.clone())),
            domain: "search".into(),
            inference_hash: 7,
        }),
        highlight_mode: hl_mode as i32,
        highlight_top_k: 10,
        max_snippets_per_candidate: 2,
        language: "en".into(),
        query_tokens: 8,
        max_tokens: 32,
        stride: 0,
        max_answer_tokens: 10,
        qa_null_threshold: Some(100.0),
    };

    // Warmup
    let _ = client.rerank_and_highlight(make_bench_req(RerankHighlightMode::RangePool, "embed", "")).await.unwrap();

    let iters = 10usize;
    let mut measure = async |req: RerankAndHighlightRequest| -> f64 {
        let t0 = Instant::now();
        for _ in 0..iters {
            let r = client.rerank_and_highlight(req.clone()).await.unwrap().into_inner();
            assert_eq!(r.results.len(), 50);
        }
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    };

    let base_us = measure(make_bench_req(RerankHighlightMode::None, "", "")).await;
    let subword_us = measure(make_bench_req(RerankHighlightMode::SubwordMaxsim, "", "")).await;
    let rangepool_us = measure(make_bench_req(RerankHighlightMode::RangePool, "embed", "")).await;
    let qa_us = measure(make_bench_req(RerankHighlightMode::ExtractiveQa, "", "qa")).await;
    println!(
        "BENCH_2342 c=50 top_k=10 rerank_only_us={base_us:.1} subword_maxsim_us={subword_us:.1} range_pool_us={rangepool_us:.1} extractive_qa_us={qa_us:.1}"
    );

    server.stop().await;
}

#[tokio::test]
async fn bundle_sidecar_calibrator_loads_once_overrides_and_reloads() {
    let rerank_dir = make_rerank_bundle("PAIR_BERT");
    let opened = Bundle::open(&rerank_dir.0).unwrap();
    let core_mref = CoreModelRef {
        id: opened.manifest.model.id.clone(),
        revision: opened.manifest.model.revision.clone(),
        sha256: opened.manifest_sha256.clone(),
    };

    let build_tcal = |temp: f64, nonconf: Vec<f64>| -> Vec<u8> {
        ProbCalibration::new(
            Identity {
                model: core_mref.id.clone(),
                revision: core_mref.revision.clone(),
                sha256: core_mref.sha256.clone(),
                task: Task::Categorization,
                domain: "search".into(),
                labels: RERANK_CALIBRATION_LABELS.iter().map(|&s| s.to_owned()).collect(),
                inference_hash: 0,
            },
            Transform::Temperature(temp),
            DataIdentity { split: "fit".into(), sha256: "1".repeat(64) },
            DataIdentity { split: "conf".into(), sha256: "2".repeat(64) },
            nonconf,
        )
        .unwrap()
        .to_bytes()
    };

    let tcal_v1 = build_tcal(2.5, vec![0.10, 0.25, 0.50, 0.75]);
    let sidecar = rerank_dir.0.join("calibration.tcal");
    std::fs::write(&sidecar, &tcal_v1).unwrap();

    let model_cfg = ModelConfig {
        name: Some("rerank".into()),
        bundle: rerank_dir.0.to_str().unwrap().into(),
        device: Device::Index(cpu()),
        sessions: 1,
        precision: 0,
        max_batch: 0,
        max_seq: 0,
    };
    let server =
        Server::start("127.0.0.1:0".parse().unwrap(), vec![model_cfg], MAX, Serve { admin: true, ..Serve::default() })
            .await
            .unwrap();
    server.load().await.unwrap();

    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX);

    let make_req = |calibration: Option<RerankCalibrationSpecProto>| RerankAndHighlightRequest {
        query: "account password reset".into(),
        rerank_model: "rerank".into(),
        candidates: vec![
            RerankCandidateInputProto {
                doc_id: "d0".into(),
                text: "Reset your account password under Security Settings.".into(),
                ..Default::default()
            },
            RerankCandidateInputProto {
                doc_id: "d1".into(),
                text: "Fresh basil grows best in warm sunny soil.".into(),
                ..Default::default()
            },
        ],
        offsets: ProtoOffsetMode::Utf8 as i32,
        calibration,
        highlight_mode: RerankHighlightMode::None as i32,
        query_tokens: 8,
        max_tokens: 32,
        ..Default::default()
    };

    // Overwrite and then delete calibration.tcal after load to prove warm calls
    // perform zero filesystem reads of the bundle sidecar.
    std::fs::write(&sidecar, b"corrupt-on-disk-after-load").unwrap();
    let warm_after_corrupt = client.rerank_and_highlight(make_req(None)).await.unwrap().into_inner();
    std::fs::remove_file(&sidecar).unwrap();
    let warm_after_delete = client.rerank_and_highlight(make_req(None)).await.unwrap().into_inner();
    assert_eq!(warm_after_corrupt, warm_after_delete);

    let expected_v1 = RerankCalibrator::from_tcal_bytes(&tcal_v1, &core_mref, "search", 0).unwrap();
    for r in &warm_after_delete.results {
        let exp = expected_v1.score(r.raw_logit).unwrap();
        assert_eq!(r.calibrated_probability, exp.calibrated_probability);
        assert_eq!(r.conformal_p_value, exp.conformal_p_value);
        assert!(r.conformal_p_value.is_some());
    }

    // Request-supplied calibration overrides the cached bundle sidecar without mutating it.
    let platt = RerankPlattSpecProto { slope: 1.4, intercept: -0.3 };
    let override_resp = client
        .rerank_and_highlight(make_req(Some(RerankCalibrationSpecProto {
            kind: Some(CalKind::Platt(platt)),
            domain: String::new(),
            inference_hash: 0,
        })))
        .await
        .unwrap()
        .into_inner();
    let expected_platt = RerankCalibrator::from_transform(Transform::Platt { slope: 1.4, intercept: -0.3 }).unwrap();
    for r in &override_resp.results {
        let exp = expected_platt.score(r.raw_logit).unwrap();
        assert_eq!(r.calibrated_probability, exp.calibrated_probability);
        assert_eq!(r.conformal_p_value, None);
    }
    let after_override = client.rerank_and_highlight(make_req(None)).await.unwrap().into_inner();
    assert_eq!(after_override, warm_after_delete);

    // Reloading the model under the same name picks up the updated calibration.tcal cleanly.
    let tcal_v2 = build_tcal(0.4, vec![0.20, 0.60]);
    std::fs::write(&sidecar, &tcal_v2).unwrap();
    client.unload_bundle(UnloadBundleRequest { name: "rerank".into() }).await.unwrap();
    client.load_bundle(load_req("rerank", rerank_dir.0.to_str().unwrap(), 1, 0)).await.unwrap();
    std::fs::remove_file(&sidecar).unwrap();

    let reloaded_resp = client.rerank_and_highlight(make_req(None)).await.unwrap().into_inner();
    let expected_v2 = RerankCalibrator::from_tcal_bytes(&tcal_v2, &core_mref, "search", 0).unwrap();
    for r in &reloaded_resp.results {
        let exp = expected_v2.score(r.raw_logit).unwrap();
        assert_eq!(r.calibrated_probability, exp.calibrated_probability);
        assert_eq!(r.conformal_p_value, exp.conformal_p_value);
    }
    assert_ne!(reloaded_resp.results[0].calibrated_probability, warm_after_delete.results[0].calibrated_probability);

    // Reloading with a malformed sidecar reports INVALID_ARGUMENT unless request-overridden.
    std::fs::write(&sidecar, b"invalid-tcal-bytes").unwrap();
    client.unload_bundle(UnloadBundleRequest { name: "rerank".into() }).await.unwrap();
    client.load_bundle(load_req("rerank", rerank_dir.0.to_str().unwrap(), 1, 0)).await.unwrap();

    let status = client.rerank_and_highlight(make_req(None)).await.expect_err("malformed sidecar must fail");
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(status.message().contains("unknown calibration format"), "{}", status.message());

    let ok_override = client
        .rerank_and_highlight(make_req(Some(RerankCalibrationSpecProto {
            kind: Some(CalKind::Temperature(1.1)),
            domain: String::new(),
            inference_hash: 0,
        })))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(ok_override.results.len(), 2);

    server.stop().await;
}
