//! End-to-end KServe v2 and AnalysisService tests for `TASK_RERANK` and
//! `TASK_EXTRACTIVE_QA` bundles.

mod common;

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use common::{make_qa_bundle, make_rerank_bundle};
use serde_json::Value;
use tonic::Code;
use turbo::qa::{Answer, DecodeOptions, ExtractiveQa};
use turbo::rerank::{RerankOptions, Reranker};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::infer_parameter::ParameterChoice;
use turbo_kserve::proto::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{CapabilitiesRequest, analysis_service_client::AnalysisServiceClient};
use turbo_kserve::proto::{InferParameter, InferTensorContents, ModelInferRequest, ModelMetadataRequest};
use turbo_kserve::{Serve, Server};

fn cfg(bundle: &Path, name: &str, max_batch: u32) -> ModelConfig {
    ModelConfig {
        bundle: bundle.to_str().unwrap().to_string(),
        name: Some(name.to_string()),
        device: Device::Select,
        precision: 0,
        max_batch,
        max_seq: 0,
        sessions: 1,
    }
}

fn param(c: ParameterChoice) -> InferParameter {
    InferParameter { parameter_choice: Some(c) }
}

fn bytes_input(name: &str, shape: &[i64], items: &[&[u8]]) -> InferInputTensor {
    let bytes_contents = items.iter().map(|b| b.to_vec()).collect();
    InferInputTensor {
        name: name.into(),
        datatype: "BYTES".into(),
        shape: shape.to_vec(),
        parameters: HashMap::new(),
        contents: Some(InferTensorContents { bytes_contents, ..Default::default() }),
    }
}

fn raw_input(name: &str, shape: &[i64]) -> InferInputTensor {
    InferInputTensor { name: name.into(), datatype: "BYTES".into(), shape: shape.to_vec(), ..Default::default() }
}

fn pack_raw_bytes(items: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::new();
    for b in items {
        out.extend((b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
    }
    out
}

fn decode_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect()
}

fn decode_f64s(bytes: &[u8]) -> Vec<f64> {
    bytes.as_chunks::<8>().0.iter().map(|&c| f64::from_le_bytes(c)).collect()
}

fn decode_i32s(bytes: &[u8]) -> Vec<i32> {
    bytes.as_chunks::<4>().0.iter().map(|&c| i32::from_le_bytes(c)).collect()
}

fn refusal_field(s: &tonic::Status) -> (Code, i32, u32) {
    let (code, tc) = common::refusal(s);
    let field = s.metadata().get("turbo-field").map_or(0, |v| v.to_str().unwrap().parse::<u32>().unwrap());
    (code, tc, field)
}

#[tokio::test]
async fn metadata_capabilities_and_stream_refusal_for_rerank_and_qa() {
    let (bert_rerank, roberta_rerank, qa) =
        (make_rerank_bundle("PAIR_BERT"), make_rerank_bundle("PAIR_ROBERTA"), make_qa_bundle());
    let models = vec![
        cfg(&bert_rerank.0, "ms-marco-minilm-l6-v2-rerank", 0),
        cfg(&roberta_rerank.0, "bge-reranker-base", 0),
        cfg(&qa.0, "qa-bert", 1),
    ];
    let s = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, Serve { admin: true }).await.unwrap();
    s.load().await.unwrap();

    let mut k = common::kserve(&s).await;
    let mut a = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();

    let caps = a.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let by_name: HashMap<_, _> = caps.bundles.iter().map(|b| (b.name.as_str(), b)).collect();
    for (name, task) in [
        ("ms-marco-minilm-l6-v2-rerank", "TASK_RERANK"),
        ("bge-reranker-base", "TASK_RERANK"),
        ("qa-bert", "TASK_EXTRACTIVE_QA"),
    ] {
        assert_eq!(by_name[name].task, task);
        assert!(by_name[name].widths.is_empty());
    }

    let meta = |n: &str| ModelMetadataRequest { name: n.into(), version: String::new() };
    let m_bert = k.model_metadata(meta("ms-marco-minilm-l6-v2-rerank")).await.unwrap().into_inner();
    assert_eq!(m_bert.properties["model_info.task"], "TASK_RERANK");
    assert_eq!(m_bert.properties["model_info.rerank_pair_template"], "PAIR_BERT");
    assert_eq!(m_bert.properties["model_info.rerank_query_tokens"], "8");
    assert_eq!(
        m_bert.inputs.iter().map(|t| (&t.name[..], &t.datatype[..], &t.shape[..])).collect::<Vec<_>>(),
        [("query", "BYTES", &[-1][..]), ("passages", "BYTES", &[-1][..])]
    );
    assert_eq!(
        m_bert.outputs.iter().map(|t| (&t.name[..], &t.datatype[..], &t.shape[..])).collect::<Vec<_>>(),
        [("scores", "FP32", &[-1, 1][..])]
    );

    let m_rob = k.model_metadata(meta("bge-reranker-base")).await.unwrap().into_inner();
    assert_eq!(m_rob.properties["model_info.rerank_pair_template"], "PAIR_ROBERTA");

    let m_qa = k.model_metadata(meta("qa-bert")).await.unwrap().into_inner();
    assert_eq!(m_qa.properties["model_info.task"], "TASK_EXTRACTIVE_QA");
    assert_eq!(
        m_qa.inputs.iter().map(|t| (&t.name[..], &t.datatype[..], &t.shape[..])).collect::<Vec<_>>(),
        [("question", "BYTES", &[-1][..]), ("context", "BYTES", &[-1][..])]
    );
    assert_eq!(
        m_qa.outputs.iter().map(|t| (&t.name[..], &t.datatype[..], &t.shape[..])).collect::<Vec<_>>(),
        [("spans", "INT32", &[-1, 2][..]), ("scores", "FP64", &[-1][..]), ("probabilities", "FP64", &[-1][..])]
    );

    let mut stream = common::open_stream(&mut a, 0, 0, 0).await;
    stream.send("doc-rerank", "ms-marco-minilm-l6-v2-rerank", "hello", None).await;
    stream.send("doc-qa", "qa-bert", "world", None).await;
    let answers = common::by_id(stream.half_close()).await;
    for id in ["doc-rerank", "doc-qa"] {
        match &answers[id] {
            Outcome::Failure(f) => assert_eq!(f.turbo_code, 514, "{id}: {f:?}"),
            other => panic!("expected failure on {id}, got {other:?}"),
        }
    }
    s.stop().await;
}

#[tokio::test]
async fn rerank_model_infer_matches_direct_reranker_bit_for_bit() {
    let (bert_rerank, roberta_rerank) = (make_rerank_bundle("PAIR_BERT"), make_rerank_bundle("PAIR_ROBERTA"));
    let models =
        vec![cfg(&bert_rerank.0, "ms-marco-minilm-l6-v2-rerank", 0), cfg(&roberta_rerank.0, "bge-reranker-base", 0)];
    let s = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, Serve::default()).await.unwrap();
    s.load().await.unwrap();
    let mut k = common::kserve(&s).await;

    for (name, dir) in [("ms-marco-minilm-l6-v2-rerank", &bert_rerank.0), ("bge-reranker-base", &roberta_rerank.0)] {
        let mut direct = Reranker::open(dir, 1).unwrap();
        let pairs = [
            ("what is rust?", "rust is a systems programming language"),
            ("what is rust?", "iron oxide forms in moist air"),
            ("capital of france", "paris is the capital of france"),
        ];
        let want = direct.score(&pairs, RerankOptions::default()).unwrap().scores;

        let req_qp = ModelInferRequest {
            model_name: name.into(),
            id: "r1".into(),
            inputs: vec![
                bytes_input("query", &[3], &[pairs[0].0.as_bytes(), pairs[1].0.as_bytes(), pairs[2].0.as_bytes()]),
                bytes_input("passages", &[3], &[pairs[0].1.as_bytes(), pairs[1].1.as_bytes(), pairs[2].1.as_bytes()]),
            ],
            ..Default::default()
        };
        let resp = k.model_infer(req_qp).await.unwrap().into_inner();
        assert_eq!((&resp.outputs[0].name[..], &resp.outputs[0].datatype[..]), ("scores", "FP32"));
        assert_eq!(resp.outputs[0].shape, [3, 1]);
        assert_eq!(decode_f32s(&resp.raw_output_contents[0]), want);
        assert_eq!(resp.parameters["task"], param(ParameterChoice::StringParam("TASK_RERANK".into())));
        assert!(resp.parameters.contains_key("stage.RERANK_STAGE_TOKENIZE"));
        assert!(resp.parameters.contains_key("stage.RERANK_STAGE_HEAD"));

        let resp_bcast = k
            .model_infer(ModelInferRequest {
                model_name: name.into(),
                inputs: vec![
                    bytes_input("query", &[1], &[pairs[0].0.as_bytes()]),
                    bytes_input("passages", &[2], &[pairs[0].1.as_bytes(), pairs[1].1.as_bytes()]),
                ],
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(decode_f32s(&resp_bcast.raw_output_contents[0]), &want[..2]);

        let flat = [
            pairs[0].0.as_bytes(),
            pairs[0].1.as_bytes(),
            pairs[1].0.as_bytes(),
            pairs[1].1.as_bytes(),
            pairs[2].0.as_bytes(),
            pairs[2].1.as_bytes(),
        ];
        let resp_raw = k
            .model_infer(ModelInferRequest {
                model_name: name.into(),
                inputs: vec![raw_input("pairs", &[3, 2])],
                raw_input_contents: vec![pack_raw_bytes(&flat)],
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(decode_f32s(&resp_raw.raw_output_contents[0]), want);

        let long_passage = "alpha beta gamma delta epsilon zeta ".repeat(12);
        let stride_opts = RerankOptions { query_tokens: 6, max_tokens: 24, stride: 4 };
        let want_stride = direct.score(&[("alpha gamma", long_passage.as_str())], stride_opts).unwrap().scores;
        let resp_stride = k
            .model_infer(ModelInferRequest {
                model_name: name.into(),
                parameters: HashMap::from([
                    ("query_tokens".into(), param(ParameterChoice::Int64Param(6))),
                    ("max_tokens".into(), param(ParameterChoice::Int64Param(24))),
                    ("stride".into(), param(ParameterChoice::Int64Param(4))),
                ]),
                inputs: vec![
                    bytes_input("query", &[1], &[b"alpha gamma"]),
                    bytes_input("passages", &[1], &[long_passage.as_bytes()]),
                ],
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(decode_f32s(&resp_stride.raw_output_contents[0]), want_stride);
    }
    s.stop().await;
}

#[tokio::test]
async fn qa_model_infer_matches_direct_extractive_qa_bit_for_bit() {
    let qa_dir = make_qa_bundle();
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![cfg(&qa_dir.0, "qa-bert", 1)], MAX, Serve::default())
        .await
        .unwrap();
    s.load().await.unwrap();
    let mut k = common::kserve(&s).await;

    let mut direct = ExtractiveQa::open(&qa_dir.0, 1).unwrap();
    let question = "What does the tiny encoder test?";
    let context = "The tiny encoder tests deterministic extractive question answering spans across windows.";

    let Answer::Spans { answers: want_answers, null_score: want_null_score, null_probability: want_null_prob } = direct
        .answer(question, context, DecodeOptions { null_threshold: -1e6, top_k: 3, ..Default::default() })
        .unwrap()
    else {
        panic!("expected spans");
    };

    let params = HashMap::from([
        ("null_threshold".into(), param(ParameterChoice::DoubleParam(-1e6))),
        ("top_k".into(), param(ParameterChoice::Int64Param(3))),
    ]);
    let resp = k
        .model_infer(ModelInferRequest {
            model_name: "qa-bert".into(),
            id: "qa1".into(),
            parameters: params.clone(),
            inputs: vec![
                bytes_input("question", &[1], &[question.as_bytes()]),
                bytes_input("context", &[1], &[context.as_bytes()]),
            ],
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();

    assert_eq!(resp.outputs.len(), 3);
    assert_eq!(resp.outputs[0].shape, [want_answers.len() as i64, 2]);
    let want_spans: Vec<i32> = want_answers.iter().flat_map(|a| [a.bytes.start as i32, a.bytes.end as i32]).collect();
    assert_eq!(decode_i32s(&resp.raw_output_contents[0]), want_spans);
    assert_eq!(decode_f64s(&resp.raw_output_contents[1]), want_answers.iter().map(|a| a.score).collect::<Vec<_>>());
    assert_eq!(
        decode_f64s(&resp.raw_output_contents[2]),
        want_answers.iter().map(|a| a.probability).collect::<Vec<_>>()
    );
    assert_eq!(resp.parameters["abstained"], param(ParameterChoice::BoolParam(false)));
    assert_eq!(resp.parameters["null_score"], param(ParameterChoice::DoubleParam(want_null_score)));
    assert_eq!(resp.parameters["null_probability"], param(ParameterChoice::DoubleParam(want_null_prob)));

    let resp_sub = k
        .model_infer(ModelInferRequest {
            model_name: "qa-bert".into(),
            parameters: params,
            inputs: vec![raw_input("question", &[1]), raw_input("context", &[1])],
            outputs: vec![InferRequestedOutputTensor { name: "spans".into(), parameters: HashMap::new() }],
            raw_input_contents: vec![pack_raw_bytes(&[question.as_bytes()]), pack_raw_bytes(&[context.as_bytes()])],
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(resp_sub.outputs.len(), 1);
    assert_eq!(decode_i32s(&resp_sub.raw_output_contents[0]), want_spans);

    let resp_abs = k
        .model_infer(ModelInferRequest {
            model_name: "qa-bert".into(),
            parameters: HashMap::from([("null_threshold".into(), param(ParameterChoice::DoubleParam(1e6)))]),
            inputs: vec![
                bytes_input("question", &[1], &[question.as_bytes()]),
                bytes_input("context", &[1], &[context.as_bytes()]),
            ],
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(resp_abs.outputs[0].shape, [0, 2]);
    assert!(resp_abs.raw_output_contents[0].is_empty());
    assert_eq!(resp_abs.parameters["abstained"], param(ParameterChoice::BoolParam(true)));
    assert_eq!(resp_abs.parameters["null_probability"], param(ParameterChoice::DoubleParam(1.0)));
    s.stop().await;
}

#[tokio::test]
async fn rerank_and_qa_refusals_report_status_code_and_field() {
    let (rerank_dir, qa_dir) = (make_rerank_bundle("PAIR_BERT"), make_qa_bundle());
    let models = vec![cfg(&rerank_dir.0, "rerank", 2), cfg(&qa_dir.0, "qa", 1)];
    let s = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, Serve { admin: true }).await.unwrap();
    s.load().await.unwrap();
    let mut k = common::kserve(&s).await;

    let base_rerank = |params: HashMap<String, InferParameter>| ModelInferRequest {
        model_name: "rerank".into(),
        parameters: params,
        inputs: vec![bytes_input("query", &[1], &[b"q"]), bytes_input("passages", &[1], &[b"p"])],
        ..Default::default()
    };
    for (name, want_code, want_tc, field, choice) in [
        ("query_tokens", Code::InvalidArgument, 256, 1, ParameterChoice::Int64Param(30)),
        ("max_tokens", Code::InvalidArgument, 256, 2, ParameterChoice::Int64Param(4)),
        ("max_tokens", Code::OutOfRange, 771, 2, ParameterChoice::Int64Param(33)),
        ("stride", Code::InvalidArgument, 256, 3, ParameterChoice::Int64Param(99)),
        ("stride", Code::InvalidArgument, 256, 3, ParameterChoice::BoolParam(true)),
        ("unknown", Code::InvalidArgument, 256, 0, ParameterChoice::Int64Param(1)),
    ] {
        let err = k.model_infer(base_rerank(HashMap::from([(name.into(), param(choice))]))).await.unwrap_err();
        assert_eq!(refusal_field(&err), (want_code, want_tc, field), "{name}");
    }

    let over_rerank = ModelInferRequest {
        inputs: vec![bytes_input("query", &[1], &[b"q"]), bytes_input("passages", &[3], &[b"p1", b"p2", b"p3"])],
        ..base_rerank(HashMap::new())
    };
    assert_eq!(refusal_field(&k.model_infer(over_rerank).await.unwrap_err()), (Code::OutOfRange, 771, 0));

    let base_qa = |params: HashMap<String, InferParameter>| ModelInferRequest {
        model_name: "qa".into(),
        parameters: params,
        inputs: vec![bytes_input("question", &[1], &[b"q"]), bytes_input("context", &[1], &[b"c"])],
        ..Default::default()
    };
    for (name, field, choice) in [
        ("max_answer_tokens", 1, ParameterChoice::StringParam("10".into())),
        ("top_k", 2, ParameterChoice::Int64Param(-1)),
        ("null_threshold", 3, ParameterChoice::DoubleParam(f64::NAN)),
        ("non_overlapping", 4, ParameterChoice::Int64Param(1)),
        ("unknown", 0, ParameterChoice::BoolParam(true)),
    ] {
        let err = k.model_infer(base_qa(HashMap::from([(name.into(), param(choice))]))).await.unwrap_err();
        assert_eq!(refusal_field(&err), (Code::InvalidArgument, 256, field), "{name}");
    }

    let bad_utf8 = ModelInferRequest {
        inputs: vec![bytes_input("question", &[1], &[&[0xff]]), bytes_input("context", &[1], &[b"c"])],
        ..base_qa(HashMap::new())
    };
    assert_eq!(refusal_field(&k.model_infer(bad_utf8).await.unwrap_err()), (Code::InvalidArgument, 258, 0));

    let over_qa = ModelInferRequest {
        inputs: vec![bytes_input("question", &[2], &[b"q1", b"q2"]), bytes_input("context", &[2], &[b"c1", b"c2"])],
        ..base_qa(HashMap::new())
    };
    assert_eq!(refusal_field(&k.model_infer(over_qa).await.unwrap_err()), (Code::OutOfRange, 771, 0));

    let bad_bundle = make_rerank_bundle("PAIR_BERT");
    let mut bad_manifest: Value =
        serde_json::from_slice(&fs::read(bad_bundle.0.join("manifest.json")).unwrap()).unwrap();
    bad_manifest["task"] = "TASK_TOKEN_CLASSIFICATION".into();
    fs::write(bad_bundle.0.join("manifest.json"), serde_json::to_vec_pretty(&bad_manifest).unwrap()).unwrap();

    let mut a = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let load_err = a.load_bundle(common::load_req("bad-task", bad_bundle.0.to_str().unwrap(), 1, 0)).await.unwrap_err();
    assert_eq!(load_err.code(), Code::Internal);
    assert!(load_err.message().contains("TASK_TOKEN_CLASSIFICATION"), "{load_err:?}");
    s.stop().await;
}
