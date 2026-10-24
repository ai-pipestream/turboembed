mod common;
use std::collections::HashMap;
use turbo::qa::{Answer, DecodeOptions, ExtractiveQa, QaOptions};
#[cfg(feature = "cuda")]
use turbo_kserve::api::{Runtime, field};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES, Device, ModelConfig};
use turbo_kserve::proto::infer_parameter::ParameterChoice;
use turbo_kserve::proto::model_infer_request::InferInputTensor;
use turbo_kserve::proto::{InferParameter, InferTensorContents, ModelInferRequest};
use turbo_kserve::{Serve, Server};

fn param(v: ParameterChoice) -> InferParameter {
    InferParameter { parameter_choice: Some(v) }
}
async fn compare(backend: &str, device: u32) {
    let bundle = common::make_qa_bundle();
    let config = ModelConfig {
        bundle: bundle.0.to_str().unwrap().into(),
        name: Some("qa".into()),
        device: Device::Index(device),
        precision: turbo::TURBO_PRECISION_EXACT,
        max_batch: 2,
        max_seq: 0,
        sessions: 1,
    };
    let server =
        Server::start("127.0.0.1:0".parse().unwrap(), vec![config], DEFAULT_MAX_MESSAGE_BYTES, Serve::default())
            .await
            .unwrap();
    server.load().await.unwrap();
    let mut client = common::kserve(&server).await;
    let mut direct = ExtractiveQa::open_with(
        &bundle.0,
        &QaOptions { backend: Some(backend.into()), max_batch: 1, ..Default::default() },
    )
    .unwrap();
    let question = "What does the encoder test?";
    let context = "The tiny encoder tests deterministic question answering across windows. ".repeat(17);
    assert!(direct.logits(question, &context).unwrap().len() > 2);
    let opts = DecodeOptions { null_threshold: -1e6, top_k: 3, ..Default::default() };
    let Answer::Spans { answers, null_score, null_probability } = direct.answer(question, &context, opts).unwrap()
    else {
        panic!("expected spans")
    };
    for raw in [false, true] {
        let mut inputs = Vec::new();
        let mut raw_input_contents = Vec::new();
        for (name, text) in [("question", question), ("context", context.as_str())] {
            inputs.push(InferInputTensor {
                name: name.into(),
                datatype: "BYTES".into(),
                shape: vec![1],
                parameters: HashMap::new(),
                contents: (!raw).then(|| InferTensorContents {
                    bytes_contents: vec![text.as_bytes().to_vec()],
                    ..Default::default()
                }),
            });
            if raw {
                let mut bytes = (text.len() as u32).to_le_bytes().to_vec();
                bytes.extend_from_slice(text.as_bytes());
                raw_input_contents.push(bytes);
            }
        }
        let got = client
            .model_infer(ModelInferRequest {
                model_name: "qa".into(),
                inputs,
                raw_input_contents,
                parameters: HashMap::from([
                    ("null_threshold".into(), param(ParameterChoice::DoubleParam(-1e6))),
                    ("top_k".into(), param(ParameterChoice::Int64Param(3))),
                ]),
                ..Default::default()
            })
            .await
            .unwrap()
            .into_inner();
        let spans: Vec<i32> =
            got.raw_output_contents[0].as_chunks::<4>().0.iter().map(|&b| i32::from_le_bytes(b)).collect();
        assert_eq!(spans, answers.iter().flat_map(|a| [a.bytes.start as i32, a.bytes.end as i32]).collect::<Vec<_>>());
        for (index, want) in
            [answers.iter().map(|a| a.score).collect::<Vec<_>>(), answers.iter().map(|a| a.probability).collect()]
                .iter()
                .enumerate()
        {
            let values: Vec<f64> =
                got.raw_output_contents[index + 1].as_chunks::<8>().0.iter().map(|&b| f64::from_le_bytes(b)).collect();
            assert_eq!(values.len(), want.len());
            for (a, b) in values.iter().zip(want) {
                assert!((a - b).abs() < 1e-6, "{backend}: {a} != {b}");
            }
        }
        assert_eq!(got.parameters["null_score"], param(ParameterChoice::DoubleParam(null_score)));
        assert_eq!(got.parameters["null_probability"], param(ParameterChoice::DoubleParam(null_probability)));
    }
    server.stop().await;
}
#[tokio::test]
async fn cpu_qa_long_context_matches_single_window_runs() {
    compare("cpu", common::cpu()).await;
}
#[cfg(feature = "cuda")]
#[tokio::test]
async fn cuda_qa_long_context_matches_single_window_runs() {
    let rt = Runtime::create().unwrap();
    let device = (0..16).find(|&i| rt.device_info(i).is_ok_and(|d| field(&d.backend) == "cuda"));
    let Some(device) = device else {
        assert!(!std::env::var("TURBO_TEST_REQUIRE_CUDA").is_ok_and(|v| !v.is_empty() && v != "0"), "CUDA required");
        eprintln!("CUDA unavailable; skipped");
        return;
    };
    compare("cuda", device).await;
}
