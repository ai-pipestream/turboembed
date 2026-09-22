//! `inference.GRPCInferenceService` through the client generated from the
//! same `proto/open_inference_grpc.proto` the server is generated from.

mod common;

use std::collections::HashMap;

use tonic::Code;
use turbo_inferstream::grpc::inference::model_infer_response::InferOutputTensor as OutputTensor;
use turbo_inferstream::grpc::inference::{
    infer_parameter, model_infer_request, InferParameter, InferTensorContents, ModelInferRequest, ModelInferResponse,
    ModelMetadataRequest, ModelReadyRequest, ServerLiveRequest, ServerMetadataRequest, ServerReadyRequest,
};
use turbo_inferstream::oip::{SERVER_NAME, SERVER_VERSION};

type Client = turbo_inferstream::grpc::inference::grpc_inference_service_client::GrpcInferenceServiceClient<
    tonic::transport::Channel,
>;

async fn client() -> Client {
    common::grpc_client(common::engine()).await
}

fn string_param(v: &str) -> InferParameter {
    InferParameter { parameter_choice: Some(infer_parameter::ParameterChoice::StringParam(v.to_string())) }
}

fn int_param(v: i64) -> InferParameter {
    InferParameter { parameter_choice: Some(infer_parameter::ParameterChoice::Int64Param(v)) }
}

fn bool_param(v: bool) -> InferParameter {
    InferParameter { parameter_choice: Some(infer_parameter::ParameterChoice::BoolParam(v)) }
}

fn bytes_input(name: &str, items: &[&str]) -> model_infer_request::InferInputTensor {
    model_infer_request::InferInputTensor {
        name: name.to_string(),
        datatype: "BYTES".to_string(),
        shape: vec![items.len() as i64],
        parameters: HashMap::new(),
        contents: Some(InferTensorContents {
            bytes_contents: items.iter().map(|s| s.as_bytes().to_vec()).collect(),
            ..Default::default()
        }),
    }
}

fn request(model: &str, inputs: Vec<model_infer_request::InferInputTensor>) -> ModelInferRequest {
    ModelInferRequest {
        model_name: model.to_string(),
        model_version: String::new(),
        id: String::new(),
        parameters: HashMap::new(),
        inputs,
        outputs: Vec::new(),
        raw_input_contents: Vec::new(),
    }
}

/// The output tensor of that name, or a panic listing what came back.
fn output<'a>(response: &'a ModelInferResponse, name: &str) -> &'a OutputTensor {
    response.outputs.iter().find(|o| o.name == name).unwrap_or_else(|| {
        let names: Vec<&str> = response.outputs.iter().map(|o| o.name.as_str()).collect();
        panic!("no output `{name}`; outputs: {names:?}")
    })
}

/// A response parameter as a string, whatever shape it arrived in.
fn param(response: &ModelInferResponse, name: &str) -> String {
    let p = response
        .parameters
        .get(name)
        .unwrap_or_else(|| panic!("no response parameter `{name}`; got {:?}", response.parameters.keys()));
    match &p.parameter_choice {
        Some(infer_parameter::ParameterChoice::StringParam(s)) => s.clone(),
        Some(infer_parameter::ParameterChoice::Int64Param(i)) => i.to_string(),
        Some(infer_parameter::ParameterChoice::Uint64Param(u)) => u.to_string(),
        Some(infer_parameter::ParameterChoice::DoubleParam(d)) => d.to_string(),
        Some(infer_parameter::ParameterChoice::BoolParam(b)) => b.to_string(),
        None => panic!("response parameter `{name}` carries no value"),
    }
}

#[tokio::test]
async fn server_live_and_ready_report_a_loaded_server() {
    let mut c = client().await;
    let live = c.server_live(ServerLiveRequest {}).await.expect("ServerLive").into_inner();
    assert!(live.live, "ServerLive said the server is not live");
    let ready = c.server_ready(ServerReadyRequest {}).await.expect("ServerReady").into_inner();
    assert!(ready.ready, "ServerReady said the server is not ready although six models are loaded");
}

#[tokio::test]
async fn server_metadata_names_the_server_and_its_extensions() {
    let mut c = client().await;
    let m = c.server_metadata(ServerMetadataRequest {}).await.expect("ServerMetadata").into_inner();
    assert_eq!(m.name, SERVER_NAME, "ServerMetadata.name");
    assert_eq!(m.version, SERVER_VERSION, "ServerMetadata.version");
    assert_eq!(m.extensions, ["turbo_parameters"], "ServerMetadata.extensions");
}

#[tokio::test]
async fn model_ready_is_true_for_a_served_model_and_not_found_for_an_unknown_one() {
    let mut c = client().await;
    let known = c
        .model_ready(ModelReadyRequest { name: "embed".into(), version: String::new() })
        .await
        .expect("ModelReady for embed")
        .into_inner();
    assert!(known.ready, "ModelReady said `embed` is not ready");
    // The same answer as REST's 404: an unknown model is not "not ready",
    // it does not exist.
    let unknown = c
        .model_ready(ModelReadyRequest { name: "nope".into(), version: String::new() })
        .await
        .expect_err("ModelReady for an unknown model is an error");
    assert_eq!(unknown.code(), tonic::Code::NotFound, "{unknown}");
}

#[tokio::test]
async fn model_metadata_carries_the_tensors_and_the_bundle_properties() {
    let mut c = client().await;
    let m = c
        .model_metadata(ModelMetadataRequest { name: "embed".into(), version: String::new() })
        .await
        .expect("ModelMetadata")
        .into_inner();
    assert_eq!(m.name, "embed", "ModelMetadata.name");
    assert_eq!(m.versions, ["1"], "ModelMetadata.versions");
    assert_eq!(m.platform, "mock", "ModelMetadata.platform");
    let inputs: Vec<(&str, &str, &[i64])> =
        m.inputs.iter().map(|t| (t.name.as_str(), t.datatype.as_str(), t.shape.as_slice())).collect();
    assert_eq!(inputs, [("text", "BYTES", [-1].as_slice())], "ModelMetadata.inputs");
    let outputs: Vec<(&str, &str, &[i64])> =
        m.outputs.iter().map(|t| (t.name.as_str(), t.datatype.as_str(), t.shape.as_slice())).collect();
    assert_eq!(outputs, [("embeddings", "FP32", [-1, 8].as_slice())], "ModelMetadata.outputs");
    for (key, want) in [
        ("model_id", "turbo/mock-embedding"),
        ("kind", "Embedding"),
        ("device", "Mock accelerator"),
        ("pooling", "mean"),
    ] {
        assert_eq!(
            m.properties.get(key).map(String::as_str),
            Some(want),
            "ModelMetadata.properties[{key}]: {:?}",
            m.properties
        );
    }
}

#[tokio::test]
async fn model_infer_embedding_returns_a_row_per_text() {
    let mut c = client().await;
    let r = c
        .model_infer(request("embed", vec![bytes_input("text", &["hello world", "goodbye"])]))
        .await
        .expect("ModelInfer on embed")
        .into_inner();
    assert_eq!(r.model_name, "embed", "the response names the model");
    assert_eq!(r.model_version, "1", "the response names the version");
    let t = output(&r, "embeddings");
    assert_eq!(t.datatype, "FP32", "the embeddings datatype");
    assert_eq!(t.shape, [2, 8], "the embeddings shape is [texts, dim]");
    let values = &t.contents.as_ref().expect("embeddings contents").fp32_contents;
    assert_eq!(values.len(), 16, "two rows of eight floats");
    assert_eq!(param(&r, "placement"), "Host", "the mock leaves its result on the host");
}

#[tokio::test]
async fn model_infer_reranker_honors_top_n_and_raw_scores() {
    let mut c = client().await;
    let mut req = request(
        "rerank",
        vec![bytes_input("query", &["alpha beta"]), bytes_input("documents", &["alpha beta gamma", "zulu", "alpha"])],
    );
    req.parameters.insert("top_n".into(), int_param(2));
    req.parameters.insert("raw_scores".into(), bool_param(true));
    let r = c.model_infer(req).await.expect("ModelInfer on rerank").into_inner();
    let scores = &output(&r, "scores").contents.as_ref().expect("scores contents").fp32_contents;
    assert_eq!(scores.len(), 3, "every document keeps a score even when top_n cuts the ranking");
    assert!(scores.iter().any(|&x| !(0.0..=1.0).contains(&x)), "raw_scores returned activated scores {scores:?}");
    let sorted = &output(&r, "sorted").contents.as_ref().expect("sorted contents").int_contents;
    assert_eq!(sorted.len(), 2, "top_n 2 keeps two ranks: {sorted:?}");
    assert_eq!(sorted[0], 2, "`alpha` overlaps the query most; scores {scores:?}");
}

#[tokio::test]
async fn model_infer_classifier_returns_scores_and_labels() {
    let mut c = client().await;
    let r = c
        .model_infer(request("classify", vec![bytes_input("text", &["good", "bad"])]))
        .await
        .expect("ModelInfer on classify")
        .into_inner();
    let scores = output(&r, "scores");
    assert_eq!(scores.shape, [2, 3], "the scores shape is [texts, labels]");
    let labels = &output(&r, "labels").contents.as_ref().expect("labels contents").bytes_contents;
    let labels: Vec<String> = labels.iter().map(|b| String::from_utf8_lossy(b).into_owned()).collect();
    assert_eq!(labels, ["negative", "neutral", "positive"], "the bundle's labels");
}

#[tokio::test]
async fn model_infer_generative_reports_the_finish_reason() {
    let mut c = client().await;
    let mut req = request("chat", vec![bytes_input("prompt", &["hello"])]);
    req.parameters.insert("max_new_tokens".into(), int_param(4));
    let r = c.model_infer(req).await.expect("ModelInfer on chat").into_inner();
    let text = &output(&r, "text").contents.as_ref().expect("text contents").bytes_contents;
    assert_eq!(text.len(), 1, "one generated text");
    assert!(!text[0].is_empty(), "the model generated nothing");
    assert_eq!(param(&r, "finish_reason"), "length", "4 new tokens hits the budget");
    assert_eq!(param(&r, "generated_tokens"), "4", "max_new_tokens 4 was honored");
}

#[tokio::test]
async fn model_infer_generic_run_model_doubles_its_input() {
    let mut c = client().await;
    let x = model_infer_request::InferInputTensor {
        name: "x".into(),
        datatype: "FP32".into(),
        shape: vec![2, 3],
        parameters: HashMap::new(),
        contents: Some(InferTensorContents { fp32_contents: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ..Default::default() }),
    };
    let r = c.model_infer(request("run", vec![x])).await.expect("ModelInfer on run").into_inner();
    let y = output(&r, "y");
    assert_eq!(y.shape, [2, 3], "the output keeps the input's shape");
    assert_eq!(
        y.contents.as_ref().expect("y contents").fp32_contents,
        [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "the mock RUN model computes y = 2x"
    );
}

#[tokio::test]
async fn model_infer_accepts_raw_input_contents_for_fp32() {
    let mut c = client().await;
    let x = model_infer_request::InferInputTensor {
        name: "x".into(),
        datatype: "FP32".into(),
        shape: vec![2, 3],
        parameters: HashMap::new(),
        contents: None,
    };
    let mut req = request("run", vec![x]);
    req.raw_input_contents = vec![[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0].iter().flat_map(|v| v.to_le_bytes()).collect()];
    let r = c.model_infer(req).await.expect("ModelInfer with raw FP32 contents").into_inner();
    assert_eq!(
        output(&r, "y").contents.as_ref().expect("y contents").fp32_contents,
        [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "raw little-endian FP32 input did not reach the model unchanged"
    );
}

#[tokio::test]
async fn model_infer_accepts_raw_input_contents_for_length_prefixed_bytes() {
    let mut c = client().await;
    let texts = ["hello world", "goodbye"];
    let mut raw = Vec::new();
    for t in texts {
        raw.extend_from_slice(&(t.len() as u32).to_le_bytes());
        raw.extend_from_slice(t.as_bytes());
    }
    let text = model_infer_request::InferInputTensor {
        name: "text".into(),
        datatype: "BYTES".into(),
        shape: vec![2],
        parameters: HashMap::new(),
        contents: None,
    };
    let mut req = request("embed", vec![text]);
    req.raw_input_contents = vec![raw];
    let raw_response = c.model_infer(req).await.expect("ModelInfer with raw BYTES contents").into_inner();

    let typed = c
        .model_infer(request("embed", vec![bytes_input("text", &texts)]))
        .await
        .expect("ModelInfer with typed contents")
        .into_inner();
    assert_eq!(
        output(&raw_response, "embeddings").contents.as_ref().expect("contents").fp32_contents,
        output(&typed, "embeddings").contents.as_ref().expect("contents").fp32_contents,
        "the 4-byte length-prefixed raw form did not decode to the same strings as typed contents"
    );
}

#[tokio::test]
async fn model_infer_with_both_contents_and_raw_contents_is_invalid_argument() {
    let mut c = client().await;
    let mut req = request("embed", vec![bytes_input("text", &["hello"])]);
    req.raw_input_contents = vec![{
        let mut raw = (5u32).to_le_bytes().to_vec();
        raw.extend_from_slice(b"hello");
        raw
    }];
    let status = c.model_infer(req).await.expect_err("the protocol allows contents or raw_input_contents, not both");
    assert_eq!(status.code(), Code::InvalidArgument, "status of a doubly-carried input: {status:?}");
    assert!(status.message().contains("raw_input_contents"), "the message names the conflict: {}", status.message());
}

#[tokio::test]
async fn model_infer_on_an_unknown_model_is_not_found() {
    let mut c = client().await;
    let status =
        c.model_infer(request("nope", vec![bytes_input("text", &["hi"])])).await.expect_err("an unknown model");
    assert_eq!(status.code(), Code::NotFound, "status of an unknown model: {status:?}");
    assert!(status.message().contains("embed"), "the message lists the served models: {}", status.message());
}

#[tokio::test]
async fn model_infer_with_an_unsupported_option_is_unimplemented() {
    let mut c = client().await;
    let mut req = request("embed", vec![bytes_input("text", &["hi"])]);
    req.parameters.insert("pooling".into(), string_param("cls"));
    let status = c.model_infer(req).await.expect_err("the mock does not carry TURBO_CAP_OPT_POOLING_OVERRIDE");
    assert_eq!(status.code(), Code::Unimplemented, "status of an unsupported option: {status:?}");
    assert!(status.message().contains("field 6"), "the message names the pooling field: {}", status.message());
}

#[tokio::test]
async fn model_version_2_is_not_found_on_every_call_that_names_a_version() {
    let mut c = client().await;
    let mut req = request("embed", vec![bytes_input("text", &["hi"])]);
    req.model_version = "2".into();
    let infer = c.model_infer(req).await.expect_err("ModelInfer on version 2");
    assert_eq!(infer.code(), Code::NotFound, "ModelInfer version 2: {infer:?}");

    let metadata = c
        .model_metadata(ModelMetadataRequest { name: "embed".into(), version: "2".into() })
        .await
        .expect_err("ModelMetadata on version 2");
    assert_eq!(metadata.code(), Code::NotFound, "ModelMetadata version 2: {metadata:?}");

    let ready = c
        .model_ready(ModelReadyRequest { name: "embed".into(), version: "2".into() })
        .await
        .expect_err("ModelReady on version 2");
    assert_eq!(ready.code(), Code::NotFound, "ModelReady version 2: {ready:?}");
}
