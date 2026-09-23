//! `turbo.inferstream.InferstreamExtension` through the client generated
//! from the same `proto/turbo_inferstream.proto` the server is generated
//! from: streamed inference, the model repository, and the reflection
//! service that hands a client both protos.

mod common;

use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};

use tonic::Code;
use tonic_reflection::pb::v1::server_reflection_client::ServerReflectionClient;
use tonic_reflection::pb::v1::{server_reflection_request, server_reflection_response, ServerReflectionRequest};
use turbo_inferstream::grpc::ext::{
    ModelStreamInferResponse, RepositoryIndexRequest, RepositoryModelLoadRequest, RepositoryModelUnloadRequest,
};
use turbo_inferstream::grpc::inference::{
    infer_parameter, model_infer_request, InferParameter, InferTensorContents, ModelInferRequest, ModelReadyRequest,
};

fn int_param(v: i64) -> InferParameter {
    InferParameter { parameter_choice: Some(infer_parameter::ParameterChoice::Int64Param(v)) }
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

/// The strings of the `text` output of a streamed chunk.
fn chunk_text(chunk: &ModelStreamInferResponse) -> Vec<String> {
    let response =
        chunk.infer_response.as_ref().unwrap_or_else(|| panic!("a chunk without an infer_response: {chunk:?}"));
    let text = response.outputs.iter().find(|o| o.name == "text").unwrap_or_else(|| {
        panic!("no `text` output; outputs: {:?}", response.outputs.iter().map(|o| &o.name).collect::<Vec<_>>())
    });
    assert_eq!(text.datatype, "BYTES", "the `text` output is not BYTES");
    assert_eq!(text.shape, [1], "the `text` output holds one string");
    text.contents
        .as_ref()
        .unwrap_or_else(|| panic!("the `text` output carries no contents"))
        .bytes_contents
        .iter()
        .map(|b| String::from_utf8_lossy(b).into_owned())
        .collect()
}

/// A chunk parameter as a string, whatever shape it arrived in.
fn chunk_param(chunk: &ModelStreamInferResponse, name: &str) -> String {
    let response = chunk.infer_response.as_ref().expect("an infer_response");
    let p = response
        .parameters
        .get(name)
        .unwrap_or_else(|| panic!("no chunk parameter `{name}`; got {:?}", response.parameters.keys()));
    match &p.parameter_choice {
        Some(infer_parameter::ParameterChoice::StringParam(s)) => s.clone(),
        Some(infer_parameter::ParameterChoice::Int64Param(i)) => i.to_string(),
        Some(infer_parameter::ParameterChoice::Uint64Param(u)) => u.to_string(),
        Some(infer_parameter::ParameterChoice::DoubleParam(d)) => d.to_string(),
        Some(infer_parameter::ParameterChoice::BoolParam(b)) => b.to_string(),
        None => panic!("chunk parameter `{name}` carries no value"),
    }
}

/// Every chunk of a `ModelStreamInfer` call, in order.
async fn stream_chunks(
    client: &mut common::ExtClient,
    req: ModelInferRequest,
) -> std::result::Result<Vec<ModelStreamInferResponse>, tonic::Status> {
    let mut stream = client.model_stream_infer(req).await?.into_inner();
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.message().await? {
        chunks.push(chunk);
    }
    Ok(chunks)
}

#[tokio::test]
async fn model_stream_infer_sends_a_chunk_per_step_and_the_counts_on_the_last() {
    let mut c = common::ext_client(common::engine()).await;
    let mut req = request("chat", vec![bytes_input("prompt", &["hello"])]);
    req.parameters.insert("max_new_tokens".into(), int_param(4));
    let chunks = stream_chunks(&mut c, req).await.expect("ModelStreamInfer on chat");
    assert!(chunks.len() > 1, "a four-token generation arrived in one chunk: {chunks:?}");
    for chunk in &chunks {
        assert_eq!(chunk.error_message, "", "a chunk carries an error");
        let response = chunk.infer_response.as_ref().expect("every chunk carries an infer_response");
        assert_eq!(response.model_name, "chat", "the chunk names another model");
        assert_eq!(response.model_version, "1", "the chunk names another version");
        assert_eq!(chunk_text(chunk).len(), 1, "every chunk holds one `text` string");
    }
    // The steps before the last carry the new text; the last one closes the
    // generation and reports what it cost.
    let (last, steps) = chunks.split_last().expect("at least one chunk");
    for chunk in steps {
        assert!(!chunk_text(chunk)[0].is_empty(), "a step generated no text");
        assert!(
            !chunk.infer_response.as_ref().expect("an infer_response").parameters.contains_key("finish_reason"),
            "a step before the last already reports a finish reason"
        );
    }
    assert_eq!(chunk_param(last, "finish_reason"), "length", "4 new tokens hits the budget");
    assert_eq!(chunk_param(last, "generated_tokens"), "4", "max_new_tokens 4 was honored");
    assert_eq!(chunk_param(last, "prompt_tokens"), "4", "the prompt is one `hello` turn, templated and tokenized");

    // The chunks join to exactly what the unstreamed call returns.
    let mut plain = common::grpc_client(common::engine()).await;
    let mut req = request("chat", vec![bytes_input("prompt", &["hello"])]);
    req.parameters.insert("max_new_tokens".into(), int_param(4));
    let whole = plain.model_infer(req).await.expect("ModelInfer on chat").into_inner();
    let joined: String = chunks.iter().map(|c| chunk_text(c).remove(0)).collect();
    let text = whole.outputs.iter().find(|o| o.name == "text").expect("a `text` output");
    let text = String::from_utf8_lossy(&text.contents.as_ref().expect("text contents").bytes_contents[0]).into_owned();
    assert_eq!(joined, text, "the streamed chunks do not join to the unstreamed reply");
}

#[tokio::test]
async fn model_stream_infer_on_a_non_generative_model_sends_one_whole_response() {
    let mut c = common::ext_client(common::engine()).await;
    let chunks = stream_chunks(&mut c, request("embed", vec![bytes_input("text", &["one", "two"])]))
        .await
        .expect("ModelStreamInfer on embed");
    assert_eq!(chunks.len(), 1, "an embedding model answers in one response");
    assert_eq!(chunks[0].error_message, "", "the response carries an error");
    let response = chunks[0].infer_response.as_ref().expect("an infer_response");
    let embeddings = response.outputs.iter().find(|o| o.name == "embeddings").unwrap_or_else(|| {
        panic!("no `embeddings` output; outputs: {:?}", response.outputs.iter().map(|o| &o.name).collect::<Vec<_>>())
    });
    assert_eq!(embeddings.shape.len(), 2, "the embeddings shape is [texts, dim]");
    assert_eq!(embeddings.shape[0], 2, "both texts came back");
    let floats = &embeddings.contents.as_ref().expect("embeddings contents").fp32_contents;
    assert_eq!(floats.len() as i64, embeddings.shape[0] * embeddings.shape[1], "the data does not fill the shape");
    // The one response is the response: the same vectors `ModelInfer`
    // returns for the same request, not a differently shaped stand-in.
    let mut plain = common::grpc_client(common::engine()).await;
    let whole = plain
        .model_infer(request("embed", vec![bytes_input("text", &["one", "two"])]))
        .await
        .expect("ModelInfer on embed")
        .into_inner();
    let want = whole.outputs.iter().find(|o| o.name == "embeddings").expect("an `embeddings` output");
    assert_eq!(embeddings.shape, want.shape, "the streamed shape differs from the unary one");
    assert_eq!(
        floats,
        &want.contents.as_ref().expect("embeddings contents").fp32_contents,
        "the streamed vectors differ from the unary ones"
    );
}

#[tokio::test]
async fn a_provider_error_mid_stream_arrives_as_error_message_on_a_chunk() {
    // `server/README.md`: `error_message` is set when the stream ended in an
    // error. The mock rejects a top_k past its vocabulary on the
    // generation's own task, after the stream has already been handed to the
    // client, so the failure can only reach it on a chunk.
    let mut c = common::ext_client(common::engine()).await;
    let mut req = request("chat", vec![bytes_input("prompt", &["hello"])]);
    req.parameters.insert("top_k".into(), int_param(5000));
    let chunks = stream_chunks(&mut c, req).await.expect("the stream itself starts");
    let failed: Vec<&ModelStreamInferResponse> = chunks.iter().filter(|c| !c.error_message.is_empty()).collect();
    assert_eq!(failed.len(), 1, "exactly one chunk carries the failure: {chunks:?}");
    let failed = failed[0];
    assert!(
        failed.error_message.contains("TURBO_E_INVALID_ARGUMENT"),
        "the chunk does not carry the Turbo status: {}",
        failed.error_message
    );
    assert!(
        failed.error_message.contains("field 6"),
        "the chunk does not name top_k, generation field 6: {}",
        failed.error_message
    );
    assert!(failed.infer_response.is_none(), "a failed chunk carries no result");
    assert_eq!(
        chunks.last().map(|c| c.error_message.as_str()),
        Some(failed.error_message.as_str()),
        "the failure is the last thing the stream says"
    );
}

#[tokio::test]
async fn model_stream_infer_on_an_unknown_model_is_not_found() {
    let mut c = common::ext_client(common::engine()).await;
    let err = stream_chunks(&mut c, request("nope", vec![bytes_input("prompt", &["hello"])]))
        .await
        .expect_err("ModelStreamInfer on an unknown model is an error");
    assert_eq!(err.code(), Code::NotFound, "{err}");
    assert!(err.message().contains("no model named `nope`"), "the status does not name the model: {err}");
}

#[tokio::test]
async fn repository_index_lists_every_served_model() {
    let mut c = common::ext_client(common::engine()).await;
    let index = c.repository_index(RepositoryIndexRequest {}).await.expect("RepositoryIndex").into_inner();
    let names: Vec<&str> = index.models.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["chat", "classify", "embed", "rerank", "run", "tag"], "RepositoryIndex.models, in name order");
    for m in &index.models {
        assert_eq!(m.version, "1", "model `{}` version", m.name);
        assert_eq!(m.state, "READY", "model `{}` state", m.name);
        assert_eq!(m.reason, "", "a READY model has no reason: `{}`", m.name);
        assert_eq!(m.provider, "mock", "model `{}` provider", m.name);
        assert_eq!(m.device, "Mock accelerator", "model `{}` device", m.name);
    }
    let kinds: Vec<(&str, &str)> = index.models.iter().map(|m| (m.name.as_str(), m.kind.as_str())).collect();
    assert_eq!(
        kinds,
        [
            ("chat", "Generative"),
            ("classify", "Classifier"),
            ("embed", "Embedding"),
            ("rerank", "Reranker"),
            ("run", "Generic"),
            ("tag", "TokenClassifier"),
        ],
        "RepositoryIndex.models kinds"
    );
    let embed = index.models.iter().find(|m| m.name == "embed").expect("`embed` in the index");
    assert_eq!(embed.bundle, common::bundle("embedding"), "the index names the bundle it was loaded from");
}

#[tokio::test]
async fn repository_load_serves_a_new_model_and_unload_takes_it_away() {
    // Its own engine: loading and unloading must not disturb the shared one.
    let engine =
        common::engine_of(&[&format!("name=embed,bundle={},provider=mock,ordinal=1", common::bundle("embedding"))]);
    let addr = common::serve_grpc(engine).await;
    let channel = common::grpc_channel(addr).await;
    let mut c = common::ExtClient::new(channel.clone());
    let mut oip =
        turbo_inferstream::grpc::inference::grpc_inference_service_client::GrpcInferenceServiceClient::new(channel);

    let loaded = c
        .repository_model_load(RepositoryModelLoadRequest {
            model_name: "rerank-2".into(),
            bundle: common::bundle("reranker"),
            provider: "mock".into(),
            ordinal: 1,
            buckets: Vec::new(),
            sessions: 0,
            generations: 0,
        })
        .await
        .expect("RepositoryModelLoad of the reranker bundle")
        .into_inner();
    assert_eq!(loaded.model_name, "rerank-2", "the load answers with the name it served");

    let ready = oip
        .model_ready(ModelReadyRequest { name: "rerank-2".into(), version: String::new() })
        .await
        .expect("ModelReady after the load")
        .into_inner();
    assert!(ready.ready, "the freshly loaded model is not ready");

    let index = c.repository_index(RepositoryIndexRequest {}).await.expect("RepositoryIndex").into_inner();
    let names: Vec<&str> = index.models.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["embed", "rerank-2"], "the index does not show the loaded model");

    let r = oip
        .model_infer(request(
            "rerank-2",
            vec![bytes_input("query", &["a query"]), bytes_input("documents", &["one", "two", "three"])],
        ))
        .await
        .expect("ModelInfer on the freshly loaded model")
        .into_inner();
    let scores = r.outputs.iter().find(|o| o.name == "scores").expect("a `scores` output");
    assert_eq!(scores.shape, [3], "one score per document");

    c.repository_model_unload(RepositoryModelUnloadRequest { model_name: "rerank-2".into() })
        .await
        .expect("RepositoryModelUnload");
    let err = oip
        .model_ready(ModelReadyRequest { name: "rerank-2".into(), version: String::new() })
        .await
        .expect_err("ModelReady after the unload is an error");
    assert_eq!(err.code(), Code::NotFound, "{err}");

    let index = c.repository_index(RepositoryIndexRequest {}).await.expect("RepositoryIndex").into_inner();
    let names: Vec<&str> = index.models.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["embed"], "the unloaded model is still in the index");
}

#[tokio::test]
async fn repository_load_of_a_served_name_is_rejected() {
    let engine =
        common::engine_of(&[&format!("name=embed,bundle={},provider=mock,ordinal=1", common::bundle("embedding"))]);
    let mut c = common::ext_client(engine).await;
    let err = c
        .repository_model_load(RepositoryModelLoadRequest {
            model_name: "embed".into(),
            bundle: common::bundle("reranker"),
            provider: "mock".into(),
            ordinal: 1,
            buckets: Vec::new(),
            sessions: 0,
            generations: 0,
        })
        .await
        .expect_err("loading a name that is already served is an error");
    assert_eq!(err.code(), Code::InvalidArgument, "{err}");
    assert!(err.message().contains("already served"), "the status does not say the name is taken: {err}");
    // The model that was there is the one still served.
    let index = c.repository_index(RepositoryIndexRequest {}).await.expect("RepositoryIndex").into_inner();
    let kinds: Vec<(&str, &str)> = index.models.iter().map(|m| (m.name.as_str(), m.kind.as_str())).collect();
    assert_eq!(kinds, [("embed", "Embedding")], "the rejected load replaced the served model");
}

#[tokio::test]
async fn repository_load_of_a_missing_bundle_names_the_turbo_status() {
    let engine =
        common::engine_of(&[&format!("name=embed,bundle={},provider=mock,ordinal=1", common::bundle("embedding"))]);
    let mut c = common::ext_client(engine).await;
    let err = c
        .repository_model_load(RepositoryModelLoadRequest {
            model_name: "gone".into(),
            bundle: format!("{}-missing", common::bundle("reranker")),
            provider: "mock".into(),
            ordinal: 1,
            buckets: Vec::new(),
            sessions: 0,
            generations: 0,
        })
        .await
        .expect_err("loading a bundle directory that is not there is an error");
    assert_eq!(err.code(), Code::NotFound, "{err}");
    assert!(err.message().contains("TURBO_E_BUNDLE_NOT_FOUND"), "the status does not carry the Turbo code name: {err}");
}

#[tokio::test]
async fn repository_unload_of_an_unknown_model_is_not_found() {
    let mut c = common::ext_client(common::engine()).await;
    let err = c
        .repository_model_unload(RepositoryModelUnloadRequest { model_name: "nope".into() })
        .await
        .expect_err("unloading a model that is not served is an error");
    assert_eq!(err.code(), Code::NotFound, "{err}");
    assert!(err.message().contains("no model named `nope`"), "the status does not name the model: {err}");
}

// ---------------------------------------------------------------------------
// Reflection
// ---------------------------------------------------------------------------

/// A receiver as the request `Stream` the reflection client takes, without
/// another crate.
struct Requests {
    rx: tokio::sync::mpsc::Receiver<ServerReflectionRequest>,
}

impl tonic::codegen::tokio_stream::Stream for Requests {
    type Item = ServerReflectionRequest;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

/// A run of reflection queries over one connection, each answered or a panic.
async fn reflect(
    requests: Vec<server_reflection_request::MessageRequest>,
) -> Vec<server_reflection_response::MessageResponse> {
    let addr = common::serve_grpc(common::engine()).await;
    let mut client = ServerReflectionClient::new(common::grpc_channel(addr).await);
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    let mut stream = client.server_reflection_info(Requests { rx }).await.expect("ServerReflectionInfo").into_inner();
    let mut answers = Vec::new();
    for request in requests {
        tx.send(ServerReflectionRequest { host: String::new(), message_request: Some(request) })
            .await
            .expect("send the reflection request");
        let response = stream
            .message()
            .await
            .expect("a reflection response")
            .expect("the reflection stream ended without a response");
        answers.push(response.message_response.expect("the reflection response carries no answer"));
    }
    answers
}

/// The files of a `FileDescriptorResponse`, decoded.
fn descriptors(answer: server_reflection_response::MessageResponse) -> Vec<prost_types::FileDescriptorProto> {
    match answer {
        server_reflection_response::MessageResponse::FileDescriptorResponse(r) => r
            .file_descriptor_proto
            .iter()
            .map(|bytes| {
                <prost_types::FileDescriptorProto as prost::Message>::decode(bytes.as_slice())
                    .expect("a serialized FileDescriptorProto")
            })
            .collect(),
        other => panic!("the query was not answered with descriptors: {other:?}"),
    }
}

#[tokio::test]
async fn reflection_lists_the_protocol_service_and_the_extension() {
    let answer = reflect(vec![server_reflection_request::MessageRequest::ListServices(String::new())]).await.remove(0);
    let mut names = match answer {
        server_reflection_response::MessageResponse::ListServicesResponse(r) => {
            r.service.into_iter().map(|s| s.name).collect::<Vec<_>>()
        }
        other => panic!("ListServices was not answered with a service list: {other:?}"),
    };
    names.sort();
    assert!(
        names.iter().any(|n| n == "inference.GRPCInferenceService"),
        "reflection does not list the protocol service: {names:?}"
    );
    assert!(
        names.iter().any(|n| n == "turbo.inferstream.InferstreamExtension"),
        "reflection does not list the extension service: {names:?}"
    );
}

#[tokio::test]
async fn reflection_answers_for_the_extension_and_serves_the_file_it_imports() {
    let mut answers = reflect(vec![
        server_reflection_request::MessageRequest::FileContainingSymbol(
            "turbo.inferstream.InferstreamExtension".to_string(),
        ),
        server_reflection_request::MessageRequest::FileByFilename("open_inference_grpc.proto".to_string()),
    ])
    .await;
    let imported = descriptors(answers.remove(1));
    let extension = descriptors(answers.remove(0));

    // tonic-reflection 0.14 answers a symbol with the one file that declares
    // it, not with its transitive imports; the file names what it imports and
    // the import is served by name, which is how grpcurl and the KServe
    // clients walk the set.
    assert_eq!(extension.len(), 1, "the answer holds more than the declaring file");
    let extension = &extension[0];
    assert_eq!(extension.name(), "turbo_inferstream.proto", "the wrong file declares the extension service");
    assert_eq!(extension.package(), "turbo.inferstream", "the extension file's package");
    let services: Vec<&str> = extension.service.iter().map(|s| s.name()).collect();
    assert_eq!(services, ["InferstreamExtension"], "the extension file's services");
    assert_eq!(
        extension.dependency,
        ["open_inference_grpc.proto"],
        "the extension file does not name the protocol file it imports"
    );

    assert_eq!(imported.len(), 1, "the import was answered with more than one file");
    let imported = &imported[0];
    assert_eq!(imported.name(), "open_inference_grpc.proto", "the import came back under another name");
    assert_eq!(imported.package(), "inference", "the imported file's package");
    let services: Vec<&str> = imported.service.iter().map(|s| s.name()).collect();
    assert_eq!(services, ["GRPCInferenceService"], "the imported file's services");
}
