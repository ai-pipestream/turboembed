//! Unary embedding smoke test against inferstream-intel routing to OVMS.
//!
//! Start the server first:
//!
//! ```text
//! cargo run -p inferstream-arch-intel -- --config config/intel.toml
//! ```
//!
//! then request a real embedding through the façade (the model name must be
//! an OVMS-served pipeline from the config, e.g. `minilm_pipeline`):
//!
//! ```text
//! INFERSTREAM_API_KEY=change-me cargo run -p inferstream-arch-intel \
//!   --example ovms_embed -- http://127.0.0.1:8461 minilm_pipeline "hello world"
//! ```
//!
//! To hit OVMS directly (bypassing the façade), point the endpoint at the
//! OVMS gRPC port and unset the API key.

use inferstream_protocol::inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use inferstream_protocol::inference::model_infer_request::InferInputTensor;
use inferstream_protocol::inference::{ModelInferRequest, ModelReadyRequest};
use inferstream_protocol::tensor::{pack_bytes, unpack_fp32};
use tonic::metadata::MetadataValue;
use tonic::Request;

fn with_auth<T>(message: T, api_key: &Option<String>) -> Request<T> {
    let mut request = Request::new(message);
    if let Some(key) = api_key {
        request.metadata_mut().insert(
            "authorization",
            MetadataValue::try_from(format!("Bearer {key}")).expect("valid metadata"),
        );
    }
    request
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let endpoint = args
        .next()
        .unwrap_or_else(|| "http://127.0.0.1:8461".to_string());
    let model = args.next().unwrap_or_else(|| "minilm_pipeline".to_string());
    let text = args
        .next()
        .unwrap_or_else(|| "the quick brown fox".to_string());
    let api_key = std::env::var("INFERSTREAM_API_KEY").ok();

    let mut client = GrpcInferenceServiceClient::connect(endpoint.clone()).await?;
    println!("connected to {endpoint}");

    let ready = client
        .model_ready(with_auth(
            ModelReadyRequest {
                name: model.clone(),
                version: String::new(),
            },
            &api_key,
        ))
        .await?
        .into_inner();
    println!("ModelReady({model}): {}", ready.ready);

    let request = ModelInferRequest {
        model_name: model.clone(),
        id: "ovms-embed-smoke".to_string(),
        inputs: vec![InferInputTensor {
            name: "strings".to_string(),
            datatype: "BYTES".to_string(),
            shape: vec![1],
            ..Default::default()
        }],
        raw_input_contents: vec![pack_bytes(&[text.as_bytes()])],
        ..Default::default()
    };
    let response = client
        .model_infer(with_auth(request, &api_key))
        .await?
        .into_inner();

    let output = response
        .outputs
        .first()
        .map(|o| format!("{} {:?} {}", o.name, o.shape, o.datatype))
        .unwrap_or_else(|| "<no output metadata>".to_string());
    let embedding = unpack_fp32(&response.raw_output_contents[0])?;
    println!(
        "ModelInfer id={} model={} output=[{output}] -> {} floats, first 4: {:?}",
        response.id,
        response.model_name,
        embedding.len(),
        &embedding[..4.min(embedding.len())]
    );
    Ok(())
}
