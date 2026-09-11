//! Minimal inferstream client example.
//!
//! Start the server first:
//!
//! ```text
//! cargo run -p inferstream-server -- --config config/example.toml
//! ```
//!
//! then run this example:
//!
//! ```text
//! cargo run -p inferstream-server --example client -- http://127.0.0.1:8461
//! # with auth enabled on the server:
//! INFERSTREAM_API_KEY=dev-key cargo run -p inferstream-server --example client -- http://127.0.0.1:8461
//! ```

use std::collections::HashMap;

use tonic::metadata::MetadataValue;
use tonic::Request;

use inferstream_protocol::inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use inferstream_protocol::inference::model_infer_request::InferInputTensor;
use inferstream_protocol::inference::{ModelInferRequest, ServerLiveRequest};
use inferstream_protocol::tensor::{pack_bytes, unpack_bytes, unpack_fp32};

fn text_request(model: &str, id: &str, text: &str) -> ModelInferRequest {
    ModelInferRequest {
        model_name: model.to_string(),
        id: id.to_string(),
        inputs: vec![InferInputTensor {
            name: "text".to_string(),
            datatype: "BYTES".to_string(),
            shape: vec![1],
            parameters: HashMap::new(),
            contents: None,
        }],
        raw_input_contents: vec![pack_bytes(&[text.as_bytes()])],
        ..Default::default()
    }
}

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
    let endpoint = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "http://127.0.0.1:8461".to_string());
    let api_key = std::env::var("INFERSTREAM_API_KEY").ok();

    let mut client = GrpcInferenceServiceClient::connect(endpoint.clone()).await?;
    println!("connected to {endpoint}");

    let live = client
        .server_live(with_auth(ServerLiveRequest {}, &api_key))
        .await?
        .into_inner();
    println!("ServerLive: {}", live.live);

    // Unary embedding.
    let response = client
        .model_infer(with_auth(
            text_request("mock-embed", "example-unary", "the quick brown fox"),
            &api_key,
        ))
        .await?
        .into_inner();
    let embedding = unpack_fp32(&response.raw_output_contents[0])?;
    println!(
        "ModelInfer id={} -> {} dims, first 4: {:?}",
        response.id,
        embedding.len(),
        &embedding[..4.min(embedding.len())]
    );

    // Bidi streaming generation.
    let requests = vec![
        text_request("mock-generate", "example-stream-1", "stream me"),
        text_request("mock-generate", "example-stream-2", "me too"),
    ];
    let mut stream = client
        .model_stream_infer(with_auth(tokio_stream::iter(requests), &api_key))
        .await?
        .into_inner();
    while let Some(message) = stream.message().await? {
        if !message.error_message.is_empty() {
            eprintln!("stream error: {}", message.error_message);
            continue;
        }
        let response = message.infer_response.expect("response set on success");
        let tokens = unpack_bytes(&response.raw_output_contents[0])?;
        println!(
            "ModelStreamInfer id={} token={}",
            response.id,
            String::from_utf8_lossy(&tokens[0])
        );
    }
    println!("stream closed");
    Ok(())
}
