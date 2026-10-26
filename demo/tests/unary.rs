//! Unary Connect calls through the demo to the real server: the page, the
//! server's own bytes back, its refusals with their turbo-code and
//! turbo-field, and a server that is not there.

mod common;

use std::collections::HashMap;

use common::*;
use turbo_kserve::Serve;
use turbo_kserve::api::{TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_CAPACITY};
use turbo_kserve::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use turbo_kserve::proto::infer_parameter::ParameterChoice;
use turbo_kserve::proto::model_infer_request::InferInputTensor;
use turbo_kserve::proto::result::{CapabilitiesRequest, CapabilitiesResponse};
use turbo_kserve::proto::search::{CollectionEvent, WatchCollectionRequest};
use turbo_kserve::proto::*;

const NAME: &str = "tiny-bert-bundle";
const INFER: &str = "/inference.GRPCInferenceService/ModelInfer";
const METADATA: &str = "/inference.GRPCInferenceService/ModelMetadata";

fn texts(texts: &[&str], parameters: &[(&str, i64)]) -> ModelInferRequest {
    ModelInferRequest {
        model_name: NAME.into(),
        parameters: parameters
            .iter()
            .map(|(k, v)| (k.to_string(), InferParameter { parameter_choice: Some(ParameterChoice::Int64Param(*v)) }))
            .collect::<HashMap<_, _>>(),
        inputs: vec![InferInputTensor {
            name: "texts".into(),
            datatype: "BYTES".into(),
            shape: vec![texts.len() as i64],
            contents: Some(InferTensorContents {
                bytes_contents: texts.iter().map(|t| t.as_bytes().to_vec()).collect(),
                ..Default::default()
            }),
            ..Default::default()
        }],
        ..Default::default()
    }
}

#[tokio::test]
async fn the_page_is_served_and_only_the_page_answers_get() {
    let s = server(&[NAME], Serve::default()).await;
    let d = demo(&s, &[]).await;
    for (path, kind, has) in [
        ("/", "text/html; charset=utf-8", "<script src=\"app.js\"></script>"),
        ("/app.js", "text/javascript; charset=utf-8", "function"),
        ("/style.css", "text/css; charset=utf-8", "prefers-color-scheme"),
    ] {
        let r = http(&d, "GET", path, &[], b"").await;
        assert_eq!((r.status, r.header("content-type")), (200, Some(kind)), "{path}");
        let body = String::from_utf8(r.all().await).unwrap();
        assert!(body.contains(has), "{path}");
        // Nothing is loaded from anywhere but the demo.
        assert!(!body.contains("http://") && !body.contains("https://"), "{path} names another host");
    }
    assert_eq!(http(&d, "GET", "/nothing", &[], b"").await.status, 404);
    assert_eq!(http(&d, "GET", "/a/b/c", &[], b"").await.status, 404);
    // The JSON calls are gone: JSON is 415, and the path is no method.
    let r = http(&d, "POST", "/api/infer", &[("Content-Type", "application/json")], b"{}").await;
    assert_eq!(r.status, 415);
    let r = http(&d, "POST", "/api/infer", &[("Content-Type", UNARY)], b"").await;
    assert_eq!(r.status, 501);
    // A method's path answers POST only.
    let r = http(&d, "GET", INFER, &[], b"").await;
    assert_eq!((r.status, r.header("allow")), (405, Some("POST")));
}

#[tokio::test]
async fn a_unary_call_is_the_servers_own_reply() {
    let s = server(&[NAME, "tiny-static-bundle"], Serve::default()).await;
    let d = demo(&s, &[]).await;
    let req = texts(&["a cat", "a dog"], &[]);
    let got: ModelInferResponse = ok(&d, INFER, &req).await;
    let mut grpc = GrpcInferenceServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let want = grpc.model_infer(req).await.unwrap().into_inner();
    assert_eq!((&got.model_name, &got.outputs), (&want.model_name, &want.outputs));
    // Exactly the server's floats, as the bytes it sent.
    assert_eq!(got.raw_output_contents, want.raw_output_contents);
    assert_eq!(got.raw_output_contents[0].len(), 2 * 32 * 4);

    let m: ModelMetadataResponse =
        ok(&d, METADATA, &ModelMetadataRequest { name: NAME.into(), version: "".into() }).await;
    assert_eq!((m.name.as_str(), m.versions.as_slice()), (NAME, ["1".to_string()].as_slice()));
    let caps: CapabilitiesResponse =
        ok(&d, "/turbo.result.v1.AnalysisService/Capabilities", &CapabilitiesRequest {}).await;
    assert_eq!(caps.bundles.len(), 2);
    let live: ServerLiveResponse = ok(&d, "/inference.GRPCInferenceService/ServerLive", &ServerLiveRequest {}).await;
    assert!(live.live);
}

#[tokio::test]
async fn a_unary_refusal_is_the_connect_error_with_the_servers_turbo_metadata() {
    let s = server(&[NAME], Serve::default()).await;
    let d = demo(&s, &[]).await;
    let mut grpc = GrpcInferenceServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();

    let req = ModelMetadataRequest { name: "no-such-model".into(), version: "".into() };
    let want = grpc.model_metadata(req.clone()).await.unwrap_err();
    let (r, body) = unary(&d, METADATA, &req).await;
    assert_eq!((r.status, r.header("content-type")), (404, Some("application/json")));
    let body = String::from_utf8(body).unwrap();
    assert!(body.starts_with(r#"{"code":"not_found","message":""#), "{body}");
    assert!(body.contains("no-such-model"), "{body}");
    assert_eq!(r.header("turbo-code"), Some(TURBO_E_BUNDLE_NOT_FOUND.to_string().as_str()));
    assert_eq!(r.header("turbo-field"), want.metadata().get("turbo-field").map(|v| v.to_str().unwrap()));

    // Over the session's max_seq: the library's TURBO_E_CAPACITY.
    let (r, body) = unary(&d, INFER, &texts(&["a cat"], &[("max_tokens", 100_000)])).await;
    let body = String::from_utf8(body).unwrap();
    assert_eq!(r.status, 400, "{body}");
    assert!(body.contains(r#""code":"out_of_range""#) && body.contains("TURBO_E_CAPACITY"), "{body}");
    assert_eq!(r.header("turbo-code"), Some(TURBO_E_CAPACITY.to_string().as_str()));
    assert!(r.header("turbo-field").is_some());
}

#[tokio::test]
async fn a_server_that_is_not_there_is_unavailable() {
    // A port nothing listens on, held for the test so no other server
    // can be given it.
    let reserved = tokio::net::TcpSocket::new_v4().unwrap();
    reserved.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let d = demo_at(reserved.local_addr().unwrap(), &[]).await;
    let (r, body) = unary(&d, INFER, &texts(&["a"], &[])).await;
    assert_eq!(r.status, 503);
    assert!(String::from_utf8(body).unwrap().starts_with(r#"{"code":"unavailable""#));
    let watch = WatchCollectionRequest { collection_name: "c".into(), min_oov_ratio: 0.0 };
    let (_, end) = stream(&d, "/turbo.search.v1.TurboSearchService/WatchCollection", &watch)
        .await
        .frames::<CollectionEvent>()
        .await;
    assert!(end.starts_with(r#"{"error":{"code":"unavailable""#), "{end}");
    d.stop().await;
}
