//! CPU-only GLiNER packages must not report readiness on a named accelerator.

mod common;
#[path = "gliner_progressive/fixture.rs"]
mod gliner;
#[path = "gliner2_progressive/fixture.rs"]
mod gliner2;

use std::path::PathBuf;

use common::{cpu_name, load_req, refusal, serve};
use tonic::Code;
use turbo_kserve::api::TURBO_E_DEVICE_NOT_FOUND;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::device::Named;
use turbo_kserve::proto::health::HealthCheckRequest;
use turbo_kserve::proto::health::health_check_response::ServingStatus;
use turbo_kserve::proto::health::health_client::HealthClient;
use turbo_kserve::proto::result::{CapabilitiesRequest, DeviceBackend, DeviceName};
use turbo_kserve::{Serve, Server};

async fn startup_refuses(name: &str, path: PathBuf, device: Named, expected: &str) {
    let model = ModelConfig {
        name: Some(name.into()),
        bundle: path.to_str().unwrap().into(),
        device: Device::Named(device),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    };
    let server = Server::start("127.0.0.1:0".parse().unwrap(), vec![model], MAX, Serve::default()).await.unwrap();
    let result = server.load().await;
    let mut health = HealthClient::connect(format!("http://{}", server.local_addr())).await.unwrap();
    let status = health.check(HealthCheckRequest { service: name.into() }).await.unwrap().into_inner().status();
    server.stop().await;
    assert_eq!(status, ServingStatus::NotServing, "{name} reported ready: {result:?}");
    let error = result.unwrap_err();
    assert!(error.contains(expected) && error.contains("TURBO_E_DEVICE_NOT_FOUND"), "{error}");
}

#[tokio::test]
async fn startup_refuses_named_npu_for_gliner() {
    startup_refuses("gliner", gliner::package(), Named::parse("npu:0").unwrap(), "npu:0").await;
}

#[tokio::test]
async fn startup_refuses_named_npu_for_gliner2() {
    startup_refuses("gliner2", gliner2::package(), Named::parse("npu:0").unwrap(), "npu:0").await;
}

#[tokio::test]
async fn startup_refuses_invalid_cpu_ordinal() {
    // Unlike CLI parsing, the Rust startup API accepts a constructed Named.
    startup_refuses("gliner", gliner::package(), Named { backend: "cpu", ordinal: 7 }, "cpu:7").await;
}

async fn load_bundle_cpu_modes(name: &str, path: PathBuf) {
    let (server, mut client) = serve().await;
    let mut health = HealthClient::connect(format!("http://{}", server.local_addr())).await.unwrap();
    let mut req = load_req(name, path.to_str().unwrap(), 1, 0);
    req.device_name = Some(DeviceName { backend: DeviceBackend::Cuda as i32, ordinal: 0 });
    let result = client.load_bundle(req.clone()).await;
    let caps = client.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert!(caps.bundles.iter().all(|b| b.name != name), "{name} registered: {result:?}");
    let error = result.unwrap_err();
    assert_eq!(refusal(&error), (Code::FailedPrecondition, TURBO_E_DEVICE_NOT_FOUND), "{error:?}");
    assert!(error.message().contains("cuda:0"), "{error:?}");
    let status = health.check(HealthCheckRequest { service: name.into() }).await.unwrap_err();
    assert_eq!(status.code(), Code::NotFound);

    for (mode, device, select) in [("cpu", cpu_name(), false), ("auto", None, true), ("legacy", None, false)] {
        req.name = format!("{name}-{mode}");
        req.device_name = device;
        req.select_device = select;
        client.load_bundle(req.clone()).await.unwrap();
        let status =
            health.check(HealthCheckRequest { service: req.name.clone() }).await.unwrap().into_inner().status();
        assert_eq!(status, ServingStatus::Serving, "{name} with {mode}");
    }
    server.stop().await;
}

#[tokio::test]
async fn load_bundle_refuses_named_cuda_but_keeps_gliner_cpu_modes() {
    load_bundle_cpu_modes("gliner", gliner::package()).await;
}

#[tokio::test]
async fn load_bundle_refuses_named_cuda_but_keeps_gliner2_cpu_modes() {
    load_bundle_cpu_modes("gliner2", gliner2::package()).await;
}
