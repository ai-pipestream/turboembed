//! A model loads on the device it names, or not at all (#2508).
//!
//! A runtime index is a position in the runtime's device list, and when a
//! linked backend lists no device every later device moves up: on a build
//! with a GPU backend and no GPU, device 0 is the CPU. A device named by
//! its backend and ordinal (`cuda:0`, `cpu`) loads there or is refused
//! with DEVICE_NOT_FOUND, and a typed index is refused while a linked
//! backend lists no device, as the numbers have moved.

mod common;

use common::*;
use tonic::Code;
use turbo_kserve::api::{Runtime, TURBO_E_DEVICE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT, field, version};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::device::Named;
use turbo_kserve::proto::ModelMetadataRequest;
use turbo_kserve::proto::result::{CapabilitiesRequest, DeviceBackend, DeviceName};
use turbo_kserve::{Serve, Server};

fn config(device: Device) -> ModelConfig {
    ModelConfig {
        bundle: tiny().to_str().unwrap().to_string(),
        name: None,
        device,
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

/// What loading the tiny bundle on `device` at startup says.
async fn start_on(device: Device) -> (Server, Result<(), String>) {
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![config(device)], MAX, Serve::default()).await.unwrap();
    let r = s.load().await;
    (s, r)
}

/// The backend the served tiny bundle runs on, read back from the runtime
/// through ModelMetadata.
async fn served_backend(s: &Server) -> String {
    let mut k = kserve(s).await;
    let m = k.model_metadata(ModelMetadataRequest { name: NAME.into(), version: String::new() }).await.unwrap();
    m.into_inner().properties["device_info.backend"].clone()
}

/// How many devices the runtime lists for `backend`.
fn listed(backend: &str) -> u32 {
    let rt = Runtime::create().unwrap();
    let n = rt.device_count().unwrap();
    (0..n).filter(|&i| field(&rt.device_info(i).unwrap().backend) == backend).count() as u32
}

/// A CUDA ordinal no machine has, so the refusal is the same whether or
/// not this build links CUDA.
const ABSENT: Named = Named { backend: "cuda", ordinal: 4096 };

#[tokio::test]
async fn the_named_cpu_loads_on_the_cpu() {
    let (s, r) = start_on(Device::Named(Named::parse("cpu").unwrap())).await;
    r.unwrap();
    assert_eq!(served_backend(&s).await, "cpu");
    s.stop().await;
}

#[tokio::test]
async fn a_named_device_the_runtime_does_not_list_is_refused_at_startup() {
    let (s, r) = start_on(Device::Named(ABSENT)).await;
    let e = r.unwrap_err();
    assert!(e.contains("device cuda:4096"), "{e}");
    assert!(e.contains("TURBO_E_DEVICE_NOT_FOUND"), "{e}");
    s.stop().await;
}

#[tokio::test]
async fn load_bundle_takes_a_device_name_and_refuses_one_not_listed() {
    let (s, mut c) = serve().await;
    let mut req = load_req("absent", tiny().to_str().unwrap(), 1, 0);
    req.device = 0;
    req.device_name = Some(DeviceName { backend: DeviceBackend::Cuda as i32, ordinal: ABSENT.ordinal });
    let e = c.load_bundle(req.clone()).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::FailedPrecondition, TURBO_E_DEVICE_NOT_FOUND), "{e:?}");
    assert!(e.message().contains("device cuda:4096"), "{e:?}");
    // Nothing was registered: the model did not load anywhere else.
    let caps = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert!(caps.bundles.iter().all(|b| b.name != "absent"), "{caps:?}");

    req.name = "on-cpu".into();
    req.device_name = Some(DeviceName { backend: DeviceBackend::Cpu as i32, ordinal: 0 });
    c.load_bundle(req.clone()).await.unwrap();
    let caps = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert!(caps.bundles.iter().any(|b| b.name == "on-cpu"), "{caps:?}");
    s.stop().await;
}

#[tokio::test]
async fn load_bundle_refuses_a_device_name_beside_another_choice() {
    let (s, mut c) = serve().await;
    let cpu_name = Some(DeviceName { backend: DeviceBackend::Cpu as i32, ordinal: 0 });
    let mut req = load_req("both", tiny().to_str().unwrap(), 1, 0);
    req.device = 0;
    req.device_name = cpu_name;
    req.select_device = true;
    let e = c.load_bundle(req.clone()).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT), "{e:?}");
    req.select_device = false;
    req.device = 1;
    let e = c.load_bundle(req.clone()).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT), "{e:?}");
    req.device = 0;
    req.device_name = Some(DeviceName { backend: DeviceBackend::Unspecified as i32, ordinal: 0 });
    let e = c.load_bundle(req).await.unwrap_err();
    assert_eq!(e.code(), Code::InvalidArgument, "{e:?}");
    s.stop().await;
}

/// The issue's case, on a build that links a backend with nothing to
/// list. The npu backend opens its loader at run time, so it builds on any
/// host, and a host without an Intel NPU lists no npu device: the runtime
/// then lists the CPU at index 0, where `device=0` used to load the model
/// and report it ready.
#[tokio::test]
async fn a_typed_index_is_refused_while_a_linked_backend_lists_no_device() {
    let linked = version();
    let empty: Vec<&str> = linked.split_whitespace().skip(1).filter(|b| listed(b) == 0).collect();
    if empty.is_empty() {
        // Every linked backend lists a device: the numbers are where they
        // were, and device 0 loads on the device it names.
        let rt = Runtime::create().unwrap();
        let want = field(&rt.device_info(0).unwrap().backend);
        let (s, r) = start_on(Device::Position(0)).await;
        r.unwrap();
        assert_eq!(served_backend(&s).await, want);
        s.stop().await;
        return;
    }
    let (s, r) = start_on(Device::Position(0)).await;
    let e = r.unwrap_err();
    assert!(e.contains(&format!("the {} backend lists no device", empty.join(" and "))), "{e}");
    s.stop().await;
    // Each empty backend's device 0 by name is refused too, and the CPU by
    // name still loads.
    for b in empty {
        let backend = turbo_kserve::device::BACKENDS.into_iter().find(|x| *x == b).unwrap();
        let (s, r) = start_on(Device::Named(Named { backend, ordinal: 0 })).await;
        let e = r.unwrap_err();
        assert!(e.contains(&format!("device {b}:0: the {b} backend lists no such device")), "{e}");
        s.stop().await;
    }
    let (s, r) = start_on(Device::Named(Named::parse("cpu").unwrap())).await;
    r.unwrap();
    assert_eq!(served_backend(&s).await, "cpu");
    s.stop().await;
}
