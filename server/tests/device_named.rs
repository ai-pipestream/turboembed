//! A model loads on the device it names, or not at all (#2508).
//!
//! A runtime index is a position in the runtime's device list, and when a
//! linked backend lists no device every later device moves up: on a build
//! with a GPU backend and no GPU, device 0 is the CPU. A device named by
//! its backend and ordinal (`cuda:0`, `cpu`) loads there or is refused
//! with DEVICE_NOT_FOUND, and a typed index is refused when a linked
//! backend ahead of its device lists none, as its number has moved.

mod common;

use common::*;
use tonic::Code;
use turbo_kserve::api::{Runtime, TURBO_E_DEVICE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT, field, version};
use turbo_kserve::config::{Args, DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
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

/// The linked backends ahead of index `i`'s own backend that list no
/// device: what moved `i`, read from the runtime.
fn moved_ahead_of(i: u32) -> Vec<String> {
    let rt = Runtime::create().unwrap();
    let own = field(&rt.device_info(i).unwrap().backend);
    let linked: Vec<String> = version().split_whitespace().skip(1).map(str::to_owned).collect();
    let at = linked.iter().position(|b| *b == own).unwrap();
    linked[..at].iter().filter(|b| listed(b) == 0).cloned().collect()
}

/// What starting on runtime index `i` must give: the device it names
/// while nothing ahead of it is missing, else the refusal naming what is.
async fn index_outcome(s: &Server, r: Result<(), String>, i: u32) {
    let moved = moved_ahead_of(i);
    if moved.is_empty() {
        r.unwrap();
        let rt = Runtime::create().unwrap();
        assert_eq!(served_backend(s).await, field(&rt.device_info(i).unwrap().backend));
    } else {
        let e = r.unwrap_err();
        assert!(e.contains(&format!("the {} backend lists no device", moved.join(" and "))), "{e}");
    }
}

/// The issue's case, on a build that links a backend with nothing to
/// list. The npu backend opens its loader at run time, so it builds on any
/// host, and a host without an Intel NPU lists no npu device: the runtime
/// then lists the CPU at index 0, where `device=0` used to load the model
/// and report it ready. On a host where every backend lists a device, the
/// same indices load where they name.
#[tokio::test]
async fn a_typed_index_is_refused_when_a_backend_ahead_of_it_lists_no_device() {
    for i in [0, cpu()] {
        let (s, r) = start_on(Device::Position(i)).await;
        index_outcome(&s, r, i).await;
        s.stop().await;
    }
    // Each empty backend's device 0 by name is refused, and the CPU by
    // name loads.
    let linked = version();
    for b in linked.split_whitespace().skip(1).filter(|b| listed(b) == 0) {
        let backend = turbo_kserve::device::BACKENDS.into_iter().find(|x| *x == b).unwrap();
        let (s, r) = start_on(Device::Named(Named { backend, ordinal: 0 })).await;
        let e = r.unwrap_err();
        assert!(e.contains(&format!("device {b}:0: the {b} backend lists no such device")), "{e}");
        s.stop().await;
    }
}

/// LoadBundle's `device` is a typed index too: refused the same way, and
/// nothing registered.
#[tokio::test]
async fn load_bundle_holds_its_device_index_to_the_same_rule() {
    let (s, mut c) = serve().await;
    let i = cpu();
    let mut req = load_req("by-index", tiny().to_str().unwrap(), 1, 0);
    req.device_name = None;
    req.device = i;
    let r = c.load_bundle(req).await;
    let moved = moved_ahead_of(i);
    if moved.is_empty() {
        r.unwrap();
    } else {
        let e = r.unwrap_err();
        assert_eq!(refusal(&e), (Code::FailedPrecondition, TURBO_E_DEVICE_NOT_FOUND), "{e:?}");
        let caps = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
        assert!(caps.bundles.iter().all(|b| b.name != "by-index"), "{caps:?}");
    }
    s.stop().await;
}

/// `--bundle-dir` serves on runtime index 0 unless `--bundle-device`
/// names a device; the index is held to the same rule.
#[tokio::test]
async fn bundle_dir_serves_on_its_named_device_or_holds_index_0_to_the_rule() {
    let testdata = tiny().parent().unwrap().to_str().unwrap().to_string();
    let parse = |extra: &[&str]| {
        let mut a = vec!["--listen", "127.0.0.1:0", "--bundle-dir", testdata.as_str()];
        a.extend_from_slice(extra);
        let parsed = Args::parse_with(a.into_iter().map(String::from), |_| None).unwrap();
        parsed.models.into_iter().filter(|m| m.name() == Some(NAME)).collect::<Vec<_>>()
    };
    let start = |models: Vec<ModelConfig>| async move {
        let s = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, Serve::default()).await.unwrap();
        let r = s.load().await;
        (s, r)
    };
    let plain = parse(&[]);
    assert_eq!(plain[0].device, Device::Position(0));
    let (s, r) = start(plain).await;
    index_outcome(&s, r, 0).await;
    s.stop().await;
    let named = parse(&["--bundle-device", "cpu"]);
    let (s, r) = start(named).await;
    r.unwrap();
    assert_eq!(served_backend(&s).await, "cpu");
    s.stop().await;
    let absent = parse(&["--bundle-device", "cuda:4096"]);
    let (s, r) = start(absent).await;
    assert!(r.unwrap_err().contains("device cuda:4096"));
    s.stop().await;
    let e = Args::parse_with(
        ["--listen", "127.0.0.1:0", "--bundle-dir", testdata.as_str(), "--bundle-device", "gpu"]
            .into_iter()
            .map(String::from),
        |_| None,
    )
    .unwrap_err();
    assert!(e.starts_with("--bundle-device"), "{e}");
}
