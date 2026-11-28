//! A GLiNER package runs its encoder on the CPU in the server today, so
//! a load that names any other device, or a precision the CPU encoder
//! does not run at, is refused instead of quietly loading on the CPU.

#[path = "gliner_progressive/fixture.rs"]
mod fixture;

use turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES as MAX;
use turbo_kserve::proto::result::LoadBundleRequest;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::{Serve, Server};

async fn load(device: u32, precision: u32) -> Result<(), tonic::Status> {
    let server =
        Server::start("127.0.0.1:0".parse().unwrap(), Vec::new(), MAX, Serve { admin: true, ..Serve::default() })
            .await
            .unwrap();
    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX)
        .max_encoding_message_size(MAX);
    let result = client
        .load_bundle(LoadBundleRequest {
            name: "gliner".into(),
            bundle: fixture::package().to_string_lossy().into_owned(),
            device,
            sessions: 1,
            precision,
            max_batch: 0,
            max_seq: 0,
            select_device: false,
            user: String::new(),
            vocabulary: None,
        })
        .await
        .map(|_| ());
    server.stop().await;
    result
}

/// The runtime index of the CPU device on this host.
fn cpu_index() -> u32 {
    let rt = turbo_kserve::api::Runtime::create().unwrap();
    (0..64)
        .find(|&i| rt.device_info(i).is_ok_and(|d| d.kind == turbo::TURBO_DEVICE_CPU))
        .expect("the runtime lists a CPU")
}

#[tokio::test]
async fn a_device_the_runtime_does_not_list_is_refused() {
    let e = load(9999, 0).await.unwrap_err();
    assert!(e.message().contains("9999"), "{e:?}");
}

#[tokio::test]
async fn a_precision_the_cpu_encoder_does_not_run_at_is_refused() {
    // PRECISION_FASTEST; refused whatever device was named, before any load.
    let e = load(9999, 1).await.unwrap_err();
    assert!(e.message().contains("PRECISION_FASTEST"), "{e:?}");
}

#[tokio::test]
async fn the_cpu_by_its_own_index_loads() {
    load(cpu_index(), 0).await.unwrap();
    // PRECISION_EXACT is the f32 the CPU encoder computes in.
    load(cpu_index(), 2).await.unwrap();
}

#[tokio::test]
async fn a_listed_device_that_is_not_the_cpu_is_refused() {
    let rt = turbo_kserve::api::Runtime::create().unwrap();
    let Some(index) = (0..64).find(|&i| rt.device_info(i).is_ok_and(|d| d.kind != turbo::TURBO_DEVICE_CPU)) else {
        eprintln!("skipped: the runtime lists no non-cpu device");
        return;
    };
    let e = load(index, 0).await.unwrap_err();
    assert!(e.message().contains("run only on cpu"), "{e:?}");
}
