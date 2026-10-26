//! The writes the demo forwards only when named with --allow, here the two
//! server-streaming ones: TrainStaticModel, including a page that goes
//! away mid-stream, and InstallModel from a catalog served from memory.

mod common;

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use std::time::Duration;

use common::*;
use tonic::Code;
use turbo_kserve::Serve;
use turbo_kserve::catalog::Catalog;
use turbo_kserve::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use turbo_kserve::proto::training::install_model_update::Update as Install;
use turbo_kserve::proto::training::train_static_model_update::Update;
use turbo_kserve::proto::training::turbo_training_service_client::TurboTrainingServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve::proto::{ModelMetadataRequest, ServerLiveResponse};
use turbo_zoo::{Index, Sha256, Sha512, Transport, Url};

const TRAIN: &str = "/turbo.training.v1.TurboTrainingService/TrainStaticModel";
const INSTALL: &str = "/turbo.training.v1.TurboCatalogService/InstallModel";

fn request(name: &str) -> TrainStaticModelRequest {
    TrainStaticModelRequest {
        name: name.into(),
        teacher: "tiny-bert-bundle".into(),
        pca_dims: 16,
        max_length: 32,
        ..Default::default()
    }
}

fn stage(u: &TrainStaticModelUpdate) -> Option<TrainStage> {
    match u.update.as_ref()? {
        Update::Progress(p) => Some(p.stage()),
        Update::Trained(_) => None,
    }
}

async fn trainer(d: &std::path::Path) -> turbo_kserve::Server {
    let serve = Serve {
        admin: true,
        vocabulary: Some(d.join("vocabulary")),
        training: Some(d.join("training")),
        ..Serve::default()
    };
    server(&["tiny-bert-bundle", "tiny-static-bundle"], serve).await
}

#[tokio::test]
async fn train_static_model_streams_its_progress_and_refusals() {
    let dir = scratch("train");
    let s = trainer(&dir).await;
    let d = demo(&s, &[TRAIN]).await;
    let (updates, end) = stream(&d, TRAIN, &request("trained")).await.frames::<TrainStaticModelUpdate>().await;
    assert_eq!(end, "{}");
    assert_eq!(stage(&updates[0]), Some(TrainStage::Admitted));
    assert!(matches!(updates.last().unwrap().update, Some(Update::Trained(_))), "{updates:?}");

    // A field the server refuses: its code and field in the end envelope.
    let bad = TrainStaticModelRequest { pca_dims: 33, ..request("x") };
    let (updates, end) = stream(&d, TRAIN, &bad).await.frames::<TrainStaticModelUpdate>().await;
    assert!(updates.is_empty());
    assert!(end.starts_with(r#"{"error":{"code":"invalid_argument","message":"#), "{end}");
    assert!(end.ends_with(r#""metadata":{"turbo-code":["256"],"turbo-field":["3"]}}"#), "{end}");
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn a_page_that_goes_away_cancels_the_training_at_the_server() {
    let dir = scratch("cancel");
    let s = trainer(&dir).await;
    let d = demo(&s, &[TRAIN]).await;
    let mut reply = stream(&d, TRAIN, &request("cancelled")).await;
    let (flags, first) = reply.frame().await.unwrap();
    assert_eq!(flags, 0);
    let first = <TrainStaticModelUpdate as prost::Message>::decode(first.as_slice()).unwrap();
    assert_eq!(stage(&first), Some(TrainStage::Admitted));
    drop(reply);
    // The demo drops its call, the training stops at its next step and
    // gives its slot back; until then another is refused as busy.
    let mut c = TurboTrainingServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let last = loop {
        match c.train_static_model(request("after")).await {
            Err(e) if e.code() == Code::Unavailable => tokio::time::sleep(Duration::from_millis(50)).await,
            Err(e) => panic!("{e:?}"),
            Ok(r) => {
                let mut st = r.into_inner();
                let mut last = None;
                loop {
                    match st.message().await {
                        Ok(Some(u)) => last = Some(u),
                        Ok(None) => break,
                        Err(e) if e.code() == Code::Unavailable => break,
                        Err(e) => panic!("{e:?}"),
                    }
                }
                if let Some(u) = last.filter(|u| matches!(u.update, Some(Update::Trained(_)))) {
                    break u;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }
    };
    assert!(matches!(last.update, Some(Update::Trained(_))));
    let names: Vec<String> =
        fs::read_dir(dir.join("training")).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    assert_eq!(names, ["after"]);
    let mut k = GrpcInferenceServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let e = k.model_metadata(ModelMetadataRequest { name: "cancelled".into(), version: "".into() }).await.unwrap_err();
    assert_eq!(e.code(), Code::NotFound);
    fs::remove_dir_all(dir).unwrap();
}

const TEXT: &[u8] = b"hello hello hello\n";
const BASE: &str = "https://example.org/models";

/// Serves fixed bytes by address.
struct Fixed(HashMap<String, Vec<u8>>);

impl Transport for Fixed {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        self.0.get(url.as_str()).cloned().ok_or_else(|| format!("{url}: not served"))
    }
}

fn catalog(home: &std::path::Path) -> Catalog {
    let json = format!(
        r#"{{"zoo_version": 1, "artifacts": [{{"id": "hello", "kind": "KIND_LEXICON", "language": "en",
          "format": "FORMAT_TEXT", "version": "1", "license": "Apache-2.0",
          "license_terms": "https://example.org/hello/LICENSE", "trained_on": [], "class": "CLASS_PERMISSIVE",
          "attribution": "Example.", "files": [{{"path": "hello.txt", "url": "{BASE}/hello.txt", "size": {},
          "sha256": "{}", "sha512": "{}"}}]}}]}}"#,
        TEXT.len(),
        Sha256::of(TEXT),
        Sha512::of(TEXT),
    );
    let files = HashMap::from([(format!("{BASE}/hello.txt"), TEXT.to_vec())]);
    Catalog {
        home: home.to_path_buf(),
        index: Arc::new(Index::parse(json.as_bytes()).unwrap()),
        transport: Arc::new(Fixed(files)),
    }
}

#[tokio::test]
async fn install_model_streams_its_progress_when_allowed() {
    let dir = scratch("install");
    let serve = Serve { admin: true, catalog: Some(catalog(&dir.join("home"))), ..Serve::default() };
    let s = server(&["tiny-bert-bundle"], serve).await;
    let req = InstallModelRequest { user: "alice".into(), artifact_id: "hello".into() };

    // Off the allowlist until named.
    let (_, end) = stream(&demo(&s, &[]).await, INSTALL, &req).await.frames::<ServerLiveResponse>().await;
    assert!(end.starts_with(r#"{"error":{"code":"permission_denied""#), "{end}");

    let d = demo(&s, &[INSTALL]).await;
    let (updates, end) = stream(&d, INSTALL, &req).await.frames::<InstallModelUpdate>().await;
    assert_eq!(end, "{}");
    assert!(updates.iter().any(|u| matches!(u.update, Some(Install::Progress(_)))), "{updates:?}");
    assert!(matches!(updates.last().unwrap().update, Some(Install::Installed(_))), "{updates:?}");

    // An artifact the catalog does not list: the server's refusal.
    let req = InstallModelRequest { user: "alice".into(), artifact_id: "nothing".into() };
    let (updates, end) = stream(&d, INSTALL, &req).await.frames::<InstallModelUpdate>().await;
    assert!(updates.is_empty());
    assert!(end.starts_with(r#"{"error":{"code":"not_found""#) && end.contains(r#""turbo-code":["1280"]"#), "{end}");
    fs::remove_dir_all(dir).unwrap();
}
