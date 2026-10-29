//! The zoo image's smoke test (scripts/test-zoo-image.sh): a turbo-zoo
//! container already running elsewhere, at TURBO_ZOO_SMOKE_ADDR, in front
//! of a store seeded with the files of the index at TURBO_ZOO_SMOKE_INDEX,
//! whose own addresses answer nothing. It holds no model at start, refuses
//! a share-alike file to a user who has not accepted it, records that
//! user's own acceptance without the admin flag, and reads every file
//! through from the store, each verified. Without the variables there is
//! nothing to test and it passes.

use std::time::{Duration, Instant};

use tonic::Code;
use tonic::transport::Channel;
use turbo_kserve::proto::training::model_file_chunk::Frame;
use turbo_kserve::proto::training::turbo_catalog_service_client::TurboCatalogServiceClient;
use turbo_kserve::proto::training::*;
use turbo_zoo::{Index, LicenceClass, Sha256, Sha512};

type Client = TurboCatalogServiceClient<Channel>;

async fn connect(addr: &str) -> Client {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        match TurboCatalogServiceClient::connect(format!("http://{addr}")).await {
            Ok(c) => return c,
            Err(e) if Instant::now() < deadline => {
                eprintln!("waiting for {addr}: {e}");
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            Err(e) => panic!("{addr}: {e}"),
        }
    }
}

async fn read(c: &mut Client, user: &str, id: &str, path: &str) -> Result<Vec<u8>, tonic::Status> {
    let req = ReadModelFileRequest { user: user.into(), artifact_id: id.into(), path: path.into(), sha256: "".into() };
    let mut stream = c.read_model_file(req).await?.into_inner();
    let mut bytes = Vec::new();
    while let Some(m) = stream.message().await? {
        if let Some(Frame::Data(d)) = m.frame {
            bytes.extend_from_slice(&d);
        }
    }
    Ok(bytes)
}

async fn installed(c: &mut Client) -> Vec<String> {
    let models = c.list_installed_models(ListInstalledModelsRequest {}).await.unwrap().into_inner().models;
    models.into_iter().map(|m| m.id).collect()
}

#[tokio::test]
async fn a_running_zoo_image_reads_accepted_models_through_its_store() {
    let (Ok(addr), Ok(index)) = (std::env::var("TURBO_ZOO_SMOKE_ADDR"), std::env::var("TURBO_ZOO_SMOKE_INDEX")) else {
        eprintln!("TURBO_ZOO_SMOKE_ADDR and TURBO_ZOO_SMOKE_INDEX are not set; no image to check");
        return;
    };
    let index = Index::parse(&std::fs::read(&index).unwrap()).unwrap();
    let mut c = connect(&addr).await;
    assert_eq!(installed(&mut c).await, Vec::<String>::new(), "the image holds no model");
    let user = "smoke";
    for a in index.list() {
        let first = &a.files[0];
        if a.class != LicenceClass::Permissive {
            let e = read(&mut c, user, a.id.as_str(), first.path.as_str()).await.unwrap_err();
            assert_eq!(e.code(), Code::FailedPrecondition, "{e:?}");
            assert!(e.message().contains(&a.license.to_string()), "{e:?}");
            let req =
                RecordLicenceAcceptanceRequest { user: user.into(), licence: a.license.to_string(), terms: "".into() };
            c.record_licence_acceptance(req).await.unwrap();
        }
        for f in &a.files {
            let bytes = read(&mut c, user, a.id.as_str(), f.path.as_str()).await.unwrap();
            assert_eq!(bytes.len() as u64, f.size, "{}/{}", a.id, f.path.as_str());
            assert_eq!(Sha256::of(&bytes), f.sha256, "{}/{}", a.id, f.path.as_str());
            assert_eq!(Sha512::of(&bytes), f.sha512, "{}/{}", a.id, f.path.as_str());
        }
    }
    let mut want: Vec<String> = index.list().iter().map(|a| a.id.to_string()).collect();
    let mut got = installed(&mut c).await;
    want.sort();
    got.sort();
    assert_eq!(got, want);
}
