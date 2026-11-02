//! Licence acceptance through the demo: the page's calls act for the
//! user the demo was started with (--user), never for a name the page
//! sends, and without one they change nothing.

mod common;

use std::fs;
use std::sync::Arc;

use common::*;
use turbo_kserve::Serve;
use turbo_kserve::catalog::Catalog;
use turbo_kserve::proto::training::turbo_catalog_service_client::TurboCatalogServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve_demo::{Allow, Options};
use turbo_zoo::{Index, Sha256, Sha512, Transport, Url};

const RECORD: &str = "/turbo.training.v1.TurboCatalogService/RecordLicenceAcceptance";
const REVOKE: &str = "/turbo.training.v1.TurboCatalogService/RevokeLicenceAcceptance";
const LIST: &str = "/turbo.training.v1.TurboCatalogService/ListLicenceAcceptances";
const CATALOG: &str = "/turbo.training.v1.TurboCatalogService/ListModelCatalog";
const SA: &[u8] = b"a share-alike word list\n";
const SA_TEXT: &str = "https://creativecommons.org/licenses/by-sa/4.0/legalcode";

struct Nothing;

impl Transport for Nothing {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        Err(format!("{url}: not served"))
    }
}

fn catalog(home: &std::path::Path) -> Catalog {
    let json = format!(
        r#"{{"zoo_version": 1, "artifacts": [{{"id": "sa", "kind": "KIND_LEXICON", "language": "en",
          "format": "FORMAT_TEXT", "version": "1", "license": "CC-BY-SA-4.0",
          "license_terms": "https://example.org/sa/LICENSE", "trained_on": [], "class": "CLASS_SHARE_ALIKE",
          "attribution": "Example.", "files": [{{"path": "sa.txt", "url": "https://example.org/sa.txt",
          "size": {}, "sha256": "{}", "sha512": "{}"}}]}}]}}"#,
        SA.len(),
        Sha256::of(SA),
        Sha512::of(SA),
    );
    Catalog {
        home: home.to_path_buf(),
        index: Arc::new(Index::parse(json.as_bytes()).unwrap()),
        transport: Arc::new(Nothing),
    }
}

fn record(user: &str) -> RecordLicenceAcceptanceRequest {
    RecordLicenceAcceptanceRequest {
        user: user.into(),
        licence: "CC-BY-SA-4.0".into(),
        terms: String::new(),
        artifact_id: "sa".into(),
        text_sha256: Sha256::of(SA_TEXT.as_bytes()).to_string(),
    }
}

#[tokio::test]
async fn the_page_accepts_and_revokes_for_the_demos_user_only() {
    let dir = scratch("licences");
    let serve = Serve { admin: true, catalog: Some(catalog(&dir.join("home"))), ..Serve::default() };
    let s = server(&["tiny-bert-bundle"], serve).await;
    let allow = Allow::with(&[RECORD.to_string(), REVOKE.to_string()]).unwrap();
    let d = demo_with(s.local_addr(), Options { allow, user: Some("alice".into()), ..Options::default() }).await;

    // The page names someone else; the acceptance is alice's.
    let r: RecordLicenceAcceptanceResponse = ok(&d, RECORD, &record("mallory")).await;
    assert_eq!(r.user, "alice");
    let mut direct = TurboCatalogServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let list = |u: &str| ListLicenceAcceptancesRequest { user: u.into() };
    assert!(direct.list_licence_acceptances(list("mallory")).await.unwrap().into_inner().accepted.is_empty());
    assert_eq!(direct.list_licence_acceptances(list("alice")).await.unwrap().into_inner().accepted.len(), 1);

    // Reads name the demo's user too, whatever the page sent.
    let mine: ListLicenceAcceptancesResponse = ok(&d, LIST, &list("mallory")).await;
    assert_eq!((mine.user.as_str(), mine.accepted.len(), mine.admin), ("alice", 1, true));
    assert_eq!(mine.accepted[0].artifact_id, "sa");
    let cat: ListModelCatalogResponse = ok(&d, CATALOG, &ListModelCatalogRequest::default()).await;
    assert!(cat.entries[0].missing_terms.is_empty(), "alice's acceptance, though the page named no one");

    let r: RevokeLicenceAcceptanceResponse = ok(
        &d,
        REVOKE,
        &RevokeLicenceAcceptanceRequest {
            user: "mallory".into(),
            licence: "CC-BY-SA-4.0".into(),
            terms: String::new(),
        },
    )
    .await;
    assert!(r.was_accepted && r.user == "alice");
    let cat: ListModelCatalogResponse = ok(&d, CATALOG, &ListModelCatalogRequest::default()).await;
    assert_eq!(cat.entries[0].missing_terms.len(), 1);
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn without_a_user_the_page_is_read_only() {
    let dir = scratch("licences-anon");
    let serve = Serve { admin: true, catalog: Some(catalog(&dir.join("home"))), ..Serve::default() };
    let s = server(&["tiny-bert-bundle"], serve).await;
    let allow = Allow::with(&[RECORD.to_string()]).unwrap();
    let d = demo_with(s.local_addr(), Options { allow, ..Options::default() }).await;
    let (r, body) = unary(&d, RECORD, &record("mallory")).await;
    let body = String::from_utf8_lossy(&body);
    assert_eq!(r.status, 403, "{body}");
    assert!(body.contains("permission_denied") && body.contains("--user"), "{body}");
    // A read names no user: the page learns it may only read.
    let none: ListLicenceAcceptancesResponse =
        ok(&d, LIST, &ListLicenceAcceptancesRequest { user: "alice".into() }).await;
    assert_eq!((none.user.as_str(), none.accepted.len()), ("", 0));
    assert!(!dir.join("home").join("acceptances.json").exists());
    fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn a_message_that_is_not_protobuf_is_refused_before_the_server() {
    let options = Options { user: Some("alice".into()), ..Options::default() };
    let d = demo_with(nowhere().await, options).await;
    let r = http(&d, "POST", LIST, &[("Content-Type", "application/proto")], &[0x0a, 9, b'a']).await;
    let status = r.status;
    let body = String::from_utf8_lossy(&r.all().await).into_owned();
    assert_eq!(status, 400, "{body}");
    assert!(body.contains("invalid_argument") && body.contains("past the end"), "{body}");
}

/// The tiny test bundle, copied under `dir` with its licence set to
/// share-alike.
fn share_alike_bundle(dir: &std::path::Path) -> std::path::PathBuf {
    let to = dir.join("bundle-sa");
    let mut stack = vec![(testdata("tiny-bert-bundle"), to.clone())];
    while let Some((src, dst)) = stack.pop() {
        fs::create_dir_all(&dst).unwrap();
        for e in fs::read_dir(&src).unwrap() {
            let e = e.unwrap();
            if e.file_type().unwrap().is_dir() {
                stack.push((e.path(), dst.join(e.file_name())));
            } else {
                fs::copy(e.path(), dst.join(e.file_name())).unwrap();
            }
        }
    }
    let m = to.join("manifest.json");
    let text = fs::read_to_string(&m).unwrap().replace(r#""license": "Apache-2.0""#, r#""license": "CC-BY-SA-4.0""#);
    fs::write(&m, text).unwrap();
    to
}

/// LoadBundle names the demo's user too: a share-alike bundle loads once
/// that user accepted its terms, and without --user only bundles under
/// permissive terms load.
#[tokio::test]
async fn a_bundle_loads_for_the_demos_user() {
    use turbo_kserve::proto::result::{LoadBundleRequest, LoadBundleResponse};
    const LOAD: &str = "/turbo.result.v1.AnalysisService/LoadBundle";
    let dir = scratch("licences-load");
    let serve = Serve { admin: true, catalog: Some(catalog(&dir.join("home"))), ..Serve::default() };
    let s = server(&["tiny-bert-bundle"], serve).await;
    let sa = share_alike_bundle(&dir);
    let load = |name: &str, bundle: &std::path::Path| LoadBundleRequest {
        name: name.into(),
        bundle: bundle.to_str().unwrap().into(),
        sessions: 1,
        device: cpu(),
        user: "mallory".into(),
        ..Default::default()
    };
    let allow = Allow::with(&[RECORD.to_string(), LOAD.to_string()]).unwrap();
    let d =
        demo_with(s.local_addr(), Options { allow: allow.clone(), user: Some("alice".into()), ..Options::default() })
            .await;
    let (r, body) = unary(&d, LOAD, &load("sa", &sa)).await;
    let body = String::from_utf8_lossy(&body);
    assert_eq!(r.status, 400, "alice has not accepted; mallory's name is not used: {body}");
    assert!(body.contains("CC-BY-SA-4.0"), "{body}");
    let _: RecordLicenceAcceptanceResponse = ok(&d, RECORD, &record("mallory")).await;
    let _: LoadBundleResponse = ok(&d, LOAD, &load("sa", &sa)).await;

    let anon = demo_with(s.local_addr(), Options { allow, ..Options::default() }).await;
    let (r, body) = unary(&anon, LOAD, &load("sa-anon", &sa)).await;
    let body = String::from_utf8_lossy(&body);
    assert!(r.status == 400 && body.contains("12"), "no user, and the page's name is dropped: {body}");
    let _: LoadBundleResponse = ok(&anon, LOAD, &load("free", &testdata("tiny-bert-bundle"))).await;
    s.stop().await;
    fs::remove_dir_all(dir).unwrap();
}

const REGISTER: &str = "/turbo.training.v1.TurboCatalogService/RegisterModelSource";
const SOURCES: &str = "/turbo.training.v1.TurboCatalogService/ListModelSources";

/// Model sources go through the same rewrite: registered and listed for
/// the demo's user, whatever the page names.
#[tokio::test]
async fn the_page_registers_sources_for_the_demos_user_only() {
    let dir = scratch("sources");
    let serve = Serve { zoo: true, catalog: Some(catalog(&dir.join("home"))), ..Serve::default() };
    let s = server(&[], serve).await;
    let allow = Allow::with(&[REGISTER.to_string()]).unwrap();
    let d = demo_with(s.local_addr(), Options { allow, user: Some("alice".into()), ..Options::default() }).await;
    let req = RegisterModelSourceRequest {
        user: "mallory".into(),
        name: "shared".into(),
        address: "gdrive://1AbC".into(),
        api_key: "key-1".into(),
    };
    let r: RegisterModelSourceResponse = ok(&d, REGISTER, &req).await;
    assert_eq!(r.user, "alice");
    let mine: ListModelSourcesResponse = ok(&d, SOURCES, &ListModelSourcesRequest { user: "mallory".into() }).await;
    assert_eq!((mine.user.as_str(), mine.sources.len(), mine.admin), ("alice", 1, true));
    let mut direct = TurboCatalogServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let theirs = direct.list_model_sources(ListModelSourcesRequest { user: "mallory".into() }).await.unwrap();
    assert!(theirs.into_inner().sources.is_empty());
    fs::remove_dir_all(dir).unwrap();
}
