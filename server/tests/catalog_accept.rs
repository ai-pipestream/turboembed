//! TurboCatalogService's licence acceptances end to end: recording one
//! from a page that names the artifact and the digest of the text it
//! showed, listing and revoking it, and the same acceptances.json the
//! turbo-zoo command reads and writes.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tonic::Code;
use tonic::transport::Channel;
use turbo_kserve::api::{TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE, TURBO_E_UNSUPPORTED};
use turbo_kserve::catalog::Catalog;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::training::turbo_catalog_service_client::TurboCatalogServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve::{Serve, Server};
use turbo_zoo::{Acceptances, Index, Licence, Sha256, Sha512, Transport, Url, UserId};

type Client = TurboCatalogServiceClient<Channel>;

const BASE: &str = "https://example.org/models";
const SA: &[u8] = b"a share-alike word list\n";
const SA_TEXT: &str = "https://creativecommons.org/licenses/by-sa/4.0/legalcode";

struct Fixed;

impl Transport for Fixed {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        if url.as_str() == format!("{BASE}/sa.txt") { Ok(SA.to_vec()) } else { Err(format!("{url}: not served")) }
    }
}

/// A share-alike word list and a permissive one.
fn index() -> Index {
    let artifact = |id: &str, licence: &str, class: &str| {
        format!(
            r#"{{"id": "{id}", "kind": "KIND_LEXICON", "language": "en", "format": "FORMAT_TEXT", "version": "1",
              "license": "{licence}", "license_terms": "https://example.org/{id}/LICENSE", "trained_on": [],
              "class": "{class}", "attribution": "Example attribution for {id}.",
              "files": [{{"path": "sa.txt", "url": "{BASE}/sa.txt", "size": {}, "sha256": "{}", "sha512": "{}"}}]}}"#,
            SA.len(),
            Sha256::of(SA),
            Sha512::of(SA),
        )
    };
    let json = format!(
        r#"{{"zoo_version": 1, "artifacts": [{}, {}]}}"#,
        artifact("sa", "CC-BY-SA-4.0", "CLASS_SHARE_ALIKE"),
        artifact("free", "Apache-2.0", "CLASS_PERMISSIVE"),
    );
    Index::parse(json.as_bytes()).unwrap()
}

struct Home(PathBuf);

impl Home {
    fn new(name: &str) -> Home {
        let p = std::env::temp_dir().join(format!("turbo-kserve-catalog-accept-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        Home(p)
    }

    fn acceptances(&self) -> Acceptances {
        Acceptances::load(&self.0.join("acceptances.json")).unwrap()
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

async fn start(home: &Path, admin: bool) -> (Server, Client) {
    start_in(home, Serve { admin, ..Serve::default() }).await
}

/// The server over `home` with `serve`'s flags and the test catalog.
async fn start_in(home: &Path, serve: Serve) -> (Server, Client) {
    let catalog = Catalog { home: home.to_path_buf(), index: Arc::new(index()), transport: Arc::new(Fixed) };
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let model = ModelConfig {
        bundle: bundle.to_str().unwrap().into(),
        name: None,
        device: Device::Select,
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    };
    let serve = Serve { catalog: Some(catalog), ..serve };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model], MAX, serve).await.unwrap();
    let c = TurboCatalogServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    (s, c)
}

fn refusal(s: &tonic::Status) -> (Code, i32, String) {
    let md = |k: &str| s.metadata().get(k).unwrap_or_else(|| panic!("no {k} in {s:?}")).to_str().unwrap().to_owned();
    (s.code(), md("turbo-code").parse().unwrap(), md("turbo-field"))
}

fn record(user: &str, artifact: &str, sha256: &str) -> RecordLicenceAcceptanceRequest {
    RecordLicenceAcceptanceRequest {
        user: user.into(),
        licence: "CC-BY-SA-4.0".into(),
        terms: String::new(),
        artifact_id: artifact.into(),
        text_sha256: sha256.into(),
    }
}

fn revoke(user: &str) -> RevokeLicenceAcceptanceRequest {
    RevokeLicenceAcceptanceRequest { user: user.into(), licence: "CC-BY-SA-4.0".into(), terms: String::new() }
}

async fn list(c: &mut Client, user: &str) -> ListLicenceAcceptancesResponse {
    c.list_licence_acceptances(ListLicenceAcceptancesRequest { user: user.into() }).await.unwrap().into_inner()
}

async fn missing(c: &mut Client, user: &str) -> Vec<LicenceTerms> {
    let req = ListModelCatalogRequest { user: user.into(), ..Default::default() };
    let entries = c.list_model_catalog(req).await.unwrap().into_inner().entries;
    entries.into_iter().find(|e| e.id == "sa").unwrap().missing_terms
}

/// The page's acceptance names the artifact and the digest of the text it
/// showed; the zoo's own file holds the same record at the same time.
#[tokio::test]
async fn a_page_acceptance_is_the_zoo_files_acceptance() {
    let home = Home::new("page");
    let (s, mut c) = start(&home.0, true).await;
    let shown = missing(&mut c, "alice").await;
    assert_eq!(shown.len(), 1);
    let digest = Sha256::of(shown[0].text.as_bytes()).to_string();
    let r = c.record_licence_acceptance(record("alice", "sa", &digest)).await.unwrap().into_inner();
    assert!(missing(&mut c, "alice").await.is_empty());
    let listed = list(&mut c, "alice").await;
    assert_eq!((listed.user.as_str(), listed.admin, listed.accepted.len()), ("alice", true, 1));
    let a = &listed.accepted[0];
    assert_eq!((a.accepted_at, a.artifact_id.as_str(), a.text_sha256.as_str()), (r.accepted_at, "sa", digest.as_str()));
    assert_eq!(a.terms.as_ref().unwrap().text, SA_TEXT);
    let file = home.acceptances();
    let alice: UserId = "alice".parse().unwrap();
    let rec = file.of_user(&alice).next().unwrap();
    assert_eq!((rec.accepted_at, rec.artifact.as_ref().unwrap().as_str()), (r.accepted_at, "sa"));
    let raw = fs::read_to_string(home.0.join("acceptances.json")).unwrap();
    assert!(raw.contains("\"acceptances_version\": 2,"), "{raw}");
    assert!(list(&mut c, "bob").await.accepted.is_empty());
    s.stop().await;
}

#[tokio::test]
async fn a_digest_or_artifact_that_does_not_match_is_refused() {
    let home = Home::new("bad");
    let (s, mut c) = start(&home.0, true).await;
    let other = Sha256::of(b"https://example.org/terms").to_string();
    let good = Sha256::of(SA_TEXT.as_bytes()).to_string();
    for (req, field) in [
        (record("alice", "sa", &other), "5"),
        (record("alice", "sa", &good.to_uppercase()), "5"),
        (record("alice", "sa", "abc"), "5"),
        (record("alice", "nope", &good), "4"),
        (record("alice", "free", &good), "4"),
        (record("alice", "../sa", &good), "4"),
    ] {
        let e = c.record_licence_acceptance(req).await.unwrap_err();
        assert_eq!(refusal(&e), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT, field.into()), "{e:?}");
    }
    assert!(!home.0.join("acceptances.json").exists(), "a refused acceptance writes nothing");
    s.stop().await;
}

/// An acceptance made by the turbo-zoo command shows as accepted, and a
/// revoke over gRPC removes it from the same file; the installed file is
/// served no more.
#[tokio::test]
async fn the_command_line_and_the_server_share_one_record() {
    let home = Home::new("shared");
    let alice: UserId = "alice".parse().unwrap();
    let mut file = Acceptances::default();
    file.accept(&alice, Licence::CcBySa40, 1_792_000_000).unwrap();
    fs::create_dir_all(&home.0).unwrap();
    file.save(&home.0.join("acceptances.json")).unwrap();
    let (s, mut c) = start(&home.0, true).await;
    assert!(missing(&mut c, "alice").await.is_empty(), "no second prompt");
    assert_eq!(list(&mut c, "alice").await.accepted[0].accepted_at, 1_792_000_000);
    let install = InstallModelRequest { user: "alice".into(), artifact_id: "sa".into() };
    let mut updates = c.install_model(install).await.unwrap().into_inner();
    while updates.message().await.unwrap().is_some() {}
    let read = || ReadModelFileRequest {
        user: "alice".into(),
        artifact_id: "sa".into(),
        path: "sa.txt".into(),
        sha256: String::new(),
    };
    let mut frames = c.read_model_file(read()).await.unwrap().into_inner();
    while frames.message().await.unwrap().is_some() {}
    let r = c.revoke_licence_acceptance(revoke("alice")).await.unwrap().into_inner();
    assert!(r.was_accepted);
    assert_eq!(r.revoked.unwrap().text, SA_TEXT);
    assert!(home.acceptances().of_user(&alice).next().is_none());
    assert_eq!(missing(&mut c, "alice").await.len(), 1);
    let e = c.read_model_file(read()).await.unwrap_err();
    assert_eq!(refusal(&e).1, TURBO_E_INVALID_STATE, "the installed file is not served after a revoke: {e:?}");
    assert!(!c.revoke_licence_acceptance(revoke("alice")).await.unwrap().into_inner().was_accepted);
    s.stop().await;
}

/// Without the admin flag the page lists, and is told it may not change.
#[tokio::test]
async fn without_admin_acceptances_are_listed_but_not_changed() {
    let home = Home::new("read-only");
    let (s, mut c) = start(&home.0, false).await;
    let none = list(&mut c, "").await;
    assert_eq!((none.user.as_str(), none.admin, none.accepted.len()), ("", false, 0));
    let e = c.revoke_licence_acceptance(revoke("alice")).await.unwrap_err();
    assert_eq!(refusal(&e).1, TURBO_E_UNSUPPORTED, "{e:?}");
    assert!(e.message().contains("RevokeLicenceAcceptance is admin-only"), "{e:?}");
    let e = c.record_licence_acceptance(record("alice", "", "")).await.unwrap_err();
    assert_eq!(refusal(&e).1, TURBO_E_UNSUPPORTED, "{e:?}");
    let e = c.list_licence_acceptances(ListLicenceAcceptancesRequest { user: " a".into() }).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT, "1".into()), "{e:?}");
    s.stop().await;
    // Zoo mode serves the changes without --admin, and the page is told so.
    let (s, mut c) = start_in(&home.0, Serve { zoo: true, ..Serve::default() }).await;
    assert!(list(&mut c, "alice").await.admin);
    c.record_licence_acceptance(record("alice", "", "")).await.unwrap();
    assert!(c.revoke_licence_acceptance(revoke("alice")).await.unwrap().into_inner().was_accepted);
    s.stop().await;
}

/// The tiny test bundle, copied with its manifest naming `licence`.
fn bundle_under(home: &Home, licence: &str) -> PathBuf {
    let from = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let to = home.0.join(format!("bundle-{licence}"));
    let mut stack = vec![(from, to.clone())];
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
    let text =
        fs::read_to_string(&m).unwrap().replace("\"license\": \"Apache-2.0\"", &format!("\"license\": \"{licence}\""));
    fs::write(&m, text).unwrap();
    to
}

/// LoadBundle of a share-alike bundle needs the loading user's
/// acceptance; a permissive one needs none.
#[tokio::test]
async fn load_bundle_needs_the_loading_users_acceptance() {
    use turbo_kserve::proto::result::LoadBundleRequest;
    use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
    let home = Home::new("load");
    let (s, mut c) = start(&home.0, true).await;
    let mut a = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let sa = bundle_under(&home, "CC-BY-SA-4.0");
    let rt = turbo_kserve::api::Runtime::create().unwrap();
    let cpu = (0..16).find(|&i| rt.device_info(i).is_ok_and(|d| d.kind == turbo::TURBO_DEVICE_CPU)).expect("a CPU");
    let load = |name: &str, dir: &Path, user: &str| LoadBundleRequest {
        name: name.into(),
        bundle: dir.to_str().unwrap().into(),
        sessions: 1,
        device: cpu,
        user: user.into(),
        ..Default::default()
    };
    let e = a.load_bundle(load("sa", &sa, "alice")).await.unwrap_err();
    assert_eq!(refusal(&e).1, TURBO_E_INVALID_STATE, "{e:?}");
    assert!(e.message().contains("CC-BY-SA-4.0") && e.message().contains(SA_TEXT), "{e:?}");
    let e = a.load_bundle(load("sa", &sa, "")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::FailedPrecondition, TURBO_E_INVALID_STATE, "12".into()), "{e:?}");
    c.record_licence_acceptance(record("alice", "", "")).await.unwrap();
    a.load_bundle(load("sa", &sa, "alice")).await.unwrap();
    let e = a.load_bundle(load("sa-bob", &sa, "bob")).await.unwrap_err();
    assert_eq!(refusal(&e).1, TURBO_E_INVALID_STATE, "one user's acceptance is not another's: {e:?}");
    let free = bundle_under(&home, "Apache-2.0");
    a.load_bundle(load("free", &free, "")).await.unwrap();
    s.stop().await;
}
