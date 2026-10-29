//! TurboCatalogService in zoo mode end to end: a server with no model and
//! no admin flag, where a user records their own acceptance, and a read of
//! a permitted artifact not yet cached fetches it once from its origin and
//! streams it. A read the licence check or the pin refuses fetches nothing,
//! and readers that arrive together share one fetch.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tonic::Code;
use tonic::transport::Channel;
use turbo_kserve::api::{TURBO_E_INVALID_STATE, TURBO_E_UNSUPPORTED};
use turbo_kserve::catalog::Catalog;
use turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES as MAX;
use turbo_kserve::proto::training::model_file_chunk::Frame;
use turbo_kserve::proto::training::turbo_catalog_service_client::TurboCatalogServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve::{Serve, Server};
use turbo_zoo::{Index, Sha256, Sha512, Transport, Url};

type Client = TurboCatalogServiceClient<Channel>;

const BASE: &str = "https://example.org/models";
const TEXT: &[u8] = b"hello hello hello\n";
const SA: &[u8] = b"a share-alike word list\n";

fn big() -> Vec<u8> {
    (0..200_000).map(|i| (i % 251) as u8).collect()
}

/// The origin: fixed bytes by address, counting each GET, each one slow
/// enough that readers arriving together overlap.
#[derive(Clone, Default)]
struct Origin {
    files: Arc<HashMap<String, Vec<u8>>>,
    gets: Arc<Mutex<HashMap<String, usize>>>,
    total: Arc<AtomicUsize>,
}

impl Origin {
    fn new() -> Origin {
        let mut files = HashMap::new();
        files.insert(format!("{BASE}/hello.txt"), TEXT.to_vec());
        files.insert(format!("{BASE}/weights.bin"), big());
        files.insert(format!("{BASE}/sa.txt"), SA.to_vec());
        Origin { files: Arc::new(files), ..Origin::default() }
    }

    fn gets(&self, name: &str) -> usize {
        self.gets.lock().unwrap().get(&format!("{BASE}/{name}")).copied().unwrap_or(0)
    }
}

impl Transport for Origin {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        std::thread::sleep(Duration::from_millis(50));
        *self.gets.lock().unwrap().entry(url.to_string()).or_default() += 1;
        self.total.fetch_add(1, Ordering::SeqCst);
        self.files.get(url.as_str()).cloned().ok_or_else(|| format!("{url}: not served"))
    }
}

fn file(path: &str, bytes: &[u8]) -> String {
    format!(
        r#"{{"path": "{path}", "url": "{BASE}/{path}", "size": {}, "sha256": "{}", "sha512": "{}"}}"#,
        bytes.len(),
        Sha256::of(bytes),
        Sha512::of(bytes),
    )
}

/// A permissive model of two files and a share-alike word list.
fn index() -> Index {
    let artifact = |id: &str, licence: &str, class: &str, files: String| {
        format!(
            r#"{{"id": "{id}", "kind": "KIND_LEXICON", "language": "en", "format": "FORMAT_TEXT", "version": "1",
              "license": "{licence}", "license_terms": "https://example.org/{id}/LICENSE", "trained_on": [],
              "class": "{class}", "attribution": "Example attribution for {id}.", "files": [{files}]}}"#
        )
    };
    let json = format!(
        r#"{{"zoo_version": 1, "artifacts": [{}, {}]}}"#,
        artifact(
            "tiny",
            "Apache-2.0",
            "CLASS_PERMISSIVE",
            format!("{}, {}", file("hello.txt", TEXT), file("weights.bin", &big()))
        ),
        artifact("sa", "CC-BY-SA-4.0", "CLASS_SHARE_ALIKE", file("sa.txt", SA)),
    );
    Index::parse(json.as_bytes()).unwrap()
}

struct Home(PathBuf);

impl Home {
    fn new(name: &str) -> Home {
        let p = std::env::temp_dir().join(format!("turbo-kserve-catalog-zoo-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        Home(p)
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A server with no model and no admin flag, in zoo mode or not.
async fn start(home: &Path, origin: &Origin, zoo: bool) -> (Server, Client) {
    let catalog = Catalog { home: home.to_path_buf(), index: Arc::new(index()), transport: Arc::new(origin.clone()) };
    let serve = Serve { zoo, admin: !zoo, catalog: Some(catalog), ..Serve::default() };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), Vec::new(), MAX, serve).await.unwrap();
    let c = TurboCatalogServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    (s, c)
}

fn refusal(s: &tonic::Status) -> (Code, i32) {
    let tc = s.metadata().get("turbo-code").unwrap_or_else(|| panic!("no turbo-code in {s:?}"));
    (s.code(), tc.to_str().unwrap().parse().unwrap())
}

fn read_req(user: &str, id: &str, path: &str, sha256: &str) -> ReadModelFileRequest {
    ReadModelFileRequest { user: user.into(), artifact_id: id.into(), path: path.into(), sha256: sha256.into() }
}

async fn read(mut c: Client, req: ReadModelFileRequest) -> Result<Vec<u8>, tonic::Status> {
    let mut stream = c.read_model_file(req).await?.into_inner();
    let Some(Frame::File(_)) = stream.message().await?.and_then(|m| m.frame) else { panic!("no file frame") };
    let mut bytes = Vec::new();
    while let Some(m) = stream.message().await? {
        let Some(Frame::Data(d)) = m.frame else { panic!("a second file frame") };
        bytes.extend_from_slice(&d);
    }
    Ok(bytes)
}

#[tokio::test]
async fn zoo_mode_needs_a_catalog() {
    let serve = Serve { zoo: true, ..Serve::default() };
    let Err(e) = Server::start("127.0.0.1:0".parse().unwrap(), Vec::new(), MAX, serve).await else {
        panic!("a zoo with no catalog started");
    };
    assert!(e.contains("catalog"), "{e}");
}

#[tokio::test]
async fn a_user_accepts_for_themselves_and_the_first_read_fetches_the_model_once() {
    let (home, origin) = (Home::new("accept"), Origin::new());
    let (s, mut c) = start(&home.0, &origin, true).await;
    let e = read(c.clone(), read_req("alice", "sa", "sa.txt", "")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::FailedPrecondition, TURBO_E_INVALID_STATE), "{e:?}");
    assert!(e.message().contains("CC-BY-SA-4.0"), "{e:?}");
    assert_eq!(origin.total.load(Ordering::SeqCst), 0, "a refused read fetches nothing");
    // No admin flag: in zoo mode the user records their own acceptance.
    let req = RecordLicenceAcceptanceRequest { user: "alice".into(), licence: "CC-BY-SA-4.0".into(), terms: "".into() };
    c.record_licence_acceptance(req).await.unwrap();
    assert_eq!(read(c.clone(), read_req("alice", "sa", "sa.txt", "")).await.unwrap(), SA);
    assert_eq!(read(c.clone(), read_req("alice", "sa", "sa.txt", "")).await.unwrap(), SA);
    assert_eq!(origin.gets("sa.txt"), 1, "the second read is served from the cache");
    assert_eq!(fs::read(home.0.join("artifacts/sa/sa.txt")).unwrap(), SA);
    // Cached on the server, but bob has not accepted.
    let e = read(c.clone(), read_req("bob", "sa", "sa.txt", "")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::FailedPrecondition, TURBO_E_INVALID_STATE), "{e:?}");
    s.stop().await;
}

#[tokio::test]
async fn a_read_pinned_at_another_sha256_fetches_nothing() {
    let (home, origin) = (Home::new("pin"), Origin::new());
    let (s, c) = start(&home.0, &origin, true).await;
    let other = Sha256::of(SA).to_string();
    let e = read(c.clone(), read_req("alice", "tiny", "hello.txt", &other)).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::FailedPrecondition, TURBO_E_INVALID_STATE), "{e:?}");
    assert_eq!(origin.total.load(Ordering::SeqCst), 0);
    assert!(!home.0.join("artifacts/tiny").exists());
    s.stop().await;
}

#[tokio::test]
async fn readers_arriving_together_share_one_fetch_of_the_artifact() {
    let (home, origin) = (Home::new("together"), Origin::new());
    let (s, c) = start(&home.0, &origin, true).await;
    let readers: Vec<_> = (0..4)
        .map(|i| {
            let path = if i % 2 == 0 { "weights.bin" } else { "hello.txt" };
            tokio::spawn(read(c.clone(), read_req("alice", "tiny", path, "")))
        })
        .collect();
    for (i, r) in readers.into_iter().enumerate() {
        let want = if i % 2 == 0 { big() } else { TEXT.to_vec() };
        assert_eq!(r.await.unwrap().unwrap(), want);
    }
    assert_eq!((origin.gets("weights.bin"), origin.gets("hello.txt")), (1, 1));
    s.stop().await;
}

#[tokio::test]
async fn outside_zoo_mode_a_read_never_fetches_and_accepting_needs_the_admin_flag() {
    let (home, origin) = (Home::new("plain"), Origin::new());
    let (s, c) = start(&home.0, &origin, false).await;
    let e = read(c.clone(), read_req("alice", "tiny", "hello.txt", "")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::FailedPrecondition, TURBO_E_INVALID_STATE), "{e:?}");
    assert!(e.message().contains("not installed"), "{e:?}");
    assert_eq!(origin.total.load(Ordering::SeqCst), 0);
    s.stop().await;
    // Neither admin nor zoo: acceptance is refused.
    let catalog = Catalog { home: home.0.clone(), index: Arc::new(index()), transport: Arc::new(origin.clone()) };
    let serve = Serve { catalog: Some(catalog), ..Serve::default() };
    let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let model =
        turbo_kserve::config::ModelConfig::parse(&format!("bundle={},device=select,sessions=1", model.display()))
            .unwrap();
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model], MAX, serve).await.unwrap();
    let mut c = TurboCatalogServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let req = RecordLicenceAcceptanceRequest { user: "alice".into(), licence: "CC-BY-SA-4.0".into(), terms: "".into() };
    let e = c.record_licence_acceptance(req).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Unimplemented, TURBO_E_UNSUPPORTED), "{e:?}");
    assert!(e.message().contains("--zoo"), "{e:?}");
    s.stop().await;
}
