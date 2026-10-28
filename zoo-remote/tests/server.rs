//! The library reading models from a zoo server started in-process: the
//! local zoo's cache and checks, the server's licence check, a pin both
//! indexes must share, and `TURBO_ZOO_SOURCE` lists falling through to the
//! next source.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_kserve::catalog::Catalog;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::training::turbo_catalog_service_client::TurboCatalogServiceClient;
use turbo_kserve::proto::training::{InstallModelRequest, RecordLicenceAcceptanceRequest};
use turbo_kserve::{Serve, Server};
use turbo_zoo::{
    Acceptances, Blobs, Index, Licence, Offline, Policy, Requirement, Sha256, Sha512, Transport, Url, UserId, Zoo,
    ZooError,
};
use turbo_zoo_remote::{ZooServer, sources};

const BASE: &str = "https://example.org/models";
const TEXT: &[u8] = b"hello hello hello\n";
const SA: &[u8] = b"a share-alike word list\n";

/// Over three of the server's 64 KiB frames.
fn big() -> Vec<u8> {
    (0..200_000u32).map(|i| (i % 253) as u8).collect()
}

struct Fixed(HashMap<String, Vec<u8>>);

impl Transport for Fixed {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        self.0.get(url.as_str()).cloned().ok_or_else(|| format!("{url}: not served"))
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

/// A permissive model of two files and a share-alike word list, with
/// `weights` as the bytes the index pins for the model's second file.
fn index_with(weights: &[u8]) -> Index {
    let artifact = |id: &str, licence: &str, class: &str, files: String| {
        format!(
            r#"{{"id": "{id}", "kind": "KIND_LEXICON", "format": "FORMAT_TEXT", "version": "1",
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
            format!("{}, {}", file("hello.txt", TEXT), file("weights.bin", weights))
        ),
        artifact("sa", "CC-BY-SA-4.0", "CLASS_SHARE_ALIKE", file("sa.txt", SA)),
    );
    Index::parse(json.as_bytes()).unwrap()
}

fn index() -> Index {
    index_with(&big())
}

struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Dir {
        let p = std::env::temp_dir().join(format!("turbo-zoo-remote-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        Dir(p)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A server whose catalog lives in `home`, with `tiny` installed, and its
/// grpc:// address.
async fn serve(home: &Path) -> (Server, String) {
    let files = [("hello.txt", TEXT.to_vec()), ("weights.bin", big()), ("sa.txt", SA.to_vec())]
        .into_iter()
        .map(|(p, b)| (format!("{BASE}/{p}"), b))
        .collect();
    let catalog = Catalog { home: home.to_path_buf(), index: Arc::new(index()), transport: Arc::new(Fixed(files)) };
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
    let serve = Serve { admin: true, catalog: Some(catalog), ..Serve::default() };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model], MAX, serve).await.unwrap();
    let address = format!("grpc://{}", s.local_addr());
    install(&address, "tiny").await;
    (s, address)
}

async fn admin(address: &str) -> TurboCatalogServiceClient<tonic::transport::Channel> {
    let http = address.replace("grpc://", "http://");
    TurboCatalogServiceClient::connect(http).await.unwrap()
}

async fn install(address: &str, id: &str) {
    let req = InstallModelRequest { user: "admin".into(), artifact_id: id.into() };
    let mut stream = admin(address).await.install_model(req).await.unwrap().into_inner();
    while stream.message().await.unwrap().is_some() {}
}

async fn accept_on_server(address: &str, user: &str) {
    let req = RecordLicenceAcceptanceRequest { user: user.into(), licence: "CC-BY-SA-4.0".into(), terms: "".into() };
    admin(address).await.record_licence_acceptance(req).await.unwrap();
}

fn alice() -> UserId {
    "alice".parse().unwrap()
}

fn sa_accepted() -> Acceptances {
    let mut a = Acceptances::default();
    a.accept_terms(&alice(), &Requirement::of(Licence::CcBySa40).unwrap(), 1).unwrap();
    a
}

/// The local zoo fetching `id` through `source` into `cache`, off the
/// async runtime as a blocking fetch is made.
async fn fetch(
    source: Arc<dyn Transport + Send + Sync>,
    index: Index,
    cache: PathBuf,
    id: &'static str,
    acceptances: Acceptances,
) -> Result<Vec<Vec<u8>>, ZooError> {
    tokio::task::spawn_blocking(move || {
        let zoo = Zoo::new(index, cache, Policy::default(), &*source);
        let f = zoo.fetch(id, &alice(), &acceptances)?;
        Ok(f.files.iter().map(|p| fs::read(p).unwrap()).collect())
    })
    .await
    .unwrap()
}

async fn zoo_server(address: &str) -> Arc<dyn Transport + Send + Sync> {
    let a = address.to_owned();
    tokio::task::spawn_blocking(move || {
        Arc::new(ZooServer::new(&a, alice()).unwrap()) as Arc<dyn Transport + Send + Sync>
    })
    .await
    .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_library_loads_a_model_from_the_server_and_caches_it_locally() {
    let (home, cache) = (Dir::new("load-home"), Dir::new("load-cache"));
    let (s, address) = serve(&home.0).await;
    let server = zoo_server(&address).await;
    // The source goes with the fetch, so its connection closes before the
    // server stops, which waits for open connections.
    let got = fetch(server, index(), cache.0.clone(), "tiny", Acceptances::default()).await.unwrap();
    assert_eq!(got, [TEXT.to_vec(), big()]);
    s.stop().await;
    // With the server gone the local cache still serves, every file
    // verified again; nothing reaches the network.
    let offline: Arc<dyn Transport + Send + Sync> = Arc::new(Offline);
    let again = fetch(offline, index(), cache.0.clone(), "tiny", Acceptances::default()).await.unwrap();
    assert_eq!(again, got);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_share_alike_model_needs_the_terms_accepted_on_both_sides() {
    let (home, cache) = (Dir::new("sa-home"), Dir::new("sa-cache"));
    let (s, address) = serve(&home.0).await;
    accept_on_server(&address, "admin").await;
    install(&address, "sa").await;
    let server = zoo_server(&address).await;
    // Not accepted here: refused before the server is asked.
    let e = fetch(server.clone(), index(), cache.0.clone(), "sa", Acceptances::default()).await.unwrap_err();
    assert!(matches!(e, ZooError::NotAccepted { .. }), "{e:?}");
    // Accepted here but not on the server: the server refuses.
    let e = fetch(server.clone(), index(), cache.0.clone(), "sa", sa_accepted()).await.unwrap_err();
    assert!(matches!(&e, ZooError::Transport { message, .. } if message.contains("CC-BY-SA-4.0")), "{e:?}");
    accept_on_server(&address, "alice").await;
    let got = fetch(server, index(), cache.0.clone(), "sa", sa_accepted()).await.unwrap();
    assert_eq!(got, [SA.to_vec()]);
    s.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_server_whose_index_pins_other_bytes_sends_none() {
    let (home, cache) = (Dir::new("pin-home"), Dir::new("pin-cache"));
    let (s, address) = serve(&home.0).await;
    let server = zoo_server(&address).await;
    let ours = index_with(b"other weights");
    let e = fetch(server, ours, cache.0.clone(), "tiny", Acceptances::default()).await.unwrap_err();
    assert!(matches!(&e, ZooError::Transport { message, .. } if message.contains("pins SHA-256")), "{e:?}");
    assert!(!cache.0.join("tiny/weights.bin").exists());
    s.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_source_list_falls_through_to_the_next_source() {
    let (store, cache) = (Dir::new("list-store"), Dir::new("list-cache"));
    let blobs = Blobs::new(&store.0);
    blobs.put(TEXT).unwrap();
    blobs.put(&big()).unwrap();
    // Nothing listens on port 1: the server is down, the store has both.
    let spec = format!("grpc://127.0.0.1:1, file://{}", store.0.display());
    let list = tokio::task::spawn_blocking(move || sources(&spec, &alice(), Offline).unwrap()).await.unwrap();
    assert_eq!(list.len(), 2);
    let got = fetch(Arc::new(list), index(), cache.0.clone(), "tiny", Acceptances::default()).await.unwrap();
    assert_eq!(got, [TEXT.to_vec(), big()]);
}

#[test]
fn a_source_list_names_only_what_it_can_read() {
    let ok = sources("file:///zoo, grpcs://zoo.example:8081, origin", &alice(), Offline).unwrap();
    assert_eq!(ok.len(), 3);
    for (bad, says) in [
        ("", "an empty entry"),
        ("file:///zoo,,origin", "an empty entry"),
        ("origin, origin", "named twice"),
        ("s3://bucket/zoo", "a source is file://"),
        ("https://example.org/zoo", "a source is file://"),
        ("file://zoo", "absolute directory"),
        ("grpc://zoo.example:8081/path", "host and port alone"),
        ("grpc://", "host and port alone"),
    ] {
        let e = sources(bad, &alice(), Offline).err().unwrap_or_else(|| panic!("{bad:?} was accepted"));
        assert!(e.contains(says), "{bad:?}: {e}");
    }
    assert!(ZooServer::with_tls("grpc://zoo.example:8081", alice(), Some(b"pem")).is_err());
}
