//! A zoo server serving an encoder bundle (#2213): the server lists a
//! sealed bundle imported into its index, fetches it on the first read,
//! and the library loads it over `grpc://` with every file verified
//! against the index and the bundle's own manifest. The bundle is the
//! repository's small sealed one under testdata.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_kserve::catalog::Catalog;
use turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES as MAX;
use turbo_kserve::proto::training::ListModelCatalogRequest;
use turbo_kserve::proto::training::turbo_catalog_service_client::TurboCatalogServiceClient;
use turbo_kserve::{Serve, Server};
use turbo_zoo::import::{BundleImport, Described, bundle};
use turbo_zoo::{Acceptances, Directory, Index, ModelType, Offline, Policy, Transport, UserId, Zoo, ZooError};
use turbo_zoo_remote::ZooServer;

fn sealed() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle")
}

/// An index of the one sealed bundle, imported as `tiny`.
fn index() -> Index {
    let spec = BundleImport {
        id: "tiny".parse().unwrap(),
        base: "https://models.example.org/bundles/tiny".parse().unwrap(),
        described: Described {
            license_terms: "https://models.example.org/bundles/tiny/LICENSE".parse().unwrap(),
            trained_on: Vec::new(),
            attribution: Some("Example attribution for tiny.".into()),
            origin: None,
            component: None,
            language: None,
        },
    };
    let entry = bundle(&sealed(), spec).unwrap();
    Index::parse(Index::to_canonical_json(&[entry]).as_bytes()).unwrap()
}

struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Dir {
        let p = std::env::temp_dir().join(format!("turbo-zoo-remote-bundle-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        Dir(p)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A zoo server with no model of its own, whose catalog lists `index`
/// and reads the bundle's files from the sealed directory.
async fn serve(home: &Path, index: Index) -> (Server, String) {
    let catalog = Catalog {
        home: home.to_path_buf(),
        index: Arc::new(index),
        transport: Arc::new(Directory::new(sealed())),
        data: None,
    };
    let serve = Serve { zoo: true, catalog: Some(catalog), ..Serve::default() };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), Vec::new(), MAX, serve).await.unwrap();
    let address = format!("grpc://{}", s.local_addr());
    (s, address)
}

fn alice() -> UserId {
    "alice".parse().unwrap()
}

/// The local zoo fetching `tiny` through `source` into `cache`: each
/// file's path in the bundle and its bytes.
async fn fetch(
    source: Arc<dyn Transport + Send + Sync>,
    index: Index,
    cache: PathBuf,
) -> Result<Vec<(String, Vec<u8>)>, ZooError> {
    tokio::task::spawn_blocking(move || {
        let zoo = Zoo::new(index, cache, Policy::default(), &*source);
        let f = zoo.fetch("tiny", &alice(), &Acceptances::default())?;
        Ok(f.artifact.files.iter().zip(&f.files).map(|(i, p)| (i.path.to_string(), fs::read(p).unwrap())).collect())
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
async fn the_server_lists_an_encoder_bundle_and_the_library_loads_it() {
    let (home, cache) = (Dir::new("load-home"), Dir::new("load-cache"));
    let (s, address) = serve(&home.0, index()).await;

    let mut client = TurboCatalogServiceClient::connect(address.replace("grpc://", "http://")).await.unwrap();
    let listed = client.list_model_catalog(ListModelCatalogRequest::default()).await.unwrap().into_inner();
    let entry = listed.entries.iter().find(|e| e.id == "tiny").expect("the bundle is listed");
    assert_eq!(entry.format, "FORMAT_BUNDLE");
    let paths: Vec<&str> = entry.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(
        paths,
        ["manifest.json", "reference/reference.safetensors", "tokenizer.json", "weights/model.safetensors"]
    );
    assert!(!entry.installed, "nothing is fetched until the first read");
    drop(client);

    // The first read makes the server fetch the bundle; the library keeps
    // its own verified copy, laid out as the bundle directory.
    let got = fetch(zoo_server(&address).await, index(), cache.0.clone()).await.unwrap();
    assert_eq!(got.len(), 4);
    for (path, bytes) in &got {
        assert_eq!(bytes, &fs::read(sealed().join(path)).unwrap(), "{path}");
        assert_eq!(bytes, &fs::read(cache.0.join("tiny").join(path)).unwrap(), "{path}");
        assert!(home.0.join("artifacts").join("tiny").join(path).is_file(), "the server holds {path}");
    }
    assert_eq!(index().get("tiny").unwrap().model_type, ModelType::EncoderBundle);
    s.stop().await;

    // With the server gone the local copy still serves, checked again.
    let offline: Arc<dyn Transport + Send + Sync> = Arc::new(Offline);
    assert_eq!(fetch(offline, index(), cache.0.clone()).await.unwrap(), got);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bundle_whose_manifest_is_not_the_entrys_is_not_handed_out() {
    // The same files under an entry that records another model: every
    // file verifies, and the manifest does not describe the entry.
    let other = Index::to_canonical_json(index().list()).replace("turboembed/tiny-bert", "turboembed/other");
    let other = || Index::parse(other.as_bytes()).unwrap();
    let (home, cache) = (Dir::new("other-home"), Dir::new("other-cache"));

    // A server with that index refuses the read.
    let (s, address) = serve(&home.0, other()).await;
    let e = fetch(zoo_server(&address).await, other(), cache.0.clone()).await.unwrap_err();
    assert!(matches!(e, ZooError::Transport { .. }), "{e:?}");
    assert!(e.to_string().contains("/model/id"), "{e}");
    // The files it fetched are on its disk, and it does not list the
    // bundle as installed.
    assert!(home.0.join("artifacts").join("tiny").join("manifest.json").is_file());
    let mut client = TurboCatalogServiceClient::connect(address.replace("grpc://", "http://")).await.unwrap();
    let listed = client.list_model_catalog(ListModelCatalogRequest::default()).await.unwrap().into_inner();
    assert!(!listed.entries.iter().find(|e| e.id == "tiny").unwrap().installed);
    drop(client);
    s.stop().await;

    // A library with that index refuses what a correct server serves.
    let (home, cache) = (Dir::new("mine-home"), Dir::new("mine-cache"));
    let (s, address) = serve(&home.0, index()).await;
    let e = fetch(zoo_server(&address).await, other(), cache.0.clone()).await.unwrap_err();
    s.stop().await;
    // Both sides pin the same bytes, so every file arrives and verifies;
    // the library's own manifest check then refuses the bundle.
    assert!(matches!(e, ZooError::Bundle { .. }), "{e:?}");
}
