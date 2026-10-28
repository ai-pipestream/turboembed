//! TurboCatalogService's reads end to end: DescribeModel, ResolveModel by
//! SHA-256, and ReadModelFile streaming an installed file in frames to a
//! user the licence check admits, refusing files not installed, pins that
//! differ, and cached bytes that no longer verify.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tonic::Code;
use tonic::transport::Channel;
use turbo_kserve::api::{
    TURBO_E_BUNDLE_INTEGRITY, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE,
};
use turbo_kserve::catalog::{Catalog, READ_CHUNK_BYTES};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::training::model_file_chunk::Frame;
use turbo_kserve::proto::training::turbo_catalog_service_client::TurboCatalogServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve::{Serve, Server};
use turbo_zoo::{Index, Sha256, Sha512, Transport, Url};

type Client = TurboCatalogServiceClient<Channel>;

const BASE: &str = "https://example.org/models";
const TEXT: &[u8] = b"hello hello hello\n";
const SA: &[u8] = b"a share-alike word list\n";

/// Bytes over two and a half frames, so the stream splits them.
fn big() -> Vec<u8> {
    (0..READ_CHUNK_BYTES * 5 / 2).map(|i| (i % 251) as u8).collect()
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

/// A permissive model of two files, one over several frames, and a
/// share-alike word list.
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

fn transport() -> Fixed {
    let mut files = HashMap::new();
    files.insert(format!("{BASE}/hello.txt"), TEXT.to_vec());
    files.insert(format!("{BASE}/weights.bin"), big());
    files.insert(format!("{BASE}/sa.txt"), SA.to_vec());
    Fixed(files)
}

struct Home(PathBuf);

impl Home {
    fn new(name: &str) -> Home {
        let p = std::env::temp_dir().join(format!("turbo-kserve-catalog-read-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        Home(p)
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

async fn start(home: &Path, admin: bool) -> (Server, Client) {
    let catalog = Catalog { home: home.to_path_buf(), index: Arc::new(index()), transport: Arc::new(transport()) };
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
    let serve = Serve { admin, catalog: Some(catalog), ..Serve::default() };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model], MAX, serve).await.unwrap();
    let c = TurboCatalogServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    (s, c)
}

fn refusal(s: &tonic::Status) -> (Code, i32) {
    let tc = s.metadata().get("turbo-code").unwrap_or_else(|| panic!("no turbo-code in {s:?}"));
    (s.code(), tc.to_str().unwrap().parse().unwrap())
}

async fn install(c: &mut Client, user: &str, id: &str) {
    let req = InstallModelRequest { user: user.into(), artifact_id: id.into() };
    let mut stream = c.install_model(req).await.unwrap().into_inner();
    while stream.message().await.unwrap().is_some() {}
}

async fn accept_sa(c: &mut Client, user: &str) {
    let req = RecordLicenceAcceptanceRequest { user: user.into(), licence: "CC-BY-SA-4.0".into(), terms: "".into() };
    c.record_licence_acceptance(req).await.unwrap();
}

fn read_req(user: &str, id: &str, path: &str, sha256: &str) -> ReadModelFileRequest {
    ReadModelFileRequest { user: user.into(), artifact_id: id.into(), path: path.into(), sha256: sha256.into() }
}

/// The catalog file, each data frame's length, and the bytes joined; the
/// stream's error when it ends in one.
async fn read(c: &mut Client, req: ReadModelFileRequest) -> Result<(CatalogFile, Vec<usize>, Vec<u8>), tonic::Status> {
    let mut stream = c.read_model_file(req).await?.into_inner();
    let Some(Frame::File(head)) = stream.message().await?.and_then(|m| m.frame) else { panic!("no file frame") };
    let (mut lens, mut bytes) = (Vec::new(), Vec::new());
    while let Some(m) = stream.message().await? {
        let Some(Frame::Data(d)) = m.frame else { panic!("a second file frame") };
        lens.push(d.len());
        bytes.extend_from_slice(&d);
    }
    Ok((head, lens, bytes))
}

#[tokio::test]
async fn describe_gives_the_listing_entry_and_names_what_is_not_there() {
    let home = Home::new("describe");
    let (s, mut c) = start(&home.0, false).await;
    let listed = c.list_model_catalog(ListModelCatalogRequest::default()).await.unwrap().into_inner().entries;
    let req = DescribeModelRequest { user: "".into(), artifact_id: "tiny".into() };
    let one = c.describe_model(req).await.unwrap().into_inner().entry.unwrap();
    assert_eq!(one, listed[0]);
    let req = DescribeModelRequest { user: "alice".into(), artifact_id: "sa".into() };
    let sa = c.describe_model(req).await.unwrap().into_inner().entry.unwrap();
    assert_eq!(sa.missing_terms.iter().map(|t| t.licence.as_str()).collect::<Vec<_>>(), ["CC-BY-SA-4.0"]);
    let req = DescribeModelRequest { user: "".into(), artifact_id: "nothing".into() };
    let e = c.describe_model(req).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND), "{e:?}");
    let e = c.describe_model(DescribeModelRequest::default()).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT), "{e:?}");
    s.stop().await;
}

#[tokio::test]
async fn resolve_finds_the_artifact_and_file_pinned_at_a_sha256() {
    let home = Home::new("resolve");
    let (s, mut c) = start(&home.0, false).await;
    let weights = Sha256::of(&big()).to_string();
    let req = ResolveModelRequest { sha256: weights.clone(), ..Default::default() };
    let r = c.resolve_model(req).await.unwrap().into_inner();
    assert_eq!((r.entry.unwrap().id.as_str(), r.path.as_str()), ("tiny", "weights.bin"));
    // Within another artifact the hash pins nothing.
    let req = ResolveModelRequest { sha256: weights, artifact_id: "sa".into(), ..Default::default() };
    let e = c.resolve_model(req).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND), "{e:?}");
    for bad in ["", "ABC", &Sha256::of(TEXT).to_string().to_uppercase()] {
        let req = ResolveModelRequest { sha256: bad.into(), ..Default::default() };
        let e = c.resolve_model(req).await.unwrap_err();
        assert_eq!(refusal(&e), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT), "{bad:?}: {e:?}");
    }
    s.stop().await;
}

#[tokio::test]
async fn an_installed_file_streams_in_frames_without_the_admin_flag() {
    let home = Home::new("stream");
    let (s, mut c) = start(&home.0, true).await;
    install(&mut c, "alice", "tiny").await;
    s.stop().await;
    // Reading is not an admin call.
    let (s, mut c) = start(&home.0, false).await;
    let pin = Sha256::of(&big()).to_string();
    let (head, lens, bytes) = read(&mut c, read_req("bob", "tiny", "weights.bin", &pin)).await.unwrap();
    assert_eq!(
        (head.path.as_str(), head.size, head.sha256.as_str()),
        ("weights.bin", big().len() as u64, pin.as_str())
    );
    assert_eq!(head.sha512, Sha512::of(&big()).to_string());
    assert_eq!(lens, [READ_CHUNK_BYTES, READ_CHUNK_BYTES, READ_CHUNK_BYTES / 2]);
    assert_eq!(bytes, big());
    let (_, lens, bytes) = read(&mut c, read_req("bob", "tiny", "hello.txt", "")).await.unwrap();
    assert_eq!((lens, bytes), (vec![TEXT.len()], TEXT.to_vec()));
    s.stop().await;
}

#[tokio::test]
async fn a_read_is_refused_before_any_byte_unless_admitted_listed_pinned_alike_and_installed() {
    let home = Home::new("refuse");
    let (s, mut c) = start(&home.0, true).await;
    let not_installed = read(&mut c, read_req("alice", "tiny", "hello.txt", "")).await.unwrap_err();
    assert_eq!(refusal(&not_installed), (Code::FailedPrecondition, TURBO_E_INVALID_STATE), "{not_installed:?}");
    assert!(not_installed.message().contains("not installed"), "{not_installed:?}");
    install(&mut c, "alice", "tiny").await;
    let other = Sha256::of(SA).to_string();
    let e = read(&mut c, read_req("alice", "tiny", "hello.txt", &other)).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::FailedPrecondition, TURBO_E_INVALID_STATE), "{e:?}");
    assert_eq!(e.metadata().get("turbo-field").unwrap().to_str().unwrap(), "4", "{e:?}");
    let e = read(&mut c, read_req("alice", "tiny", "nothing.txt", "")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND), "{e:?}");
    for (req, field) in [
        (read_req(" alice", "tiny", "hello.txt", ""), "1"),
        (read_req("alice", "", "hello.txt", ""), "2"),
        (read_req("alice", "tiny", "", ""), "3"),
        (read_req("alice", "tiny", "hello.txt", "xyz"), "4"),
    ] {
        let e = read(&mut c, req).await.unwrap_err();
        assert_eq!(refusal(&e), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT), "{e:?}");
        assert_eq!(e.metadata().get("turbo-field").unwrap().to_str().unwrap(), field, "{e:?}");
    }
    s.stop().await;
}

#[tokio::test]
async fn a_share_alike_file_reaches_only_a_user_who_accepted_its_terms() {
    let home = Home::new("accept");
    let (s, mut c) = start(&home.0, true).await;
    accept_sa(&mut c, "alice").await;
    install(&mut c, "alice", "sa").await;
    let (_, _, bytes) = read(&mut c, read_req("alice", "sa", "sa.txt", "")).await.unwrap();
    assert_eq!(bytes, SA);
    // Installed on the server, but bob has not accepted.
    let e = read(&mut c, read_req("bob", "sa", "sa.txt", "")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::FailedPrecondition, TURBO_E_INVALID_STATE), "{e:?}");
    assert!(e.message().contains("CC-BY-SA-4.0"), "{e:?}");
    s.stop().await;
}

#[tokio::test]
async fn cached_bytes_that_no_longer_verify_end_the_stream_with_data_loss() {
    let home = Home::new("tampered");
    let (s, mut c) = start(&home.0, true).await;
    install(&mut c, "alice", "tiny").await;
    let cached = home.0.join("artifacts/tiny/weights.bin");
    let mut bytes = fs::read(&cached).unwrap();
    bytes[READ_CHUNK_BYTES + 7] ^= 1;
    fs::write(&cached, &bytes).unwrap();
    let e = read(&mut c, read_req("alice", "tiny", "weights.bin", "")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::DataLoss, TURBO_E_BUNDLE_INTEGRITY), "{e:?}");
    s.stop().await;
}
