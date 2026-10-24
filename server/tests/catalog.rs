//! TurboCatalogService end to end: the listing, the admin opt-in, licence
//! acceptance, and installs over a transport serving fixed bytes, with
//! their progress, integrity, concurrency and disk checks.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use tonic::Code;
use tonic::transport::Channel;
use turbo_kserve::api::{
    TURBO_E_BUNDLE_INTEGRITY, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_BUSY, TURBO_E_CAPACITY, TURBO_E_INVALID_ARGUMENT,
    TURBO_E_INVALID_STATE, TURBO_E_UNSUPPORTED,
};
use turbo_kserve::catalog::Catalog;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::training::install_model_update::Update;
use turbo_kserve::proto::training::turbo_catalog_service_client::TurboCatalogServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve::{Serve, Server};
use turbo_zoo::{Index, Sha256, Sha512, Transport, Url};

type Client = TurboCatalogServiceClient<Channel>;

const TEXT: &[u8] = b"hello hello hello\n";

/// TEXT as Python's `gzip.compress(..., mtime=0)` writes it.
const TEXT_GZ: [u8; 29] = [
    0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x90,
    0x5c, 0x00, 0x3b, 0x7c, 0x8a, 0xdf, 0x12, 0x00, 0x00, 0x00,
];
const SA: &[u8] = b"a share-alike word list\n";
const BASE: &str = "https://example.org/models";

/// Serves fixed bytes by address. With a gate, each get waits for the
/// gate's signal first.
#[derive(Default)]
struct Fixed {
    files: HashMap<String, Vec<u8>>,
    gets: AtomicUsize,
    gate: Mutex<Option<mpsc::Receiver<()>>>,
}

impl Transport for Fixed {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        self.gets.fetch_add(1, Ordering::SeqCst);
        if let Some(gate) = self.gate.lock().unwrap().take() {
            gate.recv().unwrap();
        }
        self.files.get(url.as_str()).cloned().ok_or_else(|| format!("{url}: not served"))
    }
}

fn file(path: &str, bytes: &[u8], wire: Option<&[u8]>) -> String {
    let download = wire.map_or(String::new(), |w| {
        format!(
            r#", "download": {{"compression": "gzip", "download_bytes": {}, "download_sha256": "{}"}}"#,
            w.len(),
            Sha256::of(w)
        )
    });
    format!(
        r#"{{"path": "{path}", "url": "{BASE}/{path}", "size": {}, "sha256": "{}", "sha512": "{}"{download}}}"#,
        bytes.len(),
        Sha256::of(bytes),
        Sha512::of(bytes),
    )
}

/// An index of a permissive English lexicon served gzip compressed, a
/// share-alike German one, and one far larger than any disk.
fn index() -> Index {
    let artifact = |id: &str, lang: &str, licence: &str, class: &str, files: String| {
        format!(
            r#"{{"id": "{id}", "kind": "KIND_LEXICON", "language": "{lang}", "format": "FORMAT_TEXT", "version": "1",
              "license": "{licence}", "license_terms": "https://example.org/{id}/LICENSE", "trained_on": [],
              "class": "{class}", "attribution": "Example attribution for {id}.", "files": [{files}]}}"#
        )
    };
    let huge = format!(
        r#"{{"path": "huge.bin", "url": "{BASE}/huge.bin", "size": {}, "sha256": "{}", "sha512": "{}"}}"#,
        1u64 << 60,
        Sha256::of(b""),
        Sha512::of(b""),
    );
    let json = format!(
        r#"{{"zoo_version": 1, "artifacts": [{}, {}, {}]}}"#,
        artifact("hello", "en", "Apache-2.0", "CLASS_PERMISSIVE", file("hello.txt", TEXT, Some(&TEXT_GZ))),
        artifact("sa", "de", "CC-BY-SA-4.0", "CLASS_SHARE_ALIKE", file("sa.txt", SA, None)),
        artifact("huge", "en", "Apache-2.0", "CLASS_PERMISSIVE", huge),
    );
    Index::parse(json.as_bytes()).unwrap()
}

fn transport() -> Fixed {
    let mut files = HashMap::new();
    files.insert(format!("{BASE}/hello.txt"), TEXT_GZ.to_vec());
    files.insert(format!("{BASE}/sa.txt"), SA.to_vec());
    Fixed { files, ..Fixed::default() }
}

struct Home(PathBuf);

impl Home {
    fn new(name: &str) -> Home {
        let p = std::env::temp_dir().join(format!("turbo-kserve-catalog-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        Home(p)
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

async fn start(home: &Path, admin: bool, transport: Arc<Fixed>) -> (Server, Client) {
    let catalog = Catalog { home: home.to_path_buf(), index: Arc::new(index()), transport };
    start_with(Serve { admin, catalog: Some(catalog) }).await
}

async fn start_with(serve: Serve) -> (Server, Client) {
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
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model], MAX, serve).await.unwrap();
    let c = TurboCatalogServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    (s, c)
}

fn refusal(s: &tonic::Status) -> (Code, i32) {
    let tc = s.metadata().get("turbo-code").unwrap_or_else(|| panic!("no turbo-code in {s:?}"));
    (s.code(), tc.to_str().unwrap().parse().unwrap())
}

fn install_req(user: &str, id: &str) -> InstallModelRequest {
    InstallModelRequest { user: user.into(), artifact_id: id.into() }
}

/// Every update of an install, the stream's error last when it ends in one.
async fn install(c: &mut Client, user: &str, id: &str) -> Result<Vec<Update>, tonic::Status> {
    let mut stream = c.install_model(install_req(user, id)).await?.into_inner();
    let mut out = Vec::new();
    while let Some(u) = stream.message().await? {
        out.push(u.update.unwrap());
    }
    Ok(out)
}

fn stages(updates: &[Update]) -> Vec<(i32, u64)> {
    updates
        .iter()
        .filter_map(|u| match u {
            Update::Progress(p) => Some((p.stage, p.bytes)),
            Update::Installed(_) => None,
        })
        .collect()
}

fn sa_terms() -> LicenceTerms {
    LicenceTerms {
        licence: "CC-BY-SA-4.0".into(),
        text: "https://creativecommons.org/licenses/by-sa/4.0/legalcode".into(),
    }
}

#[tokio::test]
async fn the_catalog_lists_every_artifact_with_its_terms_and_files() {
    let home = Home::new("list");
    let (s, mut c) = start(&home.0, false, Arc::new(transport())).await;
    let all = c.list_model_catalog(ListModelCatalogRequest::default()).await.unwrap().into_inner().entries;
    assert_eq!(all.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["hello", "sa", "huge"]);
    let hello = &all[0];
    assert_eq!((hello.kind.as_str(), hello.format.as_str()), ("KIND_LEXICON", "FORMAT_TEXT"));
    assert_eq!((hello.licence.as_str(), hello.licence_class.as_str()), ("Apache-2.0", "CLASS_PERMISSIVE"));
    assert_eq!(hello.attribution, "Example attribution for hello.");
    assert!(hello.terms.is_empty() && !hello.installed);
    let f = &hello.files[0];
    assert_eq!((f.path.as_str(), f.size, f.compression.as_str()), ("hello.txt", TEXT.len() as u64, "gzip"));
    assert_eq!((f.download_bytes, f.download_sha256.clone()), (TEXT_GZ.len() as u64, Sha256::of(&TEXT_GZ).to_string()));
    assert_eq!(f.sha512, Sha512::of(TEXT).to_string());
    assert_eq!(all[1].terms, [sa_terms()]);
    // No user, so nothing is missing; with one, the unaccepted terms are.
    assert!(all[1].missing_terms.is_empty());
    let req = ListModelCatalogRequest { user: "alice".into(), language: "de".into(), ..Default::default() };
    let de = c.list_model_catalog(req).await.unwrap().into_inner().entries;
    assert_eq!((de.len(), de[0].missing_terms.clone()), (1, vec![sa_terms()]));
    let req = ListModelCatalogRequest { kind: "KIND_MODEL".into(), ..Default::default() };
    assert!(c.list_model_catalog(req).await.unwrap().into_inner().entries.is_empty());
    let req = ListModelCatalogRequest { user: " alice".into(), ..Default::default() };
    let e = c.list_model_catalog(req).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT), "{e:?}");
    s.stop().await;
}

#[tokio::test]
async fn mutations_need_the_admin_flag_and_every_call_needs_a_catalog() {
    let home = Home::new("admin");
    let (s, mut c) = start(&home.0, false, Arc::new(transport())).await;
    let req = RecordLicenceAcceptanceRequest { user: "alice".into(), licence: "CC-BY-SA-4.0".into(), terms: "".into() };
    let e = c.record_licence_acceptance(req).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Unimplemented, TURBO_E_UNSUPPORTED));
    let e = c.install_model(install_req("alice", "hello")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Unimplemented, TURBO_E_UNSUPPORTED));
    assert!(!home.0.exists(), "a refused call writes nothing");
    s.stop().await;
    let (s, mut c) = start_with(Serve { admin: true, catalog: None }).await;
    let e = c.list_installed_models(ListInstalledModelsRequest {}).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Unimplemented, TURBO_E_UNSUPPORTED));
    s.stop().await;
}

#[tokio::test]
async fn an_install_streams_its_progress_and_a_second_fetches_nothing() {
    let home = Home::new("install");
    let t = Arc::new(transport());
    let (s, mut c) = start(&home.0, true, t.clone()).await;
    let updates = install(&mut c, "alice", "hello").await.unwrap();
    let (n, wire) = (TEXT.len() as u64, TEXT_GZ.len() as u64);
    let want = [
        (InstallStage::Admitted as i32, n),
        (InstallStage::Downloading as i32, wire),
        (InstallStage::Downloaded as i32, wire),
        (InstallStage::Verified as i32, n),
    ];
    assert_eq!(stages(&updates), want);
    let Some(Update::Installed(m)) = updates.last() else { panic!("{updates:?}") };
    assert_eq!((m.id.as_str(), m.total_bytes), ("hello", n));
    assert_eq!(fs::read(&m.files[0]).unwrap(), TEXT, "the cache holds the unpacked file");
    let listed = c.list_installed_models(ListInstalledModelsRequest {}).await.unwrap().into_inner().models;
    assert_eq!(listed, std::slice::from_ref(m));
    let again = install(&mut c, "alice", "hello").await.unwrap();
    let want = [(InstallStage::Admitted as i32, 0), (InstallStage::Verified as i32, n)];
    assert_eq!((stages(&again), t.gets.load(Ordering::SeqCst)), (want.to_vec(), 1));
    s.stop().await;
}

#[tokio::test]
async fn a_share_alike_artifact_installs_only_after_its_terms_are_accepted() {
    let home = Home::new("accept");
    let (s, mut c) = start(&home.0, true, Arc::new(transport())).await;
    let e = install(&mut c, "alice", "sa").await.unwrap_err();
    assert_eq!(refusal(&e), (Code::FailedPrecondition, TURBO_E_INVALID_STATE), "{e:?}");
    assert!(e.message().contains("CC-BY-SA-4.0"), "{e:?}");
    let record = |licence: &str, terms: &str| RecordLicenceAcceptanceRequest {
        user: "alice".into(),
        licence: licence.into(),
        terms: terms.into(),
    };
    for (bad, field) in [
        (record("GPL-9.0", ""), "2"),
        (record("Apache-2.0", ""), "2"),
        (record("CC-BY-SA-4.0", "https://x.example/"), "3"),
    ] {
        let e = c.record_licence_acceptance(bad).await.unwrap_err();
        assert_eq!(refusal(&e), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT), "{e:?}");
        assert_eq!(e.metadata().get("turbo-field").unwrap().to_str().unwrap(), field, "{e:?}");
    }
    let first = c.record_licence_acceptance(record("CC-BY-SA-4.0", "")).await.unwrap().into_inner();
    assert_eq!((first.user.as_str(), first.accepted.clone()), ("alice", Some(sa_terms())));
    let repeat = c.record_licence_acceptance(record("CC-BY-SA-4.0", "")).await.unwrap().into_inner();
    assert_eq!(repeat.accepted_at, first.accepted_at, "accepting again keeps the first time");
    let req = ListModelCatalogRequest { user: "alice".into(), language: "de".into(), ..Default::default() };
    assert!(c.list_model_catalog(req).await.unwrap().into_inner().entries[0].missing_terms.is_empty());
    let e = install(&mut c, "bob", "sa").await.unwrap_err();
    assert_eq!(refusal(&e).1, TURBO_E_INVALID_STATE, "one user's acceptance is not another's");
    let updates = install(&mut c, "alice", "sa").await.unwrap();
    assert!(matches!(updates.last(), Some(Update::Installed(m)) if m.id == "sa"), "{updates:?}");
    s.stop().await;
}

#[tokio::test]
async fn bytes_that_do_not_verify_end_the_install_and_are_never_cached() {
    let home = Home::new("integrity");
    let mut t = transport();
    t.files.insert(format!("{BASE}/hello.txt"), b"not the file".to_vec());
    let (s, mut c) = start(&home.0, true, Arc::new(t)).await;
    let e = install(&mut c, "alice", "hello").await.unwrap_err();
    assert_eq!(refusal(&e), (Code::DataLoss, TURBO_E_BUNDLE_INTEGRITY), "{e:?}");
    assert!(!home.0.join("artifacts/hello/hello.txt").exists());
    assert!(c.list_installed_models(ListInstalledModelsRequest {}).await.unwrap().into_inner().models.is_empty());
    s.stop().await;
}

#[tokio::test]
async fn one_install_runs_at_a_time() {
    let home = Home::new("busy");
    let (release, gate) = mpsc::channel();
    let t = transport();
    *t.gate.lock().unwrap() = Some(gate);
    let (s, mut c) = start(&home.0, true, Arc::new(t)).await;
    let mut first = c.install_model(install_req("alice", "hello")).await.unwrap().into_inner();
    // Admitted, then downloading: the transport now waits on the gate.
    for stage in [InstallStage::Admitted, InstallStage::Downloading] {
        let Some(Update::Progress(p)) = first.message().await.unwrap().unwrap().update else { panic!() };
        assert_eq!(p.stage, stage as i32);
    }
    let e = c.install_model(install_req("alice", "sa")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Unavailable, TURBO_E_BUSY), "{e:?}");
    release.send(()).unwrap();
    let mut last = None;
    while let Some(u) = first.message().await.unwrap() {
        last = u.update;
    }
    assert!(matches!(last, Some(Update::Installed(_))));
    // The slot is free again once the first install ends.
    let e = install(&mut c, "alice", "nothing-by-this-id").await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND), "{e:?}");
    s.stop().await;
}

#[tokio::test]
async fn an_install_that_would_not_leave_the_disk_margin_is_refused_before_any_fetch() {
    let home = Home::new("disk");
    let t = Arc::new(transport());
    let (s, mut c) = start(&home.0, true, t.clone()).await;
    let e = install(&mut c, "alice", "huge").await.unwrap_err();
    assert_eq!(refusal(&e), (Code::OutOfRange, TURBO_E_CAPACITY), "{e:?}");
    assert!(e.message().contains("margin of 67108864 bytes"), "{e:?}");
    assert_eq!(t.gets.load(Ordering::SeqCst), 0);
    let e = install(&mut c, "", "hello").await.unwrap_err();
    assert_eq!(refusal(&e), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT), "{e:?}");
    s.stop().await;
}
