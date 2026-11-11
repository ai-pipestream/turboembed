//! Several zoo servers on one data directory (`--catalog-data`): each has
//! its own home and cache, and the directory holds the files they fetch
//! and the acceptances users record. A permissive file lands under
//! `public/`, a share-alike one under `private/`, and an acceptance
//! recorded through one server is honoured, and revoked, through another.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tonic::transport::Channel;
use turbo_kserve::catalog::{Catalog, DataDir};
use turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES as MAX;
use turbo_kserve::proto::training::model_file_chunk::Frame;
use turbo_kserve::proto::training::turbo_catalog_service_client::TurboCatalogServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve::{Serve, Server};
use turbo_zoo::{Index, Licence, Requirement, Sha256, Sha512, Shown, Transport, Url, UserId};

type Client = TurboCatalogServiceClient<Channel>;

const BASE: &str = "https://example.org/models";
const TEXT: &[u8] = b"hello hello hello\n";
const SA: &[u8] = b"a share-alike word list\n";

fn big() -> Vec<u8> {
    (0..200_000).map(|i| (i % 249) as u8).collect()
}

/// The origin: fixed bytes by address, counting each GET, each one slow
/// enough that installs started together overlap.
#[derive(Clone, Default)]
struct Origin {
    files: Arc<HashMap<String, Vec<u8>>>,
    gets: Arc<Mutex<HashMap<String, usize>>>,
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

struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Dir {
        let p = std::env::temp_dir().join(format!("turbo-kserve-catalog-data-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        Dir(p)
    }

    fn join(&self, p: &str) -> PathBuf {
        self.0.join(p)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// A zoo-mode server with its own home `home`, on the data directory `data`.
async fn start(home: &Path, data: &Path, origin: &Origin) -> (Server, Client) {
    let data = DataDir::open(data).unwrap();
    let catalog = Catalog::on_data(home.to_path_buf(), index(), Arc::new(origin.clone()), data);
    let serve = Serve { zoo: true, catalog: Some(catalog), ..Serve::default() };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), Vec::new(), MAX, serve).await.unwrap();
    let c = TurboCatalogServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    (s, c)
}

async fn read(c: &mut Client, user: &str, id: &str, path: &str) -> Result<Vec<u8>, tonic::Status> {
    let req = ReadModelFileRequest { user: user.into(), artifact_id: id.into(), path: path.into(), sha256: "".into() };
    let mut stream = c.read_model_file(req).await?.into_inner();
    let Some(Frame::File(_)) = stream.message().await?.and_then(|m| m.frame) else { panic!("no file frame") };
    let mut bytes = Vec::new();
    while let Some(m) = stream.message().await? {
        let Some(Frame::Data(d)) = m.frame else { panic!("a second file frame") };
        bytes.extend_from_slice(&d);
    }
    Ok(bytes)
}

async fn install(mut c: Client, id: &str) -> Result<InstalledModel, tonic::Status> {
    let req = InstallModelRequest { user: "alice".into(), artifact_id: id.into() };
    let mut stream = c.install_model(req).await?.into_inner();
    loop {
        let Some(u) = stream.message().await? else { panic!("the install ended with no result") };
        if let Some(install_model_update::Update::Installed(m)) = u.update {
            return Ok(m);
        }
    }
}

fn sa_terms() -> (String, String) {
    ("CC-BY-SA-4.0".into(), String::new())
}

fn blob(data: &Dir, side: &str, bytes: &[u8]) -> PathBuf {
    data.join(side).join("blobs/sha256").join(Sha256::of(bytes).to_string())
}

/// The names in `<data>/<side>/blobs/sha256`, sorted.
fn blobs(data: &Dir, side: &str) -> Vec<String> {
    let dir = data.join(side).join("blobs/sha256");
    let mut names: Vec<String> = match fs::read_dir(&dir) {
        Ok(d) => d.map(|e| e.unwrap().file_name().into_string().unwrap()).collect(),
        Err(_) => Vec::new(),
    };
    names.sort();
    names
}

#[tokio::test(flavor = "multi_thread")]
async fn an_acceptance_recorded_on_one_server_is_honoured_and_revoked_on_another() {
    let (data, a_home, b_home) = (Dir::new("accept"), Dir::new("accept-a"), Dir::new("accept-b"));
    let origin = Origin::new();
    let (sa, mut a) = start(&a_home.0, &data.0, &origin).await;
    let (sb, mut b) = start(&b_home.0, &data.0, &origin).await;

    let refused = read(&mut b, "alice", "sa", "sa.txt").await.unwrap_err();
    assert_eq!(refused.code(), tonic::Code::FailedPrecondition, "{refused:?}");
    let (licence, terms) = sa_terms();
    let req = RecordLicenceAcceptanceRequest { user: "alice".into(), licence, terms, ..Default::default() };
    let recorded = a.record_licence_acceptance(req.clone()).await.unwrap().into_inner();
    // Accepting again through the other server keeps the first time.
    let again = b.record_licence_acceptance(req).await.unwrap().into_inner();
    assert_eq!(again.accepted_at, recorded.accepted_at);
    let listed = b.list_licence_acceptances(ListLicenceAcceptancesRequest { user: "alice".into() }).await.unwrap();
    assert_eq!(listed.into_inner().accepted.len(), 1);
    assert!(!a_home.join("acceptances.json").exists() && !b_home.join("acceptances.json").exists());

    assert_eq!(read(&mut b, "alice", "sa", "sa.txt").await.unwrap(), SA);
    // Kept under private/ only, and read there by the first server.
    assert!(blob(&data, "private", SA).is_file());
    assert!(!blob(&data, "public", SA).exists());
    assert_eq!(read(&mut a, "alice", "sa", "sa.txt").await.unwrap(), SA);
    assert_eq!(origin.gets("sa.txt"), 1);
    // Bob accepted nothing; the shared file is still refused to him.
    assert!(read(&mut a, "bob", "sa", "sa.txt").await.is_err());

    let (licence, terms) = sa_terms();
    let revoke = RevokeLicenceAcceptanceRequest { user: "alice".into(), licence, terms };
    assert!(b.revoke_licence_acceptance(revoke).await.unwrap().into_inner().was_accepted);
    let refused = read(&mut a, "alice", "sa", "sa.txt").await.unwrap_err();
    assert_eq!(refused.code(), tonic::Code::FailedPrecondition, "{refused:?}");
    drop((a, b));
    sa.stop().await;
    sb.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn a_file_one_server_fetched_is_served_by_another_without_a_fetch() {
    let (data, a_home, b_home) = (Dir::new("serve"), Dir::new("serve-a"), Dir::new("serve-b"));
    let origin = Origin::new();
    let (sa, mut a) = start(&a_home.0, &data.0, &origin).await;
    let (sb, mut b) = start(&b_home.0, &data.0, &origin).await;
    assert_eq!(read(&mut a, "alice", "tiny", "hello.txt").await.unwrap(), TEXT);
    assert_eq!((origin.gets("hello.txt"), origin.gets("weights.bin")), (1, 1));
    assert_eq!(fs::read(blob(&data, "public", &big())).unwrap(), big());
    assert_eq!(read(&mut b, "alice", "tiny", "weights.bin").await.unwrap(), big());
    assert_eq!((origin.gets("hello.txt"), origin.gets("weights.bin")), (1, 1));
    assert!(b_home.join("artifacts/tiny/weights.bin").is_file());
    assert!(blobs(&data, "private").is_empty());
    drop((a, b));
    sa.stop().await;
    sb.stop().await;
}

#[tokio::test(flavor = "multi_thread")]
async fn two_servers_installing_one_artifact_at_once_both_succeed_with_one_copy() {
    let (data, a_home, b_home) = (Dir::new("race"), Dir::new("race-a"), Dir::new("race-b"));
    let origin = Origin::new();
    let (sa, a) = start(&a_home.0, &data.0, &origin).await;
    let (sb, b) = start(&b_home.0, &data.0, &origin).await;
    let (x, y) = tokio::join!(install(a, "tiny"), install(b, "tiny"));
    let (x, y) = (x.unwrap(), y.unwrap());
    assert_eq!((x.id.as_str(), y.id.as_str()), ("tiny", "tiny"));
    let mut want = vec![Sha256::of(TEXT).to_string(), Sha256::of(&big()).to_string()];
    want.sort();
    // One verified copy of each, no temporary file left behind.
    assert_eq!(blobs(&data, "public"), want);
    assert_eq!(fs::read(blob(&data, "public", TEXT)).unwrap(), TEXT);
    assert_eq!(fs::read(blob(&data, "public", &big())).unwrap(), big());
    sa.stop().await;
    sb.stop().await;
}

#[cfg(unix)]
#[test]
fn a_private_side_open_to_other_users_is_refused() {
    use std::os::unix::fs::PermissionsExt;
    let data = Dir::new("open");
    let opened = DataDir::open(&data.0).unwrap();
    assert!(opened.writable());
    let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&data.join("private")) & 0o077, 0);
    assert!(data.join("public").is_dir());
    fs::set_permissions(data.join("private"), fs::Permissions::from_mode(0o755)).unwrap();
    let e = DataDir::open(&data.0).unwrap_err();
    assert!(e.contains("open to other users"), "{e}");
    assert!(DataDir::open("relative/data").unwrap_err().contains("absolute"));
}

/// A directory this process cannot write: served from, never fetched into.
/// Root writes through any mode, so there the check has nothing to show.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread")]
async fn a_read_only_data_directory_serves_what_it_holds_and_fetches_nothing() {
    use std::os::unix::fs::PermissionsExt;
    let (data, a_home, b_home) = (Dir::new("ro"), Dir::new("ro-a"), Dir::new("ro-b"));
    let origin = Origin::new();
    let (sa, mut a) = start(&a_home.0, &data.0, &origin).await;
    assert_eq!(read(&mut a, "alice", "tiny", "hello.txt").await.unwrap(), TEXT);
    let (licence, terms) = sa_terms();
    let req = RecordLicenceAcceptanceRequest { user: "alice".into(), licence, terms, ..Default::default() };
    a.record_licence_acceptance(req.clone()).await.unwrap();
    drop(a);
    sa.stop().await;
    let dirs = [("", 0o755), ("public", 0o755), ("public/blobs/sha256", 0o755), ("private", 0o700)];
    let set = |ro: bool| {
        for (p, m) in dirs {
            let m = if ro { m & 0o555 } else { m };
            fs::set_permissions(data.join(p), fs::Permissions::from_mode(m)).unwrap();
        }
    };
    set(true);
    if DataDir::open(&data.0).unwrap().writable() {
        set(false);
        return;
    }
    let (sb, mut b) = start(&b_home.0, &data.0, &origin).await;
    assert_eq!(read(&mut b, "alice", "tiny", "weights.bin").await.unwrap(), big());
    let e = b.record_licence_acceptance(req).await.unwrap_err();
    assert!(e.message().contains("read-only"), "{e:?}");
    // Accepted, but not held: refused with the reason, and nothing fetched.
    let e = read(&mut b, "alice", "sa", "sa.txt").await.unwrap_err();
    assert!(e.message().contains("read-only"), "{e:?}");
    assert_eq!((origin.gets("weights.bin"), origin.gets("sa.txt")), (1, 0));
    drop(b);
    sb.stop().await;
    set(false);
}

/// A record revoked on one server while another lists the acceptances is
/// gone, never a bad record: the listing keeps working for every user.
#[test]
fn a_record_revoked_while_another_server_reads_is_simply_gone() {
    let data = Dir::new("revoke-race");
    let (writer, reader) = (DataDir::open(&data.0).unwrap(), DataDir::open(&data.0).unwrap());
    let r = Requirement::of(Licence::CcBySa40).unwrap();
    let (alice, bob): (UserId, UserId) = ("alice".parse().unwrap(), "bob".parse().unwrap());
    writer.record(&bob, &r, &Shown::default()).unwrap();
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flips = {
        let (done, r, alice) = (done.clone(), r.clone(), alice.clone());
        std::thread::spawn(move || {
            for _ in 0..2000 {
                writer.record(&alice, &r, &Shown::default()).unwrap();
                assert!(writer.revoke(&alice, &r).unwrap());
            }
            done.store(true, std::sync::atomic::Ordering::SeqCst);
        })
    };
    let mut reads = 0;
    while !done.load(std::sync::atomic::Ordering::SeqCst) {
        let a = reader.acceptances().unwrap_or_else(|e| panic!("read {reads}: {e}"));
        assert!(a.has(&bob, &r));
        reads += 1;
    }
    flips.join().unwrap();
    assert!(!reader.acceptances().unwrap().has(&alice, &r));
}

#[test]
fn a_store_and_a_data_directory_together_are_refused() {
    let (data, home) = (Dir::new("both"), Dir::new("both-home"));
    let e = Catalog::http(home.0.clone(), None, Some("s3://models/zoo"), Some(&data.0)).unwrap_err();
    assert_eq!(e, "--catalog-store and --catalog-data are two stores; give one");
}
