//! TurboCatalogService's model sources end to end (#1744): a user's
//! sources registered, listed without their secrets and removed, on the
//! zoo's own holders.json; what zoo mode takes from a caller; Google
//! sign-in against a token endpoint started in-process; and an install
//! that reads from the user's own store before the catalog's transport.

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tonic::Code;
use tonic::transport::Channel;
use turbo_kserve::api::{TURBO_E_INVALID_ARGUMENT, TURBO_E_UNSUPPORTED};
use turbo_kserve::catalog::{Catalog, DriveSignIn};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::training::install_model_update::Update;
use turbo_kserve::proto::training::turbo_catalog_service_client::TurboCatalogServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve::{Serve, Server};
use turbo_zoo::{Blobs, Holders, Index, Sha256, Sha512, SignIn, Transport, Url};

type Client = TurboCatalogServiceClient<Channel>;

const WORDS: &[u8] = b"a permissive word list\n";

/// The catalog's own transport has nothing: what an install gets comes
/// from the user's sources.
struct Nothing;

impl Transport for Nothing {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        Err(format!("{url}: not served"))
    }
}

fn index() -> Index {
    let json = format!(
        r#"{{"zoo_version": 1, "artifacts": [{{"id": "free", "kind": "KIND_LEXICON", "language": "en",
          "format": "FORMAT_TEXT", "version": "1", "license": "Apache-2.0",
          "license_terms": "https://example.org/free/LICENSE", "trained_on": [], "class": "CLASS_PERMISSIVE",
          "attribution": "Example.", "files": [{{"path": "words.txt", "url": "https://example.org/words.txt",
          "size": {}, "sha256": "{}", "sha512": "{}"}}]}}]}}"#,
        WORDS.len(),
        Sha256::of(WORDS),
        Sha512::of(WORDS),
    );
    Index::parse(json.as_bytes()).unwrap()
}

struct Home(PathBuf);

impl Home {
    fn new(name: &str) -> Home {
        let p = std::env::temp_dir().join(format!("turbo-kserve-catalog-sources-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        Home(p)
    }

    fn holders(&self) -> Holders {
        Holders::load(&self.0.join("zoo/holders.json")).unwrap()
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

async fn start(home: &Path, serve: Serve) -> (Server, Client) {
    let catalog = Catalog { home: home.join("zoo"), index: Arc::new(index()), transport: Arc::new(Nothing) };
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
    let models = if serve.zoo { Vec::new() } else { vec![model] };
    let serve = Serve { catalog: Some(catalog), ..serve };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, serve).await.unwrap();
    let c = TurboCatalogServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    (s, c)
}

fn refusal(s: &tonic::Status) -> (Code, i32, String) {
    let md = |k: &str| s.metadata().get(k).unwrap_or_else(|| panic!("no {k} in {s:?}")).to_str().unwrap().to_owned();
    (s.code(), md("turbo-code").parse().unwrap(), md("turbo-field"))
}

fn register(user: &str, name: &str, address: &str, api_key: &str) -> RegisterModelSourceRequest {
    RegisterModelSourceRequest {
        user: user.into(),
        name: name.into(),
        address: address.into(),
        api_key: api_key.into(),
    }
}

async fn list(c: &mut Client, user: &str) -> ListModelSourcesResponse {
    c.list_model_sources(ListModelSourcesRequest { user: user.into() }).await.unwrap().into_inner()
}

#[tokio::test]
async fn sources_are_registered_listed_without_secrets_and_removed() {
    let home = Home::new("admin");
    let (_s, mut c) = start(&home.0, Serve { admin: true, ..Serve::default() }).await;
    let store = format!("file://{}", home.0.join("store").display());
    c.register_model_source(register("alice", "disk", &store, "")).await.unwrap();
    let got = c.register_model_source(register("alice", "shared", "gdrive://1AbC_d-E/models", "key-1")).await.unwrap();
    assert_eq!(got.into_inner().source.unwrap().sign_in, SourceSignIn::ApiKey as i32);
    c.register_model_source(register("bob", "disk", &store, "")).await.unwrap();

    let r = list(&mut c, "alice").await;
    assert!(r.admin && !r.drive_sign_in);
    let names: Vec<_> = r.sources.iter().map(|s| (s.name.as_str(), s.address.as_str())).collect();
    assert_eq!(names, [("disk", store.as_str()), ("shared", "gdrive://1AbC_d-E/models")]);
    assert!(!format!("{r:?}").contains("key-1"), "a listing holds no secret");
    let kept = home.holders().list(&"alice".parse().unwrap())[1].clone();
    assert_eq!(kept.sign_in, SignIn::ApiKey { key: "key-1".into() });

    // The same name replaces the source where it was.
    c.register_model_source(register("alice", "disk", "s3://models/zoo", "")).await.unwrap();
    assert_eq!(list(&mut c, "alice").await.sources[0].address, "s3://models/zoo");
    let gone = c.remove_model_source(RemoveModelSourceRequest { user: "alice".into(), name: "disk".into() }).await;
    assert!(gone.unwrap().into_inner().was_registered);
    let again = c.remove_model_source(RemoveModelSourceRequest { user: "alice".into(), name: "disk".into() }).await;
    assert!(!again.unwrap().into_inner().was_registered);
    assert_eq!(list(&mut c, "alice").await.sources.len(), 1);
    assert_eq!(list(&mut c, "bob").await.sources.len(), 1);
    assert!(list(&mut c, "").await.sources.is_empty());

    for (req, field) in [
        (register("alice", "", &store, ""), "2"),
        (register("alice", "a,b", &store, ""), "2"),
        (register("alice", "rel", "file://relative", ""), "3"),
        (register("alice", "ftp", "ftp://example.org/zoo", ""), "3"),
        (register("alice", "bad", "gdrive://a b", "k"), "3"),
        (register("alice", "keyed", &store, "k"), "4"),
    ] {
        let e = c.register_model_source(req.clone()).await.unwrap_err();
        assert_eq!(refusal(&e), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT, field.into()), "{req:?}: {e:?}");
    }
}

#[tokio::test]
async fn zoo_mode_takes_only_drive_sources_signed_in_by_the_user() {
    let home = Home::new("zoo");
    let (_s, mut c) = start(&home.0, Serve { zoo: true, ..Serve::default() }).await;
    for address in ["file:///srv/zoo", "s3://models/zoo", "gdrive://1AbC"] {
        let e = c.register_model_source(register("alice", "x", address, "")).await.unwrap_err();
        assert_eq!(refusal(&e).1, TURBO_E_UNSUPPORTED, "{address}: {e:?}");
    }
    c.register_model_source(register("alice", "shared", "gdrive://1AbC", "key-1")).await.unwrap();
    assert_eq!(list(&mut c, "alice").await.sources.len(), 1);
    // No sign-in configured.
    let e = c
        .start_drive_sign_in(StartDriveSignInRequest {
            user: "alice".into(),
            name: "d".into(),
            address: "gdrive://root".into(),
        })
        .await
        .unwrap_err();
    assert!(e.message().contains("TURBO_KSERVE_DRIVE_CLIENT_ID"), "{e:?}");

    let home = Home::new("plain");
    let (_s, mut c) = start(&home.0, Serve::default()).await;
    let e = c.register_model_source(register("alice", "shared", "gdrive://1AbC", "k")).await.unwrap_err();
    assert!(e.message().contains("admin-only"), "{e:?}");
    assert!(!list(&mut c, "alice").await.admin);
}

/// Answers the token endpoint: `code-1` grants a refresh token.
fn token_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let mut conn = conn.unwrap();
            let mut reader = BufReader::new(conn.try_clone().unwrap());
            let mut length = 0;
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).unwrap();
                if let Some(n) = h.to_ascii_lowercase().strip_prefix("content-length: ") {
                    length = n.trim().parse().unwrap();
                }
                if h.trim_end().is_empty() {
                    break;
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let body = String::from_utf8(body).unwrap();
            let ok = body.contains("code=code-1") && body.contains("client_secret=secret-1");
            let (status, reply) = if ok {
                ("200 OK", r#"{"access_token": "tok-1", "expires_in": 3600, "refresh_token": "refresh-1"}"#)
            } else {
                ("400 Bad Request", r#"{"error": "invalid_grant"}"#)
            };
            let head = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", reply.len());
            conn.write_all(head.as_bytes()).unwrap();
            conn.write_all(reply.as_bytes()).unwrap();
        }
    });
    endpoint
}

#[tokio::test]
async fn a_drive_sign_in_registers_the_users_grant() {
    let endpoint = token_endpoint();
    let drive = DriveSignIn {
        auth_endpoint: format!("{endpoint}/auth"),
        token_endpoint: format!("{endpoint}/token"),
        ..DriveSignIn::google("client-1", Some("secret-1"), "https://demo.example/")
    };
    let home = Home::new("sign-in");
    let (_s, mut c) = start(&home.0, Serve { zoo: true, drive: Some(drive), ..Serve::default() }).await;
    assert!(list(&mut c, "alice").await.drive_sign_in);
    let begin = |user: &str, address: &str| StartDriveSignInRequest {
        user: user.into(),
        name: "My Drive".into(),
        address: address.into(),
    };
    let bad = c.start_drive_sign_in(begin("alice", "gdrive://a b")).await.unwrap_err();
    assert_eq!(refusal(&bad).2, "3");
    let r = c.start_drive_sign_in(begin("alice", "gdrive://root/models")).await.unwrap().into_inner();
    assert!(r.authorize_url.starts_with(&format!("{endpoint}/auth?client_id=client-1&")), "{}", r.authorize_url);
    assert!(r.authorize_url.contains("redirect_uri=https%3A%2F%2Fdemo.example%2F"));
    assert!(r.authorize_url.contains(&format!("state={}", r.state)));
    assert_eq!(r.expires_in, 600);

    let finish = |user: &str, state: &str, code: &str| FinishDriveSignInRequest {
        user: user.into(),
        state: state.into(),
        code: code.into(),
    };
    // Another user cannot finish it, and a wrong code does not register.
    let e = c.finish_drive_sign_in(finish("bob", &r.state, "code-1")).await.unwrap_err();
    assert_eq!(refusal(&e).2, "2");
    let r2 = c.start_drive_sign_in(begin("alice", "gdrive://root/models")).await.unwrap().into_inner();
    let e = c.finish_drive_sign_in(finish("alice", &r2.state, "code-2")).await.unwrap_err();
    assert!(e.message().contains("invalid_grant"), "{e:?}");

    let done = c.finish_drive_sign_in(finish("alice", &r.state, "code-1")).await.unwrap().into_inner();
    let source = done.source.unwrap();
    assert_eq!((source.name.as_str(), source.sign_in), ("My Drive", SourceSignIn::Google as i32));
    let kept = home.holders().list(&"alice".parse().unwrap())[0].sign_in.clone();
    let want =
        SignIn::GoogleOauth { client_id: "client-1".into(), client_secret: None, refresh_token: "refresh-1".into() };
    assert_eq!(kept, want, "the server's own secret is not written with the grant");
    let text = fs::read_to_string(home.0.join("zoo/holders.json")).unwrap();
    assert!(!text.contains("secret-1"), "{text}");
    // A state is used once.
    let e = c.finish_drive_sign_in(finish("alice", &r.state, "code-1")).await.unwrap_err();
    assert!(e.message().contains("no sign-in is waiting"), "{e:?}");
}

async fn install(c: &mut Client, user: &str) -> Result<InstalledModel, tonic::Status> {
    let req = InstallModelRequest { user: user.into(), artifact_id: "free".into() };
    let mut stream = c.install_model(req).await?.into_inner();
    while let Some(m) = stream.message().await? {
        if let Some(Update::Installed(done)) = m.update {
            return Ok(done);
        }
    }
    Err(tonic::Status::unknown("the stream ended with no install"))
}

#[tokio::test]
async fn an_install_reads_the_users_own_store_first() {
    let home = Home::new("install");
    let (_s, mut c) = start(&home.0, Serve { admin: true, ..Serve::default() }).await;
    Blobs::new(home.0.join("store")).put(WORDS).unwrap();
    // Bob has no source and the catalog's transport has nothing.
    let e = install(&mut c, "bob").await.unwrap_err();
    assert!(e.message().contains("not served"), "{e:?}");
    let store = format!("file://{}", home.0.join("store").display());
    c.register_model_source(register("alice", "disk", &store, "")).await.unwrap();
    let done = install(&mut c, "alice").await.unwrap();
    assert_eq!(fs::read(&done.files[0]).unwrap(), WORDS);
}
