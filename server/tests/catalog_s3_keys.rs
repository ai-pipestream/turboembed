//! Model sources that sign in with their own S3 keys (#1744): what the
//! server takes from a caller with the admin flag and in zoo mode alone,
//! and an install and a zoo-mode read that fetch through a user's keyed
//! bucket, against an S3 stand-in started in-process that checks each
//! request's signature.

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;

use tonic::transport::Channel;
use turbo_kserve::api::{TURBO_E_INVALID_ARGUMENT, TURBO_E_UNSUPPORTED};
use turbo_kserve::catalog::Catalog;
use turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES as MAX;
use turbo_kserve::proto::training::install_model_update::Update;
use turbo_kserve::proto::training::model_file_chunk::Frame;
use turbo_kserve::proto::training::turbo_catalog_service_client::TurboCatalogServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve::{Serve, Server};
use turbo_zoo::sigv4::{Credentials, EMPTY_SHA256, sign};
use turbo_zoo::{Holders, Index, Sha256, Sha512, SignIn, Transport, Url};

type Client = TurboCatalogServiceClient<Channel>;

const WORDS: &[u8] = b"a permissive word list\n";

/// The catalog's own transport has nothing.
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
        let p = std::env::temp_dir().join(format!("turbo-kserve-catalog-s3-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        Home(p)
    }
}

impl Drop for Home {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

async fn start(home: &Path, serve: Serve) -> (Server, Client) {
    let catalog = Catalog { home: home.join("zoo"), index: Arc::new(index()), transport: Arc::new(Nothing) };
    let serve = Serve { catalog: Some(catalog), ..serve };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), Vec::new(), MAX, serve).await.unwrap();
    let c = TurboCatalogServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    (s, c)
}

fn creds() -> Credentials {
    Credentials { key: "AKIDEXAMPLE".into(), secret: "a-secret".into(), token: None }
}

/// Answers `n` GETs from a bucket holding the word list under its
/// SHA-256 at `/models/zoo`, each refused 403 unless signed with
/// [`creds`] in us-east-1. Hands back each request's authorization.
fn stand_in(n: usize) -> (String, JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let object = format!("/models/zoo/blobs/sha256/{}", Sha256::of(WORDS));
    let handle = std::thread::spawn(move || {
        let mut seen = Vec::new();
        for _ in 0..n {
            let (mut conn, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(conn.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let path = line.split(' ').nth(1).unwrap().to_owned();
            let mut headers = HashMap::new();
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).unwrap();
                let Some((k, v)) = h.trim_end().split_once(':') else { break };
                headers.insert(k.to_ascii_lowercase(), v.trim().to_owned());
            }
            let date = headers.get("x-amz-date").cloned().unwrap_or_default();
            let at = (headers["host"].as_str(), path.as_str());
            let signed = sign("GET", &creds(), "us-east-1", at, EMPTY_SHA256, &[], &date)
                .is_ok_and(|want| want.iter().all(|(k, v)| headers.get(k) == Some(v)));
            let (status, body): (&str, &[u8]) = match (signed, path == object) {
                (false, _) => ("403 Forbidden", b""),
                (true, true) => ("200 OK", WORDS),
                (true, false) => ("404 Not Found", b""),
            };
            let head = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            conn.write_all(head.as_bytes()).unwrap();
            conn.write_all(body).unwrap();
            let _ = reader.read(&mut [0u8; 1]);
            seen.push(headers.remove("authorization").unwrap_or_default());
        }
        seen
    });
    (endpoint, handle)
}

fn keys(region: &str, endpoint: &str) -> S3Keys {
    S3Keys {
        access_key_id: creds().key,
        secret_access_key: creds().secret,
        session_token: String::new(),
        region: region.into(),
        endpoint: endpoint.into(),
    }
}

fn register(name: &str, address: &str, api_key: &str, s3_keys: Option<S3Keys>) -> RegisterModelSourceRequest {
    RegisterModelSourceRequest {
        user: "alice".into(),
        name: name.into(),
        address: address.into(),
        api_key: api_key.into(),
        s3_keys,
    }
}

/// The refusal's gRPC code, turbo code and field.
fn refusal(s: &tonic::Status) -> (i32, String) {
    let md = |k: &str| s.metadata().get(k).unwrap_or_else(|| panic!("no {k} in {s:?}")).to_str().unwrap().to_owned();
    (md("turbo-code").parse().unwrap(), md("turbo-field"))
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

async fn read(c: &mut Client, user: &str) -> Result<Vec<u8>, tonic::Status> {
    let req = ReadModelFileRequest {
        user: user.into(),
        artifact_id: "free".into(),
        path: "words.txt".into(),
        sha256: String::new(),
    };
    let mut stream = c.read_model_file(req).await?.into_inner();
    let mut bytes = Vec::new();
    while let Some(m) = stream.message().await? {
        if let Some(Frame::Data(d)) = m.frame {
            bytes.extend_from_slice(&d);
        }
    }
    Ok(bytes)
}

#[tokio::test]
async fn an_install_reads_a_bucket_with_the_sources_own_keys_and_shows_none_of_them() {
    let home = Home::new("admin");
    let (_s, mut c) = start(&home.0, Serve { admin: true, ..Serve::default() }).await;
    let e = c.register_model_source(register("b", "s3://models/zoo", "", Some(keys("", "http://h/x")))).await;
    assert_eq!(refusal(&e.unwrap_err()), (TURBO_E_INVALID_ARGUMENT, "5".into()));
    let (endpoint, bucket) = stand_in(1);
    let got = c.register_model_source(register("bucket", "s3://models/zoo", "", Some(keys("", &endpoint)))).await;
    assert_eq!(got.unwrap().into_inner().source.unwrap().sign_in, SourceSignIn::S3Keys as i32);
    let listed = c.list_model_sources(ListModelSourcesRequest { user: "alice".into() }).await.unwrap().into_inner();
    assert_eq!(listed.sources[0].sign_in, SourceSignIn::S3Keys as i32);
    let shown = format!("{listed:?}");
    assert!(!shown.contains("a-secret") && !shown.contains("AKIDEXAMPLE"), "{shown}");
    let held = Holders::load(&home.0.join("zoo/holders.json")).unwrap();
    let SignIn::S3Keys { endpoint: kept, region, .. } = &held.list(&"alice".parse().unwrap())[0].sign_in else {
        panic!("{held:?}")
    };
    assert_eq!((kept.as_deref(), region.as_deref()), (Some(endpoint.as_str()), None));

    let done = install(&mut c, "alice").await.unwrap();
    assert_eq!(fs::read(&done.files[0]).unwrap(), WORDS);
    let seen = bucket.join().unwrap();
    assert!(seen[0].starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"), "{seen:?}");
}

#[tokio::test]
async fn zoo_mode_takes_an_s3_source_only_on_aws_and_with_its_own_keys() {
    let home = Home::new("zoo");
    let (_s, mut c) = start(&home.0, Serve { zoo: true, ..Serve::default() }).await;
    let unsupported = (TURBO_E_UNSUPPORTED, "3".to_string());
    let invalid = |field: &str| (TURBO_E_INVALID_ARGUMENT, field.to_string());
    let mut empty_secret = keys("", "");
    empty_secret.secret_access_key.clear();
    for (req, want) in [
        (register("b", "s3://models/zoo", "", None), unsupported.clone()),
        (register("b", "s3://models/zoo", "", Some(keys("", "http://127.0.0.1:9000"))), unsupported),
        (register("b", "gdrive://1AbC", "", Some(keys("", ""))), invalid("5")),
        (register("b", "s3://models/zoo", "key-1", Some(keys("", ""))), invalid("5")),
        (register("b", "s3://models/zoo", "", Some(keys("EU", ""))), invalid("5")),
        (register("b", "s3://models/zoo", "", Some(empty_secret)), invalid("5")),
        (register("b", "s3://Models", "", Some(keys("", ""))), invalid("3")),
        (register("b", "s3://models/zoo", "key-1", None), invalid("4")),
    ] {
        let e = c.register_model_source(req.clone()).await.unwrap_err();
        assert_eq!(refusal(&e), want, "{req:?}: {e:?}");
    }
    let r = c.register_model_source(register("aws", "s3://models/zoo", "", Some(keys("eu-west-1", "")))).await;
    assert_eq!(r.unwrap().into_inner().source.unwrap().address, "s3://models/zoo");
}

#[tokio::test]
async fn a_zoo_mode_read_fetches_through_the_readers_own_bucket() {
    let home = Home::new("through");
    let (_s, mut c) = start(&home.0, Serve { admin: true, zoo: true, ..Serve::default() }).await;
    // Bob has no source and the catalog's transport has nothing.
    let e = read(&mut c, "bob").await.unwrap_err();
    assert!(e.message().contains("not served"), "{e:?}");
    let (endpoint, bucket) = stand_in(1);
    c.register_model_source(register("bucket", "s3://models/zoo", "", Some(keys("", &endpoint)))).await.unwrap();
    assert_eq!(read(&mut c, "alice").await.unwrap(), WORDS);
    // Cached now: the second read asks the bucket for nothing.
    assert_eq!(read(&mut c, "alice").await.unwrap(), WORDS);
    assert_eq!(bucket.join().unwrap().len(), 1);
}
