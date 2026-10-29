//! An S3 bucket as a source (#1613), against a small S3 stand-in started
//! in-process: path-style addressing as RustFS and MinIO serve it, the
//! signature checked by the stand-in, unsigned reads of a public bucket,
//! and the zoo's own checks on whatever the bucket serves.
#![cfg(feature = "s3")]

use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::thread::JoinHandle;

use turbo_zoo::{Acceptances, Blobs, Index, Offline, Policy, Sha256, Sha512, Transport, Url, UserId, Zoo, ZooError};
use turbo_zoo_remote::sigv4::{Credentials, sign_get};
use turbo_zoo_remote::{S3Bucket, sources};

const TEXT: &[u8] = b"hello hello hello\n";

fn index() -> Index {
    let json = format!(
        r#"{{"zoo_version": 1, "artifacts": [{{"id": "hello", "kind": "KIND_LEXICON", "format": "FORMAT_TEXT",
          "version": "1", "license": "Apache-2.0", "license_terms": "https://example.org/LICENSE",
          "trained_on": [], "class": "CLASS_PERMISSIVE", "attribution": "Example attribution.",
          "files": [{{"path": "hello.txt", "url": "https://example.org/hello.txt", "size": {}, "sha256": "{}",
            "sha512": "{}"}}]}}]}}"#,
        TEXT.len(),
        Sha256::of(TEXT),
        Sha512::of(TEXT),
    );
    Index::parse(json.as_bytes()).unwrap()
}

fn creds() -> Credentials {
    Credentials { key: "AKIDEXAMPLE".into(), secret: "a-secret".into(), token: Some("a-session".into()) }
}

/// What the stand-in saw of one request: its path and headers, lower-cased
/// names.
struct Seen {
    path: String,
    headers: HashMap<String, String>,
}

/// Answers `n` GETs from `objects` (path to bytes), each with 404 when the
/// path is not there or, when `creds` are given, 403 unless the request
/// carries the signature those credentials make for it.
fn stand_in(
    objects: HashMap<String, Vec<u8>>,
    creds: Option<Credentials>,
    n: usize,
) -> (String, JoinHandle<Vec<Seen>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
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
                let h = h.trim_end();
                if h.is_empty() {
                    break;
                }
                let (k, v) = h.split_once(':').unwrap();
                headers.insert(k.to_ascii_lowercase(), v.trim().to_owned());
            }
            let signed_right = creds.as_ref().is_none_or(|c| {
                let date = headers.get("x-amz-date").cloned().unwrap_or_default();
                let want = sign_get(c, "us-east-1", &headers["host"], &path, &[], &date);
                want.iter().all(|(k, v)| headers.get(k) == Some(v))
            });
            let reply: (&str, &[u8]) = match objects.get(&path) {
                _ if !signed_right => ("403 Forbidden", b""),
                Some(b) => ("200 OK", b),
                None => ("404 Not Found", b""),
            };
            let head =
                format!("HTTP/1.1 {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", reply.0, reply.1.len());
            conn.write_all(head.as_bytes()).unwrap();
            conn.write_all(reply.1).unwrap();
            let _ = reader.read(&mut [0u8; 1]);
            seen.push(Seen { path, headers });
        }
        seen
    });
    (endpoint, handle)
}

fn key(prefix: &str) -> String {
    format!("{prefix}/blobs/sha256/{}", Sha256::of(TEXT))
}

struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Dir {
        let p = std::env::temp_dir().join(format!("turbo-zoo-s3-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        Dir(p)
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn alice() -> UserId {
    "alice".parse().unwrap()
}

#[test]
fn a_model_loads_from_a_bucket_with_a_signed_path_style_get() {
    let objects = HashMap::from([(key("/models/zoo"), TEXT.to_vec())]);
    let (endpoint, server) = stand_in(objects, Some(creds()), 1);
    let bucket = S3Bucket::with("s3://models/zoo/", Some(&endpoint), "us-east-1", Some(creds())).unwrap();
    let cache = Dir::new("signed");
    let zoo = Zoo::new(index(), cache.0.clone(), Policy::default(), &bucket);
    let fetched = zoo.fetch("hello", &alice(), &Acceptances::default()).unwrap();
    assert_eq!(fs::read(fetched.single().unwrap()).unwrap(), TEXT);
    let seen = server.join().unwrap();
    assert_eq!(seen[0].path, key("/models/zoo"));
    assert_eq!(seen[0].headers["x-amz-security-token"], "a-session");
    assert!(seen[0].headers["authorization"].starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"));
}

#[test]
fn a_wrong_key_is_refused_by_the_bucket_and_a_public_bucket_is_read_unsigned() {
    let objects = HashMap::from([(key("/public"), TEXT.to_vec())]);
    let (endpoint, server) = stand_in(objects.clone(), Some(creds()), 1);
    let wrong = Credentials { secret: "another".into(), ..creds() };
    let bucket = S3Bucket::with("s3://public", Some(&endpoint), "us-east-1", Some(wrong)).unwrap();
    let index = index();
    let a = index.get("hello").unwrap();
    assert!(bucket.get_file(a, &a.files[0]).unwrap_err().contains("403"));
    server.join().unwrap();
    let (endpoint, server) = stand_in(objects, None, 1);
    let bucket = S3Bucket::with("s3://public", Some(&endpoint), "us-east-1", None).unwrap();
    assert_eq!(bucket.get_file(a, &a.files[0]).unwrap(), TEXT);
    assert!(!server.join().unwrap()[0].headers.contains_key("authorization"));
}

#[test]
fn other_bytes_from_a_bucket_are_not_kept() {
    let objects = HashMap::from([(key("/models"), b"other bytes".to_vec())]);
    let (endpoint, server) = stand_in(objects, None, 1);
    let bucket = S3Bucket::with("s3://models", Some(&endpoint), "us-east-1", None).unwrap();
    let cache = Dir::new("other");
    let zoo = Zoo::new(index(), cache.0.clone(), Policy::default(), &bucket);
    let e = zoo.fetch("hello", &alice(), &Acceptances::default()).unwrap_err();
    assert!(matches!(e, ZooError::Integrity { .. }), "{e:?}");
    server.join().unwrap();
}

#[test]
fn the_first_local_store_keeps_what_a_bucket_serves() {
    let objects = HashMap::from([(key("/models"), TEXT.to_vec())]);
    let (endpoint, server) = stand_in(objects, None, 1);
    let store = Dir::new("keep");
    let bucket = S3Bucket::with("s3://models", Some(&endpoint), "us-east-1", None).unwrap();
    let list = turbo_zoo::Sources::new().then_keep("local", Blobs::new(&store.0)).then("bucket", bucket);
    let index = index();
    let a = index.get("hello").unwrap();
    assert_eq!(list.get_file(a, &a.files[0]).unwrap(), TEXT);
    server.join().unwrap();
    // Read again with the bucket gone: the store has it.
    assert_eq!(list.get_file(a, &a.files[0]).unwrap(), TEXT);
    assert!(Blobs::new(&store.0).path(&a.files[0].sha256).is_file());
}

#[test]
fn bucket_addresses_are_checked_before_any_request() {
    let url = |b: &S3Bucket| b.url_of("ab");
    let aws = S3Bucket::with("s3://models/zoo", None, "eu-west-1", None).unwrap();
    assert_eq!(url(&aws), "https://models.s3.eu-west-1.amazonaws.com/zoo/blobs/sha256/ab");
    let dotted = S3Bucket::with("s3://models.example", None, "eu-west-1", None).unwrap();
    assert_eq!(url(&dotted), "https://s3.eu-west-1.amazonaws.com/models.example/blobs/sha256/ab");
    let local = S3Bucket::with("s3://models", Some("http://RustFS:80/"), "us-east-1", None).unwrap();
    assert_eq!(url(&local), "http://rustfs/models/blobs/sha256/ab");
    for (address, endpoint, says) in [
        ("models", None, "starts s3://"),
        ("s3://Models", None, "bucket name"),
        ("s3://ab", None, "bucket name"),
        ("s3://models/a//b", None, "a prefix"),
        ("s3://models/a/../b", None, "a prefix"),
        ("s3://models/a b", None, "a prefix"),
        ("s3://models", Some("ftp://rustfs"), "the endpoint"),
        ("s3://models", Some("http://rustfs/path"), "the endpoint"),
        ("s3://models", Some("http://user@rustfs"), "the endpoint"),
    ] {
        let e = S3Bucket::with(address, endpoint, "us-east-1", None).err().unwrap();
        assert!(e.contains(says), "{address} {endpoint:?}: {e}");
    }
    assert!(S3Bucket::with("s3://models", None, "US East", None).is_err());
    let u: Url = "https://example.org/hello.txt".parse().unwrap();
    assert!(aws.get(&u, 1).unwrap_err().contains("by its SHA-256"));
    // TURBO_ZOO_SOURCE names a bucket like any other source.
    assert_eq!(sources("file:///zoo, s3://models/zoo, origin", &alice(), Offline).unwrap().len(), 3);
}
