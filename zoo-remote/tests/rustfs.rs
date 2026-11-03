//! Real RustFS interoperability. Run with scripts/test-rustfs.sh.
//! curl independently signs fixture writes; the library signs every read.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use turbo_zoo::{Acceptances, Blobs, Index, Policy, Sha256, Sha512, Sources, Transport, Zoo, ZooError};
use turbo_zoo_remote::S3Bucket;
use turbo_zoo_remote::sigv4::Credentials;

struct Service {
    endpoint: String,
    key: String,
    secret: String,
    scratch: PathBuf,
}

impl Service {
    fn new() -> Self {
        let env = |name| std::env::var(name).expect("run scripts/test-rustfs.sh to start the service");
        Self {
            endpoint: env("TURBO_RUSTFS_ENDPOINT"),
            key: env("RUSTFS_ACCESS_KEY"),
            secret: env("RUSTFS_SECRET_KEY"),
            scratch: env("TURBO_RUSTFS_SCRATCH").into(),
        }
    }

    fn bucket(&self, prefix: &str, secret: Option<&str>) -> S3Bucket {
        let creds = secret.map(|secret| Credentials { key: self.key.clone(), secret: secret.into(), token: None });
        S3Bucket::with(&format!("s3://models/{prefix}"), Some(&self.endpoint), "us-east-1", creds).unwrap()
    }

    fn request(&self, method: &str, path: &str, body: Option<&[u8]>) {
        assert!(self.key.bytes().chain(self.secret.bytes()).all(|b| b.is_ascii_alphanumeric()));
        let mut command = Command::new("curl");
        command.args([
            "--disable",
            "--noproxy",
            "*",
            "--config",
            "-",
            "--silent",
            "--show-error",
            "--fail",
            "--max-time",
            "15",
            "--aws-sigv4",
            "aws:amz:us-east-1:s3",
            "--request",
            method,
        ]);
        let input = self.scratch.join("upload");
        if let Some(bytes) = body {
            fs::write(&input, bytes).unwrap();
            command.arg("--data-binary").arg(format!("@{}", input.display()));
        }
        let mut child = command.arg(format!("{}{path}", self.endpoint)).stdin(Stdio::piped()).spawn().unwrap();
        writeln!(child.stdin.take().unwrap(), "user = \"{}:{}\"", self.key, self.secret).unwrap();
        assert!(child.wait().unwrap().success(), "fixture {method} {path} failed");
        if body.is_some() {
            fs::remove_file(input).unwrap();
        }
    }
}

fn fixture_index(bytes: &[u8]) -> Index {
    Index::parse(
        format!(
            r#"{{"zoo_version":1,"artifacts":[{{"id":"fixture","kind":"KIND_LEXICON","format":"FORMAT_TEXT",
        "version":"1","license":"Apache-2.0","license_terms":"https://example.org/LICENSE",
        "trained_on":[],"class":"CLASS_PERMISSIVE","attribution":"Generated test bytes.",
        "files":[{{"path":"fixture.txt","url":"https://example.org/fixture.txt","size":{},
        "sha256":"{}","sha512":"{}"}}]}}]}}"#,
            bytes.len(),
            Sha256::of(bytes),
            Sha512::of(bytes),
        )
        .as_bytes(),
    )
    .unwrap()
}

#[test]
fn rustfs_signed_reads_integrity_and_offline_cache() {
    let service = Service::new();
    service.request("PUT", "/models", None);
    let bytes: Vec<u8> = (0..1_048_593).map(|i| (i % 251) as u8).collect();
    let index = fixture_index(&bytes);
    let artifact = index.get("fixture").unwrap();
    let file = &artifact.files[0];
    let path = format!("/models/zoo/blobs/sha256/{}", file.sha256);
    let bucket = service.bucket("zoo", Some(&service.secret));

    assert!(bucket.get_file(artifact, file).unwrap_err().contains("404"));
    service.request("PUT", &path, Some(&bytes));
    assert_eq!(bucket.get_file(artifact, file).unwrap(), bytes);
    assert!(service.bucket("zoo", Some("wrongsecret")).get_file(artifact, file).unwrap_err().contains("403"));
    assert!(service.bucket("zoo", None).get_file(artifact, file).unwrap_err().contains("403"));

    let store = Blobs::new(service.scratch.join("blobs"));
    let sources = Sources::new().then_keep("local", store.clone()).then("rustfs", bucket);
    assert_eq!(sources.get_file(artifact, file).unwrap(), bytes);
    assert_eq!(fs::read(store.path(&file.sha256)).unwrap(), bytes);
    // Removing the remote object proves a subsequent fetch uses the local cache.
    service.request("DELETE", &path, None);
    assert_eq!(sources.get_file(artifact, file).unwrap(), bytes);

    // An object under the expected key with the wrong digest must not be installed.
    let mut corrupt = bytes.clone();
    corrupt[0] ^= 1;
    service.request("PUT", &path, Some(&corrupt));
    let bucket = service.bucket("zoo", Some(&service.secret));
    let cache = service.scratch.join("rejected");
    let zoo = Zoo::new(index.clone(), cache.clone(), Policy::default(), &bucket);
    let error = zoo.fetch("fixture", &"test-user".parse().unwrap(), &Acceptances::default()).unwrap_err();
    assert!(matches!(error, ZooError::Integrity { .. }), "{error:?}");
    assert!(!cache.join("fixture").join("fixture.txt").exists());

    // The library also signs uploads and can keep an origin file in RustFS.
    let writer = service.bucket("written", Some(&service.secret));
    assert_eq!(writer.put(&bytes).unwrap(), file.sha256);
    assert_eq!(writer.put(&bytes).unwrap(), file.sha256);
    assert_eq!(writer.get_file(artifact, file).unwrap(), bytes);
    assert!(service.bucket("written", Some("wrongsecret")).put(&bytes).unwrap_err().contains("403"));
    let keeper = service.bucket("kept", Some(&service.secret));
    let sources = Sources::new().then_keep("rustfs", keeper).then("origin", store.clone());
    assert_eq!(sources.get_file(artifact, file).unwrap(), bytes);
    fs::remove_file(store.path(&file.sha256)).unwrap();
    assert_eq!(sources.get_file(artifact, file).unwrap(), bytes);

    // Different prefix, empty objects, and correct files through the full installer.
    let empty = fixture_index(&[]);
    let empty_artifact = empty.get("fixture").unwrap();
    let empty_file = &empty_artifact.files[0];
    let empty_path = format!("/models/nested/prefix/blobs/sha256/{}", empty_file.sha256);
    service.request("PUT", &empty_path, Some(&[]));
    let bucket = service.bucket("nested/prefix", Some(&service.secret));
    assert!(bucket.get_file(empty_artifact, empty_file).unwrap().is_empty());
    service.request("PUT", &path, Some(&bytes));
    let bucket = service.bucket("zoo", Some(&service.secret));
    let zoo = Zoo::new(index, service.scratch.join("installed"), Policy::default(), &bucket);
    let fetched = zoo.fetch("fixture", &"test-user".parse().unwrap(), &Acceptances::default()).unwrap();
    assert_eq!(fs::read(fetched.single().unwrap()).unwrap(), bytes);
}
