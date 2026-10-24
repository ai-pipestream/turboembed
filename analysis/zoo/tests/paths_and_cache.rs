use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use turbo_zoo::accept::{Acceptances, UserId};
use turbo_zoo::digest::{Sha256, Sha512};
use turbo_zoo::error::ZooError;
use turbo_zoo::fetch::{Transport, Zoo};
use turbo_zoo::index::{Index, Url};
use turbo_zoo::licence::Policy;

const BYTES: &[u8] = b"model-payload";

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> TempDir {
        static N: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "turbo-zoo-paths-test-{}-{}-{label}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed),
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct FixedTransport(&'static [u8]);

impl Transport for FixedTransport {
    fn get(&self, _url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        Ok(self.0.to_vec())
    }
}

fn file_json(path: &str, url: &str) -> String {
    format!(
        r#"{{"path": "{path}", "url": "{url}", "size": {}, "sha256": "{}", "sha512": "{}"}}"#,
        BYTES.len(),
        Sha256::of(BYTES),
        Sha512::of(BYTES),
    )
}

fn index_with_files(files_json: &str, origin_json: &str) -> String {
    let (license, class) = if origin_json.is_empty() {
        ("Apache-2.0", "CLASS_PERMISSIVE")
    } else {
        ("LicenseRef-RestrictedPublic", "CLASS_LOCAL_ONLY")
    };
    format!(
        r#"{{
  "zoo_version": 1,
  "artifacts": [
    {{
      "id": "test-art",
      "kind": "KIND_MODEL",
      "format": "FORMAT_OPENNLP_MODEL",
      "version": "1.0",
      "license": "{license}",
      "license_terms": "https://example.org/terms",
      "trained_on": [],
      "class": "{class}",
      "attribution": "Example attribution for test-art.",{origin_json}
      "files": [{files_json}]
    }}
  ]
}}"#
    )
}

#[test]
fn index_rejects_partial_file_paths() {
    for bad in ["model.bin.partial", "sub/model.partial", "sub.partial/model.bin"] {
        let json = index_with_files(&file_json(bad, "https://example.org/m.bin"), "");
        match Index::parse(json.as_bytes()) {
            Err(ZooError::Index(m)) => assert!(m.contains(".partial"), "{bad}: {m}"),
            other => panic!("expected .partial rejection for {bad}, got {other:?}"),
        }
    }
}

#[test]
fn index_rejects_colliding_file_and_directory_paths() {
    // Include "a/b-other" which sorts between "a/b" and "a/b/c" in ASCII.
    let files = [
        file_json("a/b", "https://example.org/1.bin"),
        file_json("a/b-other", "https://example.org/2.bin"),
        file_json("a/b/c", "https://example.org/3.bin"),
    ]
    .join(", ");
    let json = index_with_files(&files, "");
    match Index::parse(json.as_bytes()) {
        Err(ZooError::Index(m)) => assert!(m.contains("both a file and a directory"), "{m}"),
        other => panic!("expected file/directory collision error, got {other:?}"),
    }
}

#[test]
fn under_origin_rejects_query_or_fragment_on_trailing_slash_directory() {
    let origin = r#" "origin": "https://example.org/repo","#;
    for bad_url in [
        "https://example.org/repo/",
        "https://example.org/repo/?file=a.bin",
        "https://example.org/repo/#frag",
        "https://example.org/repo///?file=a.bin",
    ] {
        let json = index_with_files(&file_json("a.bin", bad_url), origin);
        match Index::parse(json.as_bytes()) {
            Err(ZooError::Index(m)) => assert!(m.contains("not under its origin"), "{bad_url}: {m}"),
            other => panic!("expected origin rejection for {bad_url}, got {other:?}"),
        }
    }
}

#[test]
fn fetch_rejects_directory_at_target_file_path_and_refetches_wrong_size_file() {
    let dir = TempDir::new("cache");
    let json = index_with_files(&file_json("model.bin", "https://example.org/m.bin"), "");
    let index = Index::parse(json.as_bytes()).unwrap();
    let transport = FixedTransport(BYTES);
    let zoo = Zoo::new(index, dir.0.clone(), Policy::default(), &transport);
    let user: UserId = "alice".parse().unwrap();
    let acceptances = Acceptances::default();

    // If a directory sits where the target file should be, fetch fails immediately.
    let target = dir.0.join("test-art").join("model.bin");
    fs::create_dir_all(&target).unwrap();
    match zoo.fetch("test-art", &user, &acceptances) {
        Err(ZooError::Io { path, message }) => {
            assert_eq!(path, target);
            assert!(message.contains("is not a plain file"), "{message}");
        }
        other => panic!("expected non-plain-file error, got {other:?}"),
    }

    // Replace the directory with a wrong-sized file; fetch skips reading and overwrites it.
    fs::remove_dir(&target).unwrap();
    fs::write(&target, b"wrong-size-cached-content").unwrap();
    let fetched = zoo.fetch("test-art", &user, &acceptances).unwrap();
    assert_eq!(fs::read(fetched.single().unwrap()).unwrap(), BYTES);
}
