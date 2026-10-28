//! The places a model loads from besides its address (#1613): a
//! content-addressed store, kept by SHA-256 and read by it alone, and a
//! list of sources tried in order, each answer kept only when it verifies.
//! The licence check runs before any source is asked.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use turbo_zoo::{
    Acceptances, Artifact, Blobs, File, Index, Licence, Policy, Requirement, Sha256, Sha512, Sources, Transport, Url,
    UserId, Zoo, ZooError,
};

const TEXT: &[u8] = b"hello hello hello\n";

/// TEXT as Python's `gzip.compress(..., mtime=0)` writes it.
const TEXT_GZ: [u8; 29] = [
    0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x90,
    0x5c, 0x00, 0x3b, 0x7c, 0x8a, 0xdf, 0x12, 0x00, 0x00, 0x00,
];
const SA: &[u8] = b"a share-alike word list\n";

/// A permissive file served gzip compressed, and a share-alike one.
fn index() -> Index {
    let artifact = |id: &str, licence: &str, class: &str, bytes: &[u8], download: String| {
        format!(
            r#"{{"id": "{id}", "kind": "KIND_LEXICON", "format": "FORMAT_TEXT", "version": "1",
              "license": "{licence}", "license_terms": "https://example.org/{id}/LICENSE", "trained_on": [],
              "class": "{class}", "attribution": "Example attribution for {id}.",
              "files": [{{"path": "{id}.txt", "url": "https://example.org/{id}.txt", "size": {}, "sha256": "{}",
                "sha512": "{}"{download}}}]}}"#,
            bytes.len(),
            Sha256::of(bytes),
            Sha512::of(bytes),
        )
    };
    let gz = format!(
        r#", "download": {{"compression": "gzip", "download_bytes": {}, "download_sha256": "{}"}}"#,
        TEXT_GZ.len(),
        Sha256::of(&TEXT_GZ)
    );
    let json = format!(
        r#"{{"zoo_version": 1, "artifacts": [{}, {}]}}"#,
        artifact("hello", "Apache-2.0", "CLASS_PERMISSIVE", TEXT, gz),
        artifact("sa", "CC-BY-SA-4.0", "CLASS_SHARE_ALIKE", SA, String::new()),
    );
    Index::parse(json.as_bytes()).unwrap()
}

/// Serves the same bytes for every file and counts the requests.
struct Always(Vec<u8>, AtomicUsize);

impl Transport for Always {
    fn get(&self, _url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        self.1.fetch_add(1, Ordering::SeqCst);
        Ok(self.0.clone())
    }
}

/// Answers every request with an error.
struct Down;

impl Transport for Down {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        Err(format!("{url}: connection refused"))
    }
}

struct Dir(PathBuf);

impl Dir {
    fn new(name: &str) -> Dir {
        let p = std::env::temp_dir().join(format!("turbo-zoo-sources-{name}-{}", std::process::id()));
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

fn file_of<'a>(index: &'a Index, id: &str) -> (&'a Artifact, &'a File) {
    let a = index.get(id).unwrap();
    (a, &a.files[0])
}

#[test]
fn a_blob_store_keeps_a_file_under_its_sha256_and_serves_it_by_that_alone() {
    let dir = Dir::new("blobs");
    let blobs = Blobs::new(&dir.0);
    let sha = blobs.put(TEXT).unwrap();
    assert_eq!(sha, Sha256::of(TEXT));
    assert_eq!(blobs.path(&sha), dir.0.join("blobs/sha256").join(sha.to_string()));
    assert_eq!(fs::read(blobs.path(&sha)).unwrap(), TEXT);
    // Kept again, the same file.
    assert_eq!(blobs.put(TEXT).unwrap(), sha);
    let url: Url = "https://example.org/hello.txt".parse().unwrap();
    assert!(blobs.get(&url, 18).unwrap_err().contains("by its SHA-256"));
    // A file the index serves compressed is kept unpacked and still loads.
    let cache = Dir::new("blobs-cache");
    let zoo = Zoo::new(index(), cache.0.clone(), Policy::default(), &blobs);
    let fetched = zoo.fetch("hello", &alice(), &Acceptances::default()).unwrap();
    assert_eq!(fs::read(fetched.single().unwrap()).unwrap(), TEXT);
}

#[test]
fn a_blob_that_is_a_symlink_or_holds_other_bytes_is_not_served() {
    let dir = Dir::new("bad-blobs");
    let blobs = Blobs::new(&dir.0);
    let index = index();
    let (a, f) = file_of(&index, "hello");
    assert!(blobs.get_file(a, f).is_err(), "nothing kept yet");
    let path = blobs.path(&f.sha256);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, b"other bytes").unwrap();
    let cache = Dir::new("bad-blobs-cache");
    let zoo = Zoo::new(index.clone(), cache.0.clone(), Policy::default(), &blobs);
    let e = zoo.fetch("hello", &alice(), &Acceptances::default()).unwrap_err();
    assert!(matches!(e, ZooError::Integrity { .. }), "{e:?}");
    #[cfg(unix)]
    {
        let real = dir.0.join("real");
        fs::write(&real, TEXT).unwrap();
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(&real, &path).unwrap();
        assert!(blobs.get_file(a, f).unwrap_err().contains("not a plain file"));
    }
}

#[test]
fn sources_are_tried_in_order_and_only_bytes_that_verify_are_kept() {
    let store = Dir::new("order");
    let blobs = Blobs::new(&store.0);
    blobs.put(TEXT).unwrap();
    let index = index();
    let (a, f) = file_of(&index, "hello");
    // A source that is down, then one serving other bytes, then the store.
    let sources = Sources::new()
        .then("server", Down)
        .then("wrong", Always(b"not the file".to_vec(), AtomicUsize::new(0)))
        .then("local", blobs.clone());
    assert_eq!(sources.get_file(a, f).unwrap(), TEXT);
    // The release asset itself, compressed, verifies on the wire and is
    // handed over unpacked.
    let origin = Sources::new().then("release", Always(TEXT_GZ.to_vec(), AtomicUsize::new(0)));
    assert_eq!(origin.get_file(a, f).unwrap(), TEXT);
    let none = Sources::new().then("server", Down).then("empty", Blobs::new(store.0.join("empty")));
    let e = none.get_file(a, f).unwrap_err();
    assert!(e.starts_with("server: ") && e.contains("connection refused") && e.contains("; empty: "), "{e}");
    assert_eq!(Sources::new().get_file(a, f).unwrap_err(), "no source is configured");
}

#[test]
fn the_licence_is_checked_before_any_source_is_asked() {
    let store = Dir::new("licence");
    let blobs = Blobs::new(&store.0);
    blobs.put(SA).unwrap();
    let counting = Always(SA.to_vec(), AtomicUsize::new(0));
    let sources = Sources::new().then("local", blobs);
    let cache = Dir::new("licence-cache");
    let zoo = Zoo::new(index(), cache.0.clone(), Policy::default(), &sources);
    let e = zoo.fetch("sa", &alice(), &Acceptances::default()).unwrap_err();
    assert!(matches!(e, ZooError::NotAccepted { .. }), "{e:?}");
    let zoo = Zoo::new(index(), cache.0.clone(), Policy::default(), &counting);
    assert!(zoo.fetch("sa", &alice(), &Acceptances::default()).is_err());
    assert_eq!(counting.1.load(Ordering::SeqCst), 0, "no request before the acceptance");
    let mut accepted = Acceptances::default();
    accepted.accept_terms(&alice(), &Requirement::of(Licence::CcBySa40).unwrap(), 1).unwrap();
    let zoo = Zoo::new(index(), cache.0.clone(), Policy::default(), &sources);
    let fetched = zoo.fetch("sa", &alice(), &accepted).unwrap();
    assert_eq!(fs::read(fetched.single().unwrap()).unwrap(), SA);
}
