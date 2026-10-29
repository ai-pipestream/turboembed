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
const NC: &[u8] = b"a non-commercial checkpoint\n";

/// A permissive file served gzip compressed, a share-alike one and a
/// local-only one.
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
        r#"{{"zoo_version": 1, "artifacts": [{}, {}, {}]}}"#,
        artifact("hello", "Apache-2.0", "CLASS_PERMISSIVE", TEXT, gz),
        artifact("sa", "CC-BY-SA-4.0", "CLASS_SHARE_ALIKE", SA, String::new()),
        artifact("nc", "CC-BY-NC-4.0", "CLASS_LOCAL_ONLY", NC, String::new())
            .replace(
                r#""class": "CLASS_LOCAL_ONLY""#,
                r#""class": "CLASS_LOCAL_ONLY", "origin": "https://example.org/nc""#
            )
            .replace("https://example.org/nc.txt", "https://example.org/nc/nc.txt"),
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

#[test]
fn putting_bytes_again_repairs_a_kept_file_of_the_same_length_with_other_bytes() {
    let dir = Dir::new("repair");
    let blobs = Blobs::new(&dir.0);
    let sha = blobs.put(b"good").unwrap();
    fs::write(blobs.path(&sha), b"evil").unwrap();
    assert_eq!(blobs.put(b"good").unwrap(), sha);
    assert_eq!(fs::read(blobs.path(&sha)).unwrap(), b"good");
}

#[cfg(unix)]
#[test]
fn a_symlinked_directory_inside_the_store_is_neither_written_through_nor_read() {
    use std::os::unix::fs::symlink;
    let (dir, outside) = (Dir::new("inner-link"), Dir::new("inner-link-outside"));
    fs::create_dir_all(&dir.0).unwrap();
    fs::create_dir_all(outside.0.join("sha256")).unwrap();
    let index = index();
    let (a, f) = file_of(&index, "hello");
    fs::write(outside.0.join("sha256").join(f.sha256.to_string()), TEXT).unwrap();
    // <root>/blobs leads outside the store.
    symlink(&outside.0, dir.0.join("blobs")).unwrap();
    let blobs = Blobs::new(&dir.0);
    assert!(matches!(blobs.put(b"good"), Err(ZooError::Io { .. })));
    assert!(!outside.0.join("sha256").join(Sha256::of(b"good").to_string()).exists(), "nothing written outside");
    assert!(blobs.get_file(a, f).unwrap_err().contains("is a symlink"));
    // <root>/blobs/sha256 leading outside is refused the same way.
    fs::remove_file(dir.0.join("blobs")).unwrap();
    fs::create_dir_all(dir.0.join("blobs")).unwrap();
    symlink(outside.0.join("sha256"), dir.0.join("blobs/sha256")).unwrap();
    assert!(matches!(blobs.put(b"good"), Err(ZooError::Io { .. })));
    assert!(blobs.get_file(a, f).unwrap_err().contains("is a symlink"));
    // The root itself is the caller's choice and may be a symlink.
    let (real, link) = (Dir::new("root-real"), Dir::new("root-link"));
    fs::create_dir_all(&real.0).unwrap();
    symlink(&real.0, &link.0).unwrap();
    let through = Blobs::new(&link.0);
    through.put(TEXT).unwrap();
    assert_eq!(through.get_file(a, f).unwrap(), TEXT);
    assert!(real.0.join("blobs/sha256").join(f.sha256.to_string()).is_file());
}

/// Serves TEXT_GZ and counts the requests in a counter the test keeps.
struct Remote(std::sync::Arc<AtomicUsize>);

impl Transport for Remote {
    fn get(&self, _url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(TEXT_GZ.to_vec())
    }
}

#[test]
fn the_keeping_store_keeps_what_later_sources_serve_once_it_verifies() {
    let (store, other) = (Dir::new("keep"), Dir::new("keep-other"));
    let index = index();
    let (a, f) = file_of(&index, "hello");
    let asked = std::sync::Arc::new(AtomicUsize::new(0));
    let sources = Sources::new().then_keep("local", Blobs::new(&store.0)).then("remote", Remote(asked.clone()));
    assert_eq!(sources.get_file(a, f).unwrap(), TEXT);
    // Kept unpacked, under the SHA-256 the index pins, and read from there
    // next time without asking the remote again.
    assert_eq!(fs::read(Blobs::new(&store.0).path(&f.sha256)).unwrap(), TEXT);
    assert_eq!(sources.get_file(a, f).unwrap(), TEXT);
    assert_eq!(asked.load(Ordering::SeqCst), 1);
    // Bytes that fail the check are not kept.
    let (sa, sf) = file_of(&index, "sa");
    let wrong = Sources::new()
        .then_keep("local", Blobs::new(&other.0))
        .then("wrong", Always(b"x".to_vec(), AtomicUsize::new(0)));
    assert!(wrong.get_file(sa, sf).is_err());
    assert!(!Blobs::new(&other.0).path(&sf.sha256).exists());
    // A store that cannot be written still hands the file over.
    let file_root = other.0.join("a-file");
    fs::create_dir_all(&other.0).unwrap();
    fs::write(&file_root, b"not a directory").unwrap();
    let blocked = Sources::new().then_keep("local", Blobs::new(&file_root)).then("remote", Remote(asked.clone()));
    assert_eq!(blocked.get_file(a, f).unwrap(), TEXT);
}

#[test]
fn the_keeping_store_neither_serves_nor_keeps_a_local_only_file() {
    let (store, empty) = (Dir::new("local-only"), Dir::new("local-only-empty"));
    let index = index();
    let (a, f) = file_of(&index, "nc");
    // The store already holds a copy: it is not read, the origin is asked.
    Blobs::new(&store.0).put(NC).unwrap();
    let asked = AtomicUsize::new(0);
    let origin = Always(NC.to_vec(), asked);
    let list = Sources::new().then_keep("store", Blobs::new(&store.0)).then("origin", origin);
    assert_eq!(list.get_file(a, f).unwrap(), NC);
    // An origin that is down is not covered for by the store's copy.
    let down = Sources::new().then_keep("store", Blobs::new(&store.0)).then("origin", Down);
    let e = down.get_file(a, f).unwrap_err();
    assert!(e.starts_with("origin: ") && !e.contains("store"), "{e}");
    // Fetched from the origin, nothing is kept.
    let fresh = Sources::new()
        .then_keep("store", Blobs::new(&empty.0))
        .then("origin", Always(NC.to_vec(), AtomicUsize::new(0)));
    assert_eq!(fresh.get_file(a, f).unwrap(), NC);
    assert!(!Blobs::new(&empty.0).path(&f.sha256).exists(), "no copy of a local-only file");
}
