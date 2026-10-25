//! Files the index serves compressed: the wire bytes checked by their own
//! size and SHA-256, unpacked, and the unpacked file checked by size,
//! SHA-256 and SHA-512 before the cache holds it.

use std::cell::Cell;
use std::fs;
use std::path::PathBuf;

use turbo_zoo::{Acceptances, Check, Index, Policy, Sha256, Sha512, Transport, Url, UserId, Zoo, ZooError};

const HELLO: &[u8] = b"hello hello hello\n";

/// HELLO as Python's `gzip.compress(..., mtime=0)` writes it.
const HELLO_GZ: [u8; 29] = [
    0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x90,
    0x5c, 0x00, 0x3b, 0x7c, 0x8a, 0xdf, 0x12, 0x00, 0x00, 0x00,
];

/// HELLO as one zstd frame of one raw block.
fn hello_zst() -> Vec<u8> {
    let n = HELLO.len() as u32;
    let mut out = vec![0x28, 0xb5, 0x2f, 0xfd, 0x20, n as u8];
    out.extend_from_slice(&((n << 3) | 1).to_le_bytes()[..3]);
    out.extend_from_slice(HELLO);
    out
}

const URL: &str = "https://example.org/models/hello.txt.gz";

/// Serves `wire` at URL and records the size the zoo expected.
struct One {
    wire: Vec<u8>,
    gets: Cell<usize>,
    size: Cell<u64>,
}

impl Transport for One {
    fn get(&self, url: &Url, size: u64) -> Result<Vec<u8>, String> {
        assert_eq!(url.as_str(), URL);
        self.gets.set(self.gets.get() + 1);
        self.size.set(size);
        Ok(self.wire.clone())
    }
}

/// An index of one permissive artifact whose file `hello.txt` is served
/// compressed as `wire` claims: `compression`, and the wire size and
/// digest given. The unpacked file is `unpacked`.
fn index(compression: &str, wire_size: usize, wire_sha256: Sha256, unpacked: &[u8]) -> String {
    format!(
        r#"{{"zoo_version": 1, "artifacts": [{{
      "id": "hello",
      "kind": "KIND_LEXICON",
      "format": "FORMAT_TEXT",
      "version": "1",
      "license": "Apache-2.0",
      "license_terms": "https://example.org/hello/LICENSE",
      "trained_on": [],
      "class": "CLASS_PERMISSIVE",
      "attribution": "Example attribution.",
      "files": [{{
        "path": "hello.txt",
        "url": "{URL}",
        "size": {size},
        "sha256": "{sha256}",
        "sha512": "{sha512}",
        "download": {{"compression": "{compression}", "download_bytes": {wire_size}, "download_sha256": "{wire_sha256}"}}
      }}]
    }}]}}"#,
        size = unpacked.len(),
        sha256 = Sha256::of(unpacked),
        sha512 = Sha512::of(unpacked),
    )
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("turbo-zoo-transfer-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn user() -> UserId {
    "alice".parse().unwrap()
}

fn fetch(json: &str, wire: Vec<u8>, dir: &TempDir) -> (Result<PathBuf, ZooError>, usize, u64) {
    let t = One { wire, gets: Cell::new(0), size: Cell::new(0) };
    let zoo = Zoo::new(Index::parse(json.as_bytes()).unwrap(), dir.0.clone(), Policy::default(), &t);
    let got = zoo.fetch("hello", &user(), &Acceptances::default()).map(|f| f.files[0].clone());
    (got, t.gets.get(), t.size.get())
}

#[test]
fn a_compressed_file_is_fetched_unpacked_and_cached() {
    for (name, compression, wire) in [("gz", "gzip", HELLO_GZ.to_vec()), ("zst", "zstd", hello_zst())] {
        let dir = TempDir::new(name);
        let json = index(compression, wire.len(), Sha256::of(&wire), HELLO);
        let (path, gets, size) = fetch(&json, wire.clone(), &dir);
        let path = path.unwrap();
        assert_eq!(fs::read(&path).unwrap(), HELLO, "{compression}");
        assert_eq!((gets, size), (1, wire.len() as u64), "the transport is told the wire size");
        // The cache holds the unpacked file and checks it as any other.
        let (again, gets, _) = fetch(&json, wire, &dir);
        assert_eq!((again.unwrap(), gets), (path, 0));
    }
}

fn integrity(r: Result<PathBuf, ZooError>) -> Check {
    match r {
        Err(ZooError::Integrity { check, .. }) => check,
        other => panic!("expected an integrity error, got {other:?}"),
    }
}

#[test]
fn every_check_on_the_wire_and_after_unpacking_refuses_and_writes_nothing() {
    let dir = TempDir::new("checks");
    let wire = HELLO_GZ.to_vec();
    let sha = Sha256::of(&wire);
    let cases = [
        (index("gzip", wire.len() + 1, sha, HELLO), wire.clone(), Check::DownloadSize),
        (index("gzip", wire.len(), Sha256::of(b"other"), HELLO), wire.clone(), Check::DownloadSha256),
        // The right wire bytes, which unpack to another file than the index's.
        (index("gzip", wire.len(), sha, b"hello hello hellO\n"), wire.clone(), Check::Sha256),
        (index("gzip", wire.len(), sha, b"hello\n"), wire.clone(), Check::Unpack),
        // zstd bytes recorded as gzip.
        (index("gzip", hello_zst().len(), Sha256::of(&hello_zst()), HELLO), hello_zst(), Check::Unpack),
    ];
    for (json, wire, check) in cases {
        let (got, _, _) = fetch(&json, wire, &dir);
        assert_eq!(integrity(got), check);
        assert!(!dir.0.join("hello").join("hello.txt").exists());
    }
}

#[test]
fn the_download_entry_round_trips_and_is_strict() {
    let wire = hello_zst();
    let json = index("zstd", wire.len(), Sha256::of(&wire), HELLO);
    let parsed = Index::parse(json.as_bytes()).unwrap();
    let canonical = Index::to_canonical_json(parsed.list());
    assert_eq!(Index::parse(canonical.as_bytes()).unwrap().list(), parsed.list());
    assert!(canonical.contains("\"compression\": \"zstd\""));
    assert!(Index::parse(json.replace("\"zstd\"", "\"brotli\"").as_bytes()).is_err());
    let extra = json.replace("\"compression\"", "\"level\": 3, \"compression\"");
    assert!(Index::parse(extra.as_bytes()).is_err());
    let missing = json.replace(&format!(", \"download_sha256\": \"{}\"", Sha256::of(&wire)), "");
    assert!(Index::parse(missing.as_bytes()).is_err());
}
