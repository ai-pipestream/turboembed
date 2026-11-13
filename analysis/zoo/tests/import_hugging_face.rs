//! Importing a Hugging Face repository at a pinned commit (#2213). The
//! offline tests read the repository's own small sealed bundle under
//! testdata through a transport that answers as Hugging Face does: the
//! listing of the commit, then each file at it. The ignored test reads a
//! real repository and compares the entry with the built-in index's.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use turbo_scratch::Scratch;
use turbo_zoo::import::{BundleImport, Described, HUGGING_FACE, RepoImport, bundle, hugging_face};
use turbo_zoo::{Check, Format, Index, Kind, Licence, ModelType, Sha256, Sha512, Transport, Url, ZooError};

const REPO: &str = "example-org/tiny-bert";
const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const PATHS: [&str; 4] =
    ["manifest.json", "reference/reference.safetensors", "tokenizer.json", "weights/model.safetensors"];

fn sealed() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn bytes_of(path: &str) -> Vec<u8> {
    fs::read(sealed().join(path)).unwrap()
}

/// The listing Hugging Face gives for the commit: the directories, each
/// file with its size, and the safetensors files in large file storage
/// with their SHA-256.
fn listing() -> String {
    let mut entries = vec![
        r#"{"type": "directory", "oid": "4b825dc642cb6eb9a060e54bf8d69288fbee4904", "size": 0, "path": "weights"}"#.to_owned(),
        r#"{"type": "directory", "oid": "4b825dc642cb6eb9a060e54bf8d69288fbee4904", "size": 0, "path": "reference"}"#
            .to_owned(),
        r#"{"type": "file", "oid": "a6344aac8c09253b3b630fb776ae94478aa0275b", "size": 1519, "path": ".gitattributes"}"#
            .to_owned(),
    ];
    for path in PATHS {
        let bytes = bytes_of(path);
        let lfs = if path.ends_with(".safetensors") {
            format!(r#", "lfs": {{"oid": "{}", "size": {}, "pointerSize": 134}}"#, Sha256::of(&bytes), bytes.len())
        } else {
            String::new()
        };
        entries.push(format!(
            r#"{{"type": "file", "oid": "a6344aac8c09253b3b630fb776ae94478aa0275b", "size": {}{lfs}, "path": "{path}"}}"#,
            bytes.len()
        ));
    }
    format!("[{}]", entries.join(", "))
}

/// Answers each address with what the test set for it and records the
/// addresses asked.
struct Hub {
    serve: BTreeMap<String, Vec<u8>>,
    asked: std::sync::Mutex<Vec<String>>,
}

impl Hub {
    /// The commit as Hugging Face serves it.
    fn new() -> Hub {
        let mut serve = BTreeMap::new();
        serve.insert(format!("{HUGGING_FACE}/api/models/{REPO}/tree/{COMMIT}?recursive=true"), listing().into_bytes());
        serve.insert(format!("{HUGGING_FACE}/{REPO}/resolve/{COMMIT}/.gitattributes"), vec![b'#'; 1519]);
        for path in PATHS {
            serve.insert(format!("{HUGGING_FACE}/{REPO}/resolve/{COMMIT}/{path}"), bytes_of(path));
        }
        Hub { serve, asked: std::sync::Mutex::new(Vec::new()) }
    }

    fn with(mut self, path: &str, bytes: Vec<u8>) -> Hub {
        self.serve.insert(format!("{HUGGING_FACE}/{REPO}/resolve/{COMMIT}/{path}"), bytes);
        self
    }
}

impl Transport for Hub {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        self.asked.lock().unwrap().push(url.to_string());
        self.serve.get(url.as_str()).cloned().ok_or_else(|| "404 Not Found".to_owned())
    }
}

fn described() -> Described {
    Described {
        license_terms: format!("{HUGGING_FACE}/{REPO}/blob/{COMMIT}/README.md").parse().unwrap(),
        trained_on: Vec::new(),
        attribution: Some("Example attribution for tiny.".into()),
        origin: None,
        component: None,
        language: None,
    }
}

fn spec(files: &[&str]) -> RepoImport {
    RepoImport {
        id: "tiny".parse().unwrap(),
        repo: REPO.into(),
        commit: COMMIT.into(),
        files: files.iter().map(|f| (*f).to_owned()).collect(),
        bundle: false,
        kind: Kind::Model,
        format: Format::Safetensors,
        version: None,
        license: Licence::Apache2,
        described: described(),
        keep: None,
    }
}

fn file_by_hand(path: &str) -> String {
    let bytes = bytes_of(path);
    format!(
        r#"{{"path": "{path}", "url": "{HUGGING_FACE}/{REPO}/resolve/{COMMIT}/{path}", "size": {}, "sha256": "{}",
            "sha512": "{}"}}"#,
        bytes.len(),
        Sha256::of(&bytes),
        Sha512::of(&bytes)
    )
}

#[test]
fn the_imported_entry_is_the_one_written_by_hand_for_the_same_files() {
    let chosen = ["weights/model.safetensors", "tokenizer.json"];
    let by_hand = format!(
        r#"{{"zoo_version": 1, "artifacts": [{{"id": "tiny", "kind": "KIND_MODEL", "format": "FORMAT_SAFETENSORS",
          "version": "{COMMIT}", "license": "Apache-2.0",
          "license_terms": "{HUGGING_FACE}/{REPO}/blob/{COMMIT}/README.md", "trained_on": [],
          "class": "CLASS_PERMISSIVE", "attribution": "Example attribution for tiny.", "files": [{}, {}]}}]}}"#,
        file_by_hand(chosen[0]),
        file_by_hand(chosen[1])
    );
    let by_hand = Index::parse(by_hand.as_bytes()).unwrap();
    let hub = Hub::new();
    let imported = hugging_face(&hub, HUGGING_FACE, spec(&chosen)).unwrap();
    assert_eq!(&imported, by_hand.get("tiny").unwrap());
    // The listing, then only the files named, in the order named.
    let asked = hub.asked.lock().unwrap().clone();
    assert_eq!(asked.len(), 3, "{asked:?}");
    assert!(asked[0].ends_with("/tree/0123456789abcdef0123456789abcdef01234567?recursive=true"), "{asked:?}");
    assert!(asked[1].ends_with("/weights/model.safetensors") && asked[2].ends_with("/tokenizer.json"), "{asked:?}");

    // With no file named, every file of the commit in the listing's order.
    let all = hugging_face(&Hub::new(), HUGGING_FACE, spec(&[])).unwrap();
    let paths: Vec<&str> = all.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(paths[0], ".gitattributes");
    assert_eq!(paths[1..], PATHS);
    let mut versioned = spec(&chosen);
    versioned.version = Some("1.5".into());
    assert_eq!(hugging_face(&Hub::new(), HUGGING_FACE, versioned).unwrap().version, "1.5");
}

#[test]
fn a_repository_that_holds_a_sealed_bundle_imports_as_an_encoder_bundle() {
    let mut as_bundle = spec(&PATHS);
    as_bundle.bundle = true;
    let imported = hugging_face(&Hub::new(), HUGGING_FACE, as_bundle.clone()).unwrap();
    assert_eq!(
        (imported.model_type, imported.kind, imported.format),
        (ModelType::EncoderBundle, Kind::Model, Format::Bundle)
    );
    // The same entry the sealed directory itself imports as, at that address.
    let from_dir = BundleImport {
        id: "tiny".parse().unwrap(),
        base: format!("{HUGGING_FACE}/{REPO}/resolve/{COMMIT}").parse().unwrap(),
        described: described(),
    };
    assert_eq!(imported, bundle(&sealed(), from_dir).unwrap());

    // Without the manifest, with a file the manifest does not name, or
    // under another licence than the manifest's, it is not a bundle.
    let mut short = as_bundle.clone();
    short.files.remove(0);
    let e = hugging_face(&Hub::new(), HUGGING_FACE, short).unwrap_err().to_string();
    assert!(e.contains("a bundle needs manifest.json"), "{e}");
    let mut extra = as_bundle.clone();
    extra.files.clear();
    let e = hugging_face(&Hub::new(), HUGGING_FACE, extra).unwrap_err().to_string();
    assert!(e.contains("the index lists .gitattributes, which manifest.json does not name"), "{e}");
    let mut mit = as_bundle;
    mit.license = Licence::Mit;
    let e = hugging_face(&Hub::new(), HUGGING_FACE, mit).unwrap_err().to_string();
    assert!(e.contains("at /model/license, the index says MIT"), "{e}");
}

#[test]
fn an_import_that_keeps_its_files_downloads_a_pinned_file_once() {
    let hub = Hub::new();
    let cache = Scratch::at(std::env::temp_dir().join(format!("turbo-zoo-import-hf-keep-{}", std::process::id())));
    let mut keeping = spec(&["weights/model.safetensors", "tokenizer.json"]);
    keeping.keep = Some(cache.to_path_buf());
    let entry = hugging_face(&hub, HUGGING_FACE, keeping.clone()).unwrap();
    for p in ["weights/model.safetensors", "tokenizer.json"] {
        assert_eq!(fs::read(cache.join("tiny").join(p)).unwrap(), bytes_of(p), "{p}");
    }
    // Again: the listing, and the one file the repository records no
    // SHA-256 for. The weights, which it pins, come from the cache.
    hub.asked.lock().unwrap().clear();
    assert_eq!(hugging_face(&hub, HUGGING_FACE, keeping.clone()).unwrap(), entry);
    let asked = hub.asked.lock().unwrap().clone();
    assert_eq!(asked.len(), 2, "{asked:?}");
    assert!(asked[0].contains("/tree/") && asked[1].ends_with("/tokenizer.json"), "{asked:?}");

    // Kept weights that hold other bytes are read again and put right.
    let weights = cache.join("tiny").join("weights").join("model.safetensors");
    let mut other = bytes_of("weights/model.safetensors");
    other[0] ^= 1;
    fs::write(&weights, &other).unwrap();
    hub.asked.lock().unwrap().clear();
    assert_eq!(hugging_face(&hub, HUGGING_FACE, keeping).unwrap(), entry);
    assert_eq!(hub.asked.lock().unwrap().len(), 3);
    assert_eq!(fs::read(&weights).unwrap(), bytes_of("weights/model.safetensors"));
}

/// Every file below `dir`, by its path from there; none when `dir` is
/// absent. A directory that holds no file is not listed.
fn files_below(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut dirs = vec![dir.to_path_buf()];
    while let Some(d) = dirs.pop() {
        for e in fs::read_dir(&d).into_iter().flatten().map(|e| e.unwrap()) {
            if e.file_type().unwrap().is_dir() {
                dirs.push(e.path());
            } else {
                out.push(e.path().strip_prefix(dir).unwrap().to_str().unwrap().replace('\\', "/"));
            }
        }
    }
    out.sort();
    out
}

#[test]
fn an_import_that_fails_leaves_the_cache_as_it_found_it() {
    let cache = Scratch::at(std::env::temp_dir().join(format!("turbo-zoo-import-hf-failed-{}", std::process::id())));
    let mut keeping = spec(&["weights/model.safetensors", "tokenizer.json"]);
    keeping.keep = Some(cache.to_path_buf());

    // The second file is not the size the listing gives: the first was
    // read and verified by then, and is not left behind.
    let short = Hub::new().with("tokenizer.json", b"{}".to_vec());
    let e = hugging_face(&short, HUGGING_FACE, keeping.clone());
    assert!(matches!(e, Err(ZooError::Integrity { check: Check::Size, .. })), "{e:?}");
    assert!(!cache.join("tiny").exists(), "{:?}", files_below(cache.path()));

    // Every file read and verified, and the entry then refused.
    let mut no_manifest = keeping.clone();
    no_manifest.bundle = true;
    let e = hugging_face(&Hub::new(), HUGGING_FACE, no_manifest).unwrap_err().to_string();
    assert!(e.contains("a bundle needs manifest.json"), "{e}");
    assert!(!cache.join("tiny").exists(), "{:?}", files_below(cache.path()));

    // What the cache already holds under the id, another entry's files
    // among them, is not replaced by an import that fails.
    let tokenizer = cache.join("tiny").join("tokenizer.json");
    fs::create_dir_all(cache.join("tiny").join("weights")).unwrap();
    fs::write(&tokenizer, b"another entry's tokenizer").unwrap();
    fs::write(cache.join("tiny").join("weights").join("other.bin"), b"other").unwrap();
    let mut refused = keeping.clone();
    refused.bundle = true;
    assert!(hugging_face(&Hub::new(), HUGGING_FACE, refused).is_err());
    assert!(hugging_face(&short, HUGGING_FACE, keeping.clone()).is_err());
    assert_eq!(files_below(&cache.join("tiny")), ["tokenizer.json", "weights/other.bin"]);
    assert_eq!(fs::read(&tokenizer).unwrap(), b"another entry's tokenizer");

    // The import that is accepted puts its files in place, each whole.
    hugging_face(&Hub::new(), HUGGING_FACE, keeping).unwrap();
    assert_eq!(files_below(&cache.join("tiny")), ["tokenizer.json", "weights/model.safetensors", "weights/other.bin"]);
    assert_eq!(fs::read(&tokenizer).unwrap(), bytes_of("tokenizer.json"));
}

#[test]
fn a_revision_that_may_move_is_refused_before_anything_is_asked() {
    for revision in ["main", "v1.5", "refs/pr/1", "0123456789abcdef", "0123456789ABCDEF0123456789ABCDEF01234567", ""] {
        let hub = Hub::new();
        let mut moving = spec(&[]);
        moving.commit = revision.into();
        let e = hugging_face(&hub, HUGGING_FACE, moving).unwrap_err();
        assert!(matches!(e, ZooError::Import { .. }) && e.to_string().contains("is not a commit"), "{revision}: {e}");
        assert!(hub.asked.lock().unwrap().is_empty(), "{revision}");
    }
    for repo in ["tiny-bert", "a/b/c", "../x", "a/..", "a b/c", "a/b?x", ".a/b", ""] {
        let hub = Hub::new();
        let mut named = spec(&[]);
        named.repo = repo.into();
        let e = hugging_face(&hub, HUGGING_FACE, named).unwrap_err().to_string();
        assert!(e.contains("is not a repository"), "{repo}: {e}");
        assert!(hub.asked.lock().unwrap().is_empty(), "{repo}");
    }
}

#[test]
fn a_changed_byte_or_a_file_the_commit_does_not_hold_is_refused() {
    // A file in large file storage that is not the bytes the repository
    // records for the commit.
    let mut weights = bytes_of("weights/model.safetensors");
    let last = weights.len() - 1;
    weights[last] ^= 1;
    let e = hugging_face(&Hub::new().with("weights/model.safetensors", weights.clone()), HUGGING_FACE, spec(&PATHS));
    match e {
        Err(ZooError::Integrity { path, check, .. }) => {
            assert_eq!((path.as_str(), check), ("weights/model.safetensors", Check::Sha256))
        }
        other => panic!("{other:?}"),
    }
    // Any file that is not the size the listing gives.
    weights.push(0);
    for (path, bytes) in [("weights/model.safetensors", weights), ("tokenizer.json", b"{}".to_vec())] {
        let e = hugging_face(&Hub::new().with(path, bytes), HUGGING_FACE, spec(&PATHS));
        assert!(matches!(e, Err(ZooError::Integrity { check: Check::Size, .. })), "{path}: {e:?}");
    }
    for (files, why) in [
        (&["vocab.txt"][..], "the commit holds no file \"vocab.txt\""),
        (&["tokenizer.json", "tokenizer.json"], "named twice"),
        (&["weights"], "the commit holds no file \"weights\""),
    ] {
        let e = hugging_face(&Hub::new(), HUGGING_FACE, spec(files)).unwrap_err().to_string();
        assert!(e.contains(why), "{e}");
    }
    // A commit the host does not have.
    let mut absent = spec(&[]);
    absent.commit = "f".repeat(40);
    assert!(matches!(hugging_face(&Hub::new(), HUGGING_FACE, absent), Err(ZooError::Transport { .. })));
}

#[test]
fn a_listing_the_importer_cannot_take_is_refused() {
    let tree = format!("{HUGGING_FACE}/api/models/{REPO}/tree/{COMMIT}?recursive=true");
    let file = |path: &str| format!(r#"{{"type": "file", "oid": "a6344aac", "size": 2, "path": "{path}"}}"#);
    let many: Vec<String> = (0..1000).map(|i| file(&format!("f{i}"))).collect();
    for (listing, why) in [
        (r#"{"error": "Repository not found"}"#.to_owned(), "is not a list"),
        ("not json".to_owned(), "is not JSON"),
        ("[]".to_owned(), "the commit holds no file"),
        (format!("[{}]", many.join(", ")), "may not be all of them"),
        (format!("[{}]", file("../escape.txt")), "a path the zoo does not take"),
        (r#"[{"type": "file", "oid": "a6344aac", "path": "x.txt"}]"#.to_owned(), "no size"),
        (
            r#"[{"type": "file", "size": 2, "lfs": {"oid": "abc", "size": 2}, "path": "x.bin"}]"#.to_owned(),
            "no SHA-256",
        ),
    ] {
        let mut hub = Hub::new();
        hub.serve.insert(tree.clone(), listing.into_bytes());
        let e = hugging_face(&hub, HUGGING_FACE, spec(&[])).unwrap_err().to_string();
        assert!(e.contains(why), "{why}: {e}");
    }
}

#[test]
fn a_file_the_importer_cannot_take_holds_up_only_its_own_import() {
    // The commit's listing with two files the importer cannot take: one
    // whose path leaves the repository, one in large file storage with
    // no SHA-256.
    let tree = format!("{HUGGING_FACE}/api/models/{REPO}/tree/{COMMIT}?recursive=true");
    let listed = listing();
    let unusable = r#"{"type": "file", "oid": "a6344aac", "size": 2, "path": "../escape.txt"}, {"type": "file",
        "size": 2, "lfs": {"oid": "abc", "size": 2}, "path": "x.bin"}"#;
    let mut hub = Hub::new();
    hub.serve.insert(tree, format!("[{unusable}, {}", &listed[1..]).into_bytes());
    let chosen = ["weights/model.safetensors", "tokenizer.json"];
    let imported = hugging_face(&hub, HUGGING_FACE, spec(&chosen)).unwrap();
    assert_eq!(imported, hugging_face(&Hub::new(), HUGGING_FACE, spec(&chosen)).unwrap());
    // Every file of the commit, those among them.
    let e = hugging_face(&hub, HUGGING_FACE, spec(&[])).unwrap_err().to_string();
    assert!(e.contains("\"../escape.txt\" a path the zoo does not take"), "{e}");
    let e = hugging_face(&hub, HUGGING_FACE, spec(&["x.bin"])).unwrap_err().to_string();
    assert!(e.contains("\"x.bin\" no SHA-256"), "{e}");
}

#[test]
fn a_path_an_address_does_not_carry_as_it_is_is_asked_percent_encoded() {
    let tree = format!("{HUGGING_FACE}/api/models/{REPO}/tree/{COMMIT}?recursive=true");
    let path = "vocab/Größe 1%.txt";
    let bytes = "Größe\n".as_bytes().to_vec();
    let entry = format!(r#"{{"type": "file", "oid": "a6344aac", "size": {}, "path": "{path}"}}"#, bytes.len());
    let at = format!("{HUGGING_FACE}/{REPO}/resolve/{COMMIT}/vocab/Gr%C3%B6%C3%9Fe%201%25.txt");
    let mut hub = Hub::new();
    hub.serve.insert(tree, format!("[{entry}, {}", &listing()[1..]).into_bytes());
    hub.serve.insert(at.clone(), bytes.clone());
    let imported = hugging_face(&hub, HUGGING_FACE, spec(&[path, "tokenizer.json"])).unwrap();
    let file = &imported.files[0];
    assert_eq!((file.path.as_str(), file.url.as_str(), file.size), (path, at.as_str(), bytes.len() as u64));
    assert_eq!(file.sha256, Sha256::of(&bytes));
    // The entry is one the index loads as it is written.
    let written = Index::to_canonical_json(std::slice::from_ref(&imported));
    assert_eq!(Index::parse(written.as_bytes()).unwrap().get("tiny").unwrap(), &imported);
}

/// The cache a network test keeps its downloads in: the directory
/// `TURBO_TEST_ZOO_CACHE` names, which is left in place so the next run
/// downloads nothing it holds, or else a directory of the test's own.
#[cfg(feature = "http")]
fn network_cache(name: &str) -> (PathBuf, Option<Scratch>) {
    match std::env::var_os("TURBO_TEST_ZOO_CACHE") {
        Some(dir) => (PathBuf::from(dir), None),
        None => {
            let own = Scratch::at(std::env::temp_dir().join(format!("turbo-zoo-{name}-{}", std::process::id())));
            (own.to_path_buf(), Some(own))
        }
    }
}

/// Imports `BAAI/bge-small-en-v1.5` at the commit the built-in index pins
/// and expects the built-in entry, file for file:
/// `cargo test -p turbo-zoo --test import_hugging_face -- --ignored`.
/// Needs the network. With `TURBO_TEST_ZOO_CACHE=<dir>` the files are
/// kept there, and a later run does not download again a file whose
/// SHA-256 the repository records (the weights).
#[test]
#[ignore]
#[cfg(feature = "http")]
fn a_real_repository_imports_as_the_entry_the_builtin_index_holds() {
    let index = Index::builtin();
    let held = index.get("baai-bge-small-en-v1.5").unwrap();
    let (cache, _own) = network_cache("import-hf-real");
    let spec = RepoImport {
        id: held.id.clone(),
        repo: "BAAI/bge-small-en-v1.5".into(),
        commit: held.version.clone(),
        files: held.files.iter().map(|f| f.path.to_string()).collect(),
        bundle: false,
        kind: held.kind,
        format: held.format,
        version: None,
        license: held.license,
        described: Described {
            license_terms: held.license_terms.clone(),
            trained_on: held.trained_on.clone(),
            attribution: held.attribution.clone(),
            origin: held.origin.clone(),
            component: held.component,
            language: held.language.clone(),
        },
        keep: Some(cache),
    };
    let http = turbo_zoo::Http::new();
    assert_eq!(&hugging_face(&http, HUGGING_FACE, spec.clone()).unwrap(), held);

    // A branch is refused, and so is a commit the repository does not have.
    let mut moving = spec.clone();
    moving.commit = "main".into();
    assert!(matches!(hugging_face(&http, HUGGING_FACE, moving), Err(ZooError::Import { .. })));
    let mut absent = spec;
    absent.commit = "0".repeat(40);
    assert!(matches!(hugging_face(&http, HUGGING_FACE, absent), Err(ZooError::Transport { .. })));
}
