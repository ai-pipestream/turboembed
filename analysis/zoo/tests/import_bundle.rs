//! Importing a sealed bundle directory as an index entry (#2213): the
//! entry is the one written by hand for the same files, the zoo then
//! serves the bundle from it, and a changed byte refuses the import. The
//! bundles are the repository's own small sealed ones under testdata.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_scratch::Scratch;
use turbo_zoo::import::{BundleImport, add_to, bundle, component_of};
use turbo_zoo::{
    Acceptances, Component, Directory, Index, Licence, LicenceClass, ModelType, Policy, Sha256, Sha512, Task,
    TrainingData, UserId, Zoo, ZooError,
};

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join(name)
}

/// A directory of this test's own, removed when the value is dropped.
fn scratch(name: &str) -> Scratch {
    Scratch::at(std::env::temp_dir().join(format!("turbo-zoo-import-{name}-{}", std::process::id())))
}

/// A copy of the sealed tiny BERT bundle a test may change.
fn copy_of_tiny_bert(name: &str) -> Scratch {
    let (from, to) = (testdata("tiny-bert-bundle"), scratch(name));
    for p in ["manifest.json", "tokenizer.json", "weights/model.safetensors", "reference/reference.safetensors"] {
        fs::create_dir_all(to.join(p).parent().unwrap()).unwrap();
        fs::copy(from.join(p), to.join(p)).unwrap();
    }
    to
}

fn spec(id: &str) -> BundleImport {
    BundleImport {
        id: id.parse().unwrap(),
        base: "https://models.example.org/bundles/tiny/".parse().unwrap(),
        license_terms: "https://models.example.org/bundles/tiny/LICENSE".parse().unwrap(),
        trained_on: Vec::new(),
        attribution: Some("Example attribution for tiny.".into()),
        origin: None,
        component: None,
        language: None,
    }
}

/// One file of the entry as a person writes it, from the file's own bytes.
fn file_by_hand(dir: &Path, path: &str) -> String {
    let bytes = fs::read(dir.join(path)).unwrap();
    format!(
        r#"{{"path": "{path}", "url": "https://models.example.org/bundles/tiny/{path}", "size": {}, "sha256": "{}",
            "sha512": "{}"}}"#,
        bytes.len(),
        Sha256::of(&bytes),
        Sha512::of(&bytes)
    )
}

#[test]
fn the_imported_entry_is_the_one_written_by_hand_for_the_same_files() {
    let dir = testdata("tiny-bert-bundle");
    // The manifest first, then the files in the manifest's own order.
    let files: Vec<String> =
        ["manifest.json", "reference/reference.safetensors", "tokenizer.json", "weights/model.safetensors"]
            .iter()
            .map(|p| file_by_hand(&dir, p))
            .collect();
    let by_hand = format!(
        r#"{{"zoo_version": 1, "artifacts": [{{"id": "tiny", "type": "TYPE_ENCODER_BUNDLE", "kind": "KIND_MODEL",
          "component": "COMPONENT_DENSE_ENCODER", "format": "FORMAT_BUNDLE", "version": "1", "license": "Apache-2.0",
          "license_terms": "https://models.example.org/bundles/tiny/LICENSE", "trained_on": [],
          "class": "CLASS_PERMISSIVE", "attribution": "Example attribution for tiny.",
          "bundle": {{"model": "turboembed/tiny-bert", "revision": "1", "task": "TASK_EMBED"}},
          "files": [{}]}}]}}"#,
        files.join(", ")
    );
    let by_hand = Index::parse(by_hand.as_bytes()).unwrap();
    let imported = bundle(&dir, spec("tiny")).unwrap();
    assert_eq!(&imported, by_hand.get("tiny").unwrap());
    assert_eq!(Index::to_canonical_json(std::slice::from_ref(&imported)), Index::to_canonical_json(by_hand.list()));
}

#[test]
fn the_zoo_serves_an_imported_bundle_from_its_directory() {
    for (name, task_component) in
        [("tiny-bert-bundle", Component::DenseEncoder), ("tiny-static-bundle", Component::DenseEncoder)]
    {
        let dir = testdata(name);
        let entry = bundle(&dir, spec("tiny")).unwrap();
        assert_eq!((entry.model_type, entry.component), (ModelType::EncoderBundle, Some(task_component)));
        let empty = Index::parse(Index::to_canonical_json(&[]).as_bytes()).unwrap();
        let index = Index::parse(add_to(&empty, entry).as_bytes()).unwrap();
        let cache = scratch(name);
        let source = Directory::new(&dir);
        let zoo = Zoo::new(index, cache.to_path_buf(), Policy::default(), &source);
        let user: UserId = "alice".parse().unwrap();
        let fetched = zoo.fetch("tiny", &user, &Acceptances::default()).unwrap();
        assert!(fetched.files.len() >= 4, "{name}");
        for (local, file) in fetched.files.iter().zip(&fetched.artifact.files) {
            assert_eq!(fs::read(local).unwrap(), fs::read(dir.join(file.path.as_str())).unwrap(), "{name}");
        }
    }
}

#[test]
fn a_changed_byte_a_missing_file_or_a_symlink_refuses_the_import() {
    let dir = copy_of_tiny_bert("changed");
    let weights = dir.join("weights/model.safetensors");
    let mut bytes = fs::read(&weights).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&weights, &bytes).unwrap();
    match bundle(&dir, spec("tiny")) {
        Err(ZooError::Integrity { path, check, .. }) => {
            assert_eq!((path.as_str(), check), ("weights/model.safetensors", turbo_zoo::Check::Sha256));
        }
        other => panic!("{other:?}"),
    }
    bytes.push(0);
    fs::write(&weights, &bytes).unwrap();
    assert!(matches!(bundle(&dir, spec("tiny")), Err(ZooError::Integrity { check: turbo_zoo::Check::Size, .. })));

    fs::remove_file(&weights).unwrap();
    match bundle(&dir, spec("tiny")) {
        Err(ZooError::Io { path, .. }) => assert!(path.ends_with("weights/model.safetensors")),
        other => panic!("{other:?}"),
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(testdata("tiny-bert-bundle").join("weights/model.safetensors"), &weights).unwrap();
        let e = bundle(&dir, spec("tiny")).unwrap_err().to_string();
        assert!(e.contains("is not a plain file"), "{e}");
    }
    fs::remove_dir_all(&dir).unwrap();
    assert!(matches!(bundle(&dir, spec("tiny")), Err(ZooError::Io { .. })));
}

#[test]
fn a_manifest_the_importer_cannot_take_refuses_the_import() {
    let good = fs::read_to_string(testdata("tiny-bert-bundle").join("manifest.json")).unwrap();
    for (name, from, to, why) in [
        ("task", r#""TASK_EMBED""#, r#""TASK_GENERATE""#, "is not a bundle task"),
        ("licence", r#""Apache-2.0""#, r#""Apache-9.9""#, "at /model/license"),
        ("escape", r#""path": "tokenizer.json""#, r#""path": "../tokenizer.json""#, "files[1].path"),
        ("json", "{", "", "is not JSON"),
    ] {
        let dir = copy_of_tiny_bert(name);
        assert!(good.contains(from), "{name}");
        fs::write(dir.join("manifest.json"), good.replacen(from, to, 1)).unwrap();
        let e = bundle(&dir, spec("tiny")).unwrap_err().to_string();
        assert!(e.contains(why), "{name}: {e}");
    }
}

#[test]
fn the_class_comes_from_the_licences_and_the_index_rules_hold() {
    let dir = testdata("tiny-bert-bundle");
    let marco = TrainingData {
        name: "MS MARCO".into(),
        license: Licence::RestrictedPublic,
        license_terms: "https://microsoft.github.io/msmarco/".parse().unwrap(),
    };
    // Trained on restricted data: local-only, so it needs its origin.
    let mut local = spec("tiny");
    local.trained_on = vec![marco];
    let e = bundle(&dir, local.clone()).unwrap_err().to_string();
    assert!(e.contains("CLASS_LOCAL_ONLY needs origin"), "{e}");
    local.origin = Some("https://models.example.org/bundles".parse().unwrap());
    let entry = bundle(&dir, local.clone()).unwrap();
    assert_eq!(entry.class, LicenceClass::LocalOnly);
    local.origin = Some("https://elsewhere.example.org/bundles".parse().unwrap());
    let e = bundle(&dir, local).unwrap_err().to_string();
    assert!(e.contains("is not under its origin"), "{e}");

    // The manifest's licence asks for attribution.
    let mut bare = spec("tiny");
    bare.attribution = None;
    let e = bundle(&dir, bare).unwrap_err().to_string();
    assert!(e.contains("requires attribution"), "{e}");

    // A refused licence among the training data.
    let mut refused = spec("tiny");
    refused.trained_on = vec![TrainingData {
        name: "A research-only corpus".into(),
        license: "LicenseRef-ResearchOnly".parse().unwrap(),
        license_terms: "https://models.example.org/corpus/TERMS".parse().unwrap(),
    }];
    assert!(matches!(bundle(&dir, refused), Err(ZooError::Refused { .. })));

    assert_eq!(component_of(Task::Rerank), None);
    assert_eq!(component_of(Task::ExtractiveQa), Some(Component::ExtractiveQa));
    let mut named = spec("tiny");
    named.component = Some(Component::SparseEncoder);
    assert_eq!(bundle(&dir, named).unwrap().component, Some(Component::SparseEncoder));
}

/// Runs `turbo-zoo` with its home in `home`.
fn zoo(home: &Path, args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_turbo-zoo"))
        .args(args)
        .env("TURBO_ZOO_HOME", home)
        .env("USER", "alice")
        .output()
        .unwrap();
    (out.status.success(), String::from_utf8(out.stdout).unwrap(), String::from_utf8(out.stderr).unwrap())
}

#[test]
fn the_command_adds_a_bundle_to_an_index_file_and_replaces_it_by_id() {
    let home = scratch("command");
    fs::create_dir_all(&home).unwrap();
    let bert = testdata("tiny-bert-bundle");
    let fixed = testdata("tiny-static-bundle");
    let file = home.join("zoo.json");
    let (bert, fixed, file) = (bert.to_str().unwrap(), fixed.to_str().unwrap(), file.to_str().unwrap());
    let flags = [
        "--base",
        "https://models.example.org/bundles/tiny",
        "--license-terms",
        "https://models.example.org/bundles/tiny/LICENSE",
        "--attribution",
        "Example attribution for tiny.",
    ];
    let run = |dir: &str, id: &str, more: &[&str]| {
        let mut args = vec!["import-bundle", dir, "--id", id];
        args.extend(flags);
        args.extend(more);
        zoo(&home, &args)
    };

    // Printed alone, the entry is the library's.
    let (ok, printed, err) = run(bert, "tiny", &[]);
    assert!(ok, "{err}");
    let entry = bundle(Path::new(bert), spec("tiny")).unwrap();
    assert_eq!(printed, Index::to_canonical_json(std::slice::from_ref(&entry)));
    assert_eq!(Index::parse(printed.as_bytes()).unwrap().list(), [entry]);

    let (ok, out, err) = run(bert, "tiny", &["--into", file]);
    assert!(ok && out.contains("tiny\tCLASS_PERMISSIVE\t4 files"), "{out}{err}");
    let (ok, _, err) = run(fixed, "tiny-static", &["--into", file, "--language", "en"]);
    assert!(ok, "{err}");
    // The same id again replaces the entry where it was.
    let (ok, _, err) = run(fixed, "tiny", &["--into", file]);
    assert!(ok, "{err}");
    let index = Index::parse(&fs::read(file).unwrap()).unwrap();
    let listed: Vec<(&str, &str)> =
        index.list().iter().map(|a| (a.id.as_str(), a.bundle.as_ref().unwrap().model.as_str())).collect();
    assert_eq!(
        listed,
        [("tiny", "turbo-testdata/tiny-bert-static-16"), ("tiny-static", "turbo-testdata/tiny-bert-static-16")]
    );
    assert_eq!(index.get("tiny-static").unwrap().language.as_ref().unwrap().as_str(), "en");

    // The index file reads back through the command, and `index` prints it.
    let (ok, shown, err) = zoo(&home, &["--index", file, "show", "tiny"]);
    assert!(ok && shown.contains("type: TYPE_ENCODER_BUNDLE of turbo-testdata/tiny-bert-static-16"), "{shown}{err}");
    let (ok, printed, _) = zoo(&home, &["--index", file, "index"]);
    assert!(ok);
    assert_eq!(printed, fs::read_to_string(file).unwrap());

    for bad in [&["--base"][..], &["--bogus", "x"], &["--id", "again"], &["--component", "COMPONENT_NONE"]] {
        let (ok, _, err) = run(bert, "tiny", bad);
        assert!(!ok && !err.is_empty(), "{bad:?}");
    }
    let (ok, _, err) = zoo(&home, &["import-bundle", bert, "--id", "tiny"]);
    assert!(!ok && err.contains("--base is required"), "{err}");
}
