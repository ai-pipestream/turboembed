//! The schema per model type (#2213): an entry without `type` is an
//! analysis artifact, and a `TYPE_ENCODER_BUNDLE` entry is a sealed
//! bundle directory whose manifest must describe the entry. The bundles
//! are the repository's own small sealed ones under testdata.

use std::fs;
use std::path::{Path, PathBuf};

use turbo_scratch::Scratch;
use turbo_zoo::{
    Acceptances, Directory, Format, Index, Kind, ModelType, Policy, Sha256, Sha512, Task, UserId, Zoo, ZooError,
};

fn testdata(bundle: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata").join(bundle)
}

/// Every file under `dir`, as a `/` separated path relative to it, sorted.
fn files_under(dir: &Path, under: &str, out: &mut Vec<String>) {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    entries.sort();
    for p in entries {
        let name = format!("{under}{}", p.file_name().unwrap().to_str().unwrap());
        if p.is_dir() { files_under(&p, &format!("{name}/"), out) } else { out.push(name) }
    }
}

/// An encoder bundle entry `id` for the sealed bundle in `dir`: every
/// file there with its own size and digests, and `bundle` as given.
fn entry(dir: &Path, id: &str, head: &str, bundle: &str) -> String {
    let mut paths = Vec::new();
    files_under(dir, "", &mut paths);
    let files: Vec<String> = paths
        .iter()
        .map(|p| {
            let bytes = fs::read(dir.join(p)).unwrap();
            format!(
                r#"{{"path": "{p}", "url": "https://example.org/{id}/{p}", "size": {}, "sha256": "{}", "sha512": "{}"}}"#,
                bytes.len(),
                Sha256::of(&bytes),
                Sha512::of(&bytes)
            )
        })
        .collect();
    format!(
        r#"{{"id": "{id}", {head}, "version": "1", "license": "Apache-2.0",
          "license_terms": "https://example.org/{id}/LICENSE", "trained_on": [], "class": "CLASS_PERMISSIVE",
          "attribution": "Example attribution for {id}."{bundle}, "files": [{}]}}"#,
        files.join(", ")
    )
}

const ENCODER: &str = r#""type": "TYPE_ENCODER_BUNDLE", "kind": "KIND_MODEL", "component": "COMPONENT_DENSE_ENCODER",
    "format": "FORMAT_BUNDLE""#;
const TINY_BERT: &str = r#", "bundle": {"model": "turboembed/tiny-bert", "revision": "1", "task": "TASK_EMBED"}"#;
const TINY_STATIC: &str =
    r#", "bundle": {"model": "turbo-testdata/tiny-bert-static-16", "revision": "1", "task": "TASK_EMBED"}"#;

fn index_of(entries: &[String]) -> Result<Index, ZooError> {
    Index::parse(format!(r#"{{"zoo_version": 1, "artifacts": [{}]}}"#, entries.join(", ")).as_bytes())
}

/// A directory of this test's own, removed when the value is dropped.
fn cache(name: &str) -> Scratch {
    Scratch::at(std::env::temp_dir().join(format!("turbo-zoo-bundle-type-{name}-{}", std::process::id())))
}

/// Fetches `tiny` from the sealed bundle in `dir`, as a server's or a
/// user's directory of bundles serves it: the bytes of each cached file.
fn fetch(name: &str, dir: &Path, index: Index) -> Result<Vec<Vec<u8>>, ZooError> {
    let cache = cache(name);
    let source = Directory::new(dir);
    let zoo = Zoo::new(index, cache.to_path_buf(), Policy::default(), &source);
    let user: UserId = "alice".parse().unwrap();
    let fetched = zoo.fetch("tiny", &user, &Acceptances::default())?;
    Ok(fetched.files.iter().map(|f| fs::read(f).unwrap()).collect())
}

#[test]
fn an_index_without_the_type_field_is_all_analysis_artifacts() {
    let index = Index::builtin();
    assert!(!turbo_zoo::index::BUILTIN.contains(r#""type""#));
    for a in index.list() {
        assert_eq!(a.model_type, ModelType::AnalysisArtifact, "{}", a.id);
        assert!(a.bundle.is_none(), "{}", a.id);
    }
}

#[test]
fn an_index_with_both_types_loads_and_keeps_its_canonical_form() {
    let lexicon = r#"{"id": "words", "kind": "KIND_LEXICON", "format": "FORMAT_TEXT", "version": "1",
        "license": "CC0-1.0", "license_terms": "https://example.org/words/LICENSE", "trained_on": [],
        "class": "CLASS_PERMISSIVE", "files": [{"path": "words.txt", "url": "https://example.org/words.txt",
        "size": 0, "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "sha512": "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"}]}"#;
    let bert = entry(&testdata("tiny-bert-bundle"), "tiny", ENCODER, TINY_BERT);
    let fixed = entry(&testdata("tiny-static-bundle"), "tiny-static", ENCODER, TINY_STATIC);
    let index = index_of(&[lexicon.to_owned(), bert, fixed]).unwrap();
    let types: Vec<ModelType> = index.list().iter().map(|a| a.model_type).collect();
    assert_eq!(types, [ModelType::AnalysisArtifact, ModelType::EncoderBundle, ModelType::EncoderBundle]);
    let tiny = index.get("tiny").unwrap();
    assert_eq!((tiny.kind, tiny.format), (Kind::Model, Format::Bundle));
    let bundle = tiny.bundle.as_ref().unwrap();
    assert_eq!(
        (bundle.model.as_str(), bundle.revision.as_str(), bundle.task),
        ("turboembed/tiny-bert", "1", Task::Embed)
    );

    let written = Index::to_canonical_json(index.list());
    assert_eq!(Index::parse(written.as_bytes()).unwrap(), index);
    assert_eq!(written.matches(r#""type": "TYPE_ENCODER_BUNDLE""#).count(), 2);
    assert!(!written.contains("TYPE_ANALYSIS_ARTIFACT"), "the default type is left out");
    let at = |key: &str| written.find(key).unwrap_or_else(|| panic!("{key} in {written}"));
    let second = written.find(r#""id": "tiny""#).unwrap();
    assert!(second < at(r#""type""#) && at(r#""type""#) < second + written[second..].find(r#""kind""#).unwrap());
    assert!(at(r#""bundle""#) < second + written[second..].find(r#""files""#).unwrap());

    // The default type written out reads as leaving it out.
    let explicit =
        lexicon.replace(r#""kind": "KIND_LEXICON""#, r#""type": "TYPE_ANALYSIS_ARTIFACT", "kind": "KIND_LEXICON""#);
    assert_eq!(index_of(&[explicit]).unwrap(), index_of(&[lexicon.to_owned()]).unwrap());
}

#[test]
fn an_entry_that_does_not_follow_its_types_schema_fails_to_load() {
    let dir = testdata("tiny-bert-bundle");
    let good = entry(&dir, "tiny", ENCODER, TINY_BERT);
    assert!(index_of(std::slice::from_ref(&good)).is_ok());
    let analysis = r#""kind": "KIND_MODEL", "format": "FORMAT_SAFETENSORS""#;
    for (bad, why) in [
        (entry(&dir, "tiny", analysis, TINY_BERT), "bundle is only for TYPE_ENCODER_BUNDLE"),
        (entry(&dir, "tiny", ENCODER, ""), "TYPE_ENCODER_BUNDLE needs bundle"),
        (good.replace("KIND_MODEL", "KIND_HEAD"), "is KIND_MODEL, not KIND_HEAD"),
        (good.replace("FORMAT_BUNDLE", "FORMAT_SAFETENSORS"), "is FORMAT_BUNDLE, not FORMAT_SAFETENSORS"),
        (good.replace(r#""path": "manifest.json""#, r#""path": "manifest.txt""#), "lists no manifest.json"),
        (good.replace("turboembed/tiny-bert", ""), "is empty, padded or holds a control character"),
        (good.replace(r#""revision": "1""#, r#""revision": " 1""#), "is empty, padded or holds a control character"),
        (good.replace("TYPE_ENCODER_BUNDLE", "TYPE_GENERATOR"), "is not a model type"),
        (good.replace("TASK_EMBED", "TASK_GENERATE"), "is not a bundle task"),
        (good.replace(r#""task": "TASK_EMBED""#, r#""task": "TASK_EMBED", "dim": 32"#), "unknown field `dim`"),
        (good.replace(r#", "task": "TASK_EMBED""#, ""), "missing field `task`"),
    ] {
        let e = index_of(&[bad]).unwrap_err().to_string();
        assert!(e.contains(why), "{why}: {e}");
    }
}

#[test]
fn a_sealed_bundle_is_fetched_whole_and_every_file_verified() {
    for (name, bundle, record) in
        [("bert", "tiny-bert-bundle", TINY_BERT), ("static", "tiny-static-bundle", TINY_STATIC)]
    {
        let dir = testdata(bundle);
        let index = index_of(&[entry(&dir, "tiny", ENCODER, record)]).unwrap();
        let listed: Vec<String> = index.get("tiny").unwrap().files.iter().map(|f| f.path.to_string()).collect();
        let files = fetch(name, &dir, index).unwrap();
        assert_eq!(files.len(), listed.len());
        for (local, path) in files.iter().zip(&listed) {
            assert_eq!(*local, fs::read(dir.join(path)).unwrap(), "{bundle}/{path}");
        }
    }
}

#[test]
fn a_manifest_that_describes_another_model_is_refused() {
    let dir = testdata("tiny-bert-bundle");
    for (name, record, why) in [
        ("model", TINY_BERT.replace("turboembed/tiny-bert", "turboembed/other"), "at /model/id"),
        ("revision", TINY_BERT.replace(r#""revision": "1""#, r#""revision": "2""#), "at /model/revision"),
        ("task", TINY_BERT.replace("TASK_EMBED", "TASK_RERANK"), "at /task"),
    ] {
        let index = index_of(&[entry(&dir, "tiny", ENCODER, &record)]).unwrap();
        match fetch(name, &dir, index) {
            Err(ZooError::Bundle { id, message }) => {
                assert_eq!(id.as_str(), "tiny");
                assert!(message.contains(why), "{message}");
            }
            other => panic!("{name}: {other:?}"),
        }
    }
    let mit = entry(&dir, "tiny", ENCODER, TINY_BERT).replace(r#""license": "Apache-2.0""#, r#""license": "MIT""#);
    let e = fetch("licence", &dir, index_of(&[mit]).unwrap()).unwrap_err().to_string();
    assert!(e.contains(r#""Apache-2.0" at /model/license, the index says MIT"#), "{e}");
}

/// A copy of the sealed tiny BERT bundle a test may change.
fn copy_of_tiny_bert(name: &str) -> Scratch {
    let from = testdata("tiny-bert-bundle");
    let to = cache(name);
    let mut paths = Vec::new();
    files_under(&from, "", &mut paths);
    for p in paths {
        fs::create_dir_all(to.join(&p).parent().unwrap()).unwrap();
        fs::copy(from.join(&p), to.join(&p)).unwrap();
    }
    to
}

#[test]
fn the_entry_and_the_manifest_must_name_the_same_files() {
    // A file the manifest does not name.
    let dir = copy_of_tiny_bert("extra-src");
    fs::write(dir.join("notes.txt"), b"not part of the bundle\n").unwrap();
    let e = fetch("extra", &dir, index_of(&[entry(&dir, "tiny", ENCODER, TINY_BERT)]).unwrap()).unwrap_err();
    assert!(e.to_string().contains("the index lists notes.txt, which manifest.json does not name"), "{e}");

    // A file the manifest names and the entry leaves out.
    let dir = copy_of_tiny_bert("short-src");
    fs::remove_file(dir.join("tokenizer.json")).unwrap();
    let e = fetch("short", &dir, index_of(&[entry(&dir, "tiny", ENCODER, TINY_BERT)]).unwrap()).unwrap_err();
    assert!(e.to_string().contains(r#"manifest.json names "tokenizer.json", which the index does not list"#), "{e}");

    // A manifest that pins other bytes for a file than the entry does.
    let dir = copy_of_tiny_bert("other-src");
    let mut tokenizer = fs::read(dir.join("tokenizer.json")).unwrap();
    tokenizer.push(b'\n');
    fs::write(dir.join("tokenizer.json"), tokenizer).unwrap();
    let e = fetch("other", &dir, index_of(&[entry(&dir, "tiny", ENCODER, TINY_BERT)]).unwrap()).unwrap_err();
    assert!(e.to_string().contains("manifest.json gives tokenizer.json another size or SHA-256"), "{e}");
}
