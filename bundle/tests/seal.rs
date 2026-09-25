//! The bundle tool on real files: the MiniLM recipe, cut to the small BERT
//! whose reference the upstream pipeline wrote into testdata, and the
//! upstream tokenizer.json.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use turbo_bundle::recipe::Recipe;
use turbo_bundle::{convert, reference, seal};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-bundle-test-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// The MiniLM recipe as the small BERT's reference was made from it.
fn tiny_recipe(dir: &Path) -> PathBuf {
    let mut r: Value =
        serde_json::from_slice(&fs::read(root().join("bundle/recipes/all-minilm-l6-v2.json")).unwrap()).unwrap();
    let m = &mut r["manifest"];
    m["embed"]["dim"] = json!(32);
    m["embed"]["max_seq"] = json!(64);
    m["embed"]["prefix_query"] = json!("query: ");
    m["architecture"]["layers"] = json!(2);
    m["architecture"]["hidden"] = json!(32);
    m["architecture"]["heads"] = json!(4);
    m["architecture"]["intermediate"] = json!(64);
    r["upstream"] = json!([
        { "path": "tokenizer.json", "to": "tokenizer.json" },
        { "path": "model.safetensors", "to": "weights/model.safetensors" },
        { "path": "onnx/model.onnx", "to": "onnx/model.onnx" }
    ]);
    let p = dir.join("recipe.json");
    fs::write(&p, serde_json::to_vec_pretty(&r).unwrap()).unwrap();
    p
}

/// An upstream directory: the tokenizer, a weights file and an ONNX file.
/// Sealing hashes the weights and the ONNX file and never reads them;
/// loading weights is the model loader's job, and ONNX is read only by
/// reference programs.
fn upstream(dir: &Path) -> PathBuf {
    let up = dir.join("upstream");
    fs::create_dir_all(&up).unwrap();
    fs::copy(root().join("testdata/all-minilm-l6-v2/tokenizer.json"), up.join("tokenizer.json")).unwrap();
    let header = br#"{"__metadata__":{"format":"pt"}}      "#;
    let mut w = (header.len() as u64).to_le_bytes().to_vec();
    w.extend_from_slice(header);
    fs::write(up.join("model.safetensors"), w).unwrap();
    fs::create_dir_all(up.join("onnx")).unwrap();
    fs::write(up.join("onnx/model.onnx"), ONNX).unwrap();
    up
}

/// The bytes the upstream ONNX file has here: the tool copies and hashes
/// it, and nothing in the tool or the core parses it.
const ONNX: &[u8] = b"an ONNX graph, as far as sealing is concerned";

fn reported() -> Value {
    json!({ "tool": "sentence-transformers", "tool_version": "6.1.0", "args": ["--device", "cpu"] })
}

/// The bytes the converted F16 file has here: like the upstream ONNX
/// file, it is copied and hashed and never parsed.
const ONNX_F16: &[u8] = b"the same graph in float16, as far as sealing is concerned";

/// What the conversion run reports, in the form onnx_f16.py writes it.
fn reported_f16() -> Value {
    json!({ "tool": "onnxconverter-common", "tool_version": "1.16.0 (onnx 1.23.0)", "settings": ["keep_io_types=True"] })
}

/// Each converted artifact as a run makes it: its file written, and its
/// produced_by from what the run reported.
fn converted(r: &Recipe, bundle: &Path) -> Vec<(String, Value)> {
    convert::conversions(r)
        .unwrap()
        .into_iter()
        .map(|c| {
            let file = bundle.join(&c.file);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            fs::write(file, ONNX_F16).unwrap();
            (c.name.clone(), convert::produced_by(&reported_f16(), CONTAINER, &c, true).unwrap())
        })
        .collect()
}

const CONTAINER: &str = "turbo-reference@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// Stage, put the reference in, and seal: everything `make` does after the
/// container has run.
fn sealed(name: &str, edit: impl FnOnce(&mut Recipe)) -> (PathBuf, Result<(), String>) {
    let d = scratch(name);
    let mut r = Recipe::load(&tiny_recipe(&d)).unwrap();
    edit(&mut r);
    let bundle = d.join("bundle");
    seal::stage(&r, &upstream(&d), &bundle).unwrap();
    fs::create_dir_all(bundle.join("reference")).unwrap();
    fs::copy(
        root().join("testdata/tiny-bert-reference/reference.safetensors"),
        bundle.join("reference/reference.safetensors"),
    )
    .unwrap();
    let pb = reference::produced_by(&reported(), CONTAINER).unwrap();
    let made = converted(&r, &bundle);
    let out = seal::seal(&r, &bundle, pb, made);
    (bundle, out)
}

#[test]
fn a_sealed_bundle_loads_through_the_core() {
    let (bundle, out) = sealed("loads", |_| {});
    out.expect("sealed and verified");
    let m: Value = serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    let pb = &m["reference"]["produced_by"];
    assert_eq!(pb["container"], CONTAINER);
    assert_eq!(pb["tool_version"], "6.1.0", "from the run, not the recipe");
    let paths: Vec<&str> = m["files"].as_array().unwrap().iter().map(|f| f["path"].as_str().unwrap()).collect();
    assert_eq!(
        paths,
        [
            "onnx/model-f16.onnx",
            "onnx/model.onnx",
            "reference/reference.safetensors",
            "tokenizer.json",
            "weights/model.safetensors"
        ]
    );
    // The converted artifact's produced_by is the run's, beside what the
    // recipe named.
    let f16 = m["artifacts"].as_array().unwrap().iter().find(|a| a["name"] == "onnx-f16").unwrap();
    assert_eq!(
        f16["produced_by"],
        json!({
            "tool": "onnxconverter-common",
            "tool_version": "1.16.0 (onnx 1.23.0)",
            "container": CONTAINER,
            "from": "onnx-f32",
            "args": ["onnx/model.onnx", "onnx/model-f16.onnx", "keep_io_types=True"],
            "reproducible": true
        })
    );
    // The loader opens it as a machine would.
    seal::verify(&bundle).unwrap();
    let tok = bundle.join("tokenizer.json");
    assert_eq!(
        m["files"][3]["sha256"].as_str().unwrap(),
        turbo::bundle::sha256_hex(&fs::read(tok).unwrap()),
        "hashes are computed, not copied from the recipe"
    );
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

/// The backends the core runs an artifact on.
const BACKENDS: [&str; 4] = ["cuda", "levelzero", "metal", "cpu"];

#[test]
fn the_recipe_carries_upstreams_onnx_export_for_reference_programs_only() {
    let r = Recipe::load(&root().join("bundle/recipes/all-minilm-l6-v2.json")).unwrap();
    let arts = r.manifest["artifacts"].as_array().unwrap();
    let onnx: Vec<&Value> = arts.iter().filter(|a| a["format"] == "FORMAT_ONNX").collect();
    assert_eq!(onnx.len(), 2, "upstream's ONNX artifact and its F16 copy");
    assert_eq!(onnx[0]["name"], "onnx-f32");
    assert_eq!(onnx[0]["files"], json!(["onnx/model.onnx"]));
    assert_eq!(onnx[0]["backends"], json!([]), "no backend loads it");
    assert!(onnx[0].get("produced_by").is_none(), "the upstream file, unchanged");
    assert_eq!(arts[0]["format"], "FORMAT_SAFETENSORS", "manifest order: the weights come first");
    let up = r.upstream.iter().find(|u| u.path == "onnx/model.onnx").expect("fetched from upstream");
    assert_eq!(up.to.as_deref(), Some("onnx/model.onnx"), "and carried at the path the artifact names");
    assert!(seal::named_paths(&r.manifest).unwrap().contains("onnx/model.onnx"));

    // The F16 copy: made from onnx-f32 in the reference container, for
    // programs that build F16 only from a strongly typed graph.
    assert_eq!(onnx[1]["name"], "onnx-f16");
    assert_eq!(onnx[1]["compute_dtype"], "DTYPE_F16");
    assert_eq!(onnx[1]["backends"], json!([]), "no backend loads it");
    assert_eq!(onnx[1]["produced_by"], json!({ "from": "onnx-f32" }), "the rest comes from the run");
    assert!(r.upstream.iter().all(|u| u.to.as_deref() != Some("onnx/model-f16.onnx")), "not fetched");
    let c = convert::conversions(&r).unwrap();
    assert_eq!(
        c,
        [convert::Conversion {
            name: "onnx-f16".into(),
            file: "onnx/model-f16.onnx".into(),
            from: "onnx-f32".into(),
            from_file: "onnx/model.onnx".into(),
            script: convert::ONNX_F16,
        }]
    );
}

#[test]
fn only_an_f16_onnx_file_from_the_upstream_one_is_made() {
    let d = scratch("conversions");
    let write = |edit: &dyn Fn(&mut Value)| {
        let mut r: Value = serde_json::from_slice(&fs::read(tiny_recipe(&d)).unwrap()).unwrap();
        let arts = r["manifest"]["artifacts"].as_array_mut().unwrap();
        edit(arts.iter_mut().find(|a| a["name"] == "onnx-f16").unwrap());
        let p = d.join("edited.json");
        fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
        Recipe::load(&p).map(|_| ())
    };
    write(&|_| {}).unwrap();
    let e = write(&|a| a["produced_by"]["tool"] = json!("typed by hand")).unwrap_err();
    assert!(e.contains("names only from"), "{e}");
    let e = write(&|a| a["produced_by"]["from"] = json!("onnx-f64")).unwrap_err();
    assert!(e.contains("names no artifact"), "{e}");
    let e = write(&|a| a["compute_dtype"] = json!("DTYPE_BF16")).unwrap_err();
    assert!(e.contains("only a DTYPE_F16 FORMAT_ONNX file"), "{e}");
    let e = write(&|a| a["produced_by"]["from"] = json!("weights-f32")).unwrap_err();
    assert!(e.contains("only a DTYPE_F16 FORMAT_ONNX file"), "{e}");
    let e = write(&|a| a["files"] = json!(["onnx/a.onnx", "onnx/b.onnx"])).unwrap_err();
    assert!(e.contains("one file each"), "{e}");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn a_conversion_that_did_not_run_is_not_sealed() {
    let (bundle, out) = sealed("unconverted", |_| {});
    out.unwrap();
    let r = Recipe::load(&bundle.parent().unwrap().join("recipe.json")).unwrap();
    let pb = reference::produced_by(&reported(), CONTAINER).unwrap();
    let e = seal::seal(&r, &bundle, pb, vec![]).unwrap_err();
    assert!(e.contains("the recipe converts [\"onnx-f16\"], and the runs made []"), "{e}");
    // A second run that gave other bytes is recorded as such.
    let c = &convert::conversions(&r).unwrap()[0];
    assert_eq!(convert::produced_by(&reported_f16(), CONTAINER, c, false).unwrap()["reproducible"], false);
    assert!(convert::produced_by(&json!({ "tool": "x", "tool_version": "1" }), CONTAINER, c, true).is_err());
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

#[test]
fn the_onnx_file_is_copied_and_sealed_and_no_backend_chooses_it() {
    let (bundle, out) = sealed("onnx", |_| {});
    out.expect("sealed and verified");
    assert_eq!(fs::read(bundle.join("onnx/model.onnx")).unwrap(), ONNX, "copied byte for byte");
    let b = turbo::bundle::Bundle::open(&bundle).unwrap();
    let f = b.manifest.file("onnx/model.onnx");
    assert_eq!((f.size, f.sha256.as_str()), (ONNX.len() as u64, turbo::bundle::sha256_hex(ONNX).as_str()));
    assert_eq!(b.read_verified("onnx/model.onnx").unwrap(), ONNX);
    let i = b.manifest.artifacts.iter().position(|a| a.format == turbo::manifest::Format::Onnx).unwrap();
    assert_eq!(b.manifest.artifacts[i].files, ["onnx/model.onnx"]);
    assert!(b.manifest.artifacts[i].backends.is_empty());
    for backend in BACKENDS {
        assert_ne!(turbo::model::choose(&b.manifest, backend, "any").unwrap(), i, "{backend}");
    }
    let e = turbo::model::choose(&b.manifest, "openvino", "any").unwrap_err();
    assert!(e.message.contains("onnx-f32: backends [] has no openvino"), "{}", e.message);

    // A changed ONNX file fails verification like any other.
    fs::write(bundle.join("onnx/model.onnx"), b"another graph").unwrap();
    let e = seal::verify(&bundle).unwrap_err();
    assert!(e.contains("onnx/model.onnx"), "{e}");
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

#[test]
fn a_recipe_without_the_onnx_upstream_file_is_not_sealed() {
    let (bundle, out) = sealed("no-onnx", |r| r.upstream.retain(|u| u.path != "onnx/model.onnx"));
    let e = out.unwrap_err();
    assert!(e.contains("onnx/model.onnx"), "{e}");
    assert!(!bundle.join("manifest.json").exists());
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

#[test]
fn a_reference_whose_ids_differ_is_refused() {
    // Another case text: the reference file's ids no longer match what the
    // core's tokenizer gives, which is loader rule 5.
    let (bundle, out) = sealed("ids", |r| {
        r.manifest["reference"]["cases"][1]["text"] = json!("The quick brown fox jumps over the lazy cat.");
    });
    let e = out.unwrap_err();
    assert!(e.contains("case") || e.contains("ids"), "{e}");
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

#[test]
fn a_reference_that_is_not_normalized_is_refused() {
    let (bundle, out) = sealed("norm", |_| {});
    out.unwrap();
    // Scale every vector: the ids still match, the norms no longer do.
    let path = bundle.join("reference/reference.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let [a, b] = [0, 1].map(|i| header["embeddings"]["data_offsets"][i].as_u64().unwrap() as usize + 8 + n);
    for c in bytes[a..b].as_chunks_mut::<4>().0 {
        let v = f32::from_le_bytes(*c) * 2.0;
        c.copy_from_slice(&v.to_le_bytes());
    }
    fs::write(&path, &bytes).unwrap();
    let r = Recipe::load(&bundle.parent().unwrap().join("recipe.json")).unwrap();
    let made = converted(&r, &bundle);
    let e = seal::seal(&r, &bundle, reference::produced_by(&reported(), CONTAINER).unwrap(), made).unwrap_err();
    assert!(e.contains("norm"), "{e}");
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

#[test]
fn verify_refuses_a_changed_file_and_an_unlisted_one() {
    let (bundle, out) = sealed("verify", |_| {});
    out.unwrap();
    let w = bundle.join("weights/model.safetensors");
    let mut bytes = fs::read(&w).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&w, &bytes).unwrap();
    let e = seal::verify(&bundle).unwrap_err();
    assert!(e.contains("weights/model.safetensors") && e.contains("SHA-256"), "{e}");

    bytes[last] ^= 1;
    fs::write(&w, &bytes).unwrap();
    seal::verify(&bundle).unwrap();
    fs::write(bundle.join("notes.txt"), "x").unwrap();
    let e = seal::verify(&bundle).unwrap_err();
    assert!(e.contains("notes.txt"), "{e}");
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

#[test]
fn recipes_are_checked() {
    let d = scratch("recipes");
    let write = |edit: &dyn Fn(&mut Value)| {
        let mut r: Value = serde_json::from_slice(&fs::read(tiny_recipe(&d)).unwrap()).unwrap();
        edit(&mut r);
        let p = d.join("edited.json");
        fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
        Recipe::load(&p).map(|_| ())
    };
    write(&|_| {}).unwrap();
    let e = write(&|r| r["manifest"]["files"] = json!([])).unwrap_err();
    assert!(e.contains("files"), "{e}");
    let e = write(&|r| r["manifest"]["model"]["source"]["commit"] = json!("main")).unwrap_err();
    assert!(e.contains("commit"), "{e}");
    let e = write(&|r| r["upstream"][0]["to"] = json!("../tokenizer.json")).unwrap_err();
    assert!(e.contains("relative"), "{e}");
    let e = write(&|r| r["upstream"][0]["sha256"] = json!("ABC")).unwrap_err();
    assert!(e.contains("sha256"), "{e}");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn the_container_is_pinned_by_content() {
    assert!(reference::check_pinned(CONTAINER).is_ok());
    for bad in ["turbo-reference", "turbo-reference:latest", "turbo-reference@sha256:abc", "x@sha256:ABCDEF"] {
        assert!(reference::check_pinned(bad).is_err(), "{bad}");
    }
}

#[test]
fn the_container_gets_the_cases_with_their_prefixes() {
    let d = scratch("cases");
    let r = Recipe::load(&tiny_recipe(&d)).unwrap();
    let c = reference::cases(&r).unwrap();
    assert_eq!((c["max_seq"].as_u64(), c["max_batch"].as_u64()), (Some(64), Some(64)));
    let cases = c["cases"].as_array().unwrap();
    assert_eq!(cases.len(), r.manifest["reference"]["cases"].as_array().unwrap().len());
    assert_eq!(cases[5], json!({ "text": "how do I reset a password", "prefix": "query: " }));
    assert_eq!(cases[6]["prefix"], "", "the recipe has no document prefix");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn produced_by_comes_from_the_run() {
    let pb = reference::produced_by(&reported(), CONTAINER).unwrap();
    assert_eq!(pb["reproducible"], false);
    assert!(reference::produced_by(&json!({ "tool": "x", "args": [] }), CONTAINER).is_err());
    assert!(reference::produced_by(&json!({ "tool": "x", "tool_version": "1", "args": [1] }), CONTAINER).is_err());
}

#[test]
fn upstream_files_are_fetched_at_the_commit() {
    assert_eq!(
        turbo_bundle::fetch::url(
            "https://huggingface.co/org/model/",
            "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            "1_Pooling/config.json"
        ),
        "https://huggingface.co/org/model/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/1_Pooling/config.json"
    );
}
