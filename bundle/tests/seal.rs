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
    // The recipe's own files sit beside it.
    for l in r["local"].as_array().into_iter().flatten() {
        let f = l["path"].as_str().unwrap();
        fs::copy(root().join("bundle/recipes").join(f), dir.join(f)).unwrap();
    }
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
    json!({ "tool": "onnxconverter-common", "tool_version": "1.16.0 (onnx 1.23.0)", "settings": ["keep_io_types=True", "max_finite_val=10000.0", "min_positive_val=1e-07", "float_casts_into_f16_ops=FLOAT16"] })
}

/// The bytes the compiled HEF has here: copied and hashed, never parsed
/// by the tool.
const HEF: &[u8] = b"a compiled Hailo graph, as far as sealing is concerned";

/// What the HEF compile reports, in the form hef_compile.py writes it.
fn reported_hef() -> Value {
    json!({ "tool": "hailo-dataflow-compiler", "tool_version": "5.4.0", "settings": ["hw_arch=hailo10h", "seq=128"] })
}

/// What the IR conversion reports, in the form onnx_to_openvino_ir.py writes it.
fn reported_ir() -> Value {
    json!({
        "tool": "openvino.save_model",
        "tool_version": "2026.3.0",
        "container": CONTAINER,
        "reproducible": true,
        "settings": ["seq=128", "batch=1", "max_opset=11", "compress_to_fp16=True", "output_precision=FP16"],
    })
}

/// Each converted artifact as a run makes it: its file written, and its
/// produced_by from what the run reported, in the container it ran in.
fn converted(r: &Recipe, bundle: &Path) -> Vec<(String, Value)> {
    convert::conversions(r)
        .unwrap()
        .into_iter()
        .map(|c| {
            let file = bundle.join(&c.file);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            let (bytes, reported, container) = if c.file2.is_some() {
                (b"<ir>".as_slice(), reported_ir(), CONTAINER.to_owned())
            } else {
                match &c.container {
                    Some(k) => (HEF, reported_hef(), k.clone()),
                    None => (ONNX_F16, reported_f16(), CONTAINER.to_owned()),
                }
            };
            fs::write(file, bytes).unwrap();
            if let Some(f2) = &c.file2 {
                fs::create_dir_all(bundle.join(f2).parent().unwrap()).unwrap();
                fs::write(bundle.join(f2), b"<weights>").unwrap();
            }
            (c.name.clone(), convert::produced_by(&reported, &container, &c, true).unwrap())
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

const BACKENDS: [&str; 4] = ["cuda", "levelzero", "metal", "cpu"];
#[path = "seal/artifacts.rs"]
mod artifacts;
#[path = "seal/catalogue.rs"]
mod catalogue;
#[path = "seal/staging.rs"]
mod staging;
#[path = "seal/validation.rs"]
mod validation;
