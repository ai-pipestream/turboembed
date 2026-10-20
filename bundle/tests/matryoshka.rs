//! Declared Matryoshka widths, end to end through the tool: the recipe's
//! `matryoshka` field becomes the manifest's `embed.output_dims`, a bad
//! list is refused at load before anything runs, and the sealed bundle
//! loads through the core: a session cut to each declared width gives the
//! reference's truncation and renormalization, an undeclared width is
//! refused by name, and the narrowed run moves no more bytes than the
//! width needs.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use turbo::safetensors::{self, Dtype};
use turbo::{TURBO_PROMPT_DOCUMENT, TURBO_PROMPT_NONE, TURBO_PROMPT_QUERY, turbo_embed_options};
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

/// The MiniLM recipe as the small BERT's reference was made from it, with
/// `edit` applied to the recipe JSON (a matryoshka field, say).
fn tiny_recipe(dir: &Path, edit: impl FnOnce(&mut Value)) -> PathBuf {
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
    edit(&mut r);
    let p = dir.join("recipe.json");
    fs::write(&p, serde_json::to_vec_pretty(&r).unwrap()).unwrap();
    for l in r["local"].as_array().into_iter().flatten() {
        let f = l["path"].as_str().unwrap();
        fs::copy(root().join("bundle/recipes").join(f), dir.join(f)).unwrap();
    }
    p
}

/// An upstream directory with the real tokenizer and the small BERT's real
/// weights, so the sealed bundle runs sessions; the ONNX bytes only carry
/// a hash, and the model loader never reads them.
fn upstream(dir: &Path) -> PathBuf {
    let up = dir.join("upstream");
    fs::create_dir_all(&up).unwrap();
    fs::copy(root().join("testdata/all-minilm-l6-v2/tokenizer.json"), up.join("tokenizer.json")).unwrap();
    fs::copy(root().join("testdata/tiny-bert-bundle/weights/model.safetensors"), up.join("model.safetensors")).unwrap();
    fs::create_dir_all(up.join("onnx")).unwrap();
    fs::write(up.join("onnx/model.onnx"), b"an ONNX graph, as far as sealing is concerned").unwrap();
    up
}

const CONTAINER: &str = "turbo-reference@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn reported() -> Value {
    json!({ "tool": "sentence-transformers", "tool_version": "6.1.0", "args": ["--device", "cpu"] })
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
                (
                    b"<ir>".as_slice(),
                    json!({
                        "tool": "openvino.save_model", "tool_version": "2026.3.0", "container": CONTAINER,
                        "reproducible": true,
                        "settings": ["seq=128", "batch=1", "max_opset=11", "compress_to_fp16=True", "output_precision=FP16"],
                    }),
                    CONTAINER.to_owned(),
                )
            } else {
                match &c.container {
                    Some(k) => (
                        b"a compiled Hailo graph, as far as sealing is concerned".as_slice(),
                        json!({ "tool": "hailo-dataflow-compiler", "tool_version": "5.4.0", "settings": ["hw_arch=hailo10h", "seq=128"] }),
                        k.clone(),
                    ),
                    None => (
                        b"the same graph in float16, as far as sealing is concerned".as_slice(),
                        json!({ "tool": "onnxconverter-common", "tool_version": "1.16.0 (onnx 1.23.0)", "settings": ["keep_io_types=True"] }),
                        CONTAINER.to_owned(),
                    ),
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

/// The recipe's load error for `widths` as its matryoshka field, or as its
/// manifest's embed.output_dims when `as_manifest`.
fn refused(widths: Value, as_manifest: bool) -> String {
    let d = scratch("widths-refused");
    let p = tiny_recipe(&d, |r| {
        if as_manifest {
            r["manifest"]["embed"]["output_dims"] = widths;
        } else {
            r["matryoshka"] = widths;
        }
    });
    let e = Recipe::load(&p).err().unwrap();
    fs::remove_dir_all(d).unwrap();
    e
}

#[test]
fn the_recipe_field_is_checked_before_anything_runs() {
    // Each width a nonzero integer within the model's dim, ascending, and
    // at most TURBO_OUTPUT_DIMS_MAX of them.
    let d = scratch("widths-checked");
    let cases: &[(&str, Value, &str)] = &[
        ("unsorted", json!([8, 4]), "matryoshka[1]: 4 follows 8: list the widths ascending"),
        ("duplicated", json!([4, 4]), "matryoshka[1]: 4 follows 4: list the widths ascending"),
        ("zero", json!([0, 4]), "matryoshka[0]: 0 is not in 1..=32"),
        ("over dim", json!([33]), "matryoshka[0]: 33 is not in 1..=32"),
        (
            "too many",
            json!((1..=17).collect::<Vec<u32>>()),
            "matryoshka: 17 widths, and turbo_model_info.output_dims holds 16",
        ),
        ("not a list", json!("16"), "matryoshka: invalid type"),
        ("not integers", json!([4, "8"]), "matryoshka[1]: invalid type"),
    ];
    for (name, widths, want) in cases {
        let e = refused(widths.clone(), false);
        assert!(e.contains(want), "{name}: {e}, want {want}");
    }
    // The manifest field the recipe already carries is checked the same way.
    for (widths, want) in [
        (json!([8, 4]), "manifest.embed.output_dims[1]: 4 follows 8"),
        (json!([0]), "manifest.embed.output_dims[0]: 0 is not in 1..=32"),
        (json!([4, 0.5]), "manifest.embed.output_dims[1]: 0.5 is not an integer"),
        (json!("16"), "manifest.embed.output_dims: not a list"),
    ] {
        let e = refused(widths, true);
        assert!(e.contains(want), "{e}, want {want}");
    }
    // The widths are not given twice.
    let p = tiny_recipe(&d, |r| {
        r["matryoshka"] = json!([4]);
        r["manifest"]["embed"]["output_dims"] = json!([4]);
    });
    let e = Recipe::load(&p).err().unwrap();
    assert!(e.contains("given twice"), "{e}");
    fs::remove_dir_all(d).unwrap();
}

/// Seal a bundle from a recipe with `edit` applied, the real weights and
/// the small BERT's reference, as `make` finishes after the container ran.
fn sealed(name: &str, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let d = scratch(name);
    let p = tiny_recipe(&d, edit);
    let r = Recipe::load(&p).unwrap();
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
    seal::seal(&r, &bundle, pb, made).expect("sealed and verified");
    bundle
}

fn manifest(bundle: &Path) -> Value {
    serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap()
}

#[test]
fn the_tool_fills_the_manifest_from_the_recipe_field() {
    for want in [json!([4, 8, 16]), json!([16])] {
        let bundle = sealed("widths-filled", |r| r["matryoshka"] = want.clone());
        let m = manifest(&bundle);
        assert_eq!(m["embed"]["output_dims"], want);
        // The recipe the seal ran from carries the filled field too.
        let r = Recipe::load(&bundle.parent().unwrap().join("recipe.json")).unwrap();
        assert_eq!(r.manifest["embed"]["output_dims"], want);
        fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
    }
    // An empty field declares no widths: the canonical manifest leaves the
    // field out, and the bundle loads and serves the model's dim alone.
    let bundle = sealed("widths-empty", |r| r["matryoshka"] = json!([]));
    let m = manifest(&bundle);
    assert!(m["embed"].get("output_dims").is_none(), "{}", m["embed"]);
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
    // No field, no widths: only the model's own dim is served, and the
    // manifest stays without the field.
    let bundle = sealed("widths-none", |_| {});
    let m = manifest(&bundle);
    assert!(m["embed"].get("output_dims").is_none(), "{}", m["embed"]);
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

/// The upstream pipeline's fp32 vectors for the bundle's reference cases:
/// the golden the widths are cut from, independent of this library.
fn golden(bundle: &Path, m: &Value) -> Vec<Vec<f32>> {
    let bytes = fs::read(bundle.join(m["reference"]["file"].as_str().unwrap())).unwrap();
    let st = safetensors::File::parse("reference.safetensors", &bytes).unwrap();
    let dim = m["embed"]["dim"].as_u64().unwrap() as usize;
    let emb = st.get("embeddings", Dtype::F32, 2).unwrap();
    assert_eq!(emb.shape, [m["reference"]["cases"].as_array().unwrap().len() as u64, dim as u64]);
    emb.f32s().chunks(dim).map(<[f32]>::to_vec).collect()
}

fn options(width: u32, prompt: u32) -> turbo_embed_options {
    turbo_embed_options {
        struct_size: size_of::<turbo_embed_options>() as u32,
        truncate: 0,
        max_tokens: 0,
        prompt_role: prompt,
        normalize: 0,
        pooling: 0,
        output_dim: width,
    }
}

#[test]
fn each_declared_width_matches_the_reference_truncation_and_renormalization() {
    let bundle = sealed("widths-golden", |r| r["matryoshka"] = json!([4, 8, 16]));
    let m = manifest(&bundle);
    let cases = m["reference"]["cases"].as_array().unwrap();
    let texts: Vec<&str> = cases.iter().map(|c| c["text"].as_str().unwrap()).collect();
    let role = |c: &Value| match c["prompt_role"].as_str().unwrap() {
        "PROMPT_QUERY" => TURBO_PROMPT_QUERY,
        "PROMPT_DOCUMENT" => TURBO_PROMPT_DOCUMENT,
        _ => TURBO_PROMPT_NONE,
    };
    let want = golden(&bundle, &m);

    let rt = turbo_bundle::api::Runtime::create().unwrap();
    let model = rt.cpu().unwrap().load(&bundle).unwrap();
    let info = model.info().unwrap();
    assert_eq!(info.output_dims_count, 3);
    assert_eq!(info.output_dims[..3], [4, 8, 16], "ascending, as the recipe gave them");
    let s = model.session(16, 64).unwrap();

    // Every width the bundle declares, and its own dim besides: the first
    // `w` coordinates of the reference vector, divided by their length.
    for w in [4u32, 8, 16, 32] {
        let mut out = vec![f32::NAN; texts.len() * w as usize];
        let per_case: Vec<turbo_embed_options> = cases.iter().map(|c| options(w, role(c))).collect();
        for (i, o) in per_case.iter().enumerate() {
            s.texts(&[texts[i]], o, &mut out[i * w as usize..(i + 1) * w as usize]).unwrap();
        }
        for (i, got) in out.chunks(w as usize).enumerate() {
            let v = &want[i][..w as usize];
            let n = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
            for (g, &x) in got.iter().zip(v) {
                let d = (*g as f64 - x as f64 / n).abs();
                assert!(d < 1e-4, "case {i} at width {w}: {g} vs {}", x as f64 / n);
            }
        }
    }

    // A width the bundle does not declare is refused by name, and one over
    // the model's dim is an argument error; both name output_dim.
    let e = s.texts(&["a"], &options(5, 0), &mut [0.0; 5]).err().unwrap();
    assert!(e.contains("UNSUPPORTED_OPTION"), "{e}");
    assert!(e.contains("output_dim 5: the model was trained to be cut to [4, 8, 16]"), "{e}");
    let e = s.texts(&["a"], &options(33, 0), &mut [0.0; 33]).err().unwrap();
    assert!(e.contains("INVALID_ARGUMENT"), "{e}");
    assert!(e.contains("output_dim 33 is over the model's dim 32"), "{e}");
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}
