//! The bundle side of the learned sparse output (#708): the recipe's
//! `sparse` field fills `manifest.embed.sparse` the way the Matryoshka
//! widths field fills `embed.output_dims`, checked before anything is
//! fetched; the MLM head's tensors are named in `tensor_names`, and seal
//! loads the head through the core, so a declaration the weights cannot
//! serve is refused at seal with the tensor named.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use turbo_bundle::recipe::Recipe;
use turbo_bundle::seal;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-bundle-sparse-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// The MiniLM recipe as the small BERT's reference was made from it
/// (bundle/tests/seal.rs's helper), with `edit` applied before it is
/// written, so Recipe::load reads exactly what `edit` made.
fn recipe(dir: &Path, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let mut r: Value =
        serde_json::from_slice(&fs::read(root().join("bundle/recipes/all-minilm-l6-v2.json")).unwrap()).unwrap();
    edit(&mut r);
    let p = dir.join("recipe.json");
    fs::write(&p, serde_json::to_vec_pretty(&r).unwrap()).unwrap();
    p
}

/// The raw-weights artifact of the recipe's manifest.
fn raw_weights(m: &mut Value) -> &mut Value {
    m["artifacts"].as_array_mut().unwrap().iter_mut().find(|a| a["format"] == "FORMAT_SAFETENSORS").unwrap()
}

/// The MLM head's tensor names in the manifest's raw-weights artifact.
fn name_mlm(m: &mut Value) {
    let tn = raw_weights(m)["tensor_names"].as_object_mut().unwrap();
    tn.insert("mlm_transform_weight".into(), json!("mlm.transform.weight"));
    tn.insert("mlm_transform_bias".into(), json!("mlm.transform.bias"));
    tn.insert("mlm_transform_ln_weight".into(), json!("mlm.transform.LayerNorm.weight"));
    tn.insert("mlm_transform_ln_bias".into(), json!("mlm.transform.LayerNorm.bias"));
    tn.insert("mlm_decoder_bias".into(), json!("mlm.decoder.bias"));
}

/// The manifest edit a sparse bundle on disk makes: the declaration and
/// the head's tensor names.
fn declare_mlm(m: &mut Value) {
    m["embed"]["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
    name_mlm(m);
}

#[test]
fn the_sparse_field_fills_the_manifest_before_anything_is_fetched() {
    let d = scratch("fill");
    let p = recipe(&d, |r| {
        name_mlm(&mut r["manifest"]);
        r["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
    });
    let r = Recipe::load(&p).unwrap();
    assert_eq!(r.manifest["embed"]["sparse"], json!({ "pooling": "SPARSE_POOLING_MAXLOG" }));

    // The manifest's own block is checked the same way, and giving both is
    // refused.
    let p = recipe(&d, |r| {
        declare_mlm(&mut r["manifest"]);
    });
    let r = Recipe::load(&p).unwrap();
    assert_eq!(r.manifest["embed"]["sparse"]["pooling"], "SPARSE_POOLING_MAXLOG");

    let p = recipe(&d, |r| {
        declare_mlm(&mut r["manifest"]);
        r["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
    });
    let err = Recipe::load(&p).unwrap_err();
    assert!(err.contains("given twice"), "{err}");

    // An unknown pooling value is refused, naming the field.
    let p = recipe(&d, |r| {
        declare_mlm(&mut r["manifest"]);
        r["sparse"] = json!({ "pooling": "SPARSE_POOLING_MEAN" });
    });
    let err = Recipe::load(&p).unwrap_err();
    assert!(err.contains("unknown variant"), "{err}");

    // The declaration without the head's tensor names is refused, naming
    // the role, before anything is fetched.
    let p = recipe(&d, |r| {
        r["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
    });
    let err = Recipe::load(&p).unwrap_err();
    assert!(err.contains("mlm_transform_weight"), "{err}");

    // No declaration at all: no block appears, and no error.
    let p = recipe(&d, |_| {});
    let r = Recipe::load(&p).unwrap();
    assert!(r.manifest["embed"].get("sparse").is_none());
}

/// Hand-built MLM head tensors for the small BERT (hidden 32, vocab
/// 30522), the decoder tied to the word embeddings.
fn mlm_tensors() -> Vec<(String, Vec<u64>, Vec<u8>)> {
    let lin = |start: f32, end: f32, n: usize| {
        (0..n)
            .flat_map(|i| {
                let t = i as f32 / (n - 1) as f32;
                (start + (end - start) * t).to_le_bytes()
            })
            .collect()
    };
    let v = 30522usize;
    vec![
        ("mlm.transform.weight".into(), vec![32, 32], lin(-0.2, 0.3, 1024)),
        ("mlm.transform.bias".into(), vec![32], lin(-0.1, 0.1, 32)),
        (
            "mlm.transform.LayerNorm.weight".into(),
            vec![32],
            [1.0f32.to_le_bytes()].into_iter().flatten().cycle().take(32 * 4).collect(),
        ),
        ("mlm.transform.LayerNorm.bias".into(), vec![32], vec![0u8; 32 * 4]),
        ("mlm.decoder.bias".into(), vec![v as u64], lin(-0.05, 0.05, v)),
    ]
}

/// Append `tensors` to the safetensors file at `path`, rewriting its
/// header, and return the new bytes.
fn append_safetensors(path: &Path, tensors: &[(String, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
    let bytes = fs::read(path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let mut data = bytes[8 + n..].to_vec();
    for (name, shape, vals) in tensors {
        let at = data.len();
        data.extend_from_slice(vals);
        header.insert(name.clone(), json!({ "dtype": "F32", "shape": shape, "data_offsets": [at, data.len()] }));
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    out
}

/// `testdata/tiny-bert-bundle` copied to a scratch directory, its weights
/// given `extra` tensors, its manifest `edit`ed, and its `files` entry
/// for the weights updated to the new bytes: what a sparse bundle looks
/// like on disk.
fn sparse_bundle(name: &str, extra: &[(String, Vec<u64>, Vec<u8>)], edit: impl FnOnce(&mut Value)) -> PathBuf {
    let dir = scratch(name);
    for f in ["tokenizer.json", "reference/reference.safetensors", "weights/model.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(root().join("testdata/tiny-bert-bundle").join(f), dir.join(f)).unwrap();
    }
    let weights = append_safetensors(&dir.join("weights/model.safetensors"), extra);
    fs::write(dir.join("weights/model.safetensors"), &weights).unwrap();
    let mut m: Value =
        serde_json::from_slice(&fs::read(root().join("testdata/tiny-bert-bundle/manifest.json")).unwrap()).unwrap();
    edit(&mut m);
    let files = m["files"].as_array_mut().unwrap();
    let w = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(turbo::bundle::sha256_hex(&weights));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

#[test]
fn seal_loads_the_head_and_refuses_a_bundle_the_weights_cannot_serve() {
    let ok = sparse_bundle("ok", &mlm_tensors(), declare_mlm);
    seal::verify(&ok).unwrap_or_else(|e| panic!("{e}"));

    // The declaration with no tensors behind it: refused at verify, with
    // the tensor named.
    let bad = sparse_bundle("bad", &[], declare_mlm);
    let err = seal::verify(&bad).unwrap_err();
    assert!(err.contains("mlm.transform.weight"), "{err}");

    // And the core's own parse refuses the roles with no declaration.
    let dir = sparse_bundle("nodecl", &mlm_tensors(), |m| {
        let tn = raw_weights(m)["tensor_names"].as_object_mut().unwrap();
        tn.insert("mlm_transform_weight".into(), json!("mlm.transform.weight"));
    });
    let err = seal::verify(&dir).unwrap_err();
    assert!(err.contains("embed.sparse"), "{err}");
}
