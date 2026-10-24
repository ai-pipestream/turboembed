//! The masked language model head behind `embed.sparse`
//! (`turbo::mlm::MlmHead`): loader rules 2 and 8 like the classification
//! head's (core/tests/head.rs), the tied decoder (BERT ties it to the word
//! embeddings), refusals naming the field or tensor, and the head's
//! arithmetic against an independent numpy reference
//! (testdata/splade-golden/make_head_golden.py; no SPLADE checkpoint is
//! fetchable here, see that script and the PR).

mod common;

use std::fs;
use std::path::PathBuf;

use common::*;
use serde_json::{Value, json};
use turbo::bundle::Bundle;
use turbo::head::Linear;
use turbo::mlm::MlmHead;
use turbo::safetensors::{Dtype, File};
use turbo::status::*;
use turbo::*;

/// Copy `testdata/tiny-bert-bundle` into a temporary directory, append
/// `extra` tensors to `weights/model.safetensors`, apply `edit` to
/// `manifest.json`, and update `files[]`: core/tests/head.rs's helper.
fn tiny_with_mlm(name: &str, extra: Vec<Tensor>, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-mlm-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(tiny_bundle().join(f), dir.join(f)).unwrap();
    }
    let src_bytes = fs::read(tiny_bundle().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src_bytes[..8].try_into().unwrap()) as usize;
    let raw_header: serde_json::Map<String, Value> = serde_json::from_slice(&src_bytes[8..8 + n]).unwrap();
    let data = &src_bytes[8 + n..];
    let mut tensors = Vec::new();
    for (k, v) in raw_header {
        if k == "__metadata__" {
            continue;
        }
        let shape: Vec<u64> = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
        let offs = v["data_offsets"].as_array().unwrap();
        let (a, b) = (offs[0].as_u64().unwrap() as usize, offs[1].as_u64().unwrap() as usize);
        tensors.push(Tensor { name: k, dtype: "F32", shape, data: data[a..b].to_vec() });
    }
    tensors.extend(extra);
    let weights_bytes = safetensors_file(&tensors);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights_bytes).unwrap();

    let mut m: Value = serde_json::from_slice(&fs::read(tiny_bundle().join("manifest.json")).unwrap()).unwrap();
    edit(&mut m);
    let files = m["files"].as_array_mut().unwrap();
    let w_entry = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w_entry["size"] = json!(weights_bytes.len());
    w_entry["sha256"] = json!(sha256_hex(&weights_bytes));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

fn slice_f32(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn linspace_f32(start: f32, end: f32, count: usize) -> Vec<u8> {
    (0..count)
        .flat_map(|i| {
            let t = i as f32 / (count - 1) as f32;
            (start + (end - start) * t).to_le_bytes()
        })
        .collect()
}

/// The manifest edit every sparse test makes: declare `embed.sparse` and
/// name the MLM tensors, `decoder_weight` only when `tied` is false.
fn declare_sparse(tied: bool) -> impl FnOnce(&mut Value) {
    move |m: &mut Value| {
        m["embed"]["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
        let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("mlm_transform_weight".into(), json!("mlm.transform.weight"));
        tn.insert("mlm_transform_bias".into(), json!("mlm.transform.bias"));
        tn.insert("mlm_transform_ln_weight".into(), json!("mlm.transform.LayerNorm.weight"));
        tn.insert("mlm_transform_ln_bias".into(), json!("mlm.transform.LayerNorm.bias"));
        tn.insert("mlm_decoder_bias".into(), json!("mlm.decoder.bias"));
        if !tied {
            tn.insert("mlm_decoder_weight".into(), json!("mlm.decoder.weight"));
        }
    }
}

/// Hand-built MLM head tensors for the tiny BERT (hidden 32, vocab 30522),
/// the decoder weight only when `tied` is false.
fn mlm_tensors(tied: bool) -> Vec<Tensor> {
    let v = 30522usize;
    let mut out = vec![
        Tensor {
            name: "mlm.transform.weight".into(),
            dtype: "F32",
            shape: vec![32, 32],
            data: linspace_f32(-0.2, 0.3, 1024),
        },
        Tensor { name: "mlm.transform.bias".into(), dtype: "F32", shape: vec![32], data: linspace_f32(-0.1, 0.1, 32) },
        Tensor {
            name: "mlm.transform.LayerNorm.weight".into(),
            dtype: "F32",
            shape: vec![32],
            data: slice_f32(&[1.0; 32]),
        },
        Tensor {
            name: "mlm.transform.LayerNorm.bias".into(),
            dtype: "F32",
            shape: vec![32],
            data: slice_f32(&[0.0; 32]),
        },
        Tensor {
            name: "mlm.decoder.bias".into(),
            dtype: "F32",
            shape: vec![v as u64],
            data: linspace_f32(-0.05, 0.05, v),
        },
    ];
    if !tied {
        out.push(Tensor {
            name: "mlm.decoder.weight".into(),
            dtype: "F32",
            shape: vec![v as u64, 32],
            data: linspace_f32(-0.15, 0.15, v * 32),
        });
    }
    out
}

fn states() -> Vec<f32> {
    linspace_f32(-1.0, 1.0, 64).as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// `Bundle::open`'s error, where an open is wanted to fail; Bundle has no
/// Debug, so unwrap_err is out.
fn open_err(dir: &std::path::Path) -> turbo::status::Error {
    match Bundle::open(dir) {
        Ok(_) => panic!("{}: opened, and an error is wanted", dir.display()),
        Err(e) => e,
    }
}

#[test]
fn the_head_matches_the_numpy_reference() {
    let path = testdata().join("splade-golden/head_golden.json");
    let g: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let flat = |k: &str, nested: bool| -> Vec<f32> {
        let mut out = Vec::new();
        for r in g[k].as_array().unwrap() {
            if nested {
                out.extend(r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32));
            } else {
                out.push(r.as_f64().unwrap() as f32);
            }
        }
        out
    };
    let hidden = g["hidden"].as_u64().unwrap() as usize;
    let vocab = g["vocab"].as_u64().unwrap() as usize;
    let head = MlmHead::new(
        hidden,
        vocab,
        Linear::new(hidden, hidden, flat("transform_weight", true), flat("transform_bias", false)).unwrap(),
        flat("ln_weight", false),
        flat("ln_bias", false),
        g["eps"].as_f64().unwrap() as f32,
        Linear::new(hidden, vocab, flat("decoder_weight", true), flat("decoder_bias", false)).unwrap(),
    );
    let states = flat("states", true);
    let logits = head.logits(&states).unwrap();
    let want = flat("logits", true);
    assert_eq!(logits.len(), want.len());
    for (i, (got, want)) in logits.iter().zip(&want).enumerate() {
        let d = (*got as f64 - *want as f64).abs();
        assert!(d <= 1e-5 * (*want as f64).abs().max(1.0), "logit {i} is {got}, want {want}");
    }
    // The same through the sparse path: terms are a subsequence of the ids,
    // sorted, every weight the max-pooled transform of the golden logits.
    let terms = head.sparse_terms(&states, &sparse::SparseOptions::all()).unwrap();
    assert!(terms.windows(2).all(|w| w[0].term < w[1].term));
    let mut pool = vec![0.0f32; vocab];
    sparse::pool_logits_maxlog(&logits, vocab, &mut pool).unwrap();
    for t in &terms {
        let d = (t.weight - pool[t.term as usize]).abs();
        assert!(d <= 1e-6 * pool[t.term as usize].max(1.0));
    }
}

#[test]
fn the_tied_decoder_is_the_word_embedding_table() {
    // Untied, with the decoder weight set to the word embeddings' bytes:
    // logits must equal the tied bundle's, which names no decoder weight.
    let src = fs::read(tiny_bundle().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let raw: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    let v = &raw["embeddings.word_embeddings.weight"];
    let shape: Vec<u64> = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
    let offs = v["data_offsets"].as_array().unwrap();
    let (a, b) = (offs[0].as_u64().unwrap() as usize, offs[1].as_u64().unwrap() as usize);
    let tied_embeddings = src[8 + n + a..8 + n + b].to_vec();

    let mut untied = mlm_tensors(false);
    untied.retain(|t| t.name != "mlm.decoder.weight");
    untied.push(Tensor { name: "mlm.decoder.weight".into(), dtype: "F32", shape, data: tied_embeddings });
    let dir_u = tiny_with_mlm("untied", untied, declare_sparse(false));
    let dir_t = tiny_with_mlm("tied", mlm_tensors(true), declare_sparse(true));

    let hu = MlmHead::load(&Bundle::open(&dir_u).unwrap()).unwrap();
    let ht = MlmHead::load(&Bundle::open(&dir_t).unwrap()).unwrap();
    assert_eq!(hu.decoder.weight, ht.decoder.weight);
    let s = states();
    assert_eq!(hu.logits(&s).unwrap(), ht.logits(&s).unwrap());
}

#[test]
fn loader_rule_2_a_sparse_bundle_without_the_mlm_tensors_is_named() {
    let dir = tiny_with_mlm("no-roles", vec![], |m| {
        m["embed"]["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
    });
    let err = open_err(&dir);
    assert_eq!(err.code, BUNDLE_INVALID);
    assert!(err.message.contains("mlm_transform_weight"), "{}", err.message);
    assert!(err.message.contains("embed.sparse"), "{}", err.message);

    // Decoder bias alone is not the head.
    let dir = tiny_with_mlm("bias-only", vec![mlm_tensors(true).pop().unwrap()], |m| {
        m["embed"]["sparse"] = json!({ "pooling": "SPARSE_POOLING_MAXLOG" });
        m["artifacts"][0]["tensor_names"]
            .as_object_mut()
            .unwrap()
            .insert("mlm_decoder_bias".into(), json!("mlm.decoder.bias"));
    });
    let err = open_err(&dir);
    assert_eq!(err.code, BUNDLE_INVALID);
    assert!(err.message.contains("mlm_transform_weight"), "{}", err.message);

    // MLM roles without the declaration are named too.
    let dir = tiny_with_mlm("no-sparse", mlm_tensors(false), |m| {
        let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        tn.insert("mlm_transform_weight".into(), json!("mlm.transform.weight"));
    });
    let err = open_err(&dir);
    assert_eq!(err.code, BUNDLE_INVALID);
    assert!(err.message.contains("embed.sparse"), "{}", err.message);
}

#[test]
fn loader_rule_8_wrong_shapes_and_missing_sparse_are_refused_by_name() {
    let dir = tiny_with_mlm(
        "bad-shape",
        {
            // [16, 16] of 256 values where the architecture implies [32, 32].
            let mut t = mlm_tensors(true);
            t[0].shape = vec![16, 16];
            t[0].data = linspace_f32(-0.2, 0.3, 256);
            t
        },
        declare_sparse(true),
    );
    let b = Bundle::open(&dir).unwrap();
    let err = MlmHead::load(&b).unwrap_err();
    assert_eq!(err.code, BUNDLE_INVALID, "{}", err.message);
    assert!(err.message.contains("mlm.transform.weight"), "{}", err.message);

    // A bundle with no sparse declaration refuses the head, naming it.
    let dir = tiny_with_mlm("no-decl", mlm_tensors(true), |m| {
        let tn = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
        for (k, v) in [
            ("mlm_transform_weight", "mlm.transform.weight"),
            ("mlm_transform_bias", "mlm.transform.bias"),
            ("mlm_transform_ln_weight", "mlm.transform.LayerNorm.weight"),
            ("mlm_transform_ln_bias", "mlm.transform.LayerNorm.bias"),
            ("mlm_decoder_bias", "mlm.decoder.bias"),
        ] {
            tn.insert(k.into(), json!(v));
        }
    });
    // The manifest is refused first: MLM roles require embed.sparse.
    let err = open_err(&dir);
    assert_eq!(err.code, BUNDLE_INVALID);
    assert!(err.message.contains("embed.sparse"), "{}", err.message);

    // An out-of-bounds artifact index names the artifact.
    let dir = tiny_with_mlm("oob", mlm_tensors(true), declare_sparse(true));
    let b = Bundle::open(&dir).unwrap();
    let err = MlmHead::load_artifact(&b, b.manifest.artifacts.len()).unwrap_err();
    assert_eq!(err.code, INVALID_ARGUMENT);
}

#[test]
fn logits_refuse_non_finite_states_and_bad_shapes() {
    let dir = tiny_with_mlm("ok", mlm_tensors(true), declare_sparse(true));
    let b = Bundle::open(&dir).unwrap();
    let head = MlmHead::load(&b).unwrap();

    let mut s = states();
    assert_eq!(head.logits(&s).unwrap().len(), 2 * 30522);
    s[3] = f32::NAN;
    assert_eq!(head.logits(&s).unwrap_err().code, INVALID_ARGUMENT);
    s[3] = f32::INFINITY;
    assert_eq!(head.logits_into(&s, &mut vec![0.0; 2 * 30522]).unwrap_err().code, INVALID_ARGUMENT);
    let short = vec![0.0f32; 31];
    assert_eq!(head.logits(&short).unwrap_err().code, INVALID_SHAPE);

    // The weights file is what the head loaded: the same bytes through the
    // safetensors reader.
    let bytes = b.read_verified("weights/model.safetensors").unwrap();
    let f = File::parse("weights/model.safetensors", &bytes).unwrap();
    let bias = f.get("mlm.decoder.bias", Dtype::F32, 1).unwrap();
    assert_eq!(bias.shape, [30522]);
    assert_eq!(head.decoder.bias, bias.f32s());
}
