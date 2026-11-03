//! Token tagger heads in a bundle (`manifest.heads`, docs/bundle.md
//! "Heads"): the manifest round trip, the loader reading the same weights
//! `from_tensors` takes, and every refusal the manifest and loader make.

mod common;

use std::fs;
use std::path::PathBuf;

use common::*;
use serde_json::{Value, json};
use turbo::bundle::Bundle;
use turbo::head::TokenTagger;
use turbo::head::tagger::{NO_WORD, Work, tensors};
use turbo::manifest::{HeadSpec, Manifest};
use turbo::status::*;

const LABELS: [&str; 5] = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"];

fn head_json() -> Value {
    json!({
        "name": "ner",
        "layer": "turboembed:entities",
        "kind": "HEAD_TOKEN_TAGGER",
        "labels": LABELS,
        "scheme": "SCHEME_BIO",
        "word_pooling": "WORD_POOLING_MEAN",
        "adapter": {"bilstm": {"hidden": 6, "layers": 2}},
        "decode": "DECODE_CRF",
        "artifact": "weights-f32",
        "tensor_prefix": "heads.ner"
    })
}

/// Deterministic values in [-0.5, 0.5).
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            ((s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn spec() -> HeadSpec {
    serde_json::from_value(head_json()).unwrap()
}

/// Each head tensor by suffix, with its values.
fn head_weights(spec: &HeadSpec) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    tensors(spec, 32)
        .into_iter()
        .enumerate()
        .map(|(i, (suffix, shape))| {
            let n = shape.iter().product::<u64>() as usize;
            (suffix, shape, values(i as u64 + 1, n))
        })
        .collect()
}

/// The tiny bundle with the tensors `extra` appended to its weights and
/// `edit` applied to its manifest, files[] updated to match.
fn bundle(name: &str, extra: &[(String, Vec<u64>, Vec<f32>)], edit: impl FnOnce(&mut Value)) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-tagger-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(tiny_bundle().join(f), dir.join(f)).unwrap();
    }
    let src = fs::read(tiny_bundle().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    let data = &src[8 + n..];
    let mut all = Vec::new();
    for (k, v) in header {
        if k == "__metadata__" {
            continue;
        }
        let shape = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
        let offs = v["data_offsets"].as_array().unwrap();
        let (a, b) = (offs[0].as_u64().unwrap() as usize, offs[1].as_u64().unwrap() as usize);
        all.push(Tensor { name: k, dtype: "F32", shape, data: data[a..b].to_vec() });
    }
    for (suffix, shape, v) in extra {
        all.push(Tensor {
            name: format!("heads.ner.{suffix}"),
            dtype: "F32",
            shape: shape.clone(),
            data: v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        });
    }
    let weights = safetensors_file(&all);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights).unwrap();
    let mut m: Value = serde_json::from_slice(&fs::read(tiny_bundle().join("manifest.json")).unwrap()).unwrap();
    m["heads"] = json!([head_json()]);
    edit(&mut m);
    let files = m["files"].as_array_mut().unwrap();
    let w = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(sha256_hex(&weights));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

fn open(name: &str, edit: impl FnOnce(&mut Value)) -> Result<Bundle> {
    Bundle::open(&bundle(name, &head_weights(&spec()), edit))
}

fn refused(name: &str, edit: impl FnOnce(&mut Value), code: i32, containing: &str) {
    let Err(e) = open(name, edit) else { panic!("{name}: the bundle opened") };
    assert_eq!(e.code, code, "{name}: {e:?}");
    assert!(e.message.contains(containing), "{name}: {:?} lacks {containing:?}", e.message);
}

#[test]
fn heads_round_trip_through_the_manifest() {
    let b = open("round-trip", |_| {}).unwrap();
    assert_eq!(b.manifest.heads, [spec()]);
    let text = serde_json::to_string(&b.manifest).unwrap();
    let back: Manifest = serde_json::from_str(&text).unwrap();
    assert_eq!(back.heads, [spec()]);
    // A bundle without heads writes no `heads` field.
    let plain: Value = serde_json::from_slice(&fs::read(tiny_bundle().join("manifest.json")).unwrap()).unwrap();
    let m: Manifest = serde_json::from_value(plain).unwrap();
    assert!(!serde_json::to_string(&m).unwrap().contains("\"heads\":["));
}

#[test]
fn loaded_tagger_matches_the_one_built_from_the_same_weights() {
    let s = spec();
    let w = head_weights(&s);
    let b = Bundle::open(&bundle("load", &w, |_| {})).unwrap();
    let loaded = TokenTagger::load(&b, "ner").unwrap();
    let get = |suffix: &str| w.iter().find(|(k, _, _)| k == suffix).map(|(_, _, v)| v.clone());
    let built = TokenTagger::from_tensors(&s, 32, &get).unwrap();
    assert_eq!(loaded.hidden(), 32);
    assert_eq!(loaded.types(), ["PER", "LOC"]);
    // Seven rows: [CLS], three words of 1, 2 and 2 subwords, [SEP].
    let states = values(99, 7 * 32);
    let words = [NO_WORD, 0, 1, 1, 2, 2, NO_WORD];
    let (mut a, mut c) = (Vec::new(), Vec::new());
    let mut work = Work::default();
    assert_eq!(loaded.scores(&states, &words, &mut a, &mut work).unwrap(), 3);
    built.scores(&states, &words, &mut c, &mut work).unwrap();
    assert_eq!(a, c);
    let (mut ta, mut tc) = (Vec::new(), Vec::new());
    loaded.decode(&a, 3, &mut ta).unwrap();
    built.decode(&c, 3, &mut tc).unwrap();
    assert_eq!(ta, tc);
    assert_eq!(TokenTagger::load(&b, "pos").unwrap_err().code, INVALID_ARGUMENT);
}

#[test]
fn manifest_refuses_ill_formed_heads() {
    fn h(m: &mut Value) -> &mut Value {
        &mut m["heads"][0]
    }
    refused("kind", |m| h(m)["kind"] = json!("HEAD_PARSER"), BUNDLE_INVALID, "HEAD_PARSER");
    refused("unknown-field", |m| h(m)["dropout"] = json!(0.1), BUNDLE_INVALID, "dropout");
    refused("bad-label", |m| h(m)["labels"] = json!(["O", "L-PER"]), BUNDLE_INVALID, "L-PER");
    refused("no-outside", |m| h(m)["labels"] = json!(["B-PER", "I-PER"]), BUNDLE_INVALID, "needs the label");
    refused("twice", |m| h(m)["labels"] = json!(["O", "O"]), BUNDLE_INVALID, "listed twice");
    refused("empty", |m| h(m)["labels"] = json!([]), BUNDLE_INVALID, "0 labels");
    refused(
        "too-many",
        |m| {
            let many: Vec<String> = (0..4097).map(|i| format!("L{i}")).collect();
            h(m)["labels"] = json!(many);
            h(m)["scheme"] = json!("SCHEME_TAGS");
        },
        BUNDLE_INVALID,
        "4097 labels",
    );
    refused("name", |m| h(m)["name"] = json!("NER"), BUNDLE_INVALID, "lowercase");
    refused(
        "duplicate",
        |m| {
            let again = m["heads"][0].clone();
            m["heads"].as_array_mut().unwrap().push(again);
        },
        BUNDLE_INVALID,
        "names two heads",
    );
    refused("artifact", |m| h(m)["artifact"] = json!("weights-f16"), BUNDLE_INVALID, "no artifact \"weights-f16\"");
    refused("artifact-index", |m| h(m)["artifact"] = json!(0), BUNDLE_INVALID, "artifact");
    refused("layer-namespace", |m| h(m)["layer"] = json!("turbo:pos"), BUNDLE_INVALID, "turboembed:");
    refused("layer-name", |m| h(m)["layer"] = json!("turboembed:"), BUNDLE_INVALID, "turboembed:");
    refused("layer-case", |m| h(m)["layer"] = json!("turboembed:POS"), BUNDLE_INVALID, "turboembed:");
    refused("layers", |m| h(m)["adapter"]["bilstm"]["layers"] = json!(4), BUNDLE_INVALID, "at most");
    refused("hidden", |m| h(m)["adapter"]["bilstm"]["hidden"] = json!(0), BUNDLE_INVALID, "hidden");
    refused("prefix", |m| h(m)["tensor_prefix"] = json!(""), BUNDLE_INVALID, "tensor_prefix");
}

#[test]
fn loader_refuses_missing_and_misshapen_tensors() {
    let s = spec();
    let mut w = head_weights(&s);
    w.retain(|(k, _, _)| k != "crf.end");
    let b = Bundle::open(&bundle("missing", &w, |_| {})).unwrap();
    let e = TokenTagger::load(&b, "ner").unwrap_err();
    assert!(e.message.contains("heads.ner.crf.end"), "{e:?}");

    let mut w = head_weights(&s);
    let out = w.iter_mut().find(|(k, _, _)| k == "out.weight").unwrap();
    out.1 = vec![out.1[1], out.1[0]];
    let b = Bundle::open(&bundle("shape", &w, |_| {})).unwrap();
    let e = TokenTagger::load(&b, "ner").unwrap_err();
    assert!(e.message.contains("heads.ner.out.weight"), "{e:?}");
}
