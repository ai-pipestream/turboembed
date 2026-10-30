//! A biaffine parser head in a bundle (`manifest.heads`, docs/bundle.md
//! "Heads"): the loader reading the weights `from_tensors` takes, and the
//! manifest refusing a parser spec that mixes in a tagger's fields.

mod common;

use std::fs;
use std::path::PathBuf;

use common::*;
use serde_json::{Value, json};
use turbo::bundle::Bundle;
use turbo::head::parser::{Work, tensors};
use turbo::head::tagger::NO_WORD;
use turbo::head::{BiaffineParser, TokenTagger};
use turbo::manifest::HeadSpec;
use turbo::status::*;

fn head_json() -> Value {
    json!({
        "name": "deps",
        "layer": "turboembed:dependencies",
        "kind": "HEAD_BIAFFINE_PARSER",
        "labels": ["root", "nsubj", "obj", "det", "punct"],
        "scheme": "SCHEME_TAGS",
        "word_pooling": "WORD_POOLING_FIRST",
        "adapter": {"bilstm": {"hidden": 4, "layers": 1}},
        "decode": "DECODE_MST",
        "biaffine": {"arc": 6, "rel": 3},
        "artifact": "weights-f32",
        "tensor_prefix": "heads.deps"
    })
}

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

fn head_weights() -> Vec<(String, Vec<u64>, Vec<f32>)> {
    tensors(&spec(), 32)
        .into_iter()
        .enumerate()
        .map(|(i, (suffix, shape))| {
            let n = shape.iter().product::<u64>() as usize;
            (suffix, shape, values(i as u64 + 1, n))
        })
        .collect()
}

/// The tiny bundle with the parser's tensors appended and `edit` applied
/// to its manifest, files[] updated to match.
fn bundle(name: &str, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-parser-{}-{name}", std::process::id()));
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
    for (k, v) in header.into_iter().filter(|(k, _)| k != "__metadata__") {
        let shape = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
        let o = v["data_offsets"].as_array().unwrap();
        let (a, b) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
        all.push(Tensor { name: k, dtype: "F32", shape, data: data[a..b].to_vec() });
    }
    for (suffix, shape, v) in head_weights() {
        let data = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        all.push(Tensor { name: format!("heads.deps.{suffix}"), dtype: "F32", shape, data });
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

#[test]
fn loaded_parser_matches_the_one_built_from_the_same_weights() {
    let b = Bundle::open(&bundle("load", |_| {})).unwrap();
    assert_eq!(b.manifest.heads, [spec()]);
    let loaded = BiaffineParser::load(&b, "deps").unwrap();
    let w = head_weights();
    let get = |s: &str| w.iter().find(|(k, _, _)| k == s).map(|(_, _, v)| v.clone());
    let built = BiaffineParser::from_tensors(&spec(), 32, &get).unwrap();
    let states = values(99, 6 * 32);
    let words = [NO_WORD, 0, 1, 1, 2, NO_WORD];
    let (mut a, mut c, mut wa, mut wc) = (Vec::new(), Vec::new(), Work::default(), Work::default());
    assert_eq!(loaded.arcs(&states, &words, &mut a, &mut wa).unwrap(), 3);
    built.arcs(&states, &words, &mut c, &mut wc).unwrap();
    assert_eq!(a, c);
    let (mut ha, mut hc) = (Vec::new(), Vec::new());
    loaded.decode(&a, 3, &mut ha).unwrap();
    built.decode(&c, 3, &mut hc).unwrap();
    assert_eq!(ha, hc);
    let (mut ra, mut rc) = (Vec::new(), Vec::new());
    loaded.relations(&ha, &mut ra, &wa).unwrap();
    built.relations(&hc, &mut rc, &wc).unwrap();
    assert_eq!(ra, rc);
    assert!(TokenTagger::load(&b, "deps").unwrap_err().message.contains("not TokenTagger"));
}

#[test]
fn manifest_refuses_a_parser_with_a_taggers_fields() {
    let refused = |name: &str, edit: fn(&mut Value), containing: &str| {
        let Err(e) = Bundle::open(&bundle(name, edit)) else { panic!("{name}: the bundle opened") };
        assert_eq!(e.code, BUNDLE_INVALID, "{name}: {e:?}");
        assert!(e.message.contains(containing), "{name}: {:?}", e.message);
    };
    refused("crf", |m| m["heads"][0]["decode"] = json!("DECODE_CRF"), "HEAD_BIAFFINE_PARSER takes");
    refused("no-sizes", |m| m["heads"][0].as_object_mut().unwrap().remove("biaffine").map(drop).unwrap(), "takes");
    refused(
        "bio",
        |m| {
            m["heads"][0]["scheme"] = json!("SCHEME_BIO");
            m["heads"][0]["labels"] = json!(["O", "B-X"]);
        },
        "HEAD_BIAFFINE_PARSER takes",
    );
    refused("arc", |m| m["heads"][0]["biaffine"]["arc"] = json!(0), "biaffine.arc");
    refused("field", |m| m["heads"][0]["biaffine"]["mlp"] = json!(1), "mlp");
    refused("decode", |m| m["heads"][0]["decode"] = json!("DECODE_CHU_LIU"), "DECODE_CHU_LIU");
}
