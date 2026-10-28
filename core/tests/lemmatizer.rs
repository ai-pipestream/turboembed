//! An edit-tree lemmatizer head in a bundle (`manifest.heads`,
//! docs/bundle.md "Heads"), reading the tags of the bundle's
//! part-of-speech head: the loader reading the weights `from_tensors`
//! takes, and the manifest refusing a lemmatizer whose tags no head gives.

mod common;

use std::fs;
use std::path::PathBuf;

use common::*;
use serde_json::{Value, json};
use turbo::bundle::Bundle;
use turbo::head::lemmatizer::{Work, tensors};
use turbo::head::tagger::{self, NO_WORD};
use turbo::head::{EditTreeLemmatizer, TokenTagger};
use turbo::manifest::HeadSpec;
use turbo::status::*;

fn heads() -> Value {
    json!([
        {
            "name": "pos", "layer": "turboembed:pos", "kind": "HEAD_TOKEN_TAGGER",
            "labels": ["NOUN", "VERB", "AUX"], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
            "decode": "DECODE_ARGMAX", "artifact": "weights-f32", "tensor_prefix": "heads.pos"
        },
        {
            "name": "lemma", "layer": "turboembed:lemmas", "kind": "HEAD_EDIT_TREE_LEMMATIZER",
            "labels": ["(m 0 0 () ())", "(m 0 2 () (s 'ed' ''))", "(s 'was' 'be')"],
            "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_MEAN",
            "adapter": {"bilstm": {"hidden": 3, "layers": 1}},
            "decode": "DECODE_EDIT_TREE", "pos_input": {"head": "pos", "labels": 3, "dim": 2},
            "artifact": "weights-f32", "tensor_prefix": "heads.lemma"
        }
    ])
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

fn spec(i: usize) -> HeadSpec {
    serde_json::from_value(heads()[i].clone()).unwrap()
}

/// Each head's tensors with seeded values, as (prefix.suffix, shape, values).
fn head_weights() -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let pos = tagger::tensors(&spec(0), 32).into_iter().map(|(s, sh)| (format!("heads.pos.{s}"), sh));
    let lemma = tensors(&spec(1), 32).into_iter().map(|(s, sh)| (format!("heads.lemma.{s}"), sh));
    pos.chain(lemma)
        .enumerate()
        .map(|(i, (name, shape))| {
            let n = shape.iter().product::<u64>() as usize;
            (name, shape, values(i as u64 + 1, n))
        })
        .collect()
}

fn bundle(name: &str, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-lemmatizer-{}-{name}", std::process::id()));
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
    for (name, shape, v) in head_weights() {
        all.push(Tensor { name, dtype: "F32", shape, data: v.iter().flat_map(|x| x.to_le_bytes()).collect() });
    }
    let weights = safetensors_file(&all);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights).unwrap();
    let mut m: Value = serde_json::from_slice(&fs::read(tiny_bundle().join("manifest.json")).unwrap()).unwrap();
    m["heads"] = heads();
    edit(&mut m);
    let files = m["files"].as_array_mut().unwrap();
    let w = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(sha256_hex(&weights));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

#[test]
fn loaded_lemmatizer_matches_the_one_built_from_the_same_weights() {
    let b = Bundle::open(&bundle("load", |_| {})).unwrap();
    let loaded = EditTreeLemmatizer::load(&b, "lemma").unwrap();
    let w = head_weights();
    let get = |s: &str| w.iter().find(|(k, _, _)| *k == format!("heads.lemma.{s}")).map(|(_, _, v)| v.clone());
    let built = EditTreeLemmatizer::from_tensors(&spec(1), 32, &get).unwrap();
    let pos = TokenTagger::load(&b, "pos").unwrap();
    let states = values(99, 5 * 32);
    let words = [NO_WORD, 0, 1, 1, NO_WORD];
    let (mut ts, mut tags) = (Vec::new(), Vec::new());
    pos.scores(&states, &words, &mut ts, &mut Default::default()).unwrap();
    pos.decode(&ts, 2, &mut tags).unwrap();
    let (mut a, mut c) = (Vec::new(), Vec::new());
    assert_eq!(loaded.scores(&states, &words, Some(&tags), &mut a, &mut Work::default()).unwrap(), 2);
    built.scores(&states, &words, Some(&tags), &mut c, &mut Work::default()).unwrap();
    assert_eq!(a, c);
    let forms = ["walked", "was"];
    let (mut da, mut dc) = (Vec::new(), Vec::new());
    loaded.decode(&a, &forms, &mut da).unwrap();
    built.decode(&c, &forms, &mut dc).unwrap();
    assert_eq!(da, dc);
    assert!(da.iter().all(Option::is_some), "the identity fits every word");
    assert!(TokenTagger::load(&b, "lemma").unwrap_err().message.contains("not TokenTagger"));
}

#[test]
fn manifest_refuses_a_lemmatizer_whose_tags_no_head_gives() {
    let refused = |name: &str, edit: fn(&mut Value), containing: &str| {
        let Err(e) = Bundle::open(&bundle(name, edit)) else { panic!("{name}: the bundle opened") };
        assert_eq!(e.code, BUNDLE_INVALID, "{name}: {e:?}");
        assert!(e.message.contains(containing), "{name}: {:?}", e.message);
    };
    refused("no-head", |m| m["heads"][1]["pos_input"]["head"] = json!("upos"), "pos_input.head");
    refused("count", |m| m["heads"][1]["pos_input"]["labels"] = json!(4), "pos_input.head");
    refused(
        "not-tags",
        |m| {
            m["heads"][0]["scheme"] = json!("SCHEME_BIO");
            m["heads"][0]["labels"] = json!(["O", "B-X", "I-X"]);
        },
        "pos_input.head",
    );
    refused("bad-tree", |m| m["heads"][1]["labels"][1] = json!("(m 0 2 () (s 'ed' '')"), "labels[1]");
    refused("decode", |m| m["heads"][1]["decode"] = json!("DECODE_ARGMAX"), "HEAD_EDIT_TREE_LEMMATIZER takes");
}
