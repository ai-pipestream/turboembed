//! The sentence boundary, token split and span classifier heads in one
//! bundle (`manifest.heads`, docs/bundle.md "Heads"): each loaded from
//! the artifact scores as the head built from the same weights, and the
//! manifest refuses each kind with another kind's fields.

mod common;

use std::fs;
use std::path::PathBuf;

use common::*;
use serde_json::{Value, json};
use turbo::bundle::Bundle;
use turbo::head::span::SpanWork;
use turbo::head::tagger::{NO_WORD, Work};
use turbo::head::{SentenceBoundary, SpanClassifier, TokenSplitter, TokenTagger, boundary, span, split};
use turbo::manifest::HeadSpec;
use turbo::status::*;

const DEL: &str = "(m 0 1 () (s 'l' '')) (m 1 0 (s 'd' '') ())";

fn heads_json() -> Value {
    let common = |name: &str, layer: &str, kind: &str, labels: Value, decode: &str| {
        json!({
            "name": name,
            "layer": layer,
            "kind": kind,
            "labels": labels,
            "scheme": "SCHEME_TAGS",
            "word_pooling": "WORD_POOLING_MEAN",
            "decode": decode,
            "artifact": "weights-f32",
            "tensor_prefix": format!("heads.{name}")
        })
    };
    let mut sents =
        common("sents", "turboembed:sentences", "HEAD_SENTENCE_BOUNDARY", json!(["inside", "last"]), "DECODE_CRF");
    sents["adapter"] = json!({"bilstm": {"hidden": 4, "layers": 1}});
    let words = common("words", "turboembed:words", "HEAD_TOKEN_SPLIT", json!(["_", DEL]), "DECODE_EDIT_TREE");
    let mut spans = common(
        "mentions",
        "turboembed:mentions",
        "HEAD_SPAN_CLASSIFIER",
        json!(["O", "PER", "ORG"]),
        "DECODE_SPANS_FLAT",
    );
    spans["span"] = json!({"max_words": 3, "width_dim": 5});
    json!([sents, words, spans])
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

fn specs() -> Vec<HeadSpec> {
    serde_json::from_value(heads_json()).unwrap()
}

type Shapes = fn(&HeadSpec, usize) -> Vec<(String, Vec<u64>)>;

/// Head `k`'s tensors by suffix, shape and seeded values.
fn head_weights(k: usize) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let shapes: [Shapes; 3] = [boundary::tensors, split::tensors, span::tensors];
    shapes[k](&specs()[k], 32)
        .into_iter()
        .enumerate()
        .map(|(i, (suffix, shape))| {
            let n = shape.iter().product::<u64>() as usize;
            (suffix, shape, values(100 * k as u64 + i as u64 + 1, n))
        })
        .collect()
}

/// The tiny bundle with the three heads' tensors appended and `edit`
/// applied to its manifest, files[] updated to match.
fn bundle(name: &str, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-segment-heads-{}-{name}", std::process::id()));
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
    for (k, spec) in specs().iter().enumerate() {
        for (suffix, shape, v) in head_weights(k) {
            let data = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            all.push(Tensor { name: format!("{}.{suffix}", spec.tensor_prefix), dtype: "F32", shape, data });
        }
    }
    let weights = safetensors_file(&all);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights).unwrap();
    let mut m: Value = serde_json::from_slice(&fs::read(tiny_bundle().join("manifest.json")).unwrap()).unwrap();
    m["heads"] = heads_json();
    edit(&mut m);
    let files = m["files"].as_array_mut().unwrap();
    let w = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(sha256_hex(&weights));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

fn getter(k: usize) -> impl Fn(&str) -> Option<Vec<f32>> {
    let w = head_weights(k);
    move |s: &str| w.iter().find(|(n, _, _)| n == s).map(|(_, _, v)| v.clone())
}

/// [CLS], four words of 1, 2, 1 and 1 rows, [SEP].
const WORDS: [u32; 7] = [NO_WORD, 0, 1, 1, 2, 3, NO_WORD];

#[test]
fn each_loaded_head_matches_the_one_built_from_the_same_weights() {
    let b = Bundle::open(&bundle("load", |_| {})).unwrap();
    let s = specs();
    assert_eq!(b.manifest.heads, s);
    let states = values(99, WORDS.len() * 32);
    let (mut a, mut c, mut wa, mut wc) = (Vec::new(), Vec::new(), Work::default(), Work::default());

    let loaded = SentenceBoundary::load(&b, "sents").unwrap();
    let built = SentenceBoundary::from_tensors(&s[0], 32, &getter(0)).unwrap();
    assert_eq!(loaded.scores(&states, &WORDS, None, &mut a, &mut wa).unwrap(), 4);
    built.scores(&states, &WORDS, None, &mut c, &mut wc).unwrap();
    assert_eq!(a, c);
    let (mut ea, mut ec) = (Vec::new(), Vec::new());
    loaded.decode(&a, 4, &mut ea).unwrap();
    built.decode(&c, 4, &mut ec).unwrap();
    assert_eq!((ea.len(), ea.last()), (4, Some(&true)));
    assert_eq!(ea, ec);

    let loaded = TokenSplitter::load(&b, "words").unwrap();
    let built = TokenSplitter::from_tensors(&s[1], 32, &getter(1)).unwrap();
    loaded.scores(&states, &WORDS, None, &mut a, &mut wa).unwrap();
    built.scores(&states, &WORDS, None, &mut c, &mut wc).unwrap();
    assert_eq!(a, c);
    let forms = ["del", "mar", "al", "sol"];
    let (mut la, mut lc) = (Vec::new(), Vec::new());
    loaded.decode(&a, &forms, &mut la).unwrap();
    built.decode(&c, &forms, &mut lc).unwrap();
    assert_eq!(la, lc);
    let split = loaded.words(&la, &forms).unwrap();
    for ((&l, words), form) in la.iter().zip(&split).zip(forms) {
        // Only "del" fits the split; the others stay whole.
        let whole = [form];
        assert_eq!(words.as_slice(), if l == 1 { &["de", "el"][..] } else { &whole[..] });
    }

    let loaded = SpanClassifier::load(&b, "mentions").unwrap();
    let built = SpanClassifier::from_tensors(&s[2], 32, &getter(2)).unwrap();
    let (mut sa, mut sc) = (SpanWork::default(), SpanWork::default());
    assert_eq!(loaded.scores(&states, &WORDS, None, &mut a, &mut wa, &mut sa).unwrap(), 4);
    built.scores(&states, &WORDS, None, &mut c, &mut wc, &mut sc).unwrap();
    assert_eq!((a.len(), &a), (loaded.span_count(4) * 3, &c));
    let (mut pa, mut pc) = (Vec::new(), Vec::new());
    loaded.decode(&a, 4, &mut pa).unwrap();
    built.decode(&c, 4, &mut pc).unwrap();
    assert_eq!(pa, pc);
    assert!(pa.windows(2).all(|p| p[0].last < p[1].first), "flat spans do not overlap: {pa:?}");

    assert!(TokenTagger::load(&b, "sents").unwrap_err().message.contains("not TokenTagger"));
    assert!(SentenceBoundary::load(&b, "words").is_err());
    assert!(SpanClassifier::load(&b, "sents").is_err());
}

#[test]
fn manifest_refuses_each_kind_with_another_kind_s_fields() {
    let refused = |name: &str, edit: fn(&mut Value), containing: &str| {
        let Err(e) = Bundle::open(&bundle(name, edit)) else { panic!("{name}: the bundle opened") };
        assert_eq!(e.code, BUNDLE_INVALID, "{name}: {e:?}");
        assert!(e.message.contains(containing), "{name}: {:?}", e.message);
    };
    refused("labels", |m| m["heads"][0]["labels"] = json!(["O", "last"]), "HEAD_SENTENCE_BOUNDARY's labels");
    refused("bio", |m| m["heads"][0]["scheme"] = json!("SCHEME_BIO"), "is not a label of Bio");
    refused("keep", |m| m["heads"][1]["labels"] = json!([DEL]), "needs the label \"_\"");
    refused("split", |m| m["heads"][1]["labels"] = json!(["_", "(m 0 1 () ())"]), "split");
    refused("no-span", |m| m["heads"][2].as_object_mut().unwrap().remove("span").map(drop).unwrap(), "heads[2].span");
    refused("span", |m| m["heads"][0]["span"] = json!({"max_words": 2, "width_dim": 2}), "heads[0].span");
    refused("wide", |m| m["heads"][2]["span"]["max_words"] = json!(500), "max_words 500");
    refused("hash", |m| m["heads"][2]["hash_embed"] = json!({"dim": 2, "rows": [4, 4, 4, 4]}), "hash_embed");
    refused("decode", |m| m["heads"][2]["decode"] = json!("DECODE_ARGMAX"), "HEAD_SPAN_CLASSIFIER takes");
}
