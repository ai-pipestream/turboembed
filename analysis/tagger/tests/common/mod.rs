//! The small BERT in testdata with a part-of-speech head, an entity head,
//! a dependency parser head and a lemmatizer head reading the
//! part-of-speech tags, all of seeded weights, for the tests of this crate.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use turbo::bundle::sha256_hex;
use turbo::head::{lemmatizer, parser, span, tagger};
use turbo::manifest::{HeadKind, HeadSpec};
use turbo_analysis::{Span, Spans, ViewId};
use turbo_analysis_layers::Tokens;

pub const TEXT: &str = "Ann met Bob in Paris , and the quiet old dog slept by the river .";

pub fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

/// The lemmatizer head's trees: keep the word, drop a final s, drop a
/// final ed, and met to meet.
pub const LEMMA_TREES: [&str; 4] =
    ["(m 0 0 () ())", "(m 0 1 () (s 's' ''))", "(m 0 2 () (s 'ed' ''))", "(s 'met' 'meet')"];

pub fn heads() -> Value {
    json!([
        {
            "name": "pos", "layer": "turboembed:pos", "kind": "HEAD_TOKEN_TAGGER",
            "labels": ["NOUN", "VERB", "DET", "ADJ", "PUNCT"], "scheme": "SCHEME_TAGS",
            "word_pooling": "WORD_POOLING_MEAN", "decode": "DECODE_ARGMAX", "artifact": "weights-f32", "tensor_prefix": "heads.pos"
        },
        {
            "name": "ner", "layer": "turboembed:entities", "kind": "HEAD_TOKEN_TAGGER",
            "labels": ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"], "scheme": "SCHEME_BIO",
            "word_pooling": "WORD_POOLING_FIRST", "adapter": {"bilstm": {"hidden": 4, "layers": 1}},
            "decode": "DECODE_CRF", "artifact": "weights-f32", "tensor_prefix": "heads.ner"
        },
        {
            "name": "deps", "layer": "turboembed:dependencies", "kind": "HEAD_BIAFFINE_PARSER",
            "labels": ["root", "nsubj", "obj", "det", "amod", "punct"], "scheme": "SCHEME_TAGS",
            "word_pooling": "WORD_POOLING_FIRST", "decode": "DECODE_MST", "biaffine": {"arc": 6, "rel": 4},
            "artifact": "weights-f32", "tensor_prefix": "heads.deps"
        },
        {
            "name": "lemma", "layer": "turboembed:lemmas", "kind": "HEAD_EDIT_TREE_LEMMATIZER",
            "labels": LEMMA_TREES, "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
            "decode": "DECODE_EDIT_TREE", "pos_input": {"head": "pos", "labels": 5, "dim": 3},
            "artifact": "weights-f32", "tensor_prefix": "heads.lemma"
        }
    ])
}

/// Deterministic values in [-scale, scale).
pub fn values(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            (((s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// The tiny bundle with `edit` applied to its manifest and the tensors
/// of every head the edited manifest names, seeded.
pub fn bundle(name: &str, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-tagger-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(dir.join(f).parent().unwrap()).unwrap();
        fs::copy(tiny().join(f), dir.join(f)).unwrap();
    }
    let src = fs::read(tiny().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    header.remove("__metadata__");
    let mut data = src[8 + n..].to_vec();
    let mut m: Value = serde_json::from_slice(&fs::read(tiny().join("manifest.json")).unwrap()).unwrap();
    m["heads"] = heads();
    edit(&mut m);
    let mut seed = 1;
    for h in m["heads"].as_array().unwrap() {
        let spec: HeadSpec = serde_json::from_value(h.clone()).unwrap();
        let shapes = match spec.kind {
            HeadKind::BiaffineParser => parser::tensors(&spec, 32),
            HeadKind::EditTreeLemmatizer => lemmatizer::tensors(&spec, 32),
            HeadKind::SpanClassifier => span::tensors(&spec, 32),
            _ => tagger::tensors(&spec, 32),
        };
        for (suffix, shape) in shapes {
            let count = shape.iter().product::<u64>() as usize;
            let start = data.len();
            data.extend(values(seed, count, 0.8).iter().flat_map(|v| v.to_le_bytes()));
            seed += 1;
            let name = format!("{}.{suffix}", spec.tensor_prefix);
            if header.contains_key(&name) {
                // A head sharing another's tensors.
                data.truncate(start);
                continue;
            }
            header.insert(name, json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
        }
    }
    let mut head = serde_json::to_vec(&header).unwrap();
    head.resize(head.len().next_multiple_of(8), b' ');
    let mut weights = (head.len() as u64).to_le_bytes().to_vec();
    weights.extend(head);
    weights.extend(data);
    fs::create_dir_all(dir.join("weights")).unwrap();
    fs::write(dir.join("weights/model.safetensors"), &weights).unwrap();
    let files = m["files"].as_array_mut().unwrap();
    let w = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(sha256_hex(&weights));
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

pub fn tokens(text: &str) -> Tokens {
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut start = 0;
    for word in text.split(' ') {
        spans.push(Span::new(start as u32, (start + word.len()) as u32));
        start += word.len() + 1;
    }
    Tokens { spans, sentence: None }
}
