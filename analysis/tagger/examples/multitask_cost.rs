//! The cost of one encoder pass with five heads against one head (#1517):
//! a copy of the bundle at `<bundle>` gets five heads of seeded weights in
//! the shapes a UD and entity bundle carries (UPOS, XPOS, BIO entities with
//! a CRF, a biaffine parser, an edit-tree lemmatizer reading the UPOS
//! tags), then `MultiTask` runs one head and all five over the same
//! sentences, alternating, `runs` times each. The weights do not change
//! the cost, only the shapes do.
//!
//! ```text
//! TURBO_TEST_DEVICE=cpu|cuda|levelzero|metal TURBO_TEST_PRECISION=model|exact \
//!   cargo run --release -p turbo-tagger [--features turbo/<backend>] --example multitask_cost -- \
//!   <bundle> <scratch dir> [runs, 20] [sentences, 512]
//! ```

use std::fs;
use std::path::Path;
use std::time::Instant;

use serde_json::{Value, json};
use turbo::bundle::{Bundle, sha256_hex};
use turbo::head::{lemmatizer, parser, tagger};
use turbo::manifest::{Format, HeadKind, HeadSpec};
use turbo_analysis::{Document, Ix, Key, Span, Spans, View, ViewId};
use turbo_analysis_embed::{Device, EncoderOptions, Precision};
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, PosTags, Sentences, Tokens};
use turbo_tagger::multi::Outputs;
use turbo_tagger::{MultiTask, WordEncoder};

const WORDS: [&str; 48] = [
    "the",
    "a",
    "dog",
    "river",
    "walked",
    "quietly",
    "over",
    "bridge",
    "Paris",
    "Ann",
    "met",
    "Bob",
    "in",
    "old",
    "house",
    "and",
    "because",
    "children",
    "were",
    "singing",
    "loudly",
    "near",
    "station",
    "yesterday",
    "company",
    "announced",
    "results",
    "for",
    "quarter",
    "unexpectedly",
    "strong",
    ",",
    ".",
    "government",
    "policy",
    "changes",
    "affected",
    "markets",
    "across",
    "Europe",
    "she",
    "said",
    "that",
    "it",
    "would",
    "rain",
    "tomorrow",
    "Microsoft",
];

fn values(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .flat_map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let v = ((s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * 0.1;
            v.to_le_bytes()
        })
        .collect()
}

fn labels(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|i| format!("{prefix}{i}")).collect()
}

/// The five heads, `artifact` holding their tensors.
fn heads(artifact: &str) -> Value {
    let mut ner = vec!["O".to_owned()];
    for t in ["PER", "LOC", "ORG", "MISC"] {
        ner.extend([format!("B-{t}"), format!("I-{t}")]);
    }
    // Substitutions with nothing in common, each canonical.
    let trees: Vec<String> = (0..200).map(|i| format!("(s 'q{i}' 'z')")).collect();
    let head = |name: &str, layer: &str, kind: &str, labels: Value, more: Value| {
        let mut h = json!({
            "name": name, "layer": layer, "kind": kind, "labels": labels, "scheme": "SCHEME_TAGS",
            "word_pooling": "WORD_POOLING_FIRST", "decode": "DECODE_ARGMAX", "artifact": artifact,
            "tensor_prefix": format!("heads.{name}"),
        });
        h.as_object_mut().unwrap().extend(more.as_object().unwrap().clone());
        h
    };
    json!([
        head("upos", "turboembed:pos", "HEAD_TOKEN_TAGGER", json!(labels("U", 17)), json!({})),
        head("xpos", "turboembed:xpos", "HEAD_TOKEN_TAGGER", json!(labels("X", 45)), json!({})),
        head(
            "ner",
            "turboembed:entities",
            "HEAD_TOKEN_TAGGER",
            json!(ner),
            json!({"scheme": "SCHEME_BIO", "decode": "DECODE_CRF"})
        ),
        head(
            "deps",
            "turboembed:dependencies",
            "HEAD_BIAFFINE_PARSER",
            json!(labels("r", 37)),
            json!({"decode": "DECODE_MST", "biaffine": {"arc": 500, "rel": 100}})
        ),
        head(
            "lemma",
            "turboembed:lemmas",
            "HEAD_EDIT_TREE_LEMMATIZER",
            json!(trees),
            json!({"decode": "DECODE_EDIT_TREE", "pos_input": {"head": "upos", "labels": 17, "dim": 32}})
        ),
    ])
}

/// A copy of the bundle at `src` in `dst` with the five heads appended to
/// its safetensors artifact.
fn with_heads(src: &Path, dst: &Path) {
    let hidden = WordEncoder::open(src, &EncoderOptions::default()).unwrap().hidden();
    let bundle = Bundle::open(src).unwrap();
    let art =
        bundle.manifest.artifacts.iter().find(|a| a.format == Format::Safetensors).expect("a safetensors artifact");
    let (name, file) = (art.name.clone(), art.files[0].clone());
    let mut m: Value = serde_json::from_slice(&fs::read(src.join("manifest.json")).unwrap()).unwrap();
    m["heads"] = heads(&name);
    let _ = fs::remove_dir_all(dst);
    for f in m["files"].as_array().unwrap() {
        let p = f["path"].as_str().unwrap();
        fs::create_dir_all(dst.join(p).parent().unwrap()).unwrap();
        if p != file && fs::hard_link(src.join(p), dst.join(p)).is_err() {
            fs::copy(src.join(p), dst.join(p)).unwrap();
        }
    }
    let raw = fs::read(src.join(&file)).unwrap();
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&raw[8..8 + n]).unwrap();
    let meta = header.remove("__metadata__");
    let mut data = raw[8 + n..].to_vec();
    for (seed, h) in m["heads"].as_array().unwrap().iter().enumerate() {
        let spec: HeadSpec = serde_json::from_value(h.clone()).unwrap();
        let shapes = match spec.kind {
            HeadKind::BiaffineParser => parser::tensors(&spec, hidden),
            HeadKind::EditTreeLemmatizer => lemmatizer::tensors(&spec, hidden),
            HeadKind::TokenTagger => tagger::tensors(&spec, hidden),
            HeadKind::SentenceBoundary => turbo::head::boundary::tensors(&spec, hidden),
            HeadKind::TokenSplit => turbo::head::split::tensors(&spec, hidden),
            HeadKind::SpanClassifier => turbo::head::span::tensors(&spec, hidden),
            HeadKind::RelationClassifier => turbo::head::relation::tensors(&spec, hidden),
        };
        for (k, (suffix, shape)) in shapes.into_iter().enumerate() {
            let start = data.len();
            data.extend(values((seed * 64 + k) as u64 + 1, shape.iter().product::<u64>() as usize));
            let key = format!("{}.{suffix}", spec.tensor_prefix);
            header.insert(key, json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
        }
    }
    if let Some(meta) = meta {
        header.insert("__metadata__".into(), meta);
    }
    let mut head = serde_json::to_vec(&header).unwrap();
    head.resize(head.len().next_multiple_of(8), b' ');
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend(head);
    out.extend(data);
    fs::write(dst.join(&file), &out).unwrap();
    let entry = m["files"].as_array_mut().unwrap().iter_mut().find(|f| f["path"] == file.as_str()).unwrap();
    entry["size"] = json!(out.len());
    entry["sha256"] = json!(sha256_hex(&out));
    fs::write(dst.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
}

/// `count` sentences of 8 to 40 words, seeded, and their tokens.
fn document(count: usize) -> (String, Tokens) {
    let (mut text, mut spans, mut of) = (String::new(), Spans::new(ViewId::SOURCE), Vec::new());
    let mut s = 0x5eed_u64;
    for i in 0..count {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let len = 8 + (s >> 33) as usize % 33;
        for w in 0..len {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let word = if w + 1 == len { "." } else { WORDS[(s >> 33) as usize % WORDS.len()] };
            if !text.is_empty() {
                text.push(' ');
            }
            let start = text.len() as u32;
            text.push_str(word);
            spans.push(Span::new(start, text.len() as u32));
            of.push(Ix::<Sentences>::new(i as u32));
        }
    }
    (text, Tokens { spans, sentence: Some(of) })
}

fn summary(label: &str, ms: &mut [f64]) -> f64 {
    ms.sort_by(f64::total_cmp);
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    let at = |q: f64| ms[((ms.len() - 1) as f64 * q).round() as usize];
    println!(
        "{label}: runs {} min {:.2} ms mean {mean:.2} ms p50 {:.2} ms p99 {:.2} ms",
        ms.len(),
        ms[0],
        at(0.5),
        at(0.99)
    );
    mean
}

/// The mean of `runs` passes of the heads `names` alone.
fn alone<W: Outputs>(
    dir: &Path,
    names: &[Option<&str>],
    o: &EncoderOptions,
    view: &View,
    t: &Tokens,
    runs: usize,
) -> f64 {
    let m = MultiTask::<W>::open(dir, names, o).unwrap();
    m.layers(view, t, None, None).unwrap();
    let start = Instant::now();
    for _ in 0..runs {
        std::hint::black_box(m.layers(view, t, None, None).unwrap());
    }
    start.elapsed().as_secs_f64() * 1e3 / runs as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (src, scratch) = (Path::new(&args[1]), Path::new(&args[2]));
    let runs: usize = args.get(3).map_or(20, |a| a.parse().unwrap());
    let sentences: usize = args.get(4).map_or(512, |a| a.parse().unwrap());
    let device = match std::env::var("TURBO_TEST_DEVICE").as_deref() {
        Err(_) | Ok("cpu") => Device::Cpu,
        Ok(d) => Device::Backend(d.to_owned()),
    };
    let precision = match std::env::var("TURBO_TEST_PRECISION").as_deref() {
        Err(_) | Ok("model") => Precision::Model,
        Ok("exact") => Precision::Exact,
        Ok(p) => panic!("TURBO_TEST_PRECISION={p}: model or exact"),
    };
    let o = EncoderOptions { device, precision, ..EncoderOptions::default() };
    let dir = scratch.join("multitask-cost-bundle");
    with_heads(src, &dir);
    let t = Instant::now();
    let one = MultiTask::<(PosTags,)>::open(&dir, &[Some("upos")], &o).unwrap();
    let load_one = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    type Five = (PosTags, PosTags, Entities, Dependencies<Tokens>, Lemmas);
    let names = [Some("upos"), Some("xpos"), Some("ner"), Some("deps"), Some("lemma")];
    let five = MultiTask::<Five>::open(&dir, &names, &o).unwrap();
    let load_five = t.elapsed().as_secs_f64() * 1e3;
    let (text, tokens) = document(sentences);
    let doc = Document::new(text.as_str());
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    // Warm both, then alternate so drift falls on both alike.
    one.layers(view, &tokens, None, None).unwrap();
    five.layers(view, &tokens, None, None).unwrap();
    let (mut a, mut b) = (Vec::with_capacity(runs), Vec::with_capacity(runs));
    for r in 0..runs {
        let order: [bool; 2] = if r % 2 == 0 { [false, true] } else { [true, false] };
        for is_five in order {
            let t = Instant::now();
            if is_five {
                std::hint::black_box(five.layers(view, &tokens, None, None).unwrap());
                b.push(t.elapsed().as_secs_f64() * 1e3);
            } else {
                std::hint::black_box(one.layers(view, &tokens, None, None).unwrap());
                a.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    let e = turbo_analysis::Annotator::execution(&one).expect("an encoder runs somewhere");
    println!("bundle {} heads appended at {}", src.display(), dir.display());
    println!("execution backend {} device {} precision {:?}", e.backend, e.device, e.precision);
    println!("sentences {sentences} words {} rows a pass {}", tokens.spans.len(), five.rows() / (runs as u64 + 1));
    println!("load: one head {load_one:.1} ms, five heads {load_five:.1} ms");
    let (ma, mb) = (summary("one head (upos)", &mut a), summary("five heads", &mut b));
    let each = (runs / 4).max(1);
    println!(
        "each head alone, mean of {each}: upos {:.2} ms, xpos {:.2} ms, ner {:.2} ms, deps {:.2} ms, lemma {:.2} ms",
        alone::<(PosTags,)>(&dir, &[Some("upos")], &o, view, &tokens, each),
        alone::<(PosTags,)>(&dir, &[Some("xpos")], &o, view, &tokens, each),
        alone::<(Entities,)>(&dir, &[Some("ner")], &o, view, &tokens, each),
        alone::<(Dependencies<Tokens>,)>(&dir, &[Some("deps")], &o, view, &tokens, each),
        alone::<(Lemmas,)>(&dir, &[Some("lemma")], &o, view, &tokens, each),
    );
    let ratio = mb / ma;
    println!("five over one, by mean: {ratio:.3} (bar 1.3): {}", if ratio < 1.3 { "PASS" } else { "FAIL" });
}
