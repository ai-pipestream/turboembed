//! Parity between `turbo-analysis-embed` (`Encoder`, `SentenceVectors`,
//! `ChunkVectors`, `SentenceChunkVectors`) and OpenNLP's
//! `opennlp.dl.vectors.SentenceVectorsDL` on the shared 2-layer BERT model
//! (`testdata/tiny-bert-bundle`) and vocabulary
//! (`analysis/goldens/data/fixtures/all-minilm-l6-v2-vocab.txt`).
//!
//! `tests/data/sentence_vectors_dl_golden.json` holds what
//! `reference/regenerate.sh` (`reference/export_onnx.py` in `turbo-reference`
//! followed by `reference/SentenceVectorsDlGolden.java` in
//! `turbo-goldens-java:opennlp-cdf33c653d5d-addons-33e95a4956f1`) wrote for:
//! 1. the 12 sentences of `tests/vectors.rs`, and
//! 2. all 76 golden inputs of the `edge`, `en`, `multi`, `subwords`, and
//!    `surrogate` sets under `analysis/goldens/data/inputs`.
//!
//! Each input is evaluated on both the raw BERT ONNX graph (`cls`, where
//! `SentenceVectorsDL.getVectors` returns `last_hidden_state[0][0]`) and the
//! mean-pooled L2-normalized ONNX graph (`mean_l2`, where
//! `SentenceVectorsDL.getVectors` returns the unit mean over
//! `[CLS] + pieces + [SEP]`).

#[path = "vectors/common.rs"]
mod common;

use serde_json::Value;
use turbo::bundle::sha256_hex;
use turbo::turbo_token_range;
use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::Subwords;
use turbo_goldens::{Golden, data_dir};

use common::*;

/// The four golden inputs where upstream `tokenizers` and OpenNLP's
/// `WordpieceEncoder` produce different piece IDs (`tests/goldens.rs`).
const DIFFERENT_PIECE_IDS: [&str; 4] =
    ["subwords/accents-reordered", "subwords/cjk-extension", "subwords/final-sigma", "subwords/special-tokens"];

struct JvmFixture {
    opennlp_commit: String,
    vocab_file: String,
    vocab_sha256: String,
    weights_file: String,
    weights_sha256: String,
    mean_l2_onnx_sha256: String,
    cls_onnx_sha256: String,
    sentences: Vec<SentenceEntry>,
    golden_inputs: Vec<GoldenEntry>,
}

struct SentenceEntry {
    text: String,
    mean_l2: Vec<f32>,
    cls: Vec<f32>,
}

struct GoldenEntry {
    name: String,
    mean_l2: Vec<f32>,
    cls: Vec<f32>,
}

fn f32s(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn load_fixture() -> JvmFixture {
    let raw: Value = serde_json::from_str(include_str!("data/sentence_vectors_dl_golden.json")).unwrap();
    let sentences = raw["sentences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| SentenceEntry {
            text: s["text"].as_str().unwrap().to_owned(),
            mean_l2: f32s(&s["mean_l2"]),
            cls: f32s(&s["cls"]),
        })
        .collect();
    let golden_inputs = raw["golden_inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| GoldenEntry {
            name: g["name"].as_str().unwrap().to_owned(),
            mean_l2: f32s(&g["mean_l2"]),
            cls: f32s(&g["cls"]),
        })
        .collect();
    JvmFixture {
        opennlp_commit: raw["opennlp_commit"].as_str().unwrap().to_owned(),
        vocab_file: raw["vocab_file"].as_str().unwrap().to_owned(),
        vocab_sha256: raw["vocab_sha256"].as_str().unwrap().to_owned(),
        weights_file: raw["weights_file"].as_str().unwrap().to_owned(),
        weights_sha256: raw["weights_sha256"].as_str().unwrap().to_owned(),
        mean_l2_onnx_sha256: raw["mean_l2_onnx_sha256"].as_str().unwrap().to_owned(),
        cls_onnx_sha256: raw["cls_onnx_sha256"].as_str().unwrap().to_owned(),
        sentences,
        golden_inputs,
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let norm = |v: &[f32]| v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Runs `SubwordAnnotator` and `SentenceChunkVectors` on a single-sentence,
/// single-chunk document holding `text`.
fn annotate_single(a: &std::sync::Arc<VectorAnnotator>, text: &str) -> Document {
    let n = text.len() as u32;
    let mut docs = [document(text, &[(0, n)], &[(0, 1, 0, None)])];
    run(SentenceChunkVectors(a.clone()), &mut docs);
    let [d] = docs;
    d
}

#[test]
fn fixture_hashes_match_the_bundle_and_the_vocabulary() {
    let f = load_fixture();
    assert_eq!(f.opennlp_commit, "cdf33c653d5d61d5af7d37ffee189a83bd8392ea");
    let vocab = std::fs::read(data_dir().join("fixtures").join(&f.vocab_file)).unwrap();
    assert_eq!(sha256_hex(&vocab), f.vocab_sha256);
    let weights = std::fs::read(bert().join("weights/model.safetensors")).unwrap();
    assert_eq!(f.weights_file, "testdata/tiny-bert-bundle/weights/model.safetensors");
    assert_eq!(sha256_hex(&weights), f.weights_sha256);
    assert_eq!(
        f.mean_l2_onnx_sha256,
        "57ec273eae030d2c810d2e1df214684f49d9c0b81da08cc52de952aa985fed49"
    );
    assert_eq!(
        f.cls_onnx_sha256,
        "7dfd15ea3e880ed3a6808df03b442b07a1d84cd55afeadd758a6108338176f0d"
    );
    assert_eq!(f.sentences.len(), SENTENCES.len());
    assert_eq!(f.golden_inputs.len(), 76);
}

/// On the 12 test sentences and all 56 golden inputs that fit the 64-token
/// window with matching piece IDs, `Encoder::run_cls` and `Encoder::run_rows`
/// match OpenNLP's `SentenceVectorsDL` on the raw BERT ONNX graph (`cls`) and
/// the mean-pooled L2-normalized ONNX graph (`mean_l2`) within float32
/// tolerance (`1e-5`).
#[test]
fn cls_and_whole_row_mean_match_opennlp_sentence_vectors_dl() {
    let f = load_fixture();
    let a = open(0);
    let (head, tail) = a.row_ends();
    let an = Analyzer::builder().add(SubwordAnnotator::open(&bert()).unwrap()).build().unwrap();
    let mut enc = Encoder::open(&bert(), &EncoderOptions::default()).unwrap();
    let mut rows = Rows::new(enc.info().max_seq, 0);
    let (mut got_cls, mut got_mean) = (Vec::new(), Vec::new());
    let (mut worst_cls, mut worst_mean) = (0.0f32, 0.0f32);
    let (mut checked, mut over_window) = (0usize, 0usize);

    let mut check_one = |label: &str, text: &str, want_cls: &[f32], want_mean: &[f32]| {
        let mut docs = [Document::new(text)];
        an.run(&mut docs).remove(0).unwrap();
        let pieces = &docs[0].get::<Subwords>().unwrap().ids;
        if pieces.len() + head.len() + tail.len() > enc.info().max_seq as usize {
            over_window += 1;
            return;
        }
        rows.clear();
        rows.push(&[head, pieces, tail]);
        enc.run_cls(&rows, &mut got_cls).unwrap();
        enc.run_rows(&rows, true, &mut got_mean).unwrap();
        let dc = max_abs_diff(&got_cls, want_cls);
        let dm = max_abs_diff(&got_mean, want_mean);
        assert!(dc <= 1e-5, "{label} cls max_abs_diff {dc:e}");
        assert!(dm <= 1e-5, "{label} mean_l2 max_abs_diff {dm:e}");
        assert!(cosine(&got_cls, want_cls) >= 0.999_999, "{label} cls cosine");
        assert!(cosine(&got_mean, want_mean) >= 0.999_999, "{label} mean_l2 cosine");
        worst_cls = worst_cls.max(dc);
        worst_mean = worst_mean.max(dm);
        checked += 1;
    };

    for (s, want) in SENTENCES.iter().zip(&f.sentences) {
        assert_eq!(want.text, *s);
        check_one(s, s, &want.cls, &want.mean_l2);
    }
    for g in &f.golden_inputs {
        if DIFFERENT_PIECE_IDS.contains(&g.name.as_str()) {
            continue;
        }
        let path = data_dir().join("subwords/all-minilm-l6-v2-vocab").join(format!("{}.golden", g.name));
        let golden = Golden::read(&path).unwrap();
        let text = golden.input_text(data_dir()).unwrap();
        check_one(&g.name, &text, &g.cls, &g.mean_l2);
    }
    assert_eq!(checked, 68);
    assert_eq!(checked + over_window + DIFFERENT_PIECE_IDS.len(), SENTENCES.len() + 76);
    println!("{checked} inputs matched SentenceVectorsDL: worst cls diff {worst_cls:e}, mean_l2 diff {worst_mean:e}");
}

/// Compares `SentenceVectors`, `ChunkVectors`, and `SentenceChunkVectors`
/// against OpenNLP's `SentenceVectorsDL` on the shared mean-pooled ONNX model
/// across the 12 test sentences and all 76 golden inputs:
///
/// 1. For every single-window non-empty input with matching piece IDs, the
///    unnormalized `SentenceVectors` range mean over the `N` content pieces,
///    combined with the `[CLS]` and `[SEP]` token states as
///    `(N * range_mean + cls + sep) / (N + 2)` and L2-normalized, equals
///    `SentenceVectorsDL`'s whole-row mean vector within `1e-5`.
/// 2. Content-only late pooling (`SentenceVectors` / `ChunkVectors`) agrees
///    with `SentenceVectorsDL`'s whole-row pooling (`[CLS] + pieces + [SEP]`)
///    within the stated cosine floor (`>= 0.96` on the 12 sentences alone,
///    `>= 0.85` when the 12 sentences share multi-sentence windows, `>= 0.90`
///    on golden inputs of 4 or more content pieces, and `>= 0.70` on 1-3
///    piece golden inputs where `[CLS]` and `[SEP]` are up to two thirds of
///    the row).
/// 3. Inputs with zero content pieces (`edge/empty`, `edge/whitespace-only`)
///    receive the zero vector and are listed in `Vectors::empty`, whereas
///    `SentenceVectorsDL` embeds `[CLS] [SEP]`.
#[test]
fn sentence_and_chunk_vectors_against_opennlp_sentence_vectors_dl_on_golden_inputs() {
    let f = load_fixture();
    let a = open(0);
    let (head, tail) = a.row_ends();
    let mut enc = Encoder::open(&bert(), &EncoderOptions::default()).unwrap();
    let mut rows = Rows::new(enc.info().max_seq, 0);
    let mut states = Vec::new();

    // 1. Twelve sentences alone vs in a shared multi-sentence document.
    let mut lowest_alone = 1.0f64;
    for (s, want) in SENTENCES.iter().zip(&f.sentences) {
        let d = annotate_single(&a, s);
        let sv = sentence_vectors(&d).row(Ix::new(0));
        let cv = chunk_vectors(&d).row(Ix::new(0));
        assert_eq!(sv, cv, "single sentence and single chunk over it have identical bits");
        let c = cosine(sv, &want.mean_l2);
        assert!(c >= 0.96, "{s}: cosine {c} under 0.96");
        lowest_alone = lowest_alone.min(c);
    }
    let (joined, spans) = text_of(12);
    let mut shared = [document(&joined, &spans, &[(0, 12, 0, None)])];
    run(SentenceChunkVectors(a.clone()), &mut shared);
    let mut lowest_shared = 1.0f64;
    for (i, want) in f.sentences.iter().enumerate() {
        let sv = sentence_vectors(&shared[0]).row(Ix::new(i as u32));
        let c = cosine(sv, &want.mean_l2);
        assert!(c >= 0.85, "sentence {i} in shared window: cosine {c} under 0.85");
        lowest_shared = lowest_shared.min(c);
    }

    // 2. All 76 golden inputs.
    let mut empty_names = Vec::new();
    let mut long_names = Vec::new();
    let (mut lowest_ge4, mut lowest_short) = (1.0f64, 1.0f64);
    for g in &f.golden_inputs {
        let path = data_dir().join("subwords/all-minilm-l6-v2-vocab").join(format!("{}.golden", g.name));
        let golden = Golden::read(&path).unwrap();
        let text = golden.input_text(data_dir()).unwrap();
        let d = annotate_single(&a, &text);
        let vs = sentence_vectors(&d);
        let vc = chunk_vectors(&d);
        let pieces = &d.get::<Subwords>().unwrap().ids;
        if pieces.is_empty() {
            empty_names.push(g.name.as_str());
            assert_eq!(vs.empty, [Ix::new(0)], "{}", g.name);
            assert_eq!(vc.empty, [Ix::new(0)], "{}", g.name);
            assert!(vs.row(Ix::new(0)).iter().all(|&x| x == 0.0));
            assert!(is_unit(&g.mean_l2), "{}: SentenceVectorsDL embeds [CLS] [SEP]", g.name);
            continue;
        }
        assert!(vs.empty.is_empty() && vc.empty.is_empty(), "{}", g.name);
        let sv = vs.row(Ix::new(0));
        assert_eq!(sv, vc.row(Ix::new(0)), "{}", g.name);
        assert!(is_unit(sv), "{}", g.name);

        if DIFFERENT_PIECE_IDS.contains(&g.name.as_str()) {
            continue;
        }
        let n = pieces.len();
        if n + head.len() + tail.len() <= enc.info().max_seq as usize {
            rows.clear();
            rows.push(&[head, pieces, tail]);
            let ranges = [
                turbo_token_range { row: 0, start: 0, end: 1 },
                turbo_token_range { row: 0, start: 1, end: 1 + n as u32 },
                turbo_token_range { row: 0, start: 1 + n as u32, end: 2 + n as u32 },
            ];
            enc.run_ranges(&rows, &ranges, &mut states).unwrap();
            let (cls_s, mid_s, sep_s) = (&states[0..32], &states[32..64], &states[64..96]);
            let mut mid_unit = mid_s.to_vec();
            normalize(&mut mid_unit);
            assert!(max_abs_diff(sv, &mid_unit) <= 1e-5, "{}: SentenceVectors is the content range", g.name);
            let mut whole: Vec<f32> =
                (0..32).map(|k| (cls_s[k] + mid_s[k] * n as f32 + sep_s[k]) / (n as f32 + 2.0)).collect();
            normalize(&mut whole);
            assert!(
                max_abs_diff(&whole, &g.mean_l2) <= 1e-5,
                "{}: content range + [CLS] + [SEP] equals SentenceVectorsDL",
                g.name
            );
        } else {
            long_names.push(g.name.as_str());
        }
        let c = cosine(sv, &g.mean_l2);
        if n >= 4 {
            assert!(c >= 0.90, "{} ({n} pieces): cosine {c} under 0.90", g.name);
            lowest_ge4 = lowest_ge4.min(c);
        } else {
            assert!(c >= 0.70, "{} ({n} pieces): cosine {c} under 0.70", g.name);
            lowest_short = lowest_short.min(c);
        }
    }
    assert_eq!(empty_names, ["edge/empty", "edge/whitespace-only"]);
    assert!(!long_names.is_empty(), "at least one golden input spans multiple windows");
    println!(
        "SentenceVectors vs SentenceVectorsDL: alone >= {lowest_alone:.4}, shared >= {lowest_shared:.4}, \
         golden (>=4 pieces) >= {lowest_ge4:.4}, golden (1..3 pieces) >= {lowest_short:.4}"
    );
}
