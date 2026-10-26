//! GLiNER through the backend table against the CPU encoder (#883): the
//! tiny DeBERTa-v2 bundle of testdata/tiny-deberta-bundle as the encoder,
//! with the heads of tests/fixtures and a small SentencePiece model, on the
//! device TURBO_TEST_DEVICE names (`cpu`, `levelzero`, `cuda`; the CPU when
//! it is not set) at the precision TURBO_TEST_PRECISION names (model, or
//! exact). Every logit within 1e-4 of the CPU encoder's and within 1e-3 of
//! the largest logit of its input, the decoded
//! entities the same. With TURBO_TEST_REQUIRE_CUDA or
//! TURBO_TEST_REQUIRE_LEVELZERO set, the test fails unless it runs there.

use std::path::{Path, PathBuf};

use turbo::deberta::{Config as EncoderConfig, Encoder};
use turbo_analysis::ModelRef;
use turbo_analysis_embed::{Device, EncoderOptions, Precision};
use turbo_gliner::{Config, Decode, ExtractionSchema, Model, decode};

/// The largest logit difference from the CPU encoder: the F32 device bound,
/// and at most this fraction of the largest logit of the same input.
const BOUND: f32 = 1e-4;
const RELATIVE: f32 = 1e-3;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn bundle() -> PathBuf {
    root().join("testdata/tiny-deberta-bundle")
}

/// The backend a TURBO_TEST_REQUIRE_* variable requires, if any.
fn required() -> Option<&'static str> {
    let set = |v: &str| std::env::var(v).is_ok_and(|v| !v.is_empty() && v != "0");
    [("TURBO_TEST_REQUIRE_CUDA", "cuda"), ("TURBO_TEST_REQUIRE_LEVELZERO", "levelzero")]
        .into_iter()
        .find(|(v, _)| set(v))
        .map(|(_, b)| b)
}

fn options() -> EncoderOptions {
    let device = match std::env::var("TURBO_TEST_DEVICE").as_deref() {
        Err(_) | Ok("cpu") => Device::Cpu,
        Ok(d) => Device::Backend(d.to_owned()),
    };
    let precision = match std::env::var("TURBO_TEST_PRECISION").as_deref() {
        Err(_) | Ok("model") => Precision::Model,
        Ok("exact") => Precision::Exact,
        Ok(p) => panic!("TURBO_TEST_PRECISION={p}: model or exact"),
    };
    EncoderOptions { device, precision, ..EncoderOptions::default() }
}

/// The heads fixture's configuration, over the tiny encoder of width 32,
/// its marker ids past the SentencePiece model's pieces and inside the
/// encoder's vocabulary.
fn config() -> Config {
    Config { hidden: 16, max_width: 3, entity_token: 30000, separator_token: 30001, cls_token: 101, eos_token: 102 }
}

fn heads() -> Vec<u8> {
    std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/heads.safetensors")).unwrap()
}

fn spm() -> Vec<u8> {
    std::fs::read(root().join("testdata/sentencepiece/unigram-identity.model")).unwrap()
}

fn identity() -> ModelRef {
    ModelRef { id: "turboembed/tiny-deberta".into(), revision: "1".into(), sha256: String::new() }
}

/// The CPU encoder on the bundle's own weights, its longest sequence the
/// bundle's 128.
fn cpu_model() -> Model {
    let ec = EncoderConfig {
        hidden: 32,
        intermediate: 64,
        heads: 4,
        layers: 2,
        vocabulary: 30522,
        max_sequence: 128,
        position_buckets: 8,
        max_relative_positions: 512,
        layer_norm_eps: 1e-7,
    };
    let bytes = std::fs::read(bundle().join("weights/model.safetensors")).unwrap();
    let encoder = Encoder::from_safetensors(ec, &bytes).unwrap().with_threads(1).unwrap();
    Model::new(encoder, config(), &heads(), &spm(), identity()).unwrap()
}

fn texts() -> Vec<Vec<&'static str>> {
    let long: Vec<&str> = "the quick brown fox jumps over the lazy dog while seven wizards quietly judge \
                           every boxing match on the long road to the city of light and back again"
        .split(' ')
        .cycle()
        .take(90)
        .collect();
    vec![
        vec!["Alice", "visited", "Paris", "in", "May", "."],
        vec!["Zoë"],
        "Apple and Microsoft announced a joint venture in Seattle on Tuesday".split(' ').collect(),
        long,
    ]
}

#[test]
fn the_backend_table_scores_as_the_cpu_encoder() {
    let o = options();
    if let Some(b) = required() {
        assert!(!matches!(o.device, Device::Cpu), "{b} is required but TURBO_TEST_DEVICE names no device");
    }
    let device = Model::open_on(&bundle(), &o, config(), &heads(), &spm(), identity()).unwrap();
    let execution = device.execution().expect("the backend table").clone();
    if let Some(b) = required() {
        assert_eq!(execution.backend, b, "the session runs on the backend required");
    }
    let cpu = cpu_model();
    let schemas = [
        ExtractionSchema::new(vec!["person".into(), "location".into(), "date".into()], 0.0, Decode::Flat).unwrap(),
        ExtractionSchema::new(vec!["organization".into(), "city".into()], 0.4, Decode::Nested).unwrap(),
    ];
    let (mut worst, mut largest, mut row_worst) = (0f32, 0f32, 0f32);
    for words in texts() {
        for schema in &schemas {
            // A long text cut to the most words that fit the encoder's
            // longest sequence, so a row runs near it.
            let n = (1..=words.len()).rev().find(|&n| cpu.pack(&words[..n], schema).is_ok()).unwrap();
            let words = &words[..n];
            let p = cpu.pack(words, schema).unwrap();
            assert_eq!(device.pack(words, schema).unwrap().ids, p.ids, "packed ids");
            // The rows the heads read, compared directly.
            let (x, y) = (cpu.states(&p, schema).unwrap(), device.states(&p, schema).unwrap());
            assert_eq!(x.len(), y.len());
            let d = x.iter().zip(&y).map(|(u, v)| (u - v).abs()).fold(0f32, f32::max);
            assert!(d < BOUND, "{words:?}: encoder rows differ by {d:e}");
            row_worst = row_worst.max(d);
            let (want, got) = (cpu.score_packed(&p, schema).unwrap(), device.score_packed(&p, schema).unwrap());
            assert_eq!(want.len(), got.len());
            // The heads fixture's logits are small, so each difference is
            // held to the set's scale too.
            let scale = want.iter().map(|c| c.logit.abs()).fold(0f32, f32::max);
            for (a, b) in want.iter().zip(&got) {
                assert_eq!((a.start, a.end, a.label), (b.start, b.end, b.label));
                let d = (a.logit - b.logit).abs();
                assert!(
                    d < BOUND && d <= RELATIVE * scale,
                    "{words:?}: span {}..{} label {}: {} vs {}",
                    a.start,
                    a.end,
                    a.label,
                    a.logit,
                    b.logit
                );
                worst = worst.max(d);
                largest = largest.max(a.logit.abs());
            }
            let (want, got) = (decode(want, schema), decode(got, schema));
            let spans = |c: &[turbo_gliner::Candidate]| c.iter().map(|c| (c.start, c.end, c.label)).collect::<Vec<_>>();
            assert_eq!(spans(&want), spans(&got), "{words:?}: decoded entities");
        }
    }
    println!(
        "gliner on {} {} ({}): encoder rows: largest difference {row_worst:e}; largest logit difference from the \
         cpu encoder {worst:e}, of logits up to {largest:e}",
        execution.backend, execution.device, execution.precision
    );
}

/// A bundle of another family is refused by name.
#[test]
fn a_bert_bundle_is_not_a_gliner_encoder() {
    let e = Model::open_on(
        &root().join("testdata/tiny-bert-bundle"),
        &EncoderOptions::default(),
        config(),
        &heads(),
        &spm(),
        identity(),
    )
    .err()
    .unwrap();
    assert!(e.contains("not FAMILY_DEBERTA_V2"), "{e}");
}
