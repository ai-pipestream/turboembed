//! GLiNER through the backend table against the CPU encoder (#883): the
//! tiny DeBERTa-v2 bundle of testdata/tiny-deberta-bundle as the encoder,
//! with the heads of tests/fixtures and a small SentencePiece model, on the
//! device TURBO_TEST_DEVICE names (`cpu`, `levelzero`, `cuda`, `metal`; the CPU when
//! it is not set) at the precision TURBO_TEST_PRECISION names (model, or
//! exact). Every logit within 1e-4 of the CPU encoder's and within 1e-3 of
//! the largest logit of its input, the decoded
//! entities the same. With TURBO_TEST_REQUIRE_CUDA,
//! TURBO_TEST_REQUIRE_LEVELZERO or TURBO_TEST_REQUIRE_METAL set, the test fails unless it runs there.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo::deberta::{
    Config as EncoderConfig, Encoder,
    package::{PackageManifest, digest},
};
use turbo_analysis::{Annotator, Document, Key, ModelRef, Span, Spans, View, ViewId};
use turbo_analysis_embed::{Device, EncoderOptions, Precision};
use turbo_analysis_layers::Tokens;
use turbo_gliner::{Config, Decode, ExtractionSchema, GlinerAnnotator, Model, decode, package::Manifest};
use turbo_scratch::Scratch;

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
    turbo_pins::device_required(&["cuda", "levelzero", "metal"])
}

fn options() -> EncoderOptions {
    let named = std::env::var("TURBO_TEST_DEVICE").ok();
    let req = required();
    turbo_pins::check_test_device(named.as_deref(), req).unwrap_or_else(|e| panic!("{e}"));
    let device = match named.as_deref() {
        None | Some("cpu") => Device::Cpu,
        Some(d) => Device::Backend(d.to_owned()),
    };
    let precision = match std::env::var("TURBO_TEST_PRECISION").as_deref() {
        Err(_) | Ok("model") => Precision::Model,
        Ok("exact") => Precision::Exact,
        Ok(p) => panic!("TURBO_TEST_PRECISION={p}: model or exact"),
    };
    EncoderOptions { device, precision, ..EncoderOptions::default() }
}

fn expected_lstm_device(backend: &str) -> turbo::device::Device {
    match backend {
        "cuda" => turbo::device::Device::Cuda(0),
        "levelzero" => turbo::device::Device::LevelZero(0),
        "metal" => turbo::device::Device::Metal(0),
        _ => turbo::device::Device::Cpu,
    }
}

/// The heads fixture's configuration, over the tiny encoder of width 32,
/// its marker ids past the SentencePiece model's pieces and inside the
/// encoder's vocabulary.
fn config() -> Config {
    Config {
        hidden: 16,
        projection_factor: 4,
        prompt_factor: 4,
        out_project_mlp: true,
        token_level: false,
        prompt_projection: true,
        max_width: 3,
        entity_token: 30000,
        separator_token: 30001,
        cls_token: 101,
        eos_token: 102,
    }
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
    assert_eq!(cpu.lstm_device(), turbo::device::Device::Cpu);
    assert_eq!(device.lstm_device(), expected_lstm_device(&execution.backend));
    assert_eq!(device.lstm_precision(), turbo::device::Precision::Model);
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

/// Builds a checked GLiNER package directory from in-tree fixtures matching
/// `testdata/tiny-deberta-bundle` and verifies `Manifest::open_on` runs both
/// the DeBERTa-v2 encoder and the recurrent BiLSTM on the session's device.
#[test]
fn manifest_open_on_runs_encoder_and_recurrent_bilstm_on_device() {
    let o = options();
    let pkg = Scratch::at(std::env::temp_dir().join(format!("turbo-gliner-pkg-{}", std::process::id())));
    std::fs::create_dir_all(pkg.join("encoder")).unwrap();
    let enc_weights = std::fs::read(bundle().join("weights/model.safetensors")).unwrap();
    let heads_bytes = heads();
    let spm_bytes = spm();
    let enc_manifest = PackageManifest {
        version: 1,
        config: EncoderConfig {
            hidden: 32,
            intermediate: 64,
            heads: 4,
            layers: 2,
            vocabulary: 30522,
            max_sequence: 128,
            position_buckets: 8,
            max_relative_positions: 512,
            layer_norm_eps: 1e-7,
        },
        source_repository: "turboembed/tiny-deberta".into(),
        source_revision: "0000000000000000000000000000000000000001".into(),
        source_license: "Apache-2.0".into(),
        weights_sha256: digest(&enc_weights),
    };
    let enc_meta = serde_json::to_vec_pretty(&enc_manifest).unwrap();
    std::fs::write(pkg.join("encoder/deberta.json"), &enc_meta).unwrap();
    std::fs::write(pkg.join("encoder/weights.safetensors"), &enc_weights).unwrap();
    std::fs::write(pkg.join("heads.safetensors"), &heads_bytes).unwrap();
    std::fs::write(pkg.join("spm.model"), &spm_bytes).unwrap();
    let manifest = Manifest {
        version: 1,
        config: config(),
        encoder_sha256: digest(&enc_meta),
        heads_sha256: digest(&heads_bytes),
        tokenizer_sha256: digest(&spm_bytes),
    };
    std::fs::write(pkg.join("gliner.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    let cpu = Manifest::load(&pkg).unwrap();
    let dev = Manifest::open_on(&pkg, &bundle(), &o).unwrap();
    let exec = dev.execution().expect("backend table execution").clone();
    if let Some(b) = required() {
        assert_eq!(exec.backend, b);
    }
    assert_eq!(cpu.lstm_device(), turbo::device::Device::Cpu);
    assert_eq!(dev.lstm_device(), expected_lstm_device(&exec.backend));

    let schema =
        ExtractionSchema::new(vec!["person".into(), "location".into(), "date".into()], 0.0, Decode::Flat).unwrap();
    let sample = "Alice visited Paris in May .";
    let doc = Document::new(sample);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    let mut start = 0u32;
    for w in sample.split(' ') {
        tok_spans.push(Span::new(start, start + w.len() as u32));
        start += w.len() as u32 + 1;
    }
    let toks = Tokens { spans: tok_spans, sentence: None };

    let ann_cpu = GlinerAnnotator::new(Arc::new(cpu), schema.clone());
    let ann_dev = GlinerAnnotator::new(Arc::new(dev), schema);
    let (ent_c, dec_c) = ann_cpu.annotate((view, &toks, None)).unwrap();
    let (ent_d, dec_d) = ann_dev.annotate((view, &toks, None)).unwrap();
    assert_eq!((ent_c.tokens, ent_c.kind), (ent_d.tokens, ent_d.kind));
    assert_eq!(dec_c.logits.len(), dec_d.logits.len());
    for (lc, ld) in dec_c.logits.iter().zip(&dec_d.logits) {
        assert!((lc - ld).abs() < BOUND, "logit {lc} vs {ld}");
    }
}
