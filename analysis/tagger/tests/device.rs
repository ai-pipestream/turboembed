//! The tagger heads on the device TURBO_TEST_DEVICE names (`cpu`,
//! `levelzero`, `cuda`, `metal`; the CPU when it is not set), at the
//! precision TURBO_TEST_PRECISION names (model, or exact), against the
//! CPU: every word feature and every label score within 1e-4, the decoded
//! tags and entities the same. The encoder loads on the device itself,
//! and the run asserts the backend it reports. With
//! TURBO_TEST_REQUIRE_CUDA, TURBO_TEST_REQUIRE_LEVELZERO or
//! TURBO_TEST_REQUIRE_METAL set, the test fails unless it runs there.

mod common;

use common::*;
use turbo::head::tagger::Work;
use turbo_analysis::{Annotator, Document, Key, Layer, View};
use turbo_analysis_embed::{Device, EncoderOptions, Precision};
use turbo_analysis_layers::{Entities, PosTags};
use turbo_tagger::{EntityTagger, HeadRunner, PosTagger, WordEncoder};

const BOUND: f32 = 1e-4;

fn required() -> Option<&'static str> {
    let set = |v: &str| std::env::var(v).is_ok_and(|v| !v.is_empty() && v != "0");
    [
        ("TURBO_TEST_REQUIRE_CUDA", "cuda"),
        ("TURBO_TEST_REQUIRE_LEVELZERO", "levelzero"),
        ("TURBO_TEST_REQUIRE_METAL", "metal"),
    ]
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

fn sequences() -> Vec<Vec<&'static str>> {
    let long: Vec<&str> = TEXT.split(' ').cycle().take(120).collect();
    vec![TEXT.split(' ').collect(), vec!["Zoë"], "Ann met Bob".split(' ').collect(), long]
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn the_device_reads_words_and_tags_as_the_cpu() {
    let o = options();
    if let Some(b) = required() {
        assert!(!matches!(o.device, Device::Cpu), "{b} is required but TURBO_TEST_DEVICE names no device");
    }
    let dir = bundle("device", |_| {});
    let mut device = WordEncoder::open(&dir, &o).unwrap();
    let execution = device.execution().clone();
    if let Some(b) = required() {
        assert_eq!(execution.backend, b, "the session runs on the backend required");
    }
    let mut cpu = WordEncoder::open(&dir, &EncoderOptions::default()).unwrap();
    let seqs = sequences();
    let (mut worst_feature, mut worst_score) = (0f32, 0f32);
    for (name, layer) in [("pos", PosTags::ID), ("ner", Entities::ID)] {
        let run = HeadRunner::open(&dir, Some(name), layer, &EncoderOptions::default()).unwrap();
        let pooling = run.tagger().spec().word_pooling;
        let on_device = device.features(&seqs, pooling).unwrap();
        let on_cpu = cpu.features(&seqs, pooling).unwrap();
        for (s, (d, c)) in seqs.iter().zip(on_device.iter().zip(&on_cpu)) {
            worst_feature = worst_feature.max(max_diff(d, c));
            let ids: Vec<u32> = (0..s.len() as u32).collect();
            let (mut sd, mut sc, mut td, mut tc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            run.tagger().scores(d, &ids, &mut sd, &mut Work::default()).unwrap();
            run.tagger().scores(c, &ids, &mut sc, &mut Work::default()).unwrap();
            worst_score = worst_score.max(max_diff(&sd, &sc));
            run.tagger().decode(&sd, s.len(), &mut td).unwrap();
            run.tagger().decode(&sc, s.len(), &mut tc).unwrap();
            assert_eq!(td, tc, "{name}: {s:?}");
        }
    }
    eprintln!(
        "tagger device {} ({}): max word feature diff {worst_feature:e}, max label score diff {worst_score:e}",
        execution.backend, execution.device
    );
    assert!(worst_feature <= BOUND && worst_score <= BOUND, "{worst_feature} {worst_score}");

    // The annotators on the device write what they write on the CPU.
    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let pd = PosTagger::open(&dir, None, &o).unwrap();
    let pc = PosTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
    assert_eq!(pd.execution().unwrap().backend, execution.backend);
    assert_eq!(pd.annotate((view, &toks, None)).unwrap().0.tags, pc.annotate((view, &toks, None)).unwrap().0.tags);
    let ed = EntityTagger::open(&dir, None, &o).unwrap();
    let ec = EntityTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
    let (a,) = ed.annotate((view, &toks, None)).unwrap();
    let (b,) = ec.annotate((view, &toks, None)).unwrap();
    assert_eq!((a.tokens, a.kind), (b.tokens, b.kind));
}
