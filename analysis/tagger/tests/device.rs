//! The tagger heads on the device TURBO_TEST_DEVICE names (`cpu`,
//! `levelzero`, `cuda`, `metal`; the CPU when it is not set), at the
//! precision TURBO_TEST_PRECISION names (model, or exact), against the
//! CPU: every word feature and every label score within 1e-4, the decoded
//! tags and entities the same; for the parser head every arc and
//! relation score within 1e-4 of its size (at least 1), the trees and
//! relations the same; for the lemmatizer head every tree score within
//! 1e-4 and the lemmas the same; for the sentence boundary and span
//! classifier heads every score within 1e-4, the sentences and spans the
//! same. The encoder loads on the device itself,
//! and the run asserts the backend it reports. With
//! TURBO_TEST_REQUIRE_CUDA, TURBO_TEST_REQUIRE_LEVELZERO or
//! TURBO_TEST_REQUIRE_METAL set, the test fails unless it runs there.
//! The backend is a feature of `turbo`: run with `--features turbo/<backend>`.

mod common;

use common::*;
use turbo::head::tagger::Work;
use turbo_analysis::{Annotator, Document, Key, Layer, View};
use turbo_analysis_embed::{Device, EncoderOptions, Precision};
use turbo_analysis_layers::{Entities, PosTags};
use turbo_tagger::{
    DependencyParser, EntityTagger, HeadRunner, Lemmatizer, PosTagger, SentenceTagger, SpanTagger, WordEncoder,
};

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
    assert_eq!(
        pd.annotate((view, &toks, None, None)).unwrap().0.tags,
        pc.annotate((view, &toks, None, None)).unwrap().0.tags
    );
    let ed = EntityTagger::open(&dir, None, &o).unwrap();
    let ec = EntityTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
    let (a,) = ed.annotate((view, &toks, None, None)).unwrap();
    let (b,) = ec.annotate((view, &toks, None, None)).unwrap();
    assert_eq!((a.tokens, a.kind), (b.tokens, b.kind));
}

/// The largest difference of `a` from `b`, each over the size of `b` (at
/// least 1): a biaffine score is a product of two views, so its error
/// grows with it.
fn max_relative(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs() / y.abs().max(1.0)).fold(0.0, f32::max)
}

#[test]
fn the_device_parses_as_the_cpu() {
    let o = options();
    if let Some(b) = required() {
        assert!(!matches!(o.device, Device::Cpu), "{b} is required but TURBO_TEST_DEVICE names no device");
    }
    let dir = bundle("device-parse", |_| {});
    let mut device = WordEncoder::open(&dir, &o).unwrap();
    let execution = device.execution().clone();
    if let Some(b) = required() {
        assert_eq!(execution.backend, b, "the session runs on the backend required");
    }
    let mut cpu = WordEncoder::open(&dir, &EncoderOptions::default()).unwrap();
    let deps = DependencyParser::open(&dir, None, &EncoderOptions::default()).unwrap();
    let p = deps.parser();
    let seqs = sequences();
    let pooling = p.spec().word_pooling;
    let (on_device, on_cpu) = (device.features(&seqs, pooling).unwrap(), cpu.features(&seqs, pooling).unwrap());
    let (mut worst_arc, mut worst_rel) = (0f32, 0f32);
    for (s, (d, c)) in seqs.iter().zip(on_device.iter().zip(&on_cpu)) {
        let ids: Vec<u32> = (0..s.len() as u32).collect();
        let (mut wd, mut wc) = (turbo::head::parser::Work::default(), turbo::head::parser::Work::default());
        let (mut ad, mut ac, mut hd, mut hc) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let n = p.arcs(d, &ids, &mut ad, &mut wd).unwrap();
        p.arcs(c, &ids, &mut ac, &mut wc).unwrap();
        worst_arc = worst_arc.max(max_relative(&ad, &ac));
        p.decode(&ad, n, &mut hd).unwrap();
        p.decode(&ac, n, &mut hc).unwrap();
        assert_eq!(hd, hc, "{s:?}");
        let (mut rd, mut rc) = (Vec::new(), Vec::new());
        p.relations(&hc, &mut rd, &wd).unwrap();
        p.relations(&hc, &mut rc, &wc).unwrap();
        worst_rel = worst_rel.max(max_relative(&rd, &rc));
    }
    eprintln!(
        "parser device {} ({}): max arc score diff {worst_arc:e}, max relation score diff {worst_rel:e} (relative)",
        execution.backend, execution.device
    );
    assert!(worst_arc <= BOUND && worst_rel <= BOUND, "{worst_arc} {worst_rel}");

    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let on = DependencyParser::open(&dir, None, &o).unwrap();
    assert_eq!(on.execution().unwrap().backend, execution.backend);
    let (a,) = on.annotate((view, &toks, None, None)).unwrap();
    let (b,) = deps.annotate((view, &toks, None, None)).unwrap();
    assert_eq!((a.head, a.relation), (b.head, b.relation));
}

#[test]
fn the_device_lemmatizes_as_the_cpu() {
    let o = options();
    if let Some(b) = required() {
        assert!(!matches!(o.device, Device::Cpu), "{b} is required but TURBO_TEST_DEVICE names no device");
    }
    let dir = bundle("device-lemma", |_| {});
    let mut device = WordEncoder::open(&dir, &o).unwrap();
    let execution = device.execution().clone();
    if let Some(b) = required() {
        assert_eq!(execution.backend, b, "the session runs on the backend required");
    }
    let mut cpu = WordEncoder::open(&dir, &EncoderOptions::default()).unwrap();
    let cpu_lem = Lemmatizer::open(&dir, None, &EncoderOptions::default()).unwrap();
    let (head, pos) = (cpu_lem.head(), cpu_lem.pos().unwrap());
    let seqs = sequences();
    let (hp, pp) = (head.spec().word_pooling, pos.spec().word_pooling);
    let (hd, hc) = (device.features(&seqs, hp).unwrap(), cpu.features(&seqs, hp).unwrap());
    let (pd, pc) = (device.features(&seqs, pp).unwrap(), cpu.features(&seqs, pp).unwrap());
    let mut worst = 0f32;
    for (k, s) in seqs.iter().enumerate() {
        let ids: Vec<u32> = (0..s.len() as u32).collect();
        let tags = |f: &[f32]| {
            let (mut sc, mut t) = (Vec::new(), Vec::new());
            pos.scores(f, &ids, &mut sc, &mut Work::default()).unwrap();
            pos.decode(&sc, s.len(), &mut t).unwrap();
            t
        };
        let (td, tc) = (tags(&pd[k]), tags(&pc[k]));
        assert_eq!(td, tc, "{s:?}");
        let mut lw = turbo::head::lemmatizer::Work::default();
        let (mut sd, mut sc) = (Vec::new(), Vec::new());
        head.scores(&hd[k], &ids, Some(&tc), &mut sd, &mut lw).unwrap();
        head.scores(&hc[k], &ids, Some(&tc), &mut sc, &mut lw).unwrap();
        worst = worst.max(max_diff(&sd, &sc));
        let (mut trd, mut trc) = (Vec::new(), Vec::new());
        head.decode(&sd, s, &mut trd).unwrap();
        head.decode(&sc, s, &mut trc).unwrap();
        assert_eq!(trd, trc, "{s:?}");
    }
    eprintln!("lemmatizer device {} ({}): max tree score diff {worst:e}", execution.backend, execution.device);
    assert!(worst <= BOUND, "{worst}");

    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let on = Lemmatizer::open(&dir, None, &o).unwrap();
    assert_eq!(on.execution().unwrap().backend, execution.backend);
    let (a,) = on.annotate((view, &toks, None, None)).unwrap();
    let (b,) = cpu_lem.annotate((view, &toks, None, None)).unwrap();
    assert!(a.lemma.iter().eq(b.lemma.iter()));
}

#[test]
fn the_device_segments_as_the_cpu() {
    let o = options();
    if let Some(b) = required() {
        assert!(!matches!(o.device, Device::Cpu), "{b} is required but TURBO_TEST_DEVICE names no device");
    }
    let sents = serde_json::json!({
        "name": "sents", "layer": "turboembed:sentences", "kind": "HEAD_SENTENCE_BOUNDARY",
        "labels": ["inside", "last"], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_FIRST",
        "decode": "DECODE_CRF", "artifact": "weights-f32", "tensor_prefix": "heads.sents"
    });
    let spans = serde_json::json!({
        "name": "mentions", "layer": "turboembed:mentions", "kind": "HEAD_SPAN_CLASSIFIER",
        "labels": ["O", "PER", "LOC"], "scheme": "SCHEME_TAGS", "word_pooling": "WORD_POOLING_MEAN",
        "adapter": {"bilstm": {"hidden": 4, "layers": 1}}, "decode": "DECODE_SPANS_NESTED",
        "span": {"max_words": 4, "width_dim": 3}, "artifact": "weights-f32", "tensor_prefix": "heads.mentions"
    });
    let dir = bundle("device-segment", |m| m["heads"] = serde_json::json!([sents, spans]));
    let mut device = WordEncoder::open(&dir, &o).unwrap();
    let execution = device.execution().clone();
    if let Some(b) = required() {
        assert_eq!(execution.backend, b, "the session runs on the backend required");
    }
    let mut cpu = WordEncoder::open(&dir, &EncoderOptions::default()).unwrap();
    let st = SentenceTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
    let sp = SpanTagger::open(&dir, Some("mentions"), &EncoderOptions::default()).unwrap();
    let seqs = sequences();
    let (mut worst_boundary, mut worst_span) = (0f32, 0f32);
    let (first, mean) = (turbo::manifest::WordPooling::First, turbo::manifest::WordPooling::Mean);
    let pairs = |enc: &mut WordEncoder, p| enc.features(&seqs, p).unwrap();
    let (bd, bc, sd, sc) =
        (pairs(&mut device, first), pairs(&mut cpu, first), pairs(&mut device, mean), pairs(&mut cpu, mean));
    for (k, s) in seqs.iter().enumerate() {
        let ids: Vec<u32> = (0..s.len() as u32).collect();
        let (mut a, mut b, mut ea, mut eb) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        st.head().scores(&bd[k], &ids, None, &mut a, &mut Work::default()).unwrap();
        st.head().scores(&bc[k], &ids, None, &mut b, &mut Work::default()).unwrap();
        worst_boundary = worst_boundary.max(max_diff(&a, &b));
        st.head().decode(&a, s.len(), &mut ea).unwrap();
        st.head().decode(&b, s.len(), &mut eb).unwrap();
        assert_eq!(ea, eb, "boundaries: {s:?}");
        let (mut wa, mut wb) = (turbo::head::span::SpanWork::default(), turbo::head::span::SpanWork::default());
        let (mut ka, mut kb) = (Vec::new(), Vec::new());
        sp.head().scores(&sd[k], &ids, None, &mut a, &mut Work::default(), &mut wa).unwrap();
        sp.head().scores(&sc[k], &ids, None, &mut b, &mut Work::default(), &mut wb).unwrap();
        worst_span = worst_span.max(max_diff(&a, &b));
        sp.head().decode(&a, s.len(), &mut ka).unwrap();
        sp.head().decode(&b, s.len(), &mut kb).unwrap();
        let key = |v: &[turbo::head::span::Span]| v.iter().map(|s| (s.first, s.last, s.label)).collect::<Vec<_>>();
        assert_eq!(key(&ka), key(&kb), "spans: {s:?}");
    }
    eprintln!(
        "segment device {} ({}): max boundary score diff {worst_boundary:e}, max span score diff {worst_span:e}",
        execution.backend, execution.device
    );
    assert!(worst_boundary <= BOUND && worst_span <= BOUND, "{worst_boundary} {worst_span}");

    let doc = Document::new(TEXT);
    let view = doc.require(&Key::<View>::DEFAULT).unwrap();
    let toks = tokens(TEXT);
    let on = SentenceTagger::open(&dir, None, &o).unwrap();
    assert_eq!(on.execution().unwrap().backend, execution.backend);
    let spans_of =
        |l: &turbo_analysis_layers::Sentences| l.spans.iter().map(|(_, s)| (s.start, s.end)).collect::<Vec<_>>();
    assert_eq!(
        spans_of(&on.annotate((view, &toks, None)).unwrap().0),
        spans_of(&st.annotate((view, &toks, None)).unwrap().0)
    );
    let on = SpanTagger::open(&dir, Some("mentions"), &o).unwrap();
    let (a,) = on.annotate((view, &toks, None, None)).unwrap();
    let (b,) = sp.annotate((view, &toks, None, None)).unwrap();
    assert_eq!((a.tokens, a.kind), (b.tokens, b.kind));
}
