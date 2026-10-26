//! GLiNER2 relations and GLiREL on the real checkpoints with the encoder on
//! a device, against the CPU encoder of the same core bundle (#920): the
//! texts and labels of tests/fixtures/*/real.json, every instance count and
//! extracted relation equal and every logit within REAL_BOUND of the
//! largest logit of its pass, then a release timing interleaved against
//! the CPU path. The device is the one device_support names.
//!
//! It needs the encoders as FAMILY_DEBERTA_V2 core bundles, which are named
//! by the environment: GLINER2_BUNDLE with GLINER2_CHECKPOINT (the
//! checkpoint's directory, as `Gliner2Relations::load` reads it), and
//! GLIREL_BUNDLE with GLIREL_PACKAGE and GLIREL_SPM (as
//! `GlirelRelations::load` reads them). DEVICE_REPS passes of each text
//! are timed, default 20. Run with
//! `cargo test --release -p turbo-relation --test device_real -- --ignored --nocapture`.

mod device_support;

use std::path::PathBuf;
use std::time::Instant;

use device_support::*;
use serde_json::Value;
use turbo_relation::encoder::DebertaEncoder;
use turbo_relation::{gliner2, glirel};

/// The largest logit difference from the CPU encoder, as a fraction of the
/// largest logit of the pass: the encoder's states are within 1e-4, and the
/// heads carry that through their own products.
const REAL_BOUND: f32 = 1e-3;

fn env(name: &str) -> PathBuf {
    std::env::var_os(name).map(PathBuf::from).unwrap_or_else(|| panic!("set {name}"))
}

fn cases(name: &str) -> Vec<Value> {
    let path = format!("{}/tests/fixtures/{name}/real.json", env!("CARGO_MANIFEST_DIR"));
    let v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    v["cases"].as_array().unwrap().clone()
}

fn strs(v: &Value) -> Vec<&str> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect()
}

fn reps() -> usize {
    std::env::var("DEVICE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20)
}

/// Each case's pass on both, interleaved, `reps` times: their times.
fn timed(reps: usize, cases: usize, mut cpu: impl FnMut(usize), mut device: impl FnMut(usize)) -> [Vec<f64>; 2] {
    let mut t = [Vec::new(), Vec::new()];
    for _ in 0..reps {
        for c in 0..cases {
            let s = Instant::now();
            cpu(c);
            t[0].push(s.elapsed().as_secs_f64() * 1e3);
            let s = Instant::now();
            device(c);
            t[1].push(s.elapsed().as_secs_f64() * 1e3);
        }
    }
    t
}

fn report(what: &str, on: &str, mut t: [Vec<f64>; 2]) {
    let p50 = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let ratio = p50(&mut t[0]) / p50(&mut t[1]);
    println!("{what}: cpu encoder {}", summary(&mut t[0]));
    println!("{what}: {on} {}", summary(&mut t[1]));
    println!("{what}: the cpu encoder's p50 over the device's: {ratio:.2}x");
}

#[test]
#[ignore = "needs the real encoders as core bundles; run with --release -- --ignored"]
fn gliner2_relations_on_the_real_checkpoint() {
    let (bundle, checkpoint) = (env("GLINER2_BUNDLE"), env("GLINER2_CHECKPOINT"));
    let load = |e: DebertaEncoder| gliner2::Gliner2Relations::load_with(e, &checkpoint, "real", "1").unwrap();
    let (cpu, on) = (load(DebertaEncoder::open_cpu(&bundle).unwrap()), load(device_on(&bundle)));
    let cases = cases("gliner2");
    let mut worst = 0f32;
    for (k, c) in cases.iter().enumerate() {
        let (text, labels) = (c["text"].as_str().unwrap(), strs(&c["labels"]));
        let (a, b) = (cpu.pass(text, &labels).unwrap(), on.pass(text, &labels).unwrap());
        let scale = a.labels.iter().flat_map(|l| &l.logits).filter(|x| x.is_finite()).fold(1f32, |m, x| m.max(x.abs()));
        for (l, (x, y)) in a.labels.iter().zip(&b.labels).enumerate() {
            assert_eq!(x.count, y.count, "case {k} label {l}: instance count");
            let d = device_support::worst(&x.logits, &y.logits, text) / scale;
            assert!(d <= REAL_BOUND, "case {k} label {l}: span logits {d} of the largest");
            worst = worst.max(d);
        }
        let ends = |v: Vec<gliner2::Extracted>| {
            v.into_iter().map(|x| (x.label, x.head.start, x.head.end, x.tail.start, x.tail.end)).collect::<Vec<_>>()
        };
        let (x, y) = (cpu.extract(text, &labels, 0.5).unwrap(), on.extract(text, &labels, 0.5).unwrap());
        assert_eq!(ends(x), ends(y), "case {k}: extracted relations");
    }
    let x = on.execution().unwrap().clone();
    let on_name = format!("{} {} ({})", x.backend, x.device, x.precision);
    println!("gliner2 relations: largest logit difference from the cpu encoder {worst:e} of the pass's largest");
    let pass = |m: &gliner2::Gliner2Relations, c: usize| {
        std::hint::black_box(m.pass(cases[c]["text"].as_str().unwrap(), &strs(&cases[c]["labels"])).unwrap());
    };
    report(
        "gliner2 relations, one text a pass",
        &on_name,
        timed(reps(), cases.len(), |c| pass(&cpu, c), |c| pass(&on, c)),
    );
}

#[test]
#[ignore = "needs the real encoders as core bundles; run with --release -- --ignored"]
fn glirel_on_the_real_checkpoint() {
    let (bundle, package, spm) = (env("GLIREL_BUNDLE"), env("GLIREL_PACKAGE"), env("GLIREL_SPM"));
    let load = |e: DebertaEncoder| glirel::GlirelRelations::load_with(e, &package, &spm).unwrap();
    let (cpu, on) = (load(DebertaEncoder::open_cpu(&bundle).unwrap()), load(device_on(&bundle)));
    let cases = cases("glirel");
    let mut worst = 0f32;
    let entities = |c: &Value| -> Vec<(usize, usize)> {
        c["ner"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| (e[0].as_u64().unwrap() as usize, e[1].as_u64().unwrap() as usize))
            .collect()
    };
    for (k, c) in cases.iter().enumerate() {
        let (text, labels) = (c["text"].as_str().unwrap(), strs(&c["labels"]));
        let (a, b) = (cpu.pass(text, &labels).unwrap(), on.pass(text, &labels).unwrap());
        let ents = entities(c);
        let pairs: Vec<(usize, usize)> =
            (0..ents.len()).flat_map(|h| (0..ents.len()).filter(move |&t| t != h).map(move |t| (h, t))).collect();
        let (x, y) = (cpu.logits(&a, &ents, &pairs).unwrap(), on.logits(&b, &ents, &pairs).unwrap());
        let scale = x.iter().fold(1f32, |m, v| m.max(v.abs()));
        let d = device_support::worst(&x, &y, text) / scale;
        assert!(d <= REAL_BOUND, "case {k}: relation logits {d} of the largest");
        worst = worst.max(d);
        let (x, y) = (cpu.extract(text, &ents, &labels, 0.5).unwrap(), on.extract(text, &ents, &labels, 0.5).unwrap());
        let keys = |v: Vec<glirel::Extracted>| v.into_iter().map(|r| (r.head, r.tail, r.label)).collect::<Vec<_>>();
        assert_eq!(keys(x), keys(y), "case {k}: extracted relations");
    }
    let x = on.execution().unwrap().clone();
    let on_name = format!("{} {} ({})", x.backend, x.device, x.precision);
    println!("glirel: largest logit difference from the cpu encoder {worst:e} of the pass's largest");
    let pass = |m: &glirel::GlirelRelations, c: usize| {
        std::hint::black_box(m.pass(cases[c]["text"].as_str().unwrap(), &strs(&cases[c]["labels"])).unwrap());
    };
    report("glirel, one text a pass", &on_name, timed(reps(), cases.len(), |c| pass(&cpu, c), |c| pass(&on, c)));
}
