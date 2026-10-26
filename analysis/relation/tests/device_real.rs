//! GLiNER2 relations and GLiREL on the real checkpoints with the encoder on
//! a device, against the CPU encoder of the same core bundle (#920): the
//! texts and labels of tests/fixtures/*/real.json, DEVICE_REPS times over:
//! every instance count and extracted relation equal, every logit within
//! REAL_BOUND of the CPU's, and every decision the CPU's margin holds (each
//! query's best span, each label's best pair) the same, those picking at
//! least two choices; then a release timing interleaved against the CPU
//! path. The device is the one device_support names.
//!
//! The CPU side is the encoder as production loads it, the DeBERTa package
//! `turbo-bundle deberta-convert` writes; the device side is the encoder's
//! FAMILY_DEBERTA_V2 core bundle bound to that package. The environment
//! names them: GLINER2_PACKAGE, GLINER2_BUNDLE and GLINER2_CHECKPOINT (the
//! checkpoint's directory, as `Gliner2Relations::load` reads it), and
//! GLIREL_PACKAGE (the package `turbo-bundle glirel-convert` writes, whose
//! encoder/ is the DeBERTa package), GLIREL_BUNDLE and GLIREL_SPM. DEVICE_REPS passes of each text
//! are timed, default 20. Run with
//! `cargo test --release -p turbo-relation --test device_real -- --ignored --nocapture`.

mod device_support;

use std::path::PathBuf;
use std::time::Instant;

use device_support::*;
use serde_json::Value;
use turbo_relation::encoder::DebertaEncoder;
use turbo_relation::{gliner2, glirel};

/// The largest logit difference from the CPU encoder, absolute up to a
/// logit of 1 and relative above (device_support::worst_scaled): the F32
/// device bound the encoder itself is held to (#883). The real heads'
/// logits reach the hundreds, where F32 has no absolute 1e-4.
const REAL_BOUND: f32 = 1e-4;

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
    let (bundle, package, checkpoint) = (env("GLINER2_BUNDLE"), env("GLINER2_PACKAGE"), env("GLINER2_CHECKPOINT"));
    let load = |e: DebertaEncoder| gliner2::Gliner2Relations::load_with(e, &checkpoint, "real", "1").unwrap();
    let (cpu, on) = (load(package_cpu(&package)), load(bound(&package, &bundle)));
    let cases = cases("gliner2");
    let (mut worst_seen, mut held, mut found) = (0f32, Vec::new(), 0);
    for _ in 0..reps() {
        for (k, c) in cases.iter().enumerate() {
            let (text, labels) = (c["text"].as_str().unwrap(), strs(&c["labels"]));
            let (a, b) = (cpu.pass(text, &labels).unwrap(), on.pass(text, &labels).unwrap());
            let size = a.prompt.words.len() * a.width;
            for (l, (x, y)) in a.labels.iter().zip(&b.labels).enumerate() {
                assert_eq!(x.count, y.count, "case {k} label {l}: instance count");
                let d = worst_scaled(&x.logits, &y.logits, text);
                assert!(d <= REAL_BOUND, "case {k} label {l}: span logits {d}");
                worst_seen = worst_seen.max(d);
            }
            let rows = |p: &gliner2::Pass| -> Vec<Vec<f32>> {
                p.labels.iter().flat_map(|l| l.logits.chunks(size).map(<[f32]>::to_vec)).collect()
            };
            let (x, y) = (rows(&a), rows(&b));
            let (x, y): (Vec<&[f32]>, Vec<&[f32]>) =
                (x.iter().map(Vec::as_slice).collect(), y.iter().map(Vec::as_slice).collect());
            held.extend(same_decisions(&x, &y, REAL_BOUND, text));
            let ends = |v: Vec<gliner2::Extracted>| {
                v.into_iter().map(|x| (x.label, x.head.start, x.head.end, x.tail.start, x.tail.end)).collect::<Vec<_>>()
            };
            let (x, y) = (cpu.extract(text, &labels, 0.5).unwrap(), on.extract(text, &labels, 0.5).unwrap());
            found += x.len();
            assert_eq!(ends(x), ends(y), "case {k}: extracted relations");
        }
    }
    varied(&held, "gliner2 best spans");
    assert!(found > 0, "no relation extracted on either side");
    let worst = worst_seen;
    println!(
        "gliner2 relations: {} runs of {} cases, {} best spans held, {found} relations extracted alike",
        reps(),
        cases.len(),
        held.len()
    );
    let x = on.execution().unwrap().clone();
    let on_name = format!("{} {} ({})", x.backend, x.device, x.precision);
    println!("gliner2 relations: largest logit difference from the cpu encoder {worst:e}");
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
    let [cpu, on] = glirel_pair(&package, &bundle, &spm);
    let cases = cases("glirel");
    let entities = |c: &Value| -> Vec<(usize, usize)> {
        let e = c["ner"].as_array().unwrap();
        e.iter().map(|e| (e[0].as_u64().unwrap() as usize, e[1].as_u64().unwrap() as usize)).collect()
    };
    let (mut worst_seen, mut held) = (0f32, Vec::new());
    for _ in 0..reps() {
        for (k, c) in cases.iter().enumerate() {
            let (text, labels) = (c["text"].as_str().unwrap(), strs(&c["labels"]));
            let (a, b) = (cpu.pass(text, &labels).unwrap(), on.pass(text, &labels).unwrap());
            let ents = entities(c);
            let pairs: Vec<(usize, usize)> =
                (0..ents.len()).flat_map(|h| (0..ents.len()).filter(move |&t| t != h).map(move |t| (h, t))).collect();
            let (x, y) = (cpu.logits(&a, &ents, &pairs).unwrap(), on.logits(&b, &ents, &pairs).unwrap());
            let d = worst_scaled(&x, &y, text);
            assert!(d <= REAL_BOUND, "case {k}: relation logits {d}");
            worst_seen = worst_seen.max(d);
            // Each pair's best label.
            let m = labels.len();
            let (p, q): (Vec<&[f32]>, Vec<&[f32]>) = (x.chunks(m).collect(), y.chunks(m).collect());
            held.extend(same_decisions(&p, &q, REAL_BOUND, text));
            let (x, y) =
                (cpu.extract(text, &ents, &labels, 0.5).unwrap(), on.extract(text, &ents, &labels, 0.5).unwrap());
            let keys = |v: Vec<glirel::Extracted>| v.into_iter().map(|r| (r.head, r.tail, r.label)).collect::<Vec<_>>();
            assert_eq!(keys(x), keys(y), "case {k}: extracted relations");
        }
    }
    varied(&held, "glirel pair labels");
    let worst = worst_seen;
    let x = on.execution().unwrap().clone();
    let on_name = format!("{} {} ({})", x.backend, x.device, x.precision);
    println!("glirel: largest logit difference from the cpu encoder {worst:e}, {} pair labels held", held.len());
    let pass = |m: &glirel::GlirelRelations, c: usize| {
        std::hint::black_box(m.pass(cases[c]["text"].as_str().unwrap(), &strs(&cases[c]["labels"])).unwrap());
    };
    report("glirel, one text a pass", &on_name, timed(reps(), cases.len(), |c| pass(&cpu, c), |c| pass(&on, c)));
}
