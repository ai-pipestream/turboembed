//! GLiNER2 schema extraction on the real checkpoint with the encoder on a
//! device, against the CPU encoder of the same core bundle (#920): the
//! texts and schemas of tests/fixtures/real.json, DEVICE_REPS times over:
//! every instance count equal, every span and label logit within
//! REAL_BOUND of the CPU's, every decision the CPU's margin holds the same
//! (picking at least two choices) and the extraction decoded alike; then a
//! release timing interleaved against the CPU path, on the device the
//! relation crate's device_support names.
//!
//! GLINER2_PACKAGE names the encoder as production loads it (the DeBERTa
//! package), GLINER2_BUNDLE its FAMILY_DEBERTA_V2 core bundle, bound to
//! that package, and GLINER2_CHECKPOINT the checkpoint's directory; DEVICE_REPS passes of
//! each text are timed, default 20. Run with
//! `cargo test --release -p turbo-schema-extract --test device_real -- --ignored --nocapture`.

mod common;
#[path = "../../relation/tests/device_support/mod.rs"]
mod device_support;

use std::path::PathBuf;
use std::time::Instant;

use device_support::*;
use serde_json::Value;
use turbo_relation::encoder::DebertaEncoder;
use turbo_schema_extract::model::{Gliner2Extractor, Scores};

const REAL_BOUND: f32 = 1e-4;

fn env(name: &str) -> PathBuf {
    std::env::var_os(name).map(PathBuf::from).unwrap_or_else(|| panic!("set {name}"))
}

#[test]
#[ignore = "needs the real encoder as a core bundle; run with --release -- --ignored"]
fn schema_extraction_on_the_real_checkpoint() {
    let (bundle, package, checkpoint) = (env("GLINER2_BUNDLE"), env("GLINER2_PACKAGE"), env("GLINER2_CHECKPOINT"));
    let load = |e: DebertaEncoder| Gliner2Extractor::load_with(e, &checkpoint, "real", "1").unwrap();
    let (cpu, on) = (load(package_cpu(&package)), load(bound(&package, &bundle)));
    let path = format!("{}/tests/fixtures/real.json", env!("CARGO_MANIFEST_DIR"));
    let reference: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let cases: Vec<(String, _)> = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (c["text"].as_str().unwrap().to_owned(), common::schema(&c["schema"])))
        .collect();
    let reps = std::env::var("DEVICE_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let (mut worst_seen, mut held, mut decoded) = (0f32, Vec::new(), 0usize);
    for _ in 0..reps {
        for (k, (text, schema)) in cases.iter().enumerate() {
            let (a, b) = (cpu.pass(text, schema).unwrap(), on.pass(text, schema).unwrap());
            let logits = |s: &Scores| match s {
                Scores::Labels(v) => v.clone(),
                Scores::Spans { logits, .. } => logits.clone(),
            };
            for (t, (x, y)) in a.scores().iter().zip(b.scores()).enumerate() {
                if let (Scores::Spans { count: c, instances: i, .. }, Scores::Spans { count: d, instances: j, .. }) =
                    (x, y)
                {
                    assert_eq!((c, i), (d, j), "case {k} task {t}: counts");
                }
                let d = worst(&logits(x), &logits(y), text);
                assert!(d <= REAL_BOUND, "case {k} task {t}: logits {d}");
                worst_seen = worst_seen.max(d);
            }
            let (x, y) = (common::rows(&a), common::rows(&b));
            let (x, y): (Vec<&[f32]>, Vec<&[f32]>) =
                (x.iter().map(Vec::as_slice).collect(), y.iter().map(Vec::as_slice).collect());
            held.extend(same_decisions(&x, &y, REAL_BOUND, text));
            // Decoded as the reference test decodes, at 0.5.
            let d = |p| turbo_schema_extract::decode(p, schema, 0.5, &turbo_schema_extract::decode::Nothing).unwrap();
            let (ea, eb) = (d(&a), d(&b));
            assert_eq!(common::picked(&ea), common::picked(&eb), "case {k}: decoded");
            decoded += ea.entities.iter().map(Vec::len).sum::<usize>() + ea.records.len() + ea.classifications.len();
        }
    }
    varied(&held, "schema decisions");
    assert!(decoded > 0, "nothing decoded on either side");
    println!(
        "schema extraction: {reps} runs of {} cases, {} decisions held, {decoded} values decoded alike",
        cases.len(),
        held.len()
    );
    let x = on.execution().unwrap().clone();
    let on_name = format!("{} {} ({})", x.backend, x.device, x.precision);
    println!("schema extraction: largest logit difference from the cpu encoder {worst_seen:e}");
    let mut t = [Vec::new(), Vec::new()];
    for _ in 0..reps {
        for (text, schema) in &cases {
            for (i, m) in [&cpu, &on].into_iter().enumerate() {
                let s = Instant::now();
                std::hint::black_box(m.pass(text, schema).unwrap());
                t[i].push(s.elapsed().as_secs_f64() * 1e3);
            }
        }
    }
    let [mut a, mut b] = t;
    println!("schema extraction, one text a pass: cpu encoder {}", summary(&mut a));
    println!("schema extraction, one text a pass: {on_name} {}", summary(&mut b));
    println!("schema extraction: the cpu encoder's p50 over the device's: {:.2}x", a[a.len() / 2] / b[b.len() / 2]);
}
