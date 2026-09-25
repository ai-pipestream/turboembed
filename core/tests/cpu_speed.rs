//! How long the CPU encoder takes on a batch of 32 rows of up to 256
//! tokens, 1353 of them live, as a MiniLM-sized server sees them: on the
//! sealed small bundle, and on a model of all-MiniLM-L6-v2's shape (6
//! layers, hidden 384, 12 heads, intermediate 1536) with synthetic
//! weights. Ignored, since it measures rather than checks; run it with
//!
//! ```
//! cargo test --release -p turbo --test cpu_speed -- --ignored --nocapture
//! ```
//!
//! TURBO_SPEED_RUNS sets the timed runs (default 20). Each case also says
//! how far its vectors are from the plain f64 encoder in tests/common.

mod common;

use std::path::Path;
use std::time::Instant;

use common::*;
use serde_json::{Value, json};
use turbo::*;

const BATCH: usize = 32;
const SEQ: usize = 256;
const LIVE: usize = 1353;

/// Each row's length: one row as long as the batch is wide, the rest
/// spread short to middling, summing to LIVE.
fn lengths() -> Vec<usize> {
    let mut l: Vec<usize> = (0..BATCH).map(|r| if r == 0 { SEQ } else { 6 + (r * 29) % 53 }).collect();
    let sum: usize = l.iter().sum();
    l[BATCH - 1] = l[BATCH - 1] + LIVE - sum;
    assert_eq!(l.iter().sum::<usize>(), LIVE);
    assert!(l.iter().all(|&n| (1..=SEQ).contains(&n)));
    l
}

/// Rows of ids from the vocabulary's body, [CLS] first and [SEP] last.
fn rows() -> Vec<Vec<i32>> {
    lengths()
        .iter()
        .enumerate()
        .map(|(r, &n)| {
            (0..n)
                .map(|p| match p {
                    0 => 101,
                    p if p == n - 1 => 102,
                    p => 1000 + ((r * 7919 + p * 104729) % 28000) as i32,
                })
                .collect()
        })
        .collect()
}

fn runs() -> usize {
    std::env::var("TURBO_SPEED_RUNS").ok().and_then(|v| v.parse().ok()).unwrap_or(20)
}

/// Times the batch on `dir`'s model and compares a few rows with the
/// plain encoder.
fn measure(what: &str, dir: &Path) {
    let l = Loaded::load(dir).unwrap_or_else(|e| panic!("{e:?}"));
    let t = Instant::now();
    let s = Session::create(l.m, Some(&session_desc(BATCH as u32, SEQ as u32, TURBO_PRECISION_MODEL))).unwrap();
    let create = t.elapsed();
    let rows = rows();
    let tokens = Tokens::new(&rows, 0);
    let b = tokens.batch();
    let mut times = Vec::new();
    let mut first: Option<Vec<Vec<f32>>> = None;
    for i in 0..runs() + 3 {
        let t = Instant::now();
        s.write_tokens(&b, None).unwrap();
        let r = s.run().unwrap();
        let dt = t.elapsed();
        let got = r.rows();
        match &first {
            None => first = Some(got),
            Some(f) => assert!(
                f.iter().flatten().zip(got.iter().flatten()).all(|(a, b)| a.to_bits() == b.to_bits()),
                "run {i} differs from the first"
            ),
        }
        if i >= 3 {
            times.push(dt.as_secs_f64() * 1e3);
        }
    }
    times.sort_by(f64::total_cmp);
    let p50 = times[times.len() / 2];
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "{what}: batch {BATCH} x seq {SEQ}, {LIVE} live tokens: p50 {p50:.2} ms, mean {mean:.2} ms, \
         min {:.2} ms over {} runs; session create {:.1} ms",
        times[0],
        times.len(),
        create.as_secs_f64() * 1e3
    );

    // Against the plain f64 encoder: the longest row and a few short ones.
    let plain = PlainBert::new(dir);
    let got = first.unwrap();
    let (mut worst_cos, mut worst_abs) = (1.0f64, 0.0f64);
    for r in [0, 1, 2, BATCH - 1] {
        let ids = &rows[r];
        let mask = vec![1; ids.len()];
        let want = plain.embed(ids, &mask, &vec![0; ids.len()], TURBO_POOLING_MEAN, got[r].len(), true);
        let want: Vec<f32> = want.iter().map(|&v| v as f32).collect();
        worst_cos = worst_cos.min(cosine(&got[r], &want));
        worst_abs = worst_abs.max(max_abs_diff(&got[r], &want));
    }
    println!(
        "  against the f64 encoder on 4 rows: 1 - min cosine {:.3e}, max abs diff {worst_abs:.3e}",
        1.0 - worst_cos
    );
}

/// The sealed small bundle's model, in a bundle whose manifest takes rows
/// of SEQ tokens.
#[test]
#[ignore = "measures; run with --ignored --nocapture"]
fn small_bundle_speed() {
    let mut m: Value = serde_json::from_slice(&std::fs::read(tiny_bundle().join("manifest.json")).unwrap()).unwrap();
    m["embed"]["max_seq"] = json!(SEQ);
    m["files"] = json!([]);
    let mut f = Fixture::new("speed-small", m);
    std::fs::create_dir_all(f.dir.join("weights")).unwrap();
    std::fs::copy(tiny_bundle().join("weights/model.safetensors"), f.dir.join("weights/model.safetensors")).unwrap();
    f.list("weights/model.safetensors");
    f.write();
    measure("small bundle (2 layers, hidden 32)", &f.dir);
}

/// all-MiniLM-L6-v2's shape, with weight matrices scaled to keep the
/// activations in the range a trained model's are.
#[test]
#[ignore = "measures; run with --ignored --nocapture"]
fn minilm_shape_speed() {
    let mut m = model_manifest();
    m["architecture"] = manifest()["architecture"].clone();
    m["embed"]["dim"] = json!(384);
    let mut f = Fixture::new("speed-minilm", m);
    f.weights("weights/model.safetensors", &bert_weights(6, 384, 1536, 0, 0.05));
    f.write();
    measure("MiniLM shape (6 layers, hidden 384)", &f.dir);
}
