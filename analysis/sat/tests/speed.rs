//! How fast the SaT step finds the sentences of the reference texts:
//! tokenizing, every window through the encoder, the classifier and the
//! cuts. On the CPU, and on the device TURBO_TEST_DEVICE names at the
//! precision TURBO_TEST_PRECISION names (tests/common), each pass on one
//! alternating with a pass on the other on the same documents. Ignored,
//! since it measures rather than checks; run it with
//!
//! ```
//! cargo test --release -p turbo-sat --test speed -- --ignored --nocapture
//! ```
//!
//! TURBO_SAT_BUNDLE names the bundle (default testdata/tiny-sat-bundle,
//! whose two layers of width 32 say little about a real model);
//! TURBO_SPEED_RUNS the timed runs on each side (default 5), whose median
//! is given.

use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo_analysis::Annotator;
use turbo_analysis_embed::Device;
use turbo_sat::{SatAnnotator, SatOptions};

mod common;

fn bundle() -> PathBuf {
    match std::env::var_os("TURBO_SAT_BUNDLE") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-sat-bundle"),
    }
}

/// One pass over the documents: the seconds it took and the sentences found.
fn pass(a: &SatAnnotator, docs: &[&str]) -> (f64, usize) {
    let start = Instant::now();
    let mut out = Vec::new();
    for (d, r) in docs.iter().zip(a.scores(docs)) {
        let (spans, scores) = r.unwrap();
        a.sentences(d, &spans, &scores, &mut out);
    }
    (start.elapsed().as_secs_f64(), out.len())
}

#[test]
#[ignore]
fn sat_speed() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/texts.jsonl");
    let texts: Vec<String> =
        std::fs::read_to_string(path).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    // Documents of a few kilobytes, as a corpus holds them: the texts
    // joined in tens.
    let docs: Vec<String> = (0..4).flat_map(|_| texts.chunks(10).map(|c| c.join("\n"))).collect();
    let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
    let bytes: usize = refs.iter().map(|d| d.len()).sum();
    let runs: usize = std::env::var("TURBO_SPEED_RUNS").ok().and_then(|r| r.parse().ok()).unwrap_or(5);
    let device = common::encoder();
    let mut sides = vec![SatOptions::default()];
    if device.device != Device::Cpu {
        sides.push(SatOptions { encoder: device, ..SatOptions::default() });
    }
    let sides: Vec<SatAnnotator> = sides.iter().map(|o| SatAnnotator::open(&bundle(), o).unwrap()).collect();
    let tokens: usize = sides[0].scores(&refs).iter().map(|r| r.as_ref().unwrap().0.len()).sum();
    let mut times = vec![Vec::with_capacity(runs); sides.len()];
    let mut found = vec![0; sides.len()];
    for a in &sides {
        pass(a, &refs);
    }
    for _ in 0..runs {
        for (i, a) in sides.iter().enumerate() {
            let (secs, n) = pass(a, &refs);
            times[i].push(secs);
            found[i] = n;
        }
    }
    for ((a, t), n) in sides.iter().zip(&mut times).zip(found) {
        t.sort_by(f64::total_cmp);
        let secs = t[t.len() / 2];
        eprintln!(
            "{} on {:?}: {} documents, {bytes} bytes, {tokens} tokens, block {} stride {}: median of {runs} {:.1} ms a pass (fastest {:.1}, slowest {:.1}), {:.0} tokens/s, {:.2} MB/s, {n} sentences",
            bundle().display(),
            a.execution(),
            refs.len(),
            a.block_size(),
            a.stride(),
            secs * 1e3,
            t[0] * 1e3,
            t[t.len() - 1] * 1e3,
            tokens as f64 / secs,
            bytes as f64 / secs / 1e6,
        );
    }
}
