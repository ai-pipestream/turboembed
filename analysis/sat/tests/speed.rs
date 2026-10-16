//! How fast the SaT step finds the sentences of the reference texts on
//! the CPU: tokenizing, every window through the encoder, the classifier
//! and the cuts. Ignored, since it measures rather than checks; run it
//! with
//!
//! ```
//! cargo test --release -p turbo-sat --test speed -- --ignored --nocapture
//! ```
//!
//! TURBO_SAT_BUNDLE names the bundle (default testdata/tiny-sat-bundle,
//! whose two layers of width 32 say little about a real model);
//! TURBO_SPEED_RUNS the timed runs (default 5).

use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo_sat::{SatAnnotator, SatOptions};

fn bundle() -> PathBuf {
    match std::env::var_os("TURBO_SAT_BUNDLE") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-sat-bundle"),
    }
}

#[test]
#[ignore]
fn sat_speed_on_the_cpu() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/texts.jsonl");
    let texts: Vec<String> =
        std::fs::read_to_string(path).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    // Documents of a few kilobytes, as a corpus holds them: the texts
    // joined in tens.
    let docs: Vec<String> = (0..4).flat_map(|_| texts.chunks(10).map(|c| c.join("\n"))).collect();
    let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
    let bytes: usize = refs.iter().map(|d| d.len()).sum();
    let runs: u32 = std::env::var("TURBO_SPEED_RUNS").ok().and_then(|r| r.parse().ok()).unwrap_or(5);
    let a = SatAnnotator::open(&bundle(), &SatOptions::default()).unwrap();
    let tokens: usize = a.scores(&refs).iter().map(|r| r.as_ref().unwrap().0.len()).sum();
    let start = Instant::now();
    let mut sentences = 0;
    for _ in 0..runs {
        let mut out = Vec::new();
        for (d, r) in refs.iter().zip(a.scores(&refs)) {
            let (spans, scores) = r.unwrap();
            out.clear();
            a.sentences(d, &spans, &scores, &mut out);
            sentences += out.len();
        }
    }
    let secs = start.elapsed().as_secs_f64() / f64::from(runs);
    eprintln!(
        "{}: {} documents, {bytes} bytes, {tokens} tokens, block {} stride {}: {:.1} ms a pass, {:.0} tokens/s, {:.2} MB/s, {} sentences",
        bundle().display(),
        refs.len(),
        a.block_size(),
        a.stride(),
        secs * 1e3,
        tokens as f64 / secs,
        bytes as f64 / secs / 1e6,
        sentences / runs as usize,
    );
}
