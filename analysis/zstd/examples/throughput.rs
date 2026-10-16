//! Times decoding: each argument is a pair `<original>=<compressed>`,
//! the compressed file made from the original by the `zstd` tool. Each
//! is decoded until a second has passed, checked against the original,
//! and the best time reported as megabytes of output per second. Run
//! with `cargo run --release -p turbo-zstd --example throughput --
//! text=text.zst ...`.

use std::time::{Duration, Instant};

fn main() {
    let pairs: Vec<String> = std::env::args().skip(1).collect();
    if pairs.is_empty() {
        eprintln!("usage: throughput <original>=<compressed> ...");
        std::process::exit(2);
    }
    for pair in &pairs {
        let (original, compressed) = pair.split_once('=').expect("<original>=<compressed>");
        let want = std::fs::read(original).unwrap_or_else(|e| panic!("{original}: {e}"));
        let src = std::fs::read(compressed).unwrap_or_else(|e| panic!("{compressed}: {e}"));
        let got = turbo_zstd::decompress(&src, want.len()).unwrap_or_else(|e| panic!("{compressed}: {e}"));
        assert!(got == want, "{compressed} does not decode to {original}");
        let mut best = Duration::MAX;
        let start = Instant::now();
        let mut runs = 0;
        while start.elapsed() < Duration::from_secs(1) || runs < 3 {
            let t = Instant::now();
            let out = turbo_zstd::decompress(&src, want.len()).unwrap();
            best = best.min(t.elapsed());
            std::hint::black_box(out);
            runs += 1;
        }
        let mb = want.len() as f64 / 1e6;
        println!(
            "{compressed}: {:.2} MB from {:.2} MB, {:.0} MB/s (best of {runs})",
            mb,
            src.len() as f64 / 1e6,
            mb / best.as_secs_f64()
        );
    }
}
