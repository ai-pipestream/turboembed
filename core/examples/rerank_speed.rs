//! Fixed short-pair batch for CPU comparisons against CrossEncoder and TEI.
use std::{path::Path, time::Instant};
use turbo::rerank::{RerankOptions, Reranker};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).ok_or("usage: rerank_speed BUNDLE")?;
    let pairs = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of France?", "Berlin is the capital of Germany."),
        ("What is the capital of France?", "Madrid is the capital of Spain."),
        ("What is the capital of France?", "Tokyo is the capital of Japan."),
    ];
    let mut r = Reranker::open(Path::new(&dir), 1)?;
    for _ in 0..3 {
        std::hint::black_box(r.score(&pairs, RerankOptions::default())?);
    }
    let mut times = Vec::new();
    for _ in 0..15 {
        let start = Instant::now();
        std::hint::black_box(r.score(&pairs, RerankOptions::default())?);
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(f64::total_cmp);
    println!("turbo batch=4 threads=1 median_seconds={} pairs_per_second={}", times[7], 4.0 / times[7]);
    Ok(())
}
