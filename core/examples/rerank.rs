//! Compare a reranker bundle with its upstream logits and time repeated batches.
use std::{path::Path, time::Instant};
use turbo::{
    bundle::Bundle,
    rerank::{RerankOptions, Reranker},
    safetensors::{Dtype, File},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).ok_or("usage: rerank BUNDLE [THREADS] [REPEATS]")?);
    let threads = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1);
    let repeats = args.get(3).map(|s| s.parse::<usize>()).transpose()?.unwrap_or(3);
    let bundle = Bundle::open(dir)?;
    let pairs: Vec<_> =
        bundle.manifest.reference.cases.iter().map(|c| (c.text.as_str(), c.passage.as_deref().unwrap())).collect();
    let bytes = bundle.read_verified(&bundle.manifest.reference.file)?;
    let reference = File::parse("reference", &bytes)?.get("scores", Dtype::F32, 2)?.f32s();
    let mut reranker = Reranker::open(dir, threads)?;
    let result = reranker.score(&pairs, RerankOptions::default())?;
    let worst = result.scores.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!(
        "model={} threads={threads} pairs={} max_abs_error={worst:.9} scores={:?}",
        result.model.model_id,
        pairs.len(),
        result.scores
    );
    if worst > 1e-4 {
        return Err("scores exceed 1e-4 reference tolerance".into());
    }
    let start = Instant::now();
    for _ in 0..repeats {
        std::hint::black_box(reranker.score(&pairs, RerankOptions::default())?);
    }
    let secs = start.elapsed().as_secs_f64();
    println!("repeats={repeats} seconds={secs:.6} pairs_per_second={:.3}", repeats as f64 * pairs.len() as f64 / secs);
    Ok(())
}
