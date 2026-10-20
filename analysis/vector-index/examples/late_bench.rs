//! Times MaxSim reranking of every candidate of a token index.
//!
//! `cargo run --release -p turbo-vector-index --example late_bench --
//! DIM CANDIDATES DOC_TOKENS QUERY_TOKENS [DUMP]`. Token vectors are unit
//! length, from a fixed LCG; with DUMP the candidates and the query are
//! also written there as little-endian `f32` files, `docs.f32` and
//! `query.f32`, for another program to time the same data.

use std::time::Instant;

use turbo_analysis::ModelRef;
use turbo_vector_index::{Normalization, Space, TokenIndexBuilder};

fn tokens(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut v: Vec<f32> = (0..n * dim)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect();
    for t in v.chunks_mut(dim) {
        let len = t.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
        t.iter_mut().for_each(|x| *x = (f64::from(*x) / len) as f32);
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize| args[i].parse::<usize>().unwrap();
    let (dim, n, doc, nq) = (num(0), num(1), num(2), num(3));
    let docs: Vec<Vec<f32>> = (0..n).map(|i| tokens(i as u64 + 1, doc, dim)).collect();
    let query = tokens(1 << 40, nq, dim);
    if let Some(dir) = args.get(4) {
        let bytes: Vec<u8> = docs.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/docs.f32"), bytes).unwrap();
        let bytes: Vec<u8> = query.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/query.f32"), bytes).unwrap();
    }
    let model = ModelRef { id: "bench".into(), revision: "1".into(), sha256: "0".into() };
    let space = Space::new(model, dim as u32, Normalization::Unit);
    let mut b = TokenIndexBuilder::new(space.clone()).unwrap();
    for (i, d) in docs.iter().enumerate() {
        b.push(d, i as u32).unwrap();
    }
    let index = b.build();
    let all: Vec<u32> = (0..n as u32).collect();
    let mut times = Vec::new();
    let mut check = 0u64;
    for _ in 0..15 {
        let t = Instant::now();
        let hits = index.rerank(&space, &query, &all, 10).unwrap();
        times.push(t.elapsed().as_secs_f64());
        check = check.wrapping_add(u64::from(hits[0].candidate));
    }
    times.sort_by(f64::total_cmp);
    let median = times[times.len() / 2];
    println!(
        "rust dim={dim} candidates={n} doc_tokens={doc} query_tokens={nq} rerank_ms={:.3} candidates_per_s={:.0} check={check}",
        median * 1e3,
        n as f64 / median
    );
}
