//! Times top-k over random rows, one query at a time and as a batch.
//!
//! `cargo run --release -p turbo-vector-index --example index_bench --
//! DIM ROWS QUERIES K METRIC [DUMP]`, METRIC `dot` or `cosine`. Rows and
//! queries are unit length (so a dot index over them answers cosine), from
//! a fixed LCG; with DUMP the rows and queries are also written there as
//! little-endian `f32` files, `rows.f32` and `queries.f32`, for another
//! program to time the same data.

use std::time::Instant;

use turbo_analysis::ModelRef;
use turbo_vector_index::{IndexBuilder, Metric, Normalization, Space};

fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut v: Vec<f32> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect();
    let len = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    v.iter_mut().for_each(|x| *x = (f64::from(*x) / len) as f32);
    v
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn dump(path: &str, rows: &[Vec<f32>]) {
    let bytes: Vec<u8> = rows.iter().flatten().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize| args[i].parse::<usize>().unwrap();
    let (dim, n, nq, k) = (num(0), num(1), num(2), num(3));
    let metric = match args[4].as_str() {
        "dot" => Metric::Dot,
        "cosine" => Metric::Cosine,
        m => panic!("metric {m}"),
    };
    let rows: Vec<Vec<f32>> = (0..n).map(|i| values(i as u64 + 1, dim)).collect();
    let queries: Vec<Vec<f32>> = (0..nq).map(|i| values(1 << 40 | i as u64, dim)).collect();
    if let Some(dir) = args.get(5) {
        dump(&format!("{dir}/rows.f32"), &rows);
        dump(&format!("{dir}/queries.f32"), &queries);
    }
    let model = ModelRef { id: "bench".into(), revision: "1".into(), sha256: "0".into() };
    let space = Space::new(model, dim as u32, Normalization::Unit);
    let t = Instant::now();
    let mut b = IndexBuilder::new(space.clone(), metric).unwrap();
    b.reserve(n);
    for (i, r) in rows.iter().enumerate() {
        b.push(r, i as u32).unwrap();
    }
    let index = b.build();
    let build_ms = t.elapsed().as_secs_f64() * 1e3;

    let mut check = 0u64;
    let mut single = Vec::new();
    for _pass in 0..3 {
        for q in &queries {
            let t = Instant::now();
            let hits = index.top_k(&space, q, k).unwrap();
            single.push(t.elapsed().as_secs_f64() * 1e3);
            check = check.wrapping_add(u64::from(hits[0].row));
        }
    }
    let flat: Vec<f32> = queries.iter().flatten().copied().collect();
    let mut batch = Vec::new();
    for _pass in 0..7 {
        let t = Instant::now();
        let hits = index.top_k_batch(&space, &flat, k).unwrap();
        batch.push(t.elapsed().as_secs_f64() * 1e3 / nq as f64);
        check = check.wrapping_add(u64::from(hits[0][0].row));
    }
    println!(
        "rust dim={dim} rows={n} queries={nq} k={k} metric={} build_ms={build_ms:.1} single_ms={:.4} batch_ms_per_query={:.4} check={check}",
        args[4],
        median(single),
        median(batch)
    );
}
