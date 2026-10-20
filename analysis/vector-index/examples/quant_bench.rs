//! Recall and speed of TurboQuant storage against the exact index.
//!
//! `cargo run --release -p turbo-vector-index --example quant_bench --
//! VECTORS QUERY_EVERY K [DUMP]`. VECTORS is a text file of one vector a
//! line, a token then its values separated by spaces (the GloVe text
//! format), or `random:ROWS:DIM` for rows from a fixed LCG. Every QUERY_EVERY-th vector is held out as a query and the
//! rest are indexed, all scaled to unit length and compared by cosine.
//! With DUMP the indexed rows and the queries are also written there as
//! little-endian `f32` files, `rows.f32` and `queries.f32`, for another
//! program to time the same data.

use std::time::Instant;

use turbo_analysis::ModelRef;
use turbo_vector_index::{Estimator, IndexBuilder, Metric, Normalization, Space, TurboQuant};

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    v.iter().map(|&x| (f64::from(x) / n) as f32).collect()
}

fn random(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
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
    let every: usize = args[1].parse().unwrap();
    let k: usize = args[2].parse().unwrap();
    let (mut rows, mut queries) = (Vec::new(), Vec::new());
    let mut take = |i: usize, v: Vec<f32>| {
        if v.iter().any(|&x| x != 0.0) {
            if i.is_multiple_of(every) { queries.push(unit(&v)) } else { rows.push(unit(&v)) }
        }
    };
    if let Some(spec) = args[0].strip_prefix("random:") {
        let (n, dim) = spec.split_once(':').unwrap();
        let dim: usize = dim.parse().unwrap();
        for i in 0..n.parse().unwrap() {
            take(i, random(i as u64 + 1, dim));
        }
    } else {
        for (i, line) in std::fs::read_to_string(&args[0]).unwrap().lines().enumerate() {
            take(i, line.split(' ').skip(1).map(|x| x.parse().unwrap()).collect());
        }
    }
    let dim = rows[0].len();
    if let Some(dir) = args.get(3) {
        dump(&format!("{dir}/rows.f32"), &rows);
        dump(&format!("{dir}/queries.f32"), &queries);
    }
    let model = ModelRef { id: "bench".into(), revision: "1".into(), sha256: "0".into() };
    let space = Space::new(model, dim as u32, Normalization::Unit);
    let builder = || {
        let mut b = IndexBuilder::new(space.clone(), Metric::Cosine).unwrap();
        b.reserve(rows.len());
        for (i, r) in rows.iter().enumerate() {
            b.push(r, i as u32).unwrap();
        }
        b
    };
    let flat: Vec<f32> = queries.iter().flatten().copied().collect();
    let exact = builder().build();
    let truth = exact.top_k_batch(&space, &flat, k).unwrap();
    let time_queries = |run: &dyn Fn(&[f32]) -> u32| {
        let mut times = Vec::new();
        let mut check = 0u32;
        for _pass in 0..3 {
            for q in &queries {
                let t = Instant::now();
                check = check.wrapping_add(run(q));
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        (median(times), check)
    };
    let (ms, _) = time_queries(&|q| exact.top_k(&space, q, k).unwrap()[0].row);
    println!(
        "exact rows={} queries={} dim={dim} k={k} bytes_per_row={} query_ms={ms:.4}",
        rows.len(),
        queries.len(),
        dim * 4
    );
    for estimator in [Estimator::Mse, Estimator::InnerProduct] {
        for bits in 1..=4 {
            let config = TurboQuant::new(bits, estimator, 0x5eed).unwrap();
            let b = builder();
            let t = Instant::now();
            let index = b.build_quantized(config);
            let encode_s = t.elapsed().as_secs_f64();
            let got = index.top_k_batch(&space, &flat, k).unwrap();
            let found: usize = got
                .iter()
                .zip(&truth)
                .map(|(g, t)| g.iter().filter(|h| t.iter().any(|x| x.row == h.row)).count())
                .sum();
            let recall = found as f64 / (truth.len() * k) as f64;
            let (ms, _) = time_queries(&|q| index.top_k(&space, q, k).unwrap()[0].row);
            // The error of every estimated score of the first 20 queries.
            let (mut sum, mut squares, mut n) = (0.0f64, 0.0f64, 0.0f64);
            for q in queries.iter().take(20) {
                for hit in index.top_k(&space, q, rows.len()).unwrap() {
                    let r = &rows[hit.row as usize];
                    let exact: f64 = q.iter().zip(r).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
                    let e = f64::from(hit.score) - exact;
                    (sum, squares, n) = (sum + e, squares + e * e, n + 1.0);
                }
            }
            println!(
                "{estimator:?} bits={bits} bytes_per_row={} encode_rows_per_s={:.0} query_ms={ms:.4} recall@{k}={recall:.4} \
                 mean_error={:.6} mse={:.3e}",
                index.bytes_per_row(),
                rows.len() as f64 / encode_s,
                sum / n,
                squares / n
            );
        }
    }
}
