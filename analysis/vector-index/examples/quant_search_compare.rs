//! Real input comparison: no generated vectors. Build with turboquant.
//! prepare VECTORS.f32 DIM QUERY_STRIDE OUT creates normalized splits/truth.
//! run DIR DIM BITS CALIBRATE staged|exhaustive measures one fresh index.
//! Run only in a coordinated timing slot; see reference/quant_search_compare.mjs.
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use turbo_analysis::ModelRef;
use turbo_vector_index::{IndexBuilder, Metric, Normalization, QuantizedSearch, Space, TurboQuant};

fn floats(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty());
    chunks.iter().map(|&b| f32::from_le_bytes(b)).collect()
}
fn write_floats(path: &Path, values: &[f32]) {
    std::fs::write(path, values.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>()).unwrap();
}
fn space(dim: usize) -> Space {
    Space::new(
        ModelRef { id: "comparison-input".into(), revision: "1".into(), sha256: String::new() },
        dim as u32,
        Normalization::Unit,
    )
}
fn builder(space: &Space, rows: &[f32]) -> IndexBuilder<u32> {
    let mut b = IndexBuilder::new(space.clone(), Metric::Cosine).unwrap();
    b.reserve(rows.len() / space.dim as usize);
    for (i, row) in rows.chunks_exact(space.dim as usize).enumerate() {
        b.push(row, i as u32).unwrap();
    }
    b
}
fn rss() -> String {
    // Whole-process resident memory, including allocator-retained arenas.
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find_map(|l| l.strip_prefix("VmRSS:").map(str::to_owned)))
        .and_then(|s| s.split_whitespace().next()?.parse::<u64>().ok())
        .map_or("null".into(), |k| (k * 1024).to_string())
}
fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}
fn prepare(args: &[String]) {
    let dim: usize = args[1].parse().unwrap();
    let stride: usize = args[2].parse().unwrap();
    assert!(stride > 1 && dim > 0);
    let input = floats(Path::new(&args[0]));
    assert!(input.len().is_multiple_of(dim));
    let (mut rows, mut queries, mut zero) = (Vec::new(), Vec::new(), 0usize);
    for (i, row) in input.chunks_exact(dim).enumerate() {
        assert!(row.iter().all(|x| x.is_finite()));
        let norm = row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        if norm == 0.0 {
            zero += 1;
            continue;
        }
        let out = if i % stride == 0 { &mut queries } else { &mut rows };
        out.extend(row.iter().map(|&x| (f64::from(x) / norm) as f32));
    }
    let space = space(dim);
    let truth = builder(&space, &rows).build().top_k_batch(&space, &queries, 10).unwrap();
    let out = Path::new(&args[3]);
    std::fs::create_dir_all(out).unwrap();
    write_floats(&out.join("rows.f32"), &rows);
    write_floats(&out.join("queries.f32"), &queries);
    std::fs::write(out.join("truth.u32"), truth.iter().flatten().flat_map(|h| h.row.to_le_bytes()).collect::<Vec<_>>())
        .unwrap();
    println!(
        "{{\"rows\":{},\"queries\":{},\"dim\":{dim},\"query_stride\":{stride},\"zero_rows_skipped\":{zero},\"k\":10}}",
        rows.len() / dim,
        queries.len() / dim
    );
}
fn run(args: &[String]) {
    let dir = Path::new(&args[0]);
    let dim: usize = args[1].parse().unwrap();
    let bits: u8 = args[2].parse().unwrap();
    let calibrate: bool = args[3].parse().unwrap();
    let (name, mode) = match args[4].as_str() {
        "staged" => ("staged", QuantizedSearch::Staged),
        "exhaustive" => ("exhaustive", QuantizedSearch::Exhaustive),
        _ => panic!("choose staged or exhaustive"),
    };
    let rows = floats(&dir.join("rows.f32"));
    let queries = floats(&dir.join("queries.f32"));
    let truth = std::fs::read(dir.join("truth.u32")).unwrap();
    let (truth, rest) = truth.as_chunks::<4>();
    assert!(rest.is_empty());
    let truth: Vec<u32> = truth.iter().map(|&b| u32::from_le_bytes(b)).collect();
    let nq = queries.len() / dim;
    assert!(rows.len().is_multiple_of(dim) && queries.len().is_multiple_of(dim) && nq > 0);
    assert_eq!(truth.len(), nq * 10);
    let n = rows.len() / dim;
    assert!(n >= 32_768, "comparison must exercise staging");
    let space = space(dim);
    let before = rss();
    let index = builder(&space, &rows).build_quantized(TurboQuant::new(bits, calibrate).unwrap()).unwrap();
    drop(rows);
    let constructed = rss();
    let found = index.top_k_batch_with_search(&space, &queries, 10, mode).unwrap();
    let hits = found
        .iter()
        .zip(truth.as_chunks::<10>().0)
        .map(|(r, t)| r.iter().filter(|h| t.contains(&h.row)).count())
        .sum::<usize>();
    let ids: Vec<u32> = found.iter().flatten().map(|h| h.row).collect();
    for _ in 0..3 {
        for q in queries.chunks_exact(dim) {
            black_box(index.top_k_with_search(&space, q, 10, mode).unwrap());
        }
        black_box(index.top_k_batch_with_search(&space, &queries, 10, mode).unwrap());
    }
    let warm = rss();
    let mut single = Vec::new();
    for _ in 0..3 {
        for q in queries.chunks_exact(dim) {
            let t = Instant::now();
            black_box(index.top_k_with_search(&space, q, 10, mode).unwrap());
            single.push(t.elapsed().as_secs_f64() * 1000.0);
        }
    }
    let batch = (0..5)
        .map(|_| {
            let t = Instant::now();
            black_box(index.top_k_batch_with_search(&space, &queries, 10, mode).unwrap());
            t.elapsed().as_secs_f64() * 1000.0 / nq as f64
        })
        .collect();
    println!(
        "{{\"mode\":\"{name}\",\"bits\":{bits},\"calibrate\":{calibrate},\"rows\":{n},\"queries\":{nq},\"dim\":{dim},\"k\":10,\"recall_at_10\":{},\"query_ms\":{},\"batch_ms_per_query\":{},\"rss_before_build\":{before},\"rss_after_build\":{constructed},\"rss_warm\":{warm},\"ids\":{ids:?}}}",
        hits as f64 / (nq * 10) as f64,
        median(single),
        median(batch)
    );
}
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args[0].as_str() {
        "prepare" => prepare(&args[1..]),
        "run" => run(&args[1..]),
        _ => panic!("prepare or run"),
    }
}
