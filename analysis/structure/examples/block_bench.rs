//! Release timing on caller-supplied text; no generated document corpus.
use std::{hint::black_box, time::Instant};
use turbo_analysis::ViewId;
use turbo_analysis_layers::Block;
use turbo_structure::{Syntax, detect};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(matches!(args.len(), 4 | 5), "block_bench TEXT markdown|plain ITERATIONS [OUTPUT]");
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let syntax = Syntax::from_name(&args[2]).unwrap();
    let iterations: usize = args[3].parse().unwrap();
    let output = detect(&text, syntax, ViewId::SOURCE);
    if let Some(path) = args.get(4) {
        let mut rows = String::new();
        for (index, span) in output.spans.iter() {
            let i = index.get();
            let range = span.range();
            let parent = output.parent[i].map(|p| p.get());
            use std::fmt::Write;
            writeln!(rows, "{:?}\t{}\t{}\t{parent:?}\t{}", output.kind[i], range.start, range.end, output.depth[i])
                .unwrap();
        }
        std::fs::write(path, rows).unwrap();
    }
    println!("blocks {} tables {}", output.kind.len(), output.kind.iter().filter(|&&k| k == Block::Table).count());
    if iterations == 0 {
        assert!(args.get(4).is_some(), "zero iterations requires an output path");
        return;
    }
    let mut times = Vec::new();
    for i in 0..iterations + 3 {
        let start = Instant::now();
        let tree = detect(black_box(&text), syntax, ViewId::SOURCE);
        black_box(&tree);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if i >= 3 {
            times.push(elapsed);
        }
    }
    for (sample, ms) in times.iter().enumerate() {
        println!("sample_ms {sample} {ms:.9}");
    }
    times.sort_by(f64::total_cmp);
    println!("{} {} {:.9}", text.len(), iterations, times[times.len() / 2]);
}
