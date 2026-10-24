//! Release timing on caller-supplied text; no generated document corpus.
use std::{hint::black_box, time::Instant};
use turbo_analysis::ViewId;
use turbo_structure::{Syntax, detect};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "block_bench TEXT markdown|plain ITERATIONS");
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let syntax = Syntax::from_name(&args[2]).unwrap();
    let iterations: usize = args[3].parse().unwrap();
    assert!(iterations > 0);
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
    times.sort_by(f64::total_cmp);
    println!("{} {} {:.9}", text.len(), iterations, times[times.len() / 2]);
}
