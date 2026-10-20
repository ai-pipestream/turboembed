//! Repeated validation of a synthetic dependency forest, excluding construction.
use std::{hint::black_box, time::Instant};
use turbo_analysis::{Ix, Layer, Tag, TagSet};
use turbo_analysis_layers::{Dependencies, Tokens};

fn main() {
    let n: usize = std::env::args().nth(1).unwrap_or("1000000".into()).parse().unwrap();
    let width: usize = std::env::args().nth(2).unwrap_or("25".into()).parse().unwrap();
    assert!(n > 0 && width > 0 && n < u32::MAX as usize);
    let layer = Dependencies::<Tokens> {
        head: (0..n)
            .map(|i| if (i + 1) % width == 0 || i + 1 == n { None } else { Some(Ix::new((i + 1) as u32)) })
            .collect(),
        relation: vec![Tag(0); n],
        relations: TagSet::new("dep", &["dep"]),
    };
    for _ in 0..20 {
        black_box(black_box(&layer).check()).unwrap();
    }
    let mut timings = Vec::new();
    for _ in 0..41 {
        let start = Instant::now();
        black_box(black_box(&layer).check()).unwrap();
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    timings.sort_by(f64::total_cmp);
    println!("n={n} width={width} median_ms={:.6} p10_ms={:.6}", timings[20], timings[4]);
}
