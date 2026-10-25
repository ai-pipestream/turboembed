//! Release preprocessing timing probe; results are elapsed milliseconds.
use std::{hint::black_box, time::Instant};
fn main() {
    let text = "A sentence. \t\u{1c}\u{85}\u{a0}Another sentence.\u{2000}\u{3000}Final sentence.\n".repeat(64);
    let boundaries: Vec<u32> = text.match_indices('.').map(|(i, _)| (i + 1) as u32).collect();
    let mut out = Vec::new();
    let start = Instant::now();
    for _ in 0..100_000 {
        out.clear();
        turbo_sat::cuts::cut_points(black_box(&text), black_box(&boundaries), &mut out);
        black_box(&out);
    }
    println!("sat {:.3}", start.elapsed().as_secs_f64() * 1e3);
}
