//! Release timing of the actual whitespace normalizer, in milliseconds.
use std::{hint::black_box, time::Instant};
use turbo_normalize::{Whitespace, apply};
fn main() {
    let text = "Dr. Smith\u{b}paid\u{85}10\u{a0}euros\u{3000}yesterday.\u{1c}Fine!\n".repeat(4096);
    let start = Instant::now();
    for _ in 0..1_000 {
        black_box(apply(&Whitespace::collapse(), black_box(&text)));
    }
    println!("normalize {:.3}", start.elapsed().as_secs_f64() * 1e3);
}
