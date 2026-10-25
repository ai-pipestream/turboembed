//! Feedforward POS feature extraction for mixed-case real words.
use std::{hint::black_box, time::Instant};
use turbo_postag::neural::FeedforwardPosContext;

fn main() {
    let words = ["London", "london", "İstanbul", "ΟΔΟΣ", "\u{10400}", "ǅ", "*UNK*"];
    let expected = ["on", "on", "ul", "οσ", "\u{10428}", "ǆ", "k*"];
    for (index, suffix) in expected.into_iter().enumerate() {
        assert_eq!(FeedforwardPosContext::extract(&words, index, Some("NN"), Some("DT"))[5], suffix);
    }
    let start = Instant::now();
    for _ in 0..80_000 {
        for index in 0..words.len() {
            black_box(FeedforwardPosContext::extract(black_box(&words), index, Some("NN"), Some("DT")));
        }
    }
    println!("{:.3} ms; 560000 contexts", start.elapsed().as_secs_f64() * 1_000.0);
}
