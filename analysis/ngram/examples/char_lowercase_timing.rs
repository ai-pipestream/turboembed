//! NGramCharModel membership checks across ASCII and Unicode casing.
use std::{hint::black_box, time::Instant};
use turbo_ngram::NGramCharModel;

fn main() {
    let words = ["London", "london", "İstanbul", "ΟΔΟΣ", "\u{10400}", "ǅ", "unlisted"];
    let mut model = NGramCharModel::new();
    for word in &words[..6] {
        model.add(word);
    }
    let mut hits = 0;
    let start = Instant::now();
    for _ in 0..500_000 {
        for word in words {
            hits += usize::from(model.contains(black_box(word)));
        }
    }
    let elapsed = start.elapsed();
    assert_eq!(hits, 3_000_000);
    println!("{:.3} ms; {hits} matches", elapsed.as_secs_f64() * 1_000.0);
}
