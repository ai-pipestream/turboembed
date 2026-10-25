//! Case-insensitive dictionary lookups with ASCII and Unicode tokens.
use std::{hint::black_box, time::Instant};
use turbo_maxent::dictionary::Dictionary;

fn main() {
    let words = ["London", "london", "İstanbul", "ΟΔΟΣ", "\u{10400}", "ǅ", "unlisted"];
    let entries: Vec<Vec<String>> = words[..6].iter().map(|w| vec![w.to_string()]).collect();
    let dictionary = Dictionary::from_entries(false, entries.iter().map(Vec::as_slice));
    let mut scratch = String::new();
    let mut hits = 0;
    let start = Instant::now();
    for _ in 0..500_000 {
        for word in words {
            hits += usize::from(dictionary.contains_with(black_box(word), &mut scratch));
        }
    }
    let elapsed = start.elapsed();
    assert_eq!(hits, 3_000_000);
    println!("{:.3} ms; {hits} matches", elapsed.as_secs_f64() * 1_000.0);
}
