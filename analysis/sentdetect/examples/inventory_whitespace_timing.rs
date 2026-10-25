//! Release timing of the actual abbreviation inventory judge, in milliseconds.
use std::{hint::black_box, time::Instant};
use turbo_sentdetect::inventory::AbbreviationInventory;
fn main() {
    let text = "Dr.\u{b}Smith met Prof.\u{a0}Brown at 10 p.m.\u{3000}He left.\u{85}No. 5!\n".repeat(64);
    let stops: Vec<_> = text.match_indices('.').map(|(i, _)| i).collect();
    let inventory = AbbreviationInventory::english();
    let start = Instant::now();
    for _ in 0..10_000 {
        for &stop in &stops {
            black_box(inventory.judge(black_box(&text), black_box(stop)));
        }
    }
    println!("inventory {:.3}", start.elapsed().as_secs_f64() * 1e3);
}
