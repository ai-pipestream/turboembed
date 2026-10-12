//! Throughput of the two detectors on 10 MB of synthetic text: a few
//! repeated lines, mostly ASCII with some accented and CJK text, a blank
//! line after every fourth. Run with
//! `cargo run --release -p turbo-sentdetect --example throughput`.

use std::hint::black_box;
use std::time::Instant;

use turbo_sentdetect::{WhitespaceMode, newline_spans, paragraph_spans};

const LINES: [&str; 4] = [
    "The quick brown fox jumps over the lazy dog, and then it runs away.\n",
    "  Indented line with trailing spaces and a tab\t  \r\n",
    "Une ligne en français, avec des accents: é, è, à, ç.\n",
    "日本語の文も少し含まれています。\u{3000}全角スペースも。\n\n",
];

fn main() {
    let mut text = String::with_capacity(10 << 20);
    'fill: loop {
        for line in LINES {
            if text.len() + line.len() > 10 << 20 {
                break 'fill;
            }
            text.push_str(line);
        }
    }
    let mb = text.len() as f64 / 1e6;
    let mut out = Vec::new();
    for (name, run) in [
        ("newline", newline_spans as fn(&str, WhitespaceMode, &mut Vec<_>)),
        ("paragraph", paragraph_spans as fn(&str, WhitespaceMode, &mut Vec<_>)),
    ] {
        for mode in [WhitespaceMode::Unicode, WhitespaceMode::Legacy] {
            let mut best = f64::INFINITY;
            for _ in 0..10 {
                out.clear();
                let t = Instant::now();
                run(black_box(&text), mode, &mut out);
                best = best.min(t.elapsed().as_secs_f64());
            }
            println!("{name:9} {mode:?}: {} spans, {:.0} MB/s", out.len(), mb / best);
        }
    }
}
