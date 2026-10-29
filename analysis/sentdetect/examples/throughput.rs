//! Throughput of native detectors; no external model is needed.
//!
//! `rules` measures newline/paragraph extraction on multilingual text;
//! `sentences` measures English rule segmentation on ASCII/accented text;
//! `no-whitespace` exercises repeated terminators without spaces.
//! With no section named, all three run.
//!
//! ```text
//! cargo run --release -p turbo-sentdetect --example throughput [section]
//! ```

use std::hint::black_box;
use std::time::Instant;

use turbo_sentdetect::{RuleScratch, RuleSplitter, newline_spans, paragraph_spans};

const TEN_MB: usize = 10 << 20;

const LINES: [&str; 4] = [
    "The quick brown fox jumps over the lazy dog, and then it runs away.\n",
    "  Indented line with trailing spaces and a tab\t  \r\n",
    "Une ligne en français, avec des accents: é, è, à, ç.\n",
    "日本語の文も少し含まれています。\u{3000}全角スペースも。\n\n",
];

const PARAGRAPH: &str = "Mr. Smith arrived at the station at 9 a.m. on Monday. He was carrying a small brown \
    suitcase! Did Dr. Brown meet him on the platform? The budget was 2.5 million dollars, e.g. for the new \
    roof. Prices on the main index closed 0.3 percent higher. \"Stop,\" she said. They walked to the car \
    and drove away.\n\n";

fn main() {
    let section = std::env::args().nth(1);
    let wants = |name: &str| section.as_deref().is_none_or(|s| s == name);
    if wants("rules") {
        rules();
    }
    if wants("sentences") || wants("no-whitespace") {
        let detector = RuleSplitter::new("en");
        if wants("sentences") {
            sentences(&detector, "ASCII", &repeat(PARAGRAPH, TEN_MB), 5);
            sentences(&detector, "non-ASCII", &repeat(&PARAGRAPH.replace("the car", "the café"), TEN_MB), 5);
        }
        if wants("no-whitespace") {
            let sizes = std::env::var("NO_WHITESPACE_SIZES").unwrap_or_else(|_| "80000,10485760".into());
            for size in sizes.split(',').map(|s| s.parse::<usize>().expect("a size in bytes")) {
                sentences(&detector, "A. repeated", &repeat("A.", size), 1);
                sentences(&detector, "x.Y repeated", &repeat("x.Y", size), 1);
            }
        }
    }
}

/// `unit` repeated while the text stays within `size` bytes.
fn repeat(unit: &str, size: usize) -> String {
    unit.repeat(size / unit.len())
}

fn rules() {
    let mut text = String::with_capacity(TEN_MB);
    'fill: loop {
        for line in LINES {
            if text.len() + line.len() > TEN_MB {
                break 'fill;
            }
            text.push_str(line);
        }
    }
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    let mut out = Vec::new();
    for (name, run) in
        [("newline", newline_spans as fn(&str, &mut Vec<_>)), ("paragraph", paragraph_spans as fn(&str, &mut Vec<_>))]
    {
        let mut best = f64::INFINITY;
        for _ in 0..10 {
            out.clear();
            let t = Instant::now();
            run(black_box(&text), &mut out);
            best = best.min(t.elapsed().as_secs_f64());
        }
        println!("{name:9}: {mb:.1} MB, {} spans, {:.0} MB/s", out.len(), mb / best);
    }
}

fn sentences(detector: &RuleSplitter, what: &str, text: &str, runs: u32) {
    let mut scratch = RuleScratch::default();
    let mut out = Vec::new();
    // One warm-up run grows the buffers, then the timed runs.
    detector.split_with(text, &mut scratch, &mut out);
    let start = Instant::now();
    for _ in 0..runs {
        out.clear();
        detector.split_with(black_box(text), &mut scratch, &mut out);
    }
    let seconds = start.elapsed().as_secs_f64() / f64::from(runs);
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    println!(
        "rules {what}: {mb:.2} MB, {} sentences, {:.1} ms per run, {:.1} MB/s",
        out.len(),
        seconds * 1e3,
        mb / seconds
    );
}
