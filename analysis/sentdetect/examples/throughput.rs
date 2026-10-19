//! Throughput of the detectors, in three sections:
//!
//! - `rules`: the newline and paragraph detectors on 10 MB of synthetic
//!   text (a few repeated lines, mostly ASCII with some accented and CJK
//!   text, a blank line after every fourth), in both whitespace modes.
//! - `maxent`: the maxent detector with the published English model on
//!   10 MB of synthetic English (a paragraph of varied sentences,
//!   repeated), once all ASCII, which the context generator
//!   walks by byte, and once with one accented word per paragraph, which
//!   takes the general UTF-16 walk.
//! - `no-whitespace`: the maxent detector on texts with end characters
//!   and no whitespace at all (`A.` and `x.Y` repeated), where every word
//!   reaches back to the start of the text or on to its end.
//!
//! ```text
//! cargo run --release -p turbo-sentdetect --example throughput [section]
//! ```
//!
//! With no section named, all three run. The English model,
//! `opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin`, is trained on a CC BY-SA
//! treebank and not checked in: it is read from the directory
//! `TURBO_GOLDENS_MODELS` names, or `target/turbo-goldens/models` under the
//! repository root, where
//! `cargo run -p turbo-goldens-tool -- fetch-models target/turbo-goldens/models`
//! puts it.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo_maxent::ModelFile;
use turbo_sentdetect::{Scratch, SentenceDetector, SentenceModel, WhitespaceMode, newline_spans, paragraph_spans};

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
    if wants("maxent") || wants("no-whitespace") {
        let dir = match std::env::var_os("TURBO_GOLDENS_MODELS") {
            Some(d) => PathBuf::from(d),
            None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/turbo-goldens/models"),
        };
        let path = dir.join("opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin");
        let file = ModelFile::read(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (download it with turbo-goldens-tool fetch-models)", path.display()));
        let model = SentenceModel::open(file).expect("a sentence model");
        let detector = SentenceDetector::new(model, WhitespaceMode::Unicode);
        if wants("maxent") {
            maxent(&detector, "ASCII", &repeat(PARAGRAPH, TEN_MB), 5);
            maxent(&detector, "non-ASCII", &repeat(&PARAGRAPH.replace("the car", "the caf\u{e9}"), TEN_MB), 5);
        }
        if wants("no-whitespace") {
            let sizes = std::env::var("NO_WHITESPACE_SIZES").unwrap_or_else(|_| "80000,10485760".into());
            for size in sizes.split(',').map(|s| s.parse::<usize>().expect("a size in bytes")) {
                maxent(&detector, "A. repeated", &repeat("A.", size), 1);
                maxent(&detector, "x.Y repeated", &repeat("x.Y", size), 1);
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
            println!("{name:9} {mode:?}: {mb:.1} MB, {} spans, {:.0} MB/s", out.len(), mb / best);
        }
    }
}

fn maxent(detector: &SentenceDetector, what: &str, text: &str, runs: u32) {
    let mut scratch = Scratch::new();
    let mut out = Vec::new();
    // One warm-up run grows the buffers, then the timed runs.
    detector.detect_with(text, &mut scratch, &mut out);
    let start = Instant::now();
    for _ in 0..runs {
        out.clear();
        detector.detect_with(black_box(text), &mut scratch, &mut out);
    }
    let seconds = start.elapsed().as_secs_f64() / f64::from(runs);
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    println!(
        "maxent {what}: {mb:.2} MB, {} sentences, {:.1} ms per run, {:.1} MB/s",
        out.len(),
        seconds * 1e3,
        mb / seconds
    );
}
