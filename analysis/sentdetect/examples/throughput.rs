//! Throughput of the maxent sentence detector on 10 MB of synthetic
//! English text (a paragraph of varied sentences, repeated), with the
//! published English model in tests/models. It is measured twice: on
//! ASCII text, which the context generator walks by byte, and on the
//! same text with one accented word per paragraph, which takes the
//! general UTF-16 walk.
//!
//! ```text
//! cargo run --release -p turbo-sentdetect --example throughput
//! ```

use std::path::Path;
use std::time::Instant;

use turbo_maxent::ModelFile;
use turbo_sentdetect::{Scratch, SentenceDetector, SentenceModel, WhitespaceMode};

const PARAGRAPH: &str = "Mr. Smith arrived at the station at 9 a.m. on Monday. He was carrying a small brown \
    suitcase! Did Dr. Brown meet him on the platform? The budget was 2.5 million dollars, e.g. for the new \
    roof. Prices on the main index closed 0.3 percent higher. \"Stop,\" she said. They walked to the car \
    and drove away.\n\n";

fn main() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models/opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin");
    let model = SentenceModel::open(ModelFile::read(path).expect("the English model")).expect("a sentence model");
    let detector = SentenceDetector::new(model, WhitespaceMode::Unicode);
    run(&detector, "ASCII", PARAGRAPH);
    run(&detector, "non-ASCII", &PARAGRAPH.replace("the car", "the caf\u{e9}"));
}

fn run(detector: &SentenceDetector, what: &str, paragraph: &str) {
    let target = 10 * 1024 * 1024;
    let mut text = String::with_capacity(target + paragraph.len());
    while text.len() < target {
        text.push_str(paragraph);
    }

    let mut scratch = Scratch::new();
    let mut out = Vec::new();
    // One warm-up run grows the buffers, then the timed runs.
    detector.detect_with(&text, &mut scratch, &mut out);
    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        out.clear();
        detector.detect_with(&text, &mut scratch, &mut out);
    }
    let seconds = start.elapsed().as_secs_f64() / f64::from(runs);
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    println!("{what}: {mb:.1} MB, {} sentences, {:.1} ms per run, {:.1} MB/s", out.len(), seconds * 1e3, mb / seconds);
}
