//! Throughput of the three rule tokenizers on 10 MB of synthetic text.
//! Ignored by default; run with
//! `cargo test --release -p turbo-tokenize --test throughput -- --ignored --nocapture`.

use std::num::NonZeroU32;
use std::time::Instant;

use turbo_tokenize::{TokenizerOptions, simple_spans, whitespace_spans, word_tokens};

/// About `bytes` of text: mostly English words and punctuation, with
/// numbers, accented letters, CJK, emoji and line breaks mixed in, from a
/// fixed pseudo-random sequence so every run sees the same text.
fn synthetic(bytes: usize) -> String {
    const WORDS: [&str; 24] = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "Mr.",
        "Smith's",
        "e.g.",
        "3.14",
        "1,000",
        "caf\u{e9}",
        "na\u{ef}ve",
        "\u{65e5}\u{672c}\u{8a9e}",
        "\u{d55c}\u{ad6d}",
        "\u{1f600}",
        "don't",
        "U.S.",
        "state-of-the-art",
        "(see",
        "below)",
        "\u{201c}quoted\u{201d}",
    ];
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut out = String::with_capacity(bytes + 64);
    while out.len() < bytes {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push_str(WORDS[(state % WORDS.len() as u64) as usize]);
        out.push_str(match state >> 60 {
            0 => ".\n",
            1 => ", ",
            _ => " ",
        });
    }
    out
}

fn best_of<F: FnMut()>(mut run: F) -> f64 {
    (0..5)
        .map(|_| {
            let t = Instant::now();
            run();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
#[ignore]
fn throughput() {
    let text = synthetic(10 << 20);
    let mb = text.len() as f64 / 1e6;
    let options = TokenizerOptions::default();
    let mut spans = Vec::with_capacity(text.len() / 4);
    let mut types = Vec::with_capacity(text.len() / 4);
    let report = |name: &str, secs: f64, n: usize| println!("{name:>10}: {:7.0} MB/s, {n} tokens", mb / secs);

    let secs = best_of(|| {
        spans.clear();
        whitespace_spans(&text, options, &mut spans);
    });
    report("whitespace", secs, spans.len());

    let secs = best_of(|| {
        spans.clear();
        simple_spans(&text, options, &mut spans);
    });
    report("simple", secs, spans.len());

    let max = NonZeroU32::new(255).unwrap();
    let secs = best_of(|| {
        spans.clear();
        types.clear();
        word_tokens(&text, max, &mut spans, &mut types);
    });
    report("uax29", secs, spans.len());
}
