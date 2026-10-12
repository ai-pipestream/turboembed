//! Normalization throughput on 1 MB of mixed text. Ignored by default; run
//! it in release mode to measure:
//!
//! `cargo test --release -p turbo-unicode --test bench -- --ignored --nocapture`

use std::hint::black_box;
use std::time::Instant;

use turbo_unicode::normalize::{Form, normalize};

/// A paragraph mixing ASCII, precomposed and decomposed Latin, Greek,
/// Cyrillic, CJK, Hangul syllables and jamo, compatibility characters and
/// emoji.
const SAMPLE: &str = "The quick brown fox jumps over the lazy dog, 1234567890. \
    Caf\u{E9} na\u{EF}ve r\u{E9}sum\u{E9} and cafe\u{301} nai\u{308}ve re\u{301}sume\u{301}. \
    \u{3A4}\u{3BF} \u{3B3}\u{3C1}\u{3AE}\u{3B3}\u{3BF}\u{3C1}\u{3BF} \u{3BA}\u{3B1}\u{3C6}\u{3AD}. \
    \u{411}\u{44B}\u{441}\u{442}\u{440}\u{430}\u{44F} \u{43B}\u{438}\u{441}\u{430}. \
    \u{6F22}\u{5B57}\u{3068}\u{304B}\u{306A}\u{3002} \u{D55C}\u{AD6D}\u{C5B4} \u{1112}\u{1161}\u{11AB}. \
    \u{FB01}ne \u{FF21}\u{FF22}\u{FF23} \u{2460} \u{33A1} \u{1F44D}\u{1F3FD} \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}\n";

fn mixed_text(bytes: usize) -> String {
    let mut text = String::with_capacity(bytes + SAMPLE.len());
    while text.len() < bytes {
        text.push_str(SAMPLE);
    }
    text
}

#[test]
#[ignore]
fn normalize_one_megabyte_of_mixed_text() {
    let text = mixed_text(1 << 20);
    let megabytes = text.len() as f64 / (1 << 20) as f64;
    let mut out = String::with_capacity(text.len() * 2);
    for form in [Form::Nfc, Form::Nfd, Form::Nfkc, Form::Nfkd] {
        let rounds = 20;
        let start = Instant::now();
        for _ in 0..rounds {
            out.clear();
            normalize(form, black_box(&text), &mut out);
            black_box(&out);
        }
        let seconds = start.elapsed().as_secs_f64() / f64::from(rounds);
        println!(
            "{form:?}: {:.2} ms per MB, {:.0} MB/s ({} bytes in, {} bytes out)",
            seconds * 1000.0 / megabytes,
            megabytes / seconds,
            text.len(),
            out.len()
        );
    }
    let ascii: String = "plain ascii text only. ".repeat((1 << 20) / 23);
    let start = Instant::now();
    out.clear();
    normalize(Form::Nfc, black_box(&ascii), &mut out);
    println!("Nfc, ASCII only: {:.3} ms per MB", start.elapsed().as_secs_f64() * 1000.0);
}
