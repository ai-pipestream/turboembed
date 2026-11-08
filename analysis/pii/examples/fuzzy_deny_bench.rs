//! Times a deny list scan with fuzzy edits against the same terms matched
//! exactly (#1935).
//!
//! `cargo run --release -p turbo-pii --example fuzzy_deny_bench -- TERMS KB EDITS`:
//! TERMS terms of the form `Name0000son`, over KB kilobytes of prose drawn
//! from a fixed word list with a fixed LCG, at EDITS (`1` or `2`) maximum
//! edits. Defaults are `2000 110 2`. Both scans report their match counts.

use std::time::Instant;

use turbo_analysis::ViewId;
use turbo_pii::{CustomKinds, DenyList, PiiDetector};

const WORDS: [&str; 52] = [
    "the",
    "of",
    "and",
    "to",
    "in",
    "a",
    "is",
    "that",
    "for",
    "it",
    "as",
    "was",
    "with",
    "be",
    "by",
    "on",
    "not",
    "he",
    "this",
    "are",
    "or",
    "his",
    "from",
    "at",
    "which",
    "but",
    "have",
    "an",
    "had",
    "they",
    "you",
    "were",
    "their",
    "one",
    "all",
    "we",
    "can",
    "her",
    "has",
    "there",
    "been",
    "information",
    "government",
    "Johnson",
    "Nameson",
    "development",
    "Anderson",
    "relationship",
    "Name0007son",
    "Nane0042son",
    "Name0013sen",
    "Name13son",
];

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

fn prose(bytes: usize) -> String {
    let mut rng = Lcg(1935);
    let mut text = String::with_capacity(bytes + 16);
    let mut sentence = 0;
    while text.len() < bytes {
        let word = WORDS[(rng.next() % WORDS.len() as u64) as usize];
        if sentence == 0 {
            let mut chars = word.chars();
            if let Some(first) = chars.next() {
                text.extend(first.to_uppercase());
                text.push_str(chars.as_str());
            }
        } else {
            text.push_str(word);
        }
        sentence += 1;
        if sentence > 6 && rng.next() % 8 == 0 {
            text.push_str(". ");
            sentence = 0;
        } else {
            text.push(' ');
        }
    }
    text
}

fn detector(terms: &[String], edits: u8) -> PiiDetector {
    let kinds = CustomKinds::from_names(["person-name"]).expect("a valid kind name");
    let kind = kinds.find("person-name").expect("the kind just named");
    let deny = DenyList::from_terms(kind, terms).expect("non-blank terms").with_max_edits(edits).expect("0..=2 edits");
    PiiDetector::new(vec![]).with_custom_kinds(kinds).with_deny_list(deny).expect("a deny list of the table's kind")
}

fn time(d: &PiiDetector, text: &str) -> (f64, usize) {
    let mut best = f64::MAX;
    let mut found = 0;
    for _ in 0..5 {
        let t = Instant::now();
        found = d.layer(text, ViewId::SOURCE).mentions().count();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (best, found)
}

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().expect("a whole number")).collect();
    let terms_n = args.first().copied().unwrap_or(2000);
    let kb = args.get(1).copied().unwrap_or(110);
    let edits = args.get(2).copied().unwrap_or(2) as u8;
    let terms: Vec<String> = (0..terms_n).map(|i| format!("Name{i:04}son")).collect();
    let text = prose(kb * 1000);
    let words = text.split_whitespace().count();
    let (exact_ms, exact_n) = time(&detector(&terms, 0), &text);
    let (fuzzy_ms, fuzzy_n) = time(&detector(&terms, edits), &text);
    println!("{terms_n} terms, {} bytes, {words} words, max_edits {edits}", text.len());
    println!("exact: {exact_ms:.3} ms, {exact_n} matches");
    println!("fuzzy: {fuzzy_ms:.3} ms, {fuzzy_n} matches");
    println!("ratio: {:.1}x", fuzzy_ms / exact_ms.max(1e-9));
}
