//! Release timing on the crate's own sentences: the five built-in
//! patterns and one token pattern. Compile on each revision, then
//! interleave the two binaries.
use std::{hint::black_box, time::Instant};
use turbo_namefind::{BuiltIn, Pattern, RegexNameFinder, Tok, TokenNameFinder};

fn main() {
    let rounds: usize = std::env::args().nth(1).unwrap_or_else(|| "100".into()).parse().unwrap();
    let mut texts: Vec<String> =
        include_str!("../tests/snapshot-data/sentences.txt").lines().map(String::from).collect();
    texts.retain(|s| s.is_ascii() && !s.ends_with(['\r', '\n']));
    let tokens: Vec<Vec<&str>> = texts.iter().map(|s| s.split(' ').collect()).collect();
    let mut finders = vec![RegexNameFinder::defaults(&BuiltIn::ALL)];
    finders.push(
        RegexNameFinder::new(
            vec![Pattern::tokens(vec![Tok::text("call").opt(), Tok::text("555-123-4567")]).unwrap()],
            "T",
        )
        .unwrap(),
    );
    let mut out = Vec::new();
    let mut found = 0;
    for f in &finders {
        for t in &tokens {
            out.clear();
            f.find(t, &mut out).unwrap();
        }
    }
    let start = Instant::now();
    for _ in 0..rounds {
        for f in &finders {
            for t in &tokens {
                out.clear();
                f.find(black_box(t), &mut out).unwrap();
                found += black_box(out.len());
            }
        }
    }
    let calls = rounds * tokens.len() * finders.len();
    println!(
        "sentences={} calls={calls} matches={found} ns_per_call={:.2}",
        tokens.len(),
        start.elapsed().as_nanos() as f64 / calls as f64
    );
}
