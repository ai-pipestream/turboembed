//! Release timing on the ASCII sentences from the checked-in namefinder
//! goldens. Compile on each revision, then interleave the two binaries.
use std::{hint::black_box, time::Instant};
use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, Shape};
use turbo_namefind::{Pattern, RegexNameFinder, TokenNameFinder};

fn main() {
    let rounds: usize = std::env::args().nth(1).unwrap_or_else(|| "100".into()).parse().unwrap();
    let root = turbo_goldens::data_dir();
    let mut texts: Vec<String> = std::fs::read_to_string(root.join("inputs/namefind/sentences.txt"))
        .unwrap()
        .lines()
        .map(String::from)
        .collect();
    const COLS: &[ColumnSpec] =
        &[ColumnSpec::new("sentence", ColumnType::Str), ColumnSpec::new("token", ColumnType::Str)];
    let shape = Shape { id: "turbo:sentences".into(), anchor: GoldenAnchor::Document, columns: COLS };
    let g = Golden::read_with(root.join("namefind/generated/namefind/sentences.golden"), &[shape]).unwrap();
    let layer = g.layer("turbo:sentences").unwrap();
    let mut last = "";
    for (key, token) in layer.strs("sentence").unwrap().iter().zip(layer.strs("token").unwrap().iter()) {
        if key != last {
            texts.push(String::new());
            last = key;
        } else {
            texts.last_mut().unwrap().push(' ');
        }
        texts.last_mut().unwrap().push_str(token);
    }
    texts.retain(|s| s.is_ascii() && !s.ends_with(['\r', '\n']));
    let tokens: Vec<Vec<&str>> = texts.iter().map(|s| s.split(' ').collect()).collect();
    let finders: Vec<_> = [r"\b\d{4}\b", r"\b[a-z]+\b", r"\Bx", r"[a-z]+$", r"\d+\Z"]
        .into_iter()
        .map(|p| RegexNameFinder::new(vec![Pattern::java(p).unwrap()], "T").unwrap())
        .collect();
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
