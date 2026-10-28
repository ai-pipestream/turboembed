//! Times loading a dictionary and stemming the vocabulary sample of its
//! language: `cargo run --release -p turbo-hunspell --example hunspell_bench
//! [en_US|de_DE_frami|hu_HU]`. Dictionaries come from `fetch-models`
//! (`TURBO_GOLDENS_MODELS` or `target/turbo-goldens/models`); samples are
//! the crate-local files under `tests/inputs/`.

use std::path::PathBuf;
use std::time::Instant;

use turbo_hunspell::{Dictionary, LoadMode, Stemmer};

fn published_model(file: &str) -> Option<PathBuf> {
    let candidates = [
        std::env::var_os("TURBO_GOLDENS_MODELS").map(PathBuf::from),
        std::env::var_os("CARGO_TARGET_DIR").map(|d| PathBuf::from(d).join("turbo-goldens/models")),
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/turbo-goldens/models")),
    ];
    for dir in candidates.into_iter().flatten() {
        let path = dir.join(file);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "en_US".into());
    let sample = match name.as_str() {
        "en_US" => "en_US",
        "de_DE_frami" => "de_DE_frami",
        "hu_HU" => "hu_HU",
        other => panic!("no sample for {other}"),
    };
    let (Some(affix), Some(words)) =
        (published_model(&format!("hunspell/{name}.aff")), published_model(&format!("hunspell/{name}.dic")))
    else {
        eprintln!("dictionaries for {name} not found; set TURBO_GOLDENS_MODELS or run fetch-models");
        return;
    };
    let t = Instant::now();
    let dictionary = Dictionary::open(&affix, &words, LoadMode::Strict).expect("the dictionary loads");
    let load = t.elapsed();
    let text =
        std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("tests/inputs/{sample}.txt")))
            .unwrap();
    let mut list: Vec<String> =
        text.lines().filter(|l| !l.is_empty() && !l.starts_with('#')).map(String::from).collect();
    // Capitalized and upper case forms take the case variant paths too.
    let capitalized: Vec<String> = list
        .iter()
        .map(|w| {
            let mut c = w.chars();
            c.next().map_or(String::new(), |f| f.to_uppercase().chain(c).collect())
        })
        .collect();
    list.extend(capitalized);
    let stemmer = Stemmer::new(&dictionary);
    let mut found = 0;
    for w in &list {
        found += usize::from(!stemmer.stems(w).is_empty());
    }
    let runs = 20;
    let t = Instant::now();
    for _ in 0..runs {
        for w in &list {
            std::hint::black_box(stemmer.stems(w));
        }
    }
    let elapsed = t.elapsed();
    let per_second = (runs * list.len()) as f64 / elapsed.as_secs_f64();
    let t = Instant::now();
    for w in &list {
        std::hint::black_box(stemmer.analyze(w));
    }
    let analyze = (list.len()) as f64 / t.elapsed().as_secs_f64();
    println!(
        "{name}: {} words loaded in {load:.2?}; {} inputs, {found} read; stems: {per_second:.0} words/s; analyze: {analyze:.0} words/s",
        dictionary.word_count(),
        list.len()
    );
}
