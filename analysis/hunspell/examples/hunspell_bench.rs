//! Times loading a dictionary and stemming the Snowball vocabulary sample
//! of its language (the `stem` input set):
//! `cargo run --release -p turbo-hunspell --example hunspell_bench [en_US|de_DE_frami|hu_HU]`.
//! The dictionaries are where `turbo-goldens fetch-models` puts them.

use std::time::Instant;

use turbo_hunspell::{Dictionary, LoadMode, Stemmer};

fn main() {
    let name = std::env::args().nth(1).unwrap_or_else(|| "en_US".into());
    let sample = match name.as_str() {
        "en_US" => "english",
        "de_DE_frami" => "german",
        "hu_HU" => "hungarian",
        other => panic!("no sample for {other}"),
    };
    let (Some(affix), Some(words)) = (
        turbo_goldens::published_model(&format!("hunspell/{name}.aff")),
        turbo_goldens::published_model(&format!("hunspell/{name}.dic")),
    ) else {
        return;
    };
    let t = Instant::now();
    let dictionary = Dictionary::open(&affix, &words, LoadMode::Strict).expect("the dictionary loads");
    let load = t.elapsed();
    let text = std::fs::read_to_string(turbo_goldens::data_dir().join(format!("inputs/stem/{sample}.txt"))).unwrap();
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
