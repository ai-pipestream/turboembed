//! Throughput of the lattice tokenizer with a published MeCab dictionary,
//! for a benchmark record. It prints what a record needs to be matched by
//! another tokenizer on the same input: the dictionary's manifest digest,
//! the text's SHA-256 and size, the morpheme count, and the best of five
//! timed runs after one warm run.
//!
//! ```text
//! cargo run --release -p turbo-tokenize --example lattice_throughput [ipadic|ko-dic] [readings]
//! ```
//!
//! With no dictionary named, both run. `readings` also times the
//! analyzer, which writes the readings layer, through a document. The
//! dictionaries are read from the directory `TURBO_MECAB_DICTIONARIES`
//! names, or `target/turbo-goldens/models` under the repository root, where
//! `cargo run -p turbo-goldens-tool -- fetch-models target/turbo-goldens/models`
//! puts them.
//!
//! The text is the golden inputs of the language (`analysis/goldens/data/
//! inputs/ja` or `ko`), each file in name order followed by a line feed,
//! repeated until it holds at least 10 MiB, so every run on every machine
//! reads the same bytes.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo_analysis::{Analyzer, Document};
use turbo_maxent::sha256;
use turbo_tokenize::lattice::{DictionaryFormat, Scratch};
use turbo_tokenize::{LatticeAnalyzer, LatticeTokenizer, MecabDictionary, WhitespaceMode};

const TEN_MIB: usize = 10 << 20;

struct Setup {
    name: &'static str,
    dir: &'static str,
    inputs: &'static str,
    format: DictionaryFormat,
}

const SETUPS: [Setup; 2] = [
    Setup { name: "ipadic", dir: "mecab-ipadic-2.7.0-20070801.utf8", inputs: "ja", format: DictionaryFormat::Ipadic },
    Setup { name: "ko-dic", dir: "mecab-ko-dic-2.1.1-20180720", inputs: "ko", format: DictionaryFormat::MecabKoDic },
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let readings = args.iter().any(|a| a == "readings");
    let named: Vec<&String> = args.iter().filter(|a| *a != "readings").collect();
    let dir = match std::env::var_os("TURBO_MECAB_DICTIONARIES") {
        Some(d) => PathBuf::from(d),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/turbo-goldens/models"),
    };
    for setup in SETUPS.iter().filter(|s| named.is_empty() || named.iter().any(|n| *n == s.name)) {
        run(setup, &dir, readings);
    }
}

/// The golden inputs of `set`, in name order, repeated to at least 10 MiB.
fn text(set: &str) -> String {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/inputs").join(set);
    let mut names: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    names.sort();
    let mut one = String::new();
    for p in &names {
        one.push_str(&std::fs::read_to_string(p).unwrap());
        one.push('\n');
    }
    let mut text = String::with_capacity(TEN_MIB + one.len());
    while text.len() < TEN_MIB {
        text.push_str(&one);
    }
    text
}

/// The best of five runs after a warm one, in seconds.
fn best_of(mut f: impl FnMut()) -> f64 {
    f();
    (0..5)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn run(setup: &Setup, dir: &Path, readings: bool) {
    let path = dir.join(setup.dir);
    let t = Instant::now();
    let dictionary = MecabDictionary::load(&path, WhitespaceMode::Unicode)
        .unwrap_or_else(|e| panic!("{}: {e} (run turbo-goldens fetch-models)", path.display()));
    let load = t.elapsed().as_secs_f64();
    let tokenizer = LatticeTokenizer::new(dictionary, WhitespaceMode::Unicode);
    let text = text(setup.inputs);
    println!(
        "dictionary: {} ({}), manifest sha256 {}",
        setup.name,
        setup.dir,
        tokenizer.dictionary().sha256().to_hex()
    );
    println!("text: inputs/{}, {} bytes, sha256 {}", setup.inputs, text.len(), sha256(text.as_bytes()).to_hex());
    println!("load: {load:.3} s");

    let mut scratch = Scratch::new();
    let mut out = Vec::new();
    let secs = best_of(|| {
        out.clear();
        tokenizer.analyze_with(black_box(&text), &mut scratch, &mut out).unwrap();
    });
    let mb = text.len() as f64 / 1e6;
    println!(
        "analyze: {} morphemes, {secs:.4} s, {:.2} MB/s, {:.0} morphemes/s",
        out.len(),
        mb / secs,
        out.len() as f64 / secs
    );

    if readings {
        let analyzer = LatticeAnalyzer::new(tokenizer, setup.format).unwrap();
        let pipeline = Analyzer::builder().add(analyzer).build().unwrap();
        let secs = best_of(|| {
            let mut docs = vec![Document::new(text.as_str())];
            pipeline.run(&mut docs)[0].clone().unwrap();
            black_box(&docs);
        });
        println!("analyzer with readings: {secs:.4} s, {:.2} MB/s", mb / secs);
    }
}
