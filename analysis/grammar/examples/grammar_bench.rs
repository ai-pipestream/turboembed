//! Times the English grammar over a 100 KB text:
//! `cargo run --release -p turbo-grammar --example grammar_bench`.

#[path = "../tests/common/sample.rs"]
mod sample;

use std::time::Instant;

use turbo_analysis::{Layer, ViewId};
use turbo_grammar::engine::Lexed;
use turbo_grammar::{ENGLISH, Grammar, GrammarAnnotator, Output, Parser};

fn main() {
    let text = sample::sample(100 * 1024);
    let annotator = GrammarAnnotator::english();
    // A first run fills the parser pool and the lexicon.
    let found = annotator.find(&text, ViewId::SOURCE);
    let runs = 50;
    let t = Instant::now();
    for _ in 0..runs {
        std::hint::black_box(annotator.find(&text, ViewId::SOURCE));
    }
    let per = t.elapsed() / runs;
    let mut lexed = Lexed::default();
    let t = Instant::now();
    for _ in 0..runs {
        lexed.lex(&text, &ENGLISH.style());
    }
    let lex = t.elapsed() / runs;
    let mut parser = Parser::new(&ENGLISH);
    let t = Instant::now();
    for _ in 0..runs {
        parser.parse(&text);
    }
    let parse = t.elapsed() / runs;
    let mut out = Vec::new();
    parser.select(&text, Output::Number, &mut out);
    println!(
        "{} bytes, {} lexemes, {} chart nodes: {} numbers, {} percentages, {} ordinals",
        text.len(),
        parser.lexed().len(),
        parser.chart().nodes.len(),
        found.numbers.len(),
        found.percents.len(),
        found.ordinals.len()
    );
    println!(
        "lex {lex:?}, lex and parse {parse:?}, both and all three layers {per:?} per run ({:.1} MB/s)",
        text.len() as f64 / per.as_secs_f64() / 1e6
    );
}
