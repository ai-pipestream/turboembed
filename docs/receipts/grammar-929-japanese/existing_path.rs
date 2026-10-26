#[path = "../tests/common/sample.rs"]
mod sample;
use std::time::Instant;
use turbo_grammar::{ENGLISH, ENGLISH_UNITS, Grammar, Output, Parser};
fn main() {
    let text = sample::sample(100 * 1024);
    let mut proof = String::new();
    for grammar in [&ENGLISH as &dyn Grammar, &ENGLISH_UNITS as &dyn Grammar] {
        let mut parser = Parser::new(grammar);
        let mut values = Vec::new();
        for _ in 0..3 {
            parser.parse(&text);
            values.clear();
            for output in [Output::Number, Output::Ordinal, Output::Percent, Output::Quantity] {
                parser.select(&text, output, &mut values);
            }
        }
        proof.push_str(&format!("{}:{values:?}\n", grammar.name()));
        let start = Instant::now();
        for _ in 0..20 {
            parser.parse(std::hint::black_box(&text));
            values.clear();
            for output in [Output::Number, Output::Ordinal, Output::Percent, Output::Quantity] {
                parser.select(&text, output, &mut values);
            }
            std::hint::black_box(&values);
        }
        println!("{} bytes={} mentions={} ns={}", grammar.name(), text.len(), values.len(), start.elapsed().as_nanos()/20);
    }
    if let Some(path) = std::env::args().nth(1) { std::fs::write(path, proof).unwrap(); }
}
