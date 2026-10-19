//! The grammar over a 100 KB text: the layers are valid, a second run
//! gives the same result, and the time is printed (a debug build is far
//! slower than a release one; `cargo run --release -p turbo-grammar
//! --example grammar_bench` measures the release build).

mod common;

use std::time::Instant;

use turbo_analysis::{Layer, ViewId};
use turbo_grammar::GrammarAnnotator;

#[test]
fn a_hundred_kilobytes() {
    let text = common::sample::sample(100 * 1024);
    let annotator = GrammarAnnotator::english();
    let t = Instant::now();
    let first = annotator.find(&text, ViewId::SOURCE);
    let elapsed = t.elapsed();
    println!(
        "{} bytes: {} numbers, {} percentages, {} ordinals in {elapsed:?}",
        text.len(),
        first.numbers.len(),
        first.percents.len(),
        first.ordinals.len()
    );
    assert_eq!(first.numbers.check(), Ok(()));
    assert_eq!(first.percents.check(), Ok(()));
    assert_eq!(first.ordinals.check(), Ok(()));
    assert!(first.numbers.len() > 1000);
    let second = annotator.find(&text, ViewId::SOURCE);
    assert_eq!(first.numbers.spans.columns(), second.numbers.spans.columns());
    assert_eq!(
        first.numbers.coefficient.iter().collect::<Vec<_>>(),
        second.numbers.coefficient.iter().collect::<Vec<_>>()
    );
    assert_eq!(first.ordinals.offset, second.ordinals.offset);
    assert_eq!(first.percents.spans.columns(), second.percents.spans.columns());
}
