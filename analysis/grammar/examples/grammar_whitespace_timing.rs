//! Release timing of the actual grammar lexer, in milliseconds.
use std::{hint::black_box, time::Instant};
use turbo_grammar::engine::{Lexed, NumeralStyle};
fn main() {
    let text = "Dr. Smith\u{b}paid\u{85}10\u{a0}euros\u{3000}yesterday.\u{1c}Fine! https://example.org/a\n".repeat(64);
    let style =
        NumeralStyle { mixed_separators: false, space_groups: false, indian_groups: false, decimal_comma: false };
    let mut lexed = Lexed::default();
    let start = Instant::now();
    for _ in 0..10_000 {
        lexed.lex(black_box(&text), &style);
        black_box(&lexed);
    }
    println!("grammar {:.3}", start.elapsed().as_secs_f64() * 1e3);
}
