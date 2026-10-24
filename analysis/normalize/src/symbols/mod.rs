//! Symbols written as English words (`&` as `and`, `%` as `percent`) and
//! multi-token symbol sequences (`C++`, `.NET`, `->`) joined into one token.
//!
//! OpenNLP's `SymbolJoinerCharSequenceNormalizer` (in the runtime and,
//! identically, in the `symbol-joiner` add-on) maps an input that is
//! exactly one of ten symbols to a word and returns every other input
//! unchanged. Two forms here:
//!
//! - [`SymbolJoiner`] reads a view and its `Tokens`, joins touching token
//!   spans that form a symbol sequence from its table
//!   ([`DEFAULT_SYMBOL_SEQUENCES`] by default, or caller-configured), and
//!   writes a new view (with remaining standalone mapped symbols replaced
//!   by their words) and the joined `Tokens` mapped onto that view.
//! - [`SymbolWords`] applies the ten-symbol map to running text: a mapped
//!   symbol with whitespace or the edge of the text on both sides is
//!   replaced.

mod joiner;
mod words;

pub use joiner::{DEFAULT_SYMBOL_SEQUENCES, SymbolJoiner};
pub use words::SymbolWords;

/// The word for a mapped symbol. Currency symbols are not mapped.
pub fn symbol_word(c: char) -> Option<&'static str> {
    Some(match c {
        '&' => "and",
        '+' => "plus",
        '@' => "at",
        '%' => "percent",
        '\u{00A7}' => "section",
        '\u{00B6}' => "paragraph",
        '\u{00B0}' => "degree",
        '\u{00A9}' => "copyright",
        '\u{00AE}' => "registered",
        '\u{2122}' => "trademark",
        _ => return None,
    })
}

/// The word for `text` when it is exactly one mapped symbol.
pub(crate) fn whole_symbol_word(text: &str) -> Option<&'static str> {
    let mut chars = text.chars();
    let c = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    symbol_word(c)
}
