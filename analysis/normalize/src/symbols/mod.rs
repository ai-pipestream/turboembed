//! Symbols written as English words: `&` as `and`, `%` as `percent`.
//!
//! OpenNLP's `SymbolJoinerCharSequenceNormalizer` (in the runtime and,
//! identically, in the `symbol-joiner` add-on) maps an input that is
//! exactly one of ten symbols to a word and returns every other input
//! unchanged. It is meant to run on one token at a time; it does not join
//! symbol sequences such as `C++` or `->`, despite its name. Two forms
//! here:
//!
//! - [`SymbolJoiner`] applies the map to tokens: it reads a view and its
//!   `Tokens` and writes a view in which every token that is exactly one
//!   mapped symbol is replaced by its word.
//! - [`SymbolWords`] applies it to running text: a mapped symbol with
//!   whitespace or the edge of the text on both sides is replaced.

mod joiner;
mod words;

pub use joiner::SymbolJoiner;
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
