//! Readers for the OpenNLP line sample formats the training fixtures use.
//!
//! The stripped corpus crate no longer ships these, so the training tests
//! keep their own here: one sentence or document per line, tokens cut at
//! Unicode whitespace, borrowing the source text. The samples they build
//! and the spans they keep are the same, so the fixture models trained
//! from them stay the checked-in bytes.

use turbo_formats::Piece;

mod columns;
mod name;
mod simple;
mod spans;

pub use columns::{ChunkReader, LemmaReader};
pub use name::NameReader;
pub use simple::{DocumentReader, LanguageReader, WordTagReader};
pub use spans::{SentenceReader, TokenReader};

/// Whether `c` is removed by a Java `String.trim`: U+0000 to U+0020.
fn is_trimmed(c: char) -> bool {
    c <= ' '
}

/// A Java `String.trim()` of a piece: every leading and trailing character
/// up to U+0020 removed.
fn java_trim(field: Piece<'_>) -> Piece<'_> {
    let t = field.text;
    let start = t.len() - t.trim_start_matches(is_trimmed).len();
    let end = t.trim_end_matches(is_trimmed).len().max(start);
    field.slice(start, end)
}

/// The runs of characters that are not Unicode whitespace in `field`.
fn whitespace_tokens(field: Piece<'_>) -> impl Iterator<Item = Piece<'_>> {
    let text = field.text;
    let mut chars = text.char_indices().peekable();
    std::iter::from_fn(move || {
        while chars.next_if(|&(_, c)| turbo_unicode::whitespace::is_whitespace(c as u32)).is_some() {}
        let (start, _) = chars.next()?;
        let mut end = text.len();
        while let Some(&(i, c)) = chars.peek() {
            if turbo_unicode::whitespace::is_whitespace(c as u32) {
                end = i;
                break;
            }
            chars.next();
        }
        Some(field.slice(start, end))
    })
}

/// A Java `StringUtil.split(field, separator)` with limit 0: the pieces
/// between every `separator`, the empty pieces at the end removed. An
/// empty field gives one empty piece.
fn split_fields(field: Piece<'_>, separator: char) -> Vec<Piece<'_>> {
    if field.text.is_empty() {
        return vec![field];
    }
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in field.text.char_indices() {
        if c == separator {
            out.push(field.slice(start, i));
            start = i + c.len_utf8();
        }
    }
    out.push(field.slice(start, field.text.len()));
    while out.last().is_some_and(|f| f.text.is_empty()) {
        out.pop();
    }
    out
}
