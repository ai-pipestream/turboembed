//! `ParserTool::parse_line_with_mode` splits a line into tokens on the
//! characters its `WhitespaceMode` counts as whitespace, as OpenNLP's
//! `WhitespaceTokenizer` does, and maps every token back to its bytes in
//! the line. The parser here returns the flat token tree it is given, so
//! the tokens the tool built are what the test reads.

use turbo_parser::{Parse, ParseError, Parser, ParserTool};
use turbo_unicode::WhitespaceMode;

/// A parser that answers the flat tree of tokens it was given.
struct Tokens;

impl Parser for Tokens {
    fn parse_n(&self, tokens: &Parse, _: usize) -> Result<Vec<Parse>, ParseError> {
        Ok(vec![tokens.clone()])
    }
}

/// The tokens of `line` under `mode`, read through the spans the tool maps
/// back to `line`; `None` when the tool gives no parse.
fn tokens(line: &str, mode: WhitespaceMode) -> Option<Vec<&str>> {
    let parses = ParserTool::parse_line_with_mode(line, &Tokens, mode, 1).unwrap();
    let p = parses.first()?;
    Some(p.children().iter().map(|c| &line[c.span().start as usize..c.span().end as usize]).collect())
}

#[test]
fn unicode_mode_splits_on_unicode_whitespace() {
    for line in ["the\u{00A0}dog", "the\u{3000}dog", "the\u{000B}dog", "the\u{2003}dog"] {
        assert_eq!(tokens(line, WhitespaceMode::Unicode), Some(vec!["the", "dog"]), "{line:?}");
    }
    assert_eq!(tokens(" the  dog\t", WhitespaceMode::Unicode), Some(vec!["the", "dog"]));
}

#[test]
fn legacy_and_unicode_modes_differ_on_information_separators() {
    assert_eq!(tokens("a\u{001C}b", WhitespaceMode::Legacy), Some(vec!["a", "b"]));
    assert_eq!(tokens("a\u{001C}b", WhitespaceMode::Unicode), Some(vec!["a\u{001C}b"]));
}

#[test]
fn a_line_of_only_whitespace_gives_no_parse() {
    assert_eq!(tokens("\u{3000}\u{3000}", WhitespaceMode::Unicode), None);
    assert_eq!(tokens("\u{001C}", WhitespaceMode::Legacy), None);
    assert_eq!(tokens("", WhitespaceMode::Unicode), None);
    // Not whitespace in Unicode mode: one token.
    assert_eq!(tokens("\u{001C}", WhitespaceMode::Unicode), Some(vec!["\u{001C}"]));
}

#[test]
fn spans_are_byte_offsets_in_the_line() {
    let line = "Zoë\u{3000}(saw)\u{00A0}it";
    let parses = ParserTool::parse_line_with_mode(line, &Tokens, WhitespaceMode::Unicode, 1).unwrap();
    let spans: Vec<(u32, u32)> = parses[0].children().iter().map(|c| (c.span().start, c.span().end)).collect();
    let zoe = "Zoë".len() as u32;
    let open = zoe + "\u{3000}".len() as u32;
    let saw = open + 1;
    let close = saw + 3;
    let it = close + 1 + "\u{00A0}".len() as u32;
    assert_eq!(spans, vec![(0, zoe), (open, open + 1), (saw, close), (close, close + 1), (it, it + 2)]);
}
