//! Collapses runs of whitespace and trims the ends of the text.

use turbo_analysis::ViewBuilder;

use crate::substitute::{decode_at, utf8_len};
use crate::tables::{is_line_break, is_whitespace};
use crate::{Normalizer, hash_config};

/// What an inner run of whitespace becomes.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum WhitespaceMode {
    /// Every run becomes one space.
    #[default]
    Collapse,
    /// A run that contains a line break becomes one "\n"; any other run
    /// becomes one space.
    KeepLineBreaks,
    /// A run that contains two or more line breaks (a blank line, so a
    /// paragraph boundary) becomes one "\n"; any other run, including one
    /// with a single line break, becomes one space. A carriage return
    /// followed by a line feed counts as one line break. This unwraps hard
    /// wrapped prose while keeping its paragraphs apart.
    KeepParagraphs,
}

/// Collapses each run of whitespace to a single character and deletes
/// whitespace at the start and end of the text.
///
/// Whitespace is the 25 code points of OpenNLP's `UnicodeWhitespace`
/// table ([`WHITESPACE`](crate::tables::WHITESPACE)), which includes the
/// no-break spaces; line breaks are its seven `LINE_BREAK` entries
/// ([`LINE_BREAKS`](crate::tables::LINE_BREAKS)). An inner run becomes " "
/// or "\n" as [`WhitespaceMode`] says; a run at either end of the text is
/// deleted. A text that is all whitespace becomes empty.
///
/// The work is one pass over the text. Each run is one edit: an inner run
/// maps to its whole source extent, so the "\n" that stands for "\n\n\t"
/// maps back to all three characters. A run that is already exactly its
/// replacement (a single space between words) is left as it is and makes
/// no edit.
///
/// OpenNLP counterparts: `WhitespaceCharSequenceNormalizer`,
/// `LineBreakPreservingWhitespaceCharSequenceNormalizer` and
/// `ParagraphPreservingWhitespaceCharSequenceNormalizer`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Whitespace {
    mode: WhitespaceMode,
}

impl Whitespace {
    pub fn new(mode: WhitespaceMode) -> Whitespace {
        Whitespace { mode }
    }

    /// Every inner run becomes one space.
    pub fn collapse() -> Whitespace {
        Whitespace::new(WhitespaceMode::Collapse)
    }

    /// Inner runs with a line break become "\n", others one space.
    pub fn keep_line_breaks() -> Whitespace {
        Whitespace::new(WhitespaceMode::KeepLineBreaks)
    }

    /// Inner runs with two or more line breaks become "\n", others one space.
    pub fn keep_paragraphs() -> Whitespace {
        Whitespace::new(WhitespaceMode::KeepParagraphs)
    }

    pub fn mode(&self) -> WhitespaceMode {
        self.mode
    }

    /// What an inner run with `breaks` logical line breaks becomes.
    fn replacement(&self, breaks: u32) -> &'static str {
        let newline = match self.mode {
            WhitespaceMode::Collapse => false,
            WhitespaceMode::KeepLineBreaks => breaks >= 1,
            WhitespaceMode::KeepParagraphs => breaks >= 2,
        };
        if newline { "\n" } else { " " }
    }
}

impl Normalizer for Whitespace {
    const ID: &'static str = "turbo:normalize:whitespace";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        let mut at = 0;
        while let Some(run) = next_run(text, at) {
            let edge = run.start == 0 || run.end == text.len();
            let with = if edge { "" } else { self.replacement(run.breaks) };
            if &text[run.start..run.end] != with {
                out.keep(run.start);
                out.replace(run.end, with);
            }
            at = run.end;
        }
    }

    fn config_hash(&self) -> u64 {
        hash_config(&self.mode)
    }
}

/// A maximal run of whitespace: its byte range and how many logical line
/// breaks it holds.
struct Run {
    start: usize,
    end: usize,
    breaks: u32,
}

/// The first run of whitespace that starts at or after byte `from`.
fn next_run(text: &str, from: usize) -> Option<Run> {
    let (start, first) = next_member(text, from)?;
    let mut run = Run { start, end: start, breaks: 0 };
    let mut c = first;
    let mut previous_was_cr = false;
    loop {
        // A line feed straight after a carriage return completes the same
        // line break, so it is not counted again.
        if is_line_break(c) && !(c == '\n' && previous_was_cr) {
            run.breaks = run.breaks.saturating_add(1);
        }
        previous_was_cr = c == '\r';
        run.end += c.len_utf8();
        if run.end == text.len() {
            break;
        }
        c = decode_at(text, run.end);
        if !is_whitespace(c) {
            break;
        }
    }
    Some(run)
}

/// The first whitespace character at or after byte `from`, with its offset.
///
/// ASCII bytes are tested directly. Of the non-ASCII members, U+0085 and
/// U+00A0 start with the byte 0xC2, U+1680 with 0xE1, U+2000 to U+205F
/// with 0xE2 and U+3000 with 0xE3; a character starting with any other
/// byte is stepped over by its encoded length without being decoded.
fn next_member(text: &str, from: usize) -> Option<(usize, char)> {
    let bytes = text.as_bytes();
    let mut i = from;
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            0x00..=0x7F => {
                if turbo_unicode::whitespace::WhitespaceMode::Unicode.is_ascii_whitespace(b) {
                    return Some((i, b as char));
                }
                i += 1;
            }
            0xC2 | 0xE1 | 0xE2 | 0xE3 => {
                let c = decode_at(text, i);
                if is_whitespace(c) {
                    return Some((i, c));
                }
                i += c.len_utf8();
            }
            _ => i += utf8_len(b),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_prefilter_covers_the_shared_unicode_table() {
        for cp in turbo_unicode::whitespace::UNICODE_WHITESPACE {
            let c = char::from_u32(cp).unwrap();
            if !c.is_ascii() {
                let mut bytes = [0; 4];
                let lead = c.encode_utf8(&mut bytes).as_bytes()[0];
                assert!(matches!(lead, 0xC2 | 0xE1 | 0xE2 | 0xE3), "prefilter omits U+{cp:04X}");
            }
        }
    }

    #[test]
    fn next_member_finds_every_table_entry_after_other_text() {
        for &w in crate::tables::WHITESPACE {
            let text = format!("a\u{E9}\u{4E2D}\u{1F600}{w}b");
            assert_eq!(next_member(&text, 0), Some((10, w)), "{w:?}");
        }
        assert_eq!(next_member("\u{E9}\u{2010}\u{1680}", 0), Some((5, '\u{1680}')));
        assert_eq!(next_member("abc\u{2030}", 0), None);
    }

    #[test]
    fn runs_count_crlf_as_one_break() {
        let run = next_run("a\r\n\r\n b", 0).unwrap();
        assert_eq!((run.start, run.end, run.breaks), (1, 6, 2));
        let run = next_run("a\r\r\n\n b", 0).unwrap();
        assert_eq!(run.breaks, 3);
        let run = next_run("a\u{2028} \u{2029}b", 0).unwrap();
        assert_eq!((run.start, run.end, run.breaks), (1, 8, 2));
    }
}
