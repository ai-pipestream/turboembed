//! Lines of a text for the line-level rules: the repeated-line artifact
//! and the document statistics of the noise scorer.

use std::ops::Range;

use turbo_unicode::props;

/// Whether `c` ends a line: the mandatory breaks of UAX #14 (classes BK,
/// CR, LF and NL). CR LF splits off an empty line between the two, which
/// `trimmed_lines` drops.
pub(crate) fn is_line_break(c: char) -> bool {
    matches!(c, '\n' | '\u{B}' | '\u{C}' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

/// The lines of `text` with `White_Space` trimmed from both ends, as byte
/// ranges, leaving out lines that are empty after trimming.
pub(crate) fn trimmed_lines(text: &str) -> impl Iterator<Item = Range<usize>> + '_ {
    text.split(is_line_break).filter_map(move |line| {
        let start = line.as_ptr() as usize - text.as_ptr() as usize;
        let trimmed = line.trim_matches(props::white_space);
        if trimmed.is_empty() {
            return None;
        }
        let offset = trimmed.as_ptr() as usize - line.as_ptr() as usize;
        Some(start + offset..start + offset + trimmed.len())
    })
}
