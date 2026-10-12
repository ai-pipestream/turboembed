//! `Shrink`: OpenNLP's `ShrinkCharSequenceNormalizer`.

use turbo_analysis::ViewBuilder;

use crate::Normalizer;

/// Shrinks runs of whitespace and of repeated characters, and trims the
/// text:
///
/// 1. A run of two or more ASCII whitespace characters (U+0009..U+000D,
///    U+0020) becomes one U+0020. A single tab or newline stays as it is.
/// 2. A run of three or more of one character, compared ASCII
///    case-insensitively, becomes the run's first character twice:
///    `Helllllo` becomes `Hello`, `aAAb` becomes `aab`. A run of a line
///    terminator (LF, CR, U+0085, U+2028, U+2029) is never shrunk.
/// 3. Every leading and trailing character at or below U+0020 is deleted,
///    as `String.trim()` does: control characters too, not only
///    whitespace.
///
/// OpenNLP counterpart: `ShrinkCharSequenceNormalizer`, which runs the
/// three steps as passes, each over the previous one's output. This
/// normalizer runs one pass and produces the same text:
///
/// - After step 1 no two whitespace characters are adjacent, and step 1
///   changes nothing else, so the runs step 2 shrinks are runs of
///   non-whitespace characters, the same in the original text.
/// - Every character steps 1 and 2 write is at or below U+0020 exactly
///   when the characters it replaces are, and a run never mixes the two
///   kinds (a character at or below U+0020 is equal, ASCII
///   case-insensitively, only to itself). So the leading and trailing
///   characters step 3 deletes are the images of the leading and trailing
///   characters at or below U+0020 of the input, which this pass deletes
///   directly.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Shrink;

/// ASCII whitespace as OpenNLP's `AsciiChars.WHITESPACE`.
fn is_ascii_space(b: u8) -> bool {
    matches!(b, b'\t'..=b'\r' | b' ')
}

/// The line terminators the repeat rule leaves alone: the ones Java's
/// regular expression `.` does not match.
fn is_line_terminator(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\u{0085}' | '\u{2028}' | '\u{2029}')
}

/// Shrinks the text between the trimmed edges, `text[start..end]`.
fn shrink_inner(text: &str, start: usize, end: usize, out: &mut ViewBuilder<'_>) {
    let bytes = text.as_bytes();
    let mut i = start;
    while i < end {
        let b = bytes[i];
        if is_ascii_space(b) {
            let run = i + bytes[i..end].iter().take_while(|&&x| is_ascii_space(x)).count();
            if run - i >= 2 {
                out.keep(i);
                out.replace(run, " ");
            }
            i = run;
            continue;
        }
        let c = text[i..].chars().next().expect("a character at a boundary");
        let width = c.len_utf8();
        if is_line_terminator(c) {
            i += width;
            continue;
        }
        let run = repeat_run(text, i, end, c);
        if run - i >= 3 * width {
            // The first character stays; the rest becomes one more copy of it.
            out.keep(i + width);
            out.replace(run, &text[i..i + width]);
        }
        i = run;
    }
}

/// The end of the run of characters equal to `c`, ASCII
/// case-insensitively, that starts at byte `i`. Only an ASCII letter has a
/// case partner, and both partners are one byte, so the run is a run of
/// `width`-byte characters.
fn repeat_run(text: &str, i: usize, end: usize, c: char) -> usize {
    let width = c.len_utf8();
    if c.is_ascii() {
        let b = c as u8;
        return i + text.as_bytes()[i..end].iter().take_while(|x| x.eq_ignore_ascii_case(&b)).count();
    }
    let unit = &text.as_bytes()[i..i + width];
    let mut run = i + width;
    while run + width <= end && &text.as_bytes()[run..run + width] == unit {
        run += width;
    }
    run
}

impl Normalizer for Shrink {
    const ID: &'static str = "turbo:normalize:shrink";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        let bytes = text.as_bytes();
        // Characters at or below U+0020 are single bytes, and no byte of a
        // multi-byte character is that low.
        let start = bytes.iter().take_while(|&&b| b <= b' ').count();
        let end = start + bytes[start..].iter().rposition(|&b| b > b' ').map_or(0, |k| k + 1);
        if start > 0 {
            out.replace(start, "");
        }
        shrink_inner(text, start, end, out);
        if end < text.len() {
            out.keep(end);
            out.replace(text.len(), "");
        }
    }
}
