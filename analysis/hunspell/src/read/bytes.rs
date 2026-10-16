//! What the reader does to the raw bytes of the files before decoding them:
//! finding the `SET` declaration, blanking the lines it does not parse, and
//! turning raw flag bytes of a UTF-8 file into characters.

use crate::LoadMode;
use crate::charset::{Charset, Lookup};
use crate::error::LoadError;
use crate::read::directive;
use crate::text::{split_fields, split_lines, trim};

/// A directive a partial load skipped: its name, the file and the line of
/// its first occurrence. OpenNLP's `UnsupportedDirective`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsupportedDirective {
    pub directive: String,
    /// The affix file's path, or `affix stream` for a dictionary loaded
    /// from bytes.
    pub source: String,
    /// The line, counted from 1.
    pub line: usize,
}

/// The name OpenNLP gives the affix file in a diagnostic when it is loaded
/// from a stream.
pub(crate) const AFFIX_SOURCE: &str = "affix stream";

/// The length of a UTF-8 byte order mark at the start of `bytes`: 3 or 0.
pub(crate) fn bom_len(bytes: &[u8]) -> usize {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) { 3 } else { 0 }
}

/// The byte-level field separators: space, tab and form feed.
fn is_field_space(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == 0x0C
}

/// Bytes read as US-ASCII by Java's `new String(bytes, US_ASCII)`: every
/// byte from 0x80 becomes U+FFFD.
fn ascii_text(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| if b < 0x80 { char::from(b) } else { '\u{FFFD}' }).collect()
}

/// The ranges of up to `max` fields of `bytes[from..to]`.
fn fields(bytes: &[u8], from: usize, to: usize, max: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = from;
    while i < to && out.len() < max {
        while i < to && is_field_space(bytes[i]) {
            i += 1;
        }
        if i == to {
            break;
        }
        let start = i;
        while i < to && !is_field_space(bytes[i]) {
            i += 1;
        }
        out.push((start, i));
    }
    out
}

/// Each line of `bytes` from `from`, as `(start, end)`, a line ending at
/// every `\n` and every `\r` taken alone (a `\r\n` makes an empty line
/// between), as the byte-level scans of the reader split them.
fn raw_lines(bytes: &[u8], from: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
    let mut start = from;
    let mut i = from;
    std::iter::from_fn(move || {
        while i <= bytes.len() {
            let at = i;
            i += 1;
            if at == bytes.len() || bytes[at] == b'\n' || bytes[at] == b'\r' {
                let line = (start, at);
                start = at + 1;
                return Some(line);
            }
        }
        None
    })
}

/// The encoding the first `SET` line declares, UTF-8 without one. The bytes
/// after any byte order mark are read as ASCII, each line trimmed, and the
/// first field after `SET` and a space or tab is the name.
pub(crate) fn declared_charset(affix: &[u8]) -> Result<Charset, LoadError> {
    let text = ascii_text(&affix[bom_len(affix)..]);
    for line in split_lines(&text) {
        let trimmed = trim(line);
        if trimmed.starts_with("SET ") || trimmed.starts_with("SET\t") {
            let name = split_fields(&trimmed[4..])[0];
            return match Charset::for_name(name) {
                Lookup::Found(c) => Ok(c),
                Lookup::NotPorted => Err(LoadError::EncodingNotPorted { name: name.into() }),
                Lookup::Unknown => Err(LoadError::UnsupportedEncoding { name: name.into() }),
            };
        }
    }
    Ok(Charset::Utf8)
}

/// Whether the first field of some line is `directive`.
fn has_directive(bytes: &[u8], directive: &str) -> bool {
    raw_lines(bytes, bom_len(bytes)).any(|(from, to)| {
        fields(bytes, from, to, 1).first().is_some_and(|&(s, e)| &bytes[s..e] == directive.as_bytes())
    })
}

/// Replaces every line the reader does not parse with spaces: comments,
/// blank lines, the directives it ignores (`REP` too, unless
/// `CHECKCOMPOUNDREP` makes it count) and, in a partial load, the ones it
/// does not support. A line ends at `\n`, `\r` or `\r\n`, so line numbers
/// agree with the decoded file's. A strict load fails at the first
/// unsupported directive; a partial one returns them, one per name.
pub(crate) fn mask_unparsed_lines(
    bytes: &mut [u8],
    mode: LoadMode,
    source: &str,
) -> Result<Vec<UnsupportedDirective>, LoadError> {
    let mut unsupported: Vec<UnsupportedDirective> = Vec::new();
    let use_replacements = has_directive(bytes, directive::CHECK_COMPOUND_REP);
    let mut line_start = 0;
    let mut line = 1;
    let mut i = 0;
    while i <= bytes.len() {
        if i == bytes.len() || bytes[i] == b'\n' || bytes[i] == b'\r' {
            let mut field_start = if line_start == 0 { bom_len(bytes).min(i) } else { line_start };
            while field_start < i && is_field_space(bytes[field_start]) {
                field_start += 1;
            }
            let mut field_end = field_start;
            while field_end < i && !is_field_space(bytes[field_end]) {
                field_end += 1;
            }
            let mut parsed = false;
            if field_start < field_end && bytes[field_start] != b'#' {
                let name = ascii_text(&bytes[field_start..field_end]);
                if directive::is_parsed(&name) && (name != directive::REPLACEMENT || use_replacements) {
                    mask_inline_comment(bytes, field_start, i, &name);
                    parsed = true;
                } else if !directive::is_ignored(&name) {
                    if mode == LoadMode::Strict {
                        return Err(LoadError::UnsupportedDirective { directive: name, source: source.into(), line });
                    }
                    if !unsupported.iter().any(|u| u.directive == name) {
                        unsupported.push(UnsupportedDirective { directive: name, source: source.into(), line });
                    }
                }
            }
            if !parsed {
                bytes[line_start..i].fill(b' ');
            }
            if i < bytes.len() && bytes[i] == b'\r' && i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                i += 1;
            }
            line_start = i + 1;
            line += 1;
        }
        i += 1;
    }
    Ok(unsupported)
}

/// Blanks a trailing comment: from the first field that starts with `#`
/// among the fields after the ones the directive reads, to the end of the
/// line. `#` is a valid flag, separator, affix and conversion text, so only
/// a field the directive does not read can start a comment.
fn mask_inline_comment(bytes: &mut [u8], from: usize, to: usize, name: &str) {
    let fs = fields(bytes, from, to, usize::MAX);
    let first = match name {
        directive::PREFIX | directive::SUFFIX => {
            if is_affix_header(bytes, &fs) {
                4
            } else {
                5
            }
        }
        directive::REPLACEMENT
        | directive::INPUT_CONVERSION
        | directive::OUTPUT_CONVERSION
        | directive::COMPOUND_PATTERN
        | directive::MORPHOLOGY_ALIAS => usize::MAX,
        directive::COMPOUND_SYLLABLE => 3,
        _ => 2,
    };
    if let Some(&(start, _)) = fs.iter().skip(first).find(|&&(s, _)| bytes[s] == b'#') {
        bytes[start..to].fill(b' ');
    }
}

/// Whether a `PFX` or `SFX` line is a block header: a third field of `Y`
/// or `N` and a fourth of digits.
fn is_affix_header(bytes: &[u8], fs: &[(usize, usize)]) -> bool {
    if fs.len() < 4 || fs[2].1 - fs[2].0 != 1 || (bytes[fs[2].0] != b'Y' && bytes[fs[2].0] != b'N') {
        return false;
    }
    bytes[fs[3].0..fs[3].1].iter().all(u8::is_ascii_digit)
}

/// Whether the first `FLAG` line declares `UTF-8` or `num`.
pub(crate) fn uses_unicode_or_numeric_flags(bytes: &[u8]) -> bool {
    for (from, to) in raw_lines(bytes, bom_len(bytes)) {
        let fs = fields(bytes, from, to, 5);
        if fs.len() >= 2 && &bytes[fs[0].0..fs[0].1] == directive::FLAG.as_bytes() {
            let mode = &bytes[fs[1].0..fs[1].1];
            return mode == b"UTF-8" || mode == b"num";
        }
    }
    false
}

/// A byte of a flag field, from 0x80, as the UTF-8 of the character of the
/// same value; every other byte as it is.
fn push_byte(out: &mut Vec<u8>, b: u8, flag: bool) {
    if flag && b >= 0x80 {
        out.push(if b < 0xC0 { 0xC2 } else { 0xC3 });
        out.push(if b < 0xC0 { b } else { b - 0x40 });
    } else {
        out.push(b);
    }
}

/// In a UTF-8 affix file whose flags are bytes (the default and `long`
/// modes), each byte from 0x80 of a flag field becomes the character of the
/// same value, so it decodes. Text, conditions and affix material stay
/// as they are and must be valid UTF-8.
pub(crate) fn normalize_affix_flags(bytes: Vec<u8>, charset: Charset) -> Vec<u8> {
    if charset != Charset::Utf8 || uses_unicode_or_numeric_flags(&bytes) {
        return bytes;
    }
    let bom = bom_len(&bytes);
    let mut out = Vec::with_capacity(bytes.len() + bytes.len() / 16);
    out.extend_from_slice(&bytes[..bom]);
    for (from, to) in raw_lines(&bytes, bom) {
        push_flag_line(&mut out, &bytes, from, to);
        if to < bytes.len() {
            out.push(bytes[to]);
        }
    }
    out
}

/// One affix file line with the high bytes of its flag fields turned into
/// characters: the second field of `AF`, `COMPOUNDRULE`, `PFX`, `SFX` and
/// the single flag directives, the continuation classes after `/` in the
/// fourth field of `PFX` and `SFX`, and the flags after `/` in the second
/// and third fields of `CHECKCOMPOUNDPATTERN`.
fn push_flag_line(out: &mut Vec<u8>, bytes: &[u8], from: usize, to: usize) {
    let fs = fields(bytes, from, to, 5);
    let mut first = (usize::MAX, usize::MAX);
    let mut continuation = (usize::MAX, usize::MAX);
    if fs.len() >= 2 {
        let name = &bytes[fs[0].0..fs[0].1];
        let is = |d: &str| name == d.as_bytes();
        let affix = is(directive::PREFIX) || is(directive::SUFFIX);
        if is(directive::ALIAS)
            || is(directive::COMPOUND_RULE)
            || affix
            || directive::takes_one_flag(&String::from_utf8_lossy(name))
        {
            first = fs[1];
        }
        if fs.len() >= 4
            && affix
            && let Some(slash) = bytes[fs[3].0..fs[3].1].iter().position(|&b| b == b'/')
        {
            continuation = (fs[3].0 + slash + 1, fs[3].1);
        }
        if fs.len() >= 3 && is(directive::COMPOUND_PATTERN) {
            if let Some(slash) = bytes[fs[1].0..fs[1].1].iter().position(|&b| b == b'/') {
                first = (fs[1].0 + slash + 1, fs[1].1);
            }
            if let Some(slash) = bytes[fs[2].0..fs[2].1].iter().position(|&b| b == b'/') {
                continuation = (fs[2].0 + slash + 1, fs[2].1);
            }
        }
    }
    for (i, &b) in bytes.iter().enumerate().take(to).skip(from) {
        let flag = (i >= first.0 && i < first.1) || (i >= continuation.0 && i < continuation.1);
        push_byte(out, b, flag);
    }
}

/// In a word list read with byte flags, each byte from 0x80 of the flags
/// after the first `/` not preceded by a backslash (and not the first byte
/// of the line), up to a space or tab, becomes the character of the same
/// value.
pub(crate) fn normalize_word_flags(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() + bytes.len() / 16);
    for (from, to) in raw_lines(bytes, 0) {
        let mut flags = (usize::MAX, usize::MAX);
        for cursor in from + 1..to {
            if bytes[cursor] == b'/' && bytes[cursor - 1] != b'\\' {
                let start = cursor + 1;
                let mut end = start;
                while end < to && bytes[end] != b' ' && bytes[end] != b'\t' {
                    end += 1;
                }
                flags = (start, end);
                break;
            }
        }
        for (i, &b) in bytes.iter().enumerate().take(to).skip(from) {
            push_byte(&mut out, b, i >= flags.0 && i < flags.1);
        }
        if to < bytes.len() {
            out.push(bytes[to]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_is_found_after_a_byte_order_mark() {
        assert_eq!(declared_charset(b"\xef\xbb\xbfSET ISO8859-1\n").unwrap().name(), "ISO-8859-1");
        assert_eq!(declared_charset(b"# SET x\n").unwrap(), Charset::Utf8);
        assert!(
            matches!(declared_charset(b"  SET\tbogus extra\n"), Err(LoadError::UnsupportedEncoding { name }) if name == "bogus")
        );
    }

    #[test]
    fn unparsed_lines_are_blanked() {
        let mut bytes = b"# c\nTRY abc\nSFX A Y 1 # note\nSFX A 0 s . # morph\nFOO x\n".to_vec();
        let skipped = mask_unparsed_lines(&mut bytes, LoadMode::AllowPartial, AFFIX_SOURCE).unwrap();
        assert_eq!(skipped, [UnsupportedDirective { directive: "FOO".into(), source: AFFIX_SOURCE.into(), line: 5 }]);
        assert_eq!(&bytes, b"   \n       \nSFX A Y 1       \nSFX A 0 s .        \n     \n");
        let mut strict = b"FOO\n".to_vec();
        let e = mask_unparsed_lines(&mut strict, LoadMode::Strict, AFFIX_SOURCE).unwrap_err();
        assert_eq!(
            e.to_string(),
            "unsupported affix directive FOO in affix stream at line 1; use LoadMode.ALLOW_PARTIAL to load without this behavior"
        );
    }

    #[test]
    fn raw_flag_bytes_become_characters() {
        let affix = b"SFX \xe9 Y 1\nSFX \xe9 0 \xc3\xa9/\xe9 .\n".to_vec();
        let out = normalize_affix_flags(affix, Charset::Utf8);
        assert_eq!(&out, "SFX \u{e9} Y 1\nSFX \u{e9} 0 \u{e9}/\u{e9} .\n".as_bytes());
        let words = normalize_word_flags(b"a\\/b/\xe9x c\n");
        assert_eq!(&words, "a\\/b/\u{e9}x c\n".as_bytes());
    }
}
