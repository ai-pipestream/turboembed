//! Reading text files line by line as the Java readers do: the lines
//! `BufferedReader.readLine` returns, and `StringUtil.trimUnicodeWhitespace`.

use turbo_unicode::WhitespaceMode;

/// The lines of `text` as `BufferedReader.readLine` returns them: ended by
/// `\n`, `\r` or `\r\n`, the end itself left out, and no empty last line
/// after a final line end.
pub(crate) struct Lines<'a> {
    rest: &'a str,
}

impl<'a> Lines<'a> {
    pub(crate) fn new(text: &'a str) -> Lines<'a> {
        Lines { rest: text }
    }

    /// The text not yet returned.
    pub(crate) fn rest(&self) -> &'a str {
        self.rest
    }
}

impl<'a> Iterator for Lines<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.rest.is_empty() {
            return None;
        }
        let bytes = self.rest.as_bytes();
        let Some(end) = bytes.iter().position(|&b| b == b'\n' || b == b'\r') else {
            let line = self.rest;
            self.rest = "";
            return Some(line);
        };
        let line = &self.rest[..end];
        let skip = if bytes[end] == b'\r' && bytes.get(end + 1) == Some(&b'\n') { 2 } else { 1 };
        self.rest = &self.rest[end + skip..];
        Some(line)
    }
}

/// `StringUtil.trimUnicodeWhitespace`: the Unicode `White_Space` set at
/// both ends, in every whitespace mode.
pub(crate) fn trim_unicode_whitespace(line: &str) -> &str {
    line.trim_matches(|c: char| WhitespaceMode::Unicode.is_whitespace(c as u32))
}

/// The fields of `line` separated by whitespace in `mode`, as OpenNLP's
/// dictionary readers split them (`splitWhitespace` over
/// `StringUtil.isWhitespace`): the runs of non-whitespace, in order.
pub(crate) fn whitespace_fields(line: &str, mode: WhitespaceMode) -> impl Iterator<Item = &str> {
    line.split(move |c: char| mode.is_whitespace(c as u32)).filter(|f| !f.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_split_as_read_line_splits_them() {
        let lines: Vec<&str> = Lines::new("a\nb\r\nc\rd\n\ne").collect();
        assert_eq!(lines, ["a", "b", "c", "d", "", "e"]);
        assert_eq!(Lines::new("a\n").collect::<Vec<_>>(), ["a"]);
        assert_eq!(Lines::new("\r\r\n").collect::<Vec<_>>(), ["", ""]);
        assert_eq!(Lines::new("").count(), 0);
        // U+2028 and U+0085 do not end a line.
        assert_eq!(Lines::new("a\u{2028}b\u{85}c").count(), 1);
        let mut lines = Lines::new("a\r\nb");
        lines.next();
        assert_eq!(lines.rest(), "b");
    }

    #[test]
    fn fields_split_in_the_mode() {
        let f = |s, m| whitespace_fields(s, m).collect::<Vec<_>>();
        assert_eq!(f("  a\tb  c ", WhitespaceMode::Unicode), ["a", "b", "c"]);
        assert_eq!(f("a\u{85}b", WhitespaceMode::Unicode), ["a", "b"]);
        assert_eq!(f("a\u{85}b", WhitespaceMode::Legacy), ["a\u{85}b"]);
        assert_eq!(f("a\u{1c}b", WhitespaceMode::Legacy), ["a", "b"]);
        assert_eq!(f("   ", WhitespaceMode::Unicode), Vec::<&str>::new());
    }

    #[test]
    fn trimming_is_unicode_in_every_mode() {
        assert_eq!(trim_unicode_whitespace("\u{3000} a b\u{85}"), "a b");
        assert_eq!(trim_unicode_whitespace("\u{1c}a"), "\u{1c}a");
    }
}
