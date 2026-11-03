//! Reading text files line by line as the Java readers do: the lines
//! `BufferedReader.readLine` returns, and `StringUtil.trimUnicodeWhitespace`.

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

pub(crate) use turbo_unicode::whitespace::trim as trim_unicode_whitespace;

/// The fields of `line` separated by Unicode whitespace: the runs of
/// non-whitespace, in order.
pub(crate) fn whitespace_fields(line: &str) -> impl Iterator<Item = &str> {
    line.split(|c: char| turbo_unicode::whitespace::is_whitespace(c as u32)).filter(|f| !f.is_empty())
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
    fn fields_split_at_unicode_whitespace() {
        let f = |s| whitespace_fields(s).collect::<Vec<_>>();
        assert_eq!(f("  a\tb  c "), ["a", "b", "c"]);
        assert_eq!(f("a\u{85}b"), ["a", "b"]);
        assert_eq!(f("a\u{1c}b"), ["a\u{1c}b"]);
        assert_eq!(f("   "), Vec::<&str>::new());
    }

    #[test]
    fn trimming_is_unicode() {
        assert_eq!(trim_unicode_whitespace("\u{3000} a b\u{85}"), "a b");
        assert_eq!(trim_unicode_whitespace("\u{1c}a"), "\u{1c}a");
    }
}
