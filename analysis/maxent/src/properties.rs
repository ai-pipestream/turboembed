//! A Java properties file read as `java.util.Properties.load(InputStream)`
//! reads it: the bytes are ISO-8859-1, so each byte is one character;
//! logical lines join physical lines that end in an odd number of
//! backslashes; lines whose first non-blank character is `#` or `!` are
//! comments; the key ends at the first unescaped `=`, `:`, space, tab or
//! form feed; and `\uXXXX`, `\t`, `\r`, `\n` and `\f` are escapes, with a
//! backslash before any other character standing for that character.
//!
//! The line reader and the key and value split follow the JDK 21 source
//! of `Properties.LineReader.readLine`, `load0` and `loadConvert`, quirks
//! included (a backslash alone at the end of the file is an entry with an
//! empty key and value; after a continuation line that ends up empty, a
//! line starting with `#` is a comment).

use crate::error::{Error, Result};

/// The entries of a properties file in the order their keys first
/// appear. A key given twice keeps its first position and its last
/// value, as `Properties.put` would leave it. Manifests have about a
/// dozen entries, so lookups are a linear scan.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Properties {
    entries: Vec<(String, String)>,
}

impl Properties {
    /// Parses properties file bytes.
    pub fn load(bytes: &[u8]) -> Result<Properties> {
        let mut reader = LineReader { data: bytes, off: 0, line: Vec::new() };
        let mut properties = Properties::default();
        while let Some(start) = reader.read_line() {
            let line = &reader.line;
            let limit = line.len();
            let mut key_len = 0;
            let mut value_start = limit;
            let mut has_separator = false;
            let mut preceding_backslash = false;
            while key_len < limit {
                let c = line[key_len];
                if (c == b'=' || c == b':') && !preceding_backslash {
                    value_start = key_len + 1;
                    has_separator = true;
                    break;
                } else if is_blank(c) && !preceding_backslash {
                    value_start = key_len + 1;
                    break;
                }
                preceding_backslash = c == b'\\' && !preceding_backslash;
                key_len += 1;
            }
            while value_start < limit {
                let c = line[value_start];
                if !is_blank(c) {
                    if !has_separator && (c == b'=' || c == b':') {
                        has_separator = true;
                    } else {
                        break;
                    }
                }
                value_start += 1;
            }
            let line_no = || line_number(bytes, start);
            let key = convert(&line[..key_len]).map_err(|message| Error::Properties { line: line_no(), message })?;
            let value =
                convert(&line[value_start..]).map_err(|message| Error::Properties { line: line_no(), message })?;
            properties.insert(key, value);
        }
        Ok(properties)
    }

    /// The value of a key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// The entries, in the order their keys first appear.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn insert(&mut self, key: String, value: String) {
        match self.entries.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = value,
            None => self.entries.push((key, value)),
        }
    }
}

fn is_blank(c: u8) -> bool {
    c == b' ' || c == b'\t' || c == 0x0C
}

/// The 1-based physical line holding byte `at`, counting `\r\n`, `\r` and
/// `\n` as line ends.
fn line_number(bytes: &[u8], at: usize) -> usize {
    let mut line = 1;
    let mut i = 0;
    while i < at {
        match bytes[i] {
            b'\n' => line += 1,
            b'\r' => {
                line += 1;
                if bytes.get(i + 1) == Some(&b'\n') {
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    line
}

/// `Properties.LineReader` over a byte slice.
struct LineReader<'a> {
    data: &'a [u8],
    off: usize,
    /// The current logical line, without its terminators, continuation
    /// backslashes or the blanks that start each joined physical line.
    line: Vec<u8>,
}

impl LineReader<'_> {
    /// Reads the next logical line into `self.line`, giving the offset
    /// where it starts, or `None` at the end of the input.
    fn read_line(&mut self) -> Option<usize> {
        self.line.clear();
        let mut start = self.off;
        let mut skip_blanks = true;
        let mut appended_line_begin = false;
        let mut preceding_backslash = false;
        loop {
            let Some(&c) = self.data.get(self.off) else {
                if self.line.is_empty() {
                    return None;
                }
                if preceding_backslash {
                    self.line.pop();
                }
                return Some(start);
            };
            self.off += 1;
            if skip_blanks {
                if is_blank(c) {
                    continue;
                }
                if !appended_line_begin && (c == b'\r' || c == b'\n') {
                    continue;
                }
                skip_blanks = false;
                appended_line_begin = false;
            }
            if self.line.is_empty() {
                if c == b'#' || c == b'!' {
                    loop {
                        let &b = self.data.get(self.off)?;
                        self.off += 1;
                        if b == b'\r' || b == b'\n' {
                            break;
                        }
                    }
                    skip_blanks = true;
                    continue;
                }
                start = self.off - 1;
            }
            if c != b'\n' && c != b'\r' {
                self.line.push(c);
                preceding_backslash = c == b'\\' && !preceding_backslash;
                continue;
            }
            if self.line.is_empty() {
                skip_blanks = true;
                continue;
            }
            if self.off >= self.data.len() {
                if preceding_backslash {
                    self.line.pop();
                }
                return Some(start);
            }
            if !preceding_backslash {
                return Some(start);
            }
            self.line.pop();
            skip_blanks = true;
            appended_line_begin = true;
            preceding_backslash = false;
            if c == b'\r' && self.data[self.off] == b'\n' {
                self.off += 1;
            }
        }
    }
}

/// `Properties.loadConvert`: resolves escapes. The input is ISO-8859-1, so
/// each byte is the character of the same value. `\uXXXX` escapes are
/// UTF-16 code units; a pair of them that forms a surrogate pair becomes
/// one character, and a lone surrogate, which a Rust string cannot hold,
/// becomes U+FFFD.
fn convert(src: &[u8]) -> std::result::Result<String, &'static str> {
    if !src.contains(&b'\\') {
        return Ok(src.iter().map(|&b| char::from(b)).collect());
    }
    let mut units: Vec<u16> = Vec::with_capacity(src.len());
    let mut i = 0;
    while i < src.len() {
        let c = src[i];
        i += 1;
        if c != b'\\' {
            units.push(u16::from(c));
            continue;
        }
        // The line reader never leaves an unescaped backslash at the end of
        // a key or a value, so a character follows.
        let Some(&escaped) = src.get(i) else { return Err("a backslash ends the line") };
        i += 1;
        let unit = match escaped {
            b'u' => {
                let Some(hex) = src.get(i..i + 4) else { return Err("malformed \\uxxxx encoding") };
                i += 4;
                let mut value = 0u16;
                for &h in hex {
                    let digit = (h as char).to_digit(16).ok_or("malformed \\uxxxx encoding")?;
                    value = value << 4 | digit as u16;
                }
                value
            }
            b't' => u16::from(b'\t'),
            b'r' => u16::from(b'\r'),
            b'n' => u16::from(b'\n'),
            b'f' => 0x0C,
            other => u16::from(other),
        };
        units.push(unit);
    }
    Ok(String::from_utf16_lossy(&units))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(text: &[u8]) -> Vec<(String, String)> {
        Properties::load(text).unwrap().entries
    }

    fn p(k: &str, v: &str) -> (String, String) {
        (k.to_string(), v.to_string())
    }

    #[test]
    fn separators_and_blanks() {
        assert_eq!(
            pairs(b"a=1\nb:2\nc 3\nd = 4\ne : = 5\n  f\t\x0c6\ng\nh=\n"),
            vec![
                p("a", "1"),
                p("b", "2"),
                p("c", "3"),
                p("d", "4"),
                p("e", "= 5"),
                p("f", "6"),
                p("g", ""),
                p("h", "")
            ]
        );
    }

    #[test]
    fn comments_and_line_ends() {
        assert_eq!(
            pairs(b"# c\r\n! c\ra=1\r\n\r\n  # indented comment\nb=2\\\n# not a comment\n"),
            vec![p("a", "1"), p("b", "2# not a comment")]
        );
    }

    #[test]
    fn continuation_lines() {
        // The blanks that start a continued line are dropped, so "two" and
        // "three" join with no space between them.
        assert_eq!(pairs(b"a=one \\\n    two\\\r\n three\n"), vec![p("a", "one twothree")]);
        // An even number of backslashes is not a continuation.
        assert_eq!(pairs(b"a=x\\\\\nb=y\n"), vec![p("a", "x\\"), p("b", "y")]);
        // A continuation followed by an empty line ends the logical line.
        assert_eq!(pairs(b"a=x\\\n\nb=y"), vec![p("a", "x"), p("b", "y")]);
    }

    #[test]
    fn escapes() {
        assert_eq!(
            pairs(b"k\\=e\\ y=\\t\\n\\r\\f\\q\\u00e9\\u00C9\\uD83D\\uDE00\n"),
            vec![p("k=e y", "\t\n\r\x0cqéÉ\u{1F600}")]
        );
        // OpenNLP manifests escape `!` and `:` in eosCharacters.
        assert_eq!(pairs(b"eosCharacters=.?\\!;\\:()\\u00AB\\u00BB'\"\n"), vec![p("eosCharacters", ".?!;:()«»'\"")]);
    }

    #[test]
    fn bytes_are_iso_8859_1() {
        assert_eq!(pairs(b"k=\xe9\xff"), vec![p("k", "\u{e9}\u{ff}")]);
    }

    #[test]
    fn a_repeated_key_keeps_its_place_and_its_last_value() {
        assert_eq!(pairs(b"a=1\nb=2\na=3\n"), vec![p("a", "3"), p("b", "2")]);
    }

    #[test]
    fn malformed_unicode_escapes_name_the_line() {
        assert!(matches!(
            Properties::load(b"a=1\n\nb=\\u12\n"),
            Err(Error::Properties { line: 3, message: "malformed \\uxxxx encoding" })
        ));
        assert!(matches!(Properties::load(b"b=\\u12g4\n"), Err(Error::Properties { line: 1, .. })));
    }
}
