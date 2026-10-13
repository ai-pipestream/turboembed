//! The frequency lexicon of the unigram segmenter, read as OpenNLP's
//! `UnigramSegmenter.load` reads it.
//!
//! One entry per line: the word, whitespace, its count, and optionally
//! whitespace and anything else (a part of speech tag in jieba's
//! `dict.txt`), which is ignored. The rules follow the Java:
//!
//! - Lines end at `\n`, `\r` or `\r\n`, as `BufferedReader.readLine`
//!   splits them. A line is trimmed of Unicode `White_Space` at both ends
//!   whatever the whitespace mode (`StringUtil.trimUnicodeWhitespace`),
//!   and an empty line is skipped.
//! - The word ends at the first whitespace character of the line in the
//!   configured mode; a line without one has no count and is refused.
//! - The count is the run of non-whitespace after it, read as
//!   `Long.parseLong` reads it: an optional `+` or `-`, then decimal
//!   digits of any script in the Basic Multilingual Plane
//!   (`Character.digit`), within the range of a Java `long`. A count that
//!   is not positive is refused.
//! - A word listed twice gets the sum of its counts. The total of all
//!   counts must fit in a `long`, and the number of distinct words may not
//!   pass the limit (OpenNLP's `ResourceLimits.MAX_ENTRIES`, ten million
//!   unless the JVM is told otherwise).
//! - Each word's score is `log(count) - log(total)`, and a character the
//!   lexicon does not list scores `log(0.5) - log(total)`, below every
//!   listed word.
//!
//! The file must be UTF-8; the Java takes any charset. A byte order mark
//! is not removed (Java's UTF-8 decoder keeps it too), so it is part of
//! the first word.

use std::collections::HashMap;
use std::fmt;
use std::hash::BuildHasherDefault;
use std::path::Path;

use turbo_maxent::{Digest, sha256};
use turbo_unicode::WhitespaceMode;
use turbo_unicode::java;

use super::trie::{TrieBuilder, WordTrie};
use crate::detok::dictionary::Fnv;

/// The most distinct words a lexicon may hold unless the caller sets
/// another limit: OpenNLP's `ResourceLimits.MAX_ENTRIES` default.
pub const DEFAULT_MAX_ENTRIES: usize = 10_000_000;

/// Why a lexicon could not be read. The messages of the line errors are
/// the Java's `IOException` messages; line numbers count from 1.
#[derive(Debug)]
pub enum LexiconError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The bytes are not UTF-8; `offset` is where the first bad sequence
    /// starts. Java throws `MalformedInputException` here.
    Utf8 { offset: usize },
    /// The line holds a word and no whitespace after it.
    NoCount { line: usize },
    /// The count is not a Java `long` in decimal digits.
    MalformedCount { line: usize },
    /// The count is zero or negative.
    CountNotPositive { line: usize },
    /// A new word would pass the limit on distinct words.
    TooManyEntries { limit: usize },
    /// The sum of the counts passes `Long.MAX_VALUE` at this line.
    TotalOverflow { line: usize },
    /// No line holds a word.
    Empty,
}

impl fmt::Display for LexiconError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexiconError::Io(e) => write!(f, "lexicon: {e}"),
            LexiconError::Utf8 { offset } => write!(f, "lexicon: not UTF-8 at byte {offset}"),
            LexiconError::NoCount { line } => write!(f, "lexicon line {line} has no count"),
            LexiconError::MalformedCount { line } => write!(f, "malformed count at lexicon line {line}"),
            LexiconError::CountNotPositive { line } => write!(f, "count must be positive at lexicon line {line}"),
            LexiconError::TooManyEntries { limit } => write!(f, "lexicon entry count exceeds safe limit of {limit}"),
            LexiconError::TotalOverflow { line } => write!(f, "lexicon count total overflows at line {line}"),
            LexiconError::Empty => f.write_str("the lexicon lists no words"),
        }
    }
}

impl std::error::Error for LexiconError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LexiconError::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// A loaded frequency lexicon: the word trie with each word's log
/// probability, the score of an unlisted character, and what identifies
/// the file.
#[derive(Debug, Clone)]
pub struct UnigramLexicon {
    trie: WordTrie,
    unknown: f64,
    total: i64,
    whitespace: WhitespaceMode,
    sha256: Digest,
}

impl UnigramLexicon {
    /// Reads the lexicon file at `path`, splitting each line in
    /// `whitespace` mode, with the default limit on distinct words.
    pub fn read(path: impl AsRef<Path>, whitespace: WhitespaceMode) -> Result<UnigramLexicon, LexiconError> {
        let bytes = std::fs::read(path).map_err(LexiconError::Io)?;
        UnigramLexicon::parse(&bytes, whitespace)
    }

    /// Reads a lexicon from its bytes with the default limit on distinct
    /// words.
    pub fn parse(bytes: &[u8], whitespace: WhitespaceMode) -> Result<UnigramLexicon, LexiconError> {
        UnigramLexicon::parse_with_limit(bytes, whitespace, DEFAULT_MAX_ENTRIES)
    }

    /// Reads a lexicon from its bytes, refusing more than `max_entries`
    /// distinct words.
    pub fn parse_with_limit(
        bytes: &[u8],
        whitespace: WhitespaceMode,
        max_entries: usize,
    ) -> Result<UnigramLexicon, LexiconError> {
        let text = std::str::from_utf8(bytes).map_err(|e| LexiconError::Utf8 { offset: e.valid_up_to() })?;
        // About one line per word, and a word line of a Chinese lexicon is
        // about 15 bytes.
        let mut builder = TrieBuilder::with_capacity(bytes.len() / 15);
        let mut total: i64 = 0;
        for (index, raw) in Lines::new(text).enumerate() {
            let line_number = index + 1;
            let line = trim_unicode_whitespace(raw);
            if line.is_empty() {
                continue;
            }
            let Some(word_end) = line.find(|c: char| whitespace.is_whitespace(c as u32)) else {
                return Err(LexiconError::NoCount { line: line_number });
            };
            let word = &line[..word_end];
            let rest = line[word_end..].trim_start_matches(|c: char| whitespace.is_whitespace(c as u32));
            let count_end = rest.find(|c: char| whitespace.is_whitespace(c as u32)).unwrap_or(rest.len());
            let count =
                parse_java_long(&rest[..count_end]).ok_or(LexiconError::MalformedCount { line: line_number })?;
            if count <= 0 {
                return Err(LexiconError::CountNotPositive { line: line_number });
            }
            let node = builder.node(word);
            if builder.count(node) == 0 && builder.words() >= max_entries {
                return Err(LexiconError::TooManyEntries { limit: max_entries });
            }
            // The merged count of one word never passes the total, which is
            // checked next, so the sum cannot overflow before it does.
            builder.add(node, count);
            total = total.checked_add(count).ok_or(LexiconError::TotalOverflow { line: line_number })?;
        }
        if builder.words() == 0 {
            return Err(LexiconError::Empty);
        }
        let log_total = java_log(total as f64);
        // Counts repeat (most words of a large lexicon have a small count),
        // so each distinct count's logarithm is computed once.
        let mut logs: HashMap<i64, f64, BuildHasherDefault<Fnv>> = HashMap::default();
        let trie = builder.finish(|count| *logs.entry(count).or_insert_with(|| java_log(count as f64)) - log_total);
        Ok(UnigramLexicon { trie, unknown: java_log(0.5) - log_total, total, whitespace, sha256: sha256(bytes) })
    }

    /// The score of one character the lexicon does not list.
    pub fn unknown_log_probability(&self) -> f64 {
        self.unknown
    }

    /// The score of `word`, if the lexicon lists it.
    pub fn log_probability(&self, word: &str) -> Option<f64> {
        self.trie.get(word)
    }

    /// The number of distinct words.
    pub fn len(&self) -> usize {
        self.trie.words()
    }

    /// Whether the lexicon is empty, which a loaded lexicon never is.
    pub fn is_empty(&self) -> bool {
        self.trie.words() == 0
    }

    /// The sum of all counts.
    pub fn total(&self) -> i64 {
        self.total
    }

    /// The whitespace mode the lines were split in.
    pub fn whitespace(&self) -> WhitespaceMode {
        self.whitespace
    }

    /// The SHA-256 of the file's bytes.
    pub fn sha256(&self) -> Digest {
        self.sha256
    }

    pub(crate) fn trie(&self) -> &WordTrie {
        &self.trie
    }
}

/// `Math.log`, which OpenNLP calls for the scores: a correctly rounded
/// logarithm, which matched the `Math.log` of the JVM build the goldens
/// are made with on the integers compared (`turbo_jvm::math` says which,
/// and the one where they differ). fdlibm, `StrictMath.log`, differs from
/// that JVM on about one integer in 150, which would change which of two
/// paths with equal exact sums wins.
#[inline]
pub(crate) fn java_log(x: f64) -> f64 {
    turbo_jvm::math::log(x)
}

/// The lines of `text` as `BufferedReader.readLine` returns them: ended by
/// `\n`, `\r` or `\r\n`, the end itself left out, and no empty last line
/// after a final line end.
struct Lines<'a> {
    rest: &'a str,
}

impl<'a> Lines<'a> {
    fn new(text: &'a str) -> Lines<'a> {
        Lines { rest: text }
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
fn trim_unicode_whitespace(line: &str) -> &str {
    line.trim_matches(|c: char| WhitespaceMode::Unicode.is_whitespace(c as u32))
}

/// `Long.parseLong(s)`: an optional sign (not alone), then one or more
/// characters `Character.digit(c, 10)` accepts, within the range of a
/// Java `long`. `Character.digit` sees UTF-16 code units, so a digit
/// outside the Basic Multilingual Plane is two units that are not digits.
/// Leading zeros are allowed, and `-9223372036854775808` is read.
fn parse_java_long(s: &str) -> Option<i64> {
    let (negative, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    // Accumulate negatively, as the Java does, so that the most negative
    // value is reachable.
    let mut value: i64 = 0;
    for c in digits.chars() {
        if c as u32 > 0xFFFF {
            return None;
        }
        let d = java::digit(c)?;
        value = value.checked_mul(10)?.checked_sub(i64::from(d))?;
    }
    if negative { Some(value) } else { value.checked_neg() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<UnigramLexicon, LexiconError> {
        UnigramLexicon::parse(text.as_bytes(), WhitespaceMode::Unicode)
    }

    fn message(text: &str) -> String {
        parse(text).expect_err("an error").to_string()
    }

    #[test]
    fn lines_split_as_read_line_splits_them() {
        let lines: Vec<&str> = Lines::new("a\nb\r\nc\rd\n\ne").collect();
        assert_eq!(lines, ["a", "b", "c", "d", "", "e"]);
        assert_eq!(Lines::new("a\n").collect::<Vec<_>>(), ["a"]);
        assert_eq!(Lines::new("\r\r\n").collect::<Vec<_>>(), ["", ""]);
        assert_eq!(Lines::new("").count(), 0);
        // U+2028 and U+0085 do not end a line.
        assert_eq!(Lines::new("a\u{2028}b\u{85}c").count(), 1);
    }

    #[test]
    fn java_long_parsing() {
        assert_eq!(parse_java_long("42"), Some(42));
        assert_eq!(parse_java_long("+42"), Some(42));
        assert_eq!(parse_java_long("-42"), Some(-42));
        assert_eq!(parse_java_long("0007"), Some(7));
        assert_eq!(parse_java_long("9223372036854775807"), Some(i64::MAX));
        assert_eq!(parse_java_long("-9223372036854775808"), Some(i64::MIN));
        assert_eq!(parse_java_long("9223372036854775808"), None);
        assert_eq!(parse_java_long("+"), None);
        assert_eq!(parse_java_long("-"), None);
        assert_eq!(parse_java_long(""), None);
        assert_eq!(parse_java_long("1a"), None);
        assert_eq!(parse_java_long("1.0"), None);
        assert_eq!(parse_java_long("++1"), None);
        // Fullwidth, Arabic-Indic and Devanagari digits are digits to Java.
        assert_eq!(parse_java_long("\u{ff11}\u{ff12}"), Some(12));
        assert_eq!(parse_java_long("\u{663}\u{660}"), Some(30));
        assert_eq!(parse_java_long("\u{967}"), Some(1));
        // A mathematical digit is outside the BMP: two code units, no digit.
        assert_eq!(parse_java_long("\u{1d7cf}"), None);
        // Chinese numerals are letters (Lo), not digits.
        assert_eq!(parse_java_long("\u{4e00}"), None);
    }

    #[test]
    fn counts_and_scores() {
        let lexicon = parse("\u{6211} 5000 r\n\u{6765}\u{5230} 2000 v\n\u{6211} 1000\n").unwrap();
        assert_eq!(lexicon.len(), 2);
        assert_eq!(lexicon.total(), 8000);
        let log_total = java_log(8000.0);
        assert_eq!(lexicon.log_probability("\u{6211}"), Some(java_log(6000.0) - log_total));
        assert_eq!(lexicon.log_probability("\u{6765}\u{5230}"), Some(java_log(2000.0) - log_total));
        assert_eq!(lexicon.log_probability("\u{6765}"), None);
        assert_eq!(lexicon.unknown_log_probability(), java_log(0.5) - log_total);
    }

    #[test]
    fn malformed_lexicons_are_refused_with_the_java_messages() {
        // The four cases of OpenNLP's testMalformedLexiconsFailLoud.
        assert_eq!(message("word\n"), "lexicon line 1 has no count");
        assert_eq!(message("word abc\n"), "malformed count at lexicon line 1");
        assert_eq!(message("word 0\n"), "count must be positive at lexicon line 1");
        assert_eq!(message("\n\n"), "the lexicon lists no words");
        assert_eq!(message("a 1\n\nb -3\n"), "count must be positive at lexicon line 3");
        let max = i64::MAX;
        assert_eq!(message(&format!("first {max}\nsecond 1\n")), "lexicon count total overflows at line 2");
        let e = UnigramLexicon::parse_with_limit(b"first 1\nsecond 1\n", WhitespaceMode::Unicode, 1).err().unwrap();
        assert_eq!(e.to_string(), "lexicon entry count exceeds safe limit of 1");
        // A word already listed does not count against the limit.
        let lexicon = UnigramLexicon::parse_with_limit(b"first 1\nfirst 2\n", WhitespaceMode::Unicode, 1).unwrap();
        assert_eq!(lexicon.total(), 3);
    }

    #[test]
    fn invalid_utf8_is_refused() {
        let e = UnigramLexicon::parse(b"w\xC3 1\n", WhitespaceMode::Unicode).err().unwrap();
        assert!(matches!(e, LexiconError::Utf8 { offset: 1 }), "{e:?}");
    }

    #[test]
    fn lines_are_trimmed_of_unicode_whitespace_in_both_modes() {
        // OpenNLP's testLeadingIdeographicSpaceIsTrimmed, and U+0085, which
        // only the Unicode mode calls whitespace, trimmed in legacy mode too.
        for mode in [WhitespaceMode::Unicode, WhitespaceMode::Legacy] {
            let lexicon = UnigramLexicon::parse("\u{3000}\u{6211} 5000 r\n\u{85}b 2\u{85}\n".as_bytes(), mode).unwrap();
            assert!(lexicon.log_probability("\u{6211}").is_some(), "{mode:?}");
            assert!(lexicon.log_probability("b").is_some(), "{mode:?}");
        }
    }

    #[test]
    fn the_mode_decides_where_the_word_ends() {
        // U+001C separates in legacy mode only, and is not trimmed.
        let text = "a\u{1c}7 3\n";
        let unicode = UnigramLexicon::parse(text.as_bytes(), WhitespaceMode::Unicode).unwrap();
        assert_eq!(unicode.len(), 1);
        assert!(unicode.log_probability("a\u{1c}7").is_some());
        let legacy = UnigramLexicon::parse(text.as_bytes(), WhitespaceMode::Legacy).unwrap();
        assert!(legacy.log_probability("a").is_some());
        assert_eq!(legacy.total(), 7);
        // U+0085 inside a line separates in Unicode mode only.
        let text = "x\u{85}5 2\n";
        assert_eq!(UnigramLexicon::parse(text.as_bytes(), WhitespaceMode::Unicode).unwrap().total(), 5);
        assert_eq!(UnigramLexicon::parse(text.as_bytes(), WhitespaceMode::Legacy).unwrap().total(), 2);
        // In legacy mode a trailing U+001C is not trimmed, so the count is
        // empty.
        let e = UnigramLexicon::parse("w\u{1c}\n".as_bytes(), WhitespaceMode::Legacy).err().unwrap();
        assert_eq!(e.to_string(), "malformed count at lexicon line 1");
    }

    #[test]
    fn a_line_starting_with_a_legacy_separator_lists_the_empty_word() {
        // U+001C is not trimmed (it is not Unicode White_Space) but ends the
        // word in legacy mode, so the word is empty. Java stores it on the
        // trie's root, where no walk scores it; it still counts in the total
        // and as an entry.
        let lexicon = UnigramLexicon::parse("\u{1c}5 3\nb 1\n".as_bytes(), WhitespaceMode::Legacy).unwrap();
        assert_eq!((lexicon.total(), lexicon.len()), (6, 2));
        assert_eq!(lexicon.log_probability(""), None);
        let e =
            UnigramLexicon::parse_with_limit("\u{1c}5 3\nb 1\n".as_bytes(), WhitespaceMode::Legacy, 1).err().unwrap();
        assert_eq!(e.to_string(), "lexicon entry count exceeds safe limit of 1");
        let s = super::super::UnigramSegmenter::new(lexicon, WhitespaceMode::Legacy);
        let mut out = Vec::new();
        s.segment("ab", &mut out);
        assert_eq!(out.len(), 2);
        // In Unicode mode the same line lists the word U+001C 5 with count 3.
        let unicode = UnigramLexicon::parse("\u{1c}5 3\nb 1\n".as_bytes(), WhitespaceMode::Unicode).unwrap();
        assert_eq!((unicode.total(), unicode.len()), (4, 2));
        assert!(unicode.log_probability("\u{1c}5").is_some());
    }

    #[test]
    fn a_byte_order_mark_is_part_of_the_first_word() {
        let lexicon = parse("\u{feff}a 1\nb 1\n").unwrap();
        assert!(lexicon.log_probability("a").is_none());
        assert!(lexicon.log_probability("\u{feff}a").is_some());
    }

    #[test]
    fn the_tag_and_anything_after_the_count_are_ignored() {
        let lexicon = parse("a 3 n extra words\nb\t\t4\u{3000}x\n").unwrap();
        assert_eq!(lexicon.total(), 7);
    }

    #[test]
    fn a_long_word_loads() {
        // OpenNLP's testLongLexiconWordLoads.
        let word = "a".repeat(20_000);
        let lexicon = parse(&format!("{word} 1\n")).unwrap();
        assert!(lexicon.log_probability(&word).is_some());
    }
}
