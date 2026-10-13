//! The plain-text frequency dictionaries OpenNLP's
//! `FrequencyDictionaryLoader` reads: `word count` per line for unigrams,
//! `w1 w2 count` per line for bigrams.
//!
//! Columns are separated by runs of TAB and space; any other whitespace
//! (a no-break space, a vertical tab) is part of a column. The count is
//! decimal digits of any script (`Character.digit`), with no sign, point
//! or exponent, and at most `Long.MAX_VALUE`. Columns after the count are
//! ignored. A U+FEFF at the start of a line is dropped; a line that is
//! empty, blank (Java `String.isBlank`: `Character.isWhitespace` only) or
//! starts with `#` is skipped but still counts for the line numbers.
//! Lines end at LF, CR or CR LF, as `BufferedReader.readLine` reads them.

use std::fmt;

use turbo_unicode::java;

use super::symspell::SymSpell;

/// Readers of the two text formats.
pub struct FrequencyDictionary;

/// Why a line could not be read. The messages are OpenNLP's.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MalformedReason {
    /// A unigram line with fewer than two columns.
    UnigramColumnsMissing,
    /// A bigram line with fewer than three columns.
    BigramColumnsMissing,
    /// A count that starts with a minus sign and is otherwise digits.
    CountNegative,
    /// A count that is not digits, or does not fit in a Java `long`.
    CountNotInteger,
}

impl MalformedReason {
    pub fn message(self) -> &'static str {
        match self {
            MalformedReason::UnigramColumnsMissing => "expected 'word<sep>count'",
            MalformedReason::BigramColumnsMissing => "expected 'w1<sep>w2<sep>count'",
            MalformedReason::CountNegative => "count must not be negative",
            MalformedReason::CountNotInteger => "count is not an integer",
        }
    }
}

/// A line that does not have the expected shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MalformedLine {
    /// The 1-based number of the line.
    pub line_number: u64,
    pub reason: MalformedReason,
    /// The line with tabs written as `\t`, cut after 120 UTF-16 code units.
    pub preview: String,
}

/// The longest preview, in UTF-16 code units as OpenNLP measures it.
const MAX_PREVIEW: usize = 120;

impl MalformedLine {
    fn new(line_number: u64, line: &str, reason: MalformedReason) -> MalformedLine {
        let escaped = line.replace('\t', "\\t");
        let mut units = 0;
        let mut cut = escaped.len();
        for (i, c) in escaped.char_indices() {
            units += c.len_utf16();
            if units > MAX_PREVIEW {
                cut = i;
                break;
            }
        }
        let preview =
            if cut < escaped.len() { format!("\"{}...\"", &escaped[..cut]) } else { format!("\"{escaped}\"") };
        MalformedLine { line_number, reason, preview }
    }
}

impl fmt::Display for MalformedLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "malformed dictionary line {} ({}): {}", self.line_number, self.reason.message(), self.preview)
    }
}

impl std::error::Error for MalformedLine {}

impl FrequencyDictionary {
    /// Reads a unigram dictionary, handing each entry to `sink`, and
    /// returns the number of entries.
    pub fn read_unigrams(text: &str, mut sink: impl FnMut(&str, u64)) -> Result<u64, MalformedLine> {
        read(text, 2, MalformedReason::UnigramColumnsMissing, |columns, count| sink(columns[0], count))
    }

    /// Reads a bigram dictionary, handing each entry to `sink`, and returns
    /// the number of entries.
    pub fn read_bigrams(text: &str, mut sink: impl FnMut(&str, &str, u64)) -> Result<u64, MalformedLine> {
        read(text, 3, MalformedReason::BigramColumnsMissing, |columns, count| sink(columns[0], columns[1], count))
    }

    /// Adds every entry of a unigram dictionary to `engine`. Entries before
    /// a malformed line stay added.
    pub fn load_unigrams(engine: &mut SymSpell, text: &str) -> Result<u64, MalformedLine> {
        FrequencyDictionary::read_unigrams(text, |word, count| {
            engine.add(word, count);
        })
    }

    /// Adds every entry of a bigram dictionary to `engine`.
    pub fn load_bigrams(engine: &mut SymSpell, text: &str) -> Result<u64, MalformedLine> {
        FrequencyDictionary::read_bigrams(text, |w1, w2, count| engine.add_bigram(w1, w2, count))
    }
}

/// The lines of `text` as `BufferedReader.readLine` returns them.
fn lines(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let end = rest.find(['\n', '\r']).unwrap_or(rest.len());
        let line = &rest[..end];
        let after = match &rest.as_bytes()[end..] {
            [b'\r', b'\n', ..] => end + 2,
            [] => end,
            _ => end + 1,
        };
        rest = &rest[after..];
        Some(line)
    })
}

fn read(
    text: &str,
    columns_needed: usize,
    missing: MalformedReason,
    mut sink: impl FnMut(&[&str], u64),
) -> Result<u64, MalformedLine> {
    let mut entries = 0;
    let mut columns: Vec<&str> = Vec::with_capacity(4);
    for (n, line) in lines(text).enumerate() {
        let line_number = n as u64 + 1;
        let content = line.strip_prefix('\u{FEFF}').unwrap_or(line);
        if content.chars().all(java::is_whitespace) || content.starts_with('#') {
            continue;
        }
        columns.clear();
        columns.extend(content.trim_matches(java::is_whitespace).split(['\t', ' ']).filter(|c| !c.is_empty()));
        if columns.len() < columns_needed {
            return Err(MalformedLine::new(line_number, line, missing));
        }
        let count = parse_count(columns[columns_needed - 1]).map_err(|r| MalformedLine::new(line_number, line, r))?;
        sink(&columns, count);
        entries += 1;
    }
    Ok(entries)
}

/// A count column: decimal digits of any script. A leading minus sign is
/// reported as negative only when the rest is digits, as OpenNLP checks it.
fn parse_count(raw: &str) -> Result<u64, MalformedReason> {
    let digits = raw.strip_prefix('-');
    let negative = digits.is_some();
    let digits = digits.unwrap_or(raw);
    if digits.is_empty() || digits.chars().any(|c| java::digit(c).is_none()) {
        return Err(MalformedReason::CountNotInteger);
    }
    if negative {
        return Err(MalformedReason::CountNegative);
    }
    let mut count: u64 = 0;
    for c in digits.chars() {
        let d = u64::from(java::digit(c).expect("checked above"));
        count = count
            .checked_mul(10)
            .and_then(|v| v.checked_add(d))
            .filter(|&v| v <= i64::MAX as u64)
            .ok_or(MalformedReason::CountNotInteger)?;
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_end_at_lf_cr_and_crlf() {
        let all: Vec<&str> = lines("a\nb\r\nc\rd\n\ne").collect();
        assert_eq!(all, ["a", "b", "c", "d", "", "e"]);
        assert_eq!(lines("a\n").collect::<Vec<_>>(), ["a"]);
        assert_eq!(lines("").count(), 0);
        assert_eq!(lines("\r\n\r\n").collect::<Vec<_>>(), ["", ""]);
    }

    #[test]
    fn preview_cuts_at_120_utf16_units() {
        let long = "x".repeat(130);
        assert_eq!(
            MalformedLine::new(1, &long, MalformedReason::CountNotInteger).preview,
            format!("\"{}...\"", "x".repeat(120))
        );
        let exact = "y".repeat(120);
        assert_eq!(MalformedLine::new(1, &exact, MalformedReason::CountNotInteger).preview, format!("\"{exact}\""));
    }
}
