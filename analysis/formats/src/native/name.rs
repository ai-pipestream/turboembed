//! The name format: `NameSampleDataStream` and `NameSample.parse`.
//!
//! One sentence per line, tokens separated by whitespace, each name
//! between a `<START:type>` (or `<START>`) token and an `<END>` token:
//!
//! ```text
//! <START:person> Pierre Vinken <END> , 61 years old , will join the board .
//! ```
//!
//! A line that is empty after `String.trim` is not a sample; it marks the
//! next sample as the first of a new document
//! ([`NameSample::clear_adaptive_data`]).
//!
//! What OpenNLP does with unusual lines, reproduced here:
//!
//! - A `<START>` without a type takes the type of the last typed start
//!   tag of the same line, or `default` when there was none: the parser
//!   keeps one type variable per line and only a typed tag sets it.
//! - A name still open at the end of the line is dropped without an
//!   error.
//! - `<START:>` is refused ("Missing a name type"), and so is a start tag
//!   inside a name and an `<END>` outside one; the stream throws and the
//!   next read goes on with the next line.
//! - A tag token must be the whole token: `<START:a>b` is a token, and so
//!   is `<START:a:b>`, since a type holds no `:` or `>`.
//! - An empty name followed at the same token by a longer one
//!   (`<START:a> <END> <START:b> x <END>`) is refused with the
//!   `RuntimeException` of `NameSample`'s overlap check, because the
//!   names are sorted longer first.

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

use crate::error::FormatError;
use crate::sample::name::{DEFAULT_NAME_TYPE, Name, NameSample};
use crate::source::{Line, Lines, Piece, java_trim, whitespace_tokens};

const END_TAG: &str = "<END>";

/// Reads name samples from the name format.
#[derive(Clone, Debug)]
pub struct NameReader<'a> {
    lines: Lines<'a>,
    mode: WhitespaceMode,
    default_type: &'a str,
}

impl<'a> NameReader<'a> {
    /// A reader of `source`, splitting lines on whitespace in `mode`.
    pub fn new(source: &'a str, mode: WhitespaceMode) -> Result<NameReader<'a>, FormatError> {
        Ok(NameReader { lines: Lines::new(source)?, mode, default_type: DEFAULT_NAME_TYPE })
    }

    /// The type of a name whose start tag has none, before any typed tag
    /// on its line (`NameSample.parse(String, String, boolean)`).
    pub fn with_default_type(mut self, default_type: &'a str) -> NameReader<'a> {
        self.default_type = default_type;
        self
    }
}

impl<'a> Iterator for NameReader<'a> {
    type Item = Result<NameSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut clear = false;
        loop {
            let line = self.lines.next()?;
            if java_trim(line.piece).text.is_empty() {
                clear = true;
                continue;
            }
            return Some(parse_line(line, clear, self.mode, self.default_type));
        }
    }
}

/// `NameSample.parse` of one line.
pub fn parse_line<'a>(
    line: Line<'a>,
    clear_adaptive_data: bool,
    mode: WhitespaceMode,
    default_type: &'a str,
) -> Result<NameSample<'a>, FormatError> {
    let parts: Vec<Piece<'a>> = whitespace_tokens(line.piece, mode).collect();
    let mut tokens = Vec::with_capacity(parts.len());
    let mut names = Vec::new();
    let mut name_type = default_type;
    // The token index and source offset of the open name's start tag.
    let mut open: Option<(u32, u32)> = None;
    for (pi, part) in parts.iter().enumerate() {
        if let Some(kind) = start_tag(part.text) {
            if open.is_some() {
                return Err(FormatError::NestedName { line: line.number, context: context(&parts, pi) });
            }
            open = Some((tokens.len() as u32, part.span.start));
            if let Some(kind) = kind {
                if kind.is_empty() {
                    return Err(FormatError::MissingNameType { line: line.number, context: context(&parts, pi) });
                }
                name_type = kind;
            }
        } else if part.text == END_TAG {
            let Some((start, from)) = open.take() else {
                return Err(FormatError::UnexpectedEnd { line: line.number, context: context(&parts, pi) });
            };
            names.push(Name {
                start,
                end: tokens.len() as u32,
                kind: Some(name_type),
                source: Span::new(from, part.span.end),
            });
        } else {
            tokens.push(part.field());
        }
    }
    NameSample::new(tokens, names, clear_adaptive_data, line.span(), line.number)
}

/// Whether `token` matches `<START(:([^:>\s]*))?>` as a whole: `None` if
/// not, `Some(None)` for `<START>` and `Some(Some(type))` for a typed tag,
/// whose type may be empty. `\s` is Java's ASCII whitespace class.
fn start_tag(token: &str) -> Option<Option<&str>> {
    let inner = token.strip_prefix("<START")?.strip_suffix('>')?;
    if inner.is_empty() {
        return Some(None);
    }
    let kind = inner.strip_prefix(':')?;
    let excluded = |c: char| matches!(c, ':' | '>' | ' ' | '\t' | '\n' | '\u{b}' | '\u{c}' | '\r');
    (!kind.contains(excluded)).then_some(Some(kind))
}

/// `NameSample.errorTokenWithContext`: up to two parts before the one at
/// `index`, that part between `###`, and up to two parts after it.
fn context(parts: &[Piece<'_>], index: usize) -> String {
    let mut out = String::new();
    if index > 1 {
        out.push_str(parts[index - 2].text);
        out.push(' ');
    }
    if index > 0 {
        out.push_str(parts[index - 1].text);
        out.push(' ');
    }
    out.push_str("###");
    out.push_str(parts[index].text);
    out.push_str("### ");
    if index + 1 < parts.len() {
        out.push_str(parts[index + 1].text);
        out.push(' ');
    }
    if index + 2 < parts.len() {
        out.push_str(parts[index + 2].text);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(s: &str) -> Vec<Result<NameSample<'_>, FormatError>> {
        NameReader::new(s, WhitespaceMode::Unicode).unwrap().collect()
    }

    fn names<'a>(s: &NameSample<'a>) -> Vec<(u32, u32, Option<&'a str>)> {
        s.names().iter().map(|n| (n.start, n.end, n.kind)).collect()
    }

    #[test]
    fn start_tags() {
        assert_eq!(start_tag("<START>"), Some(None));
        assert_eq!(start_tag("<START:per>"), Some(Some("per")));
        assert_eq!(start_tag("<START:>"), Some(Some("")));
        assert_eq!(start_tag("<START:a:b>"), None);
        assert_eq!(start_tag("<START:a>b>"), None);
        assert_eq!(start_tag("<START>>"), None);
        assert_eq!(start_tag("<START"), None);
        assert_eq!(start_tag("<STARTx>"), None);
        assert_eq!(start_tag("<START:\u{e9}\u{1F600}>"), Some(Some("\u{e9}\u{1F600}")));
    }

    #[test]
    fn reads_names_with_offsets() {
        let src = "<START:person> Pierre Vinken <END> , 61 .";
        let samples = read(src);
        let s = samples[0].as_ref().unwrap();
        let toks: Vec<&str> = s.tokens().iter().map(|t| t.as_str()).collect();
        assert_eq!(toks, ["Pierre", "Vinken", ",", "61", "."]);
        assert_eq!(names(s), [(0, 2, Some("person"))]);
        assert_eq!(&src[s.names()[0].source.range()], "<START:person> Pierre Vinken <END>");
        for t in s.tokens() {
            assert_eq!(&src[t.span.range()], t.as_str());
        }
        assert_eq!(s.to_string(), src);
    }

    #[test]
    fn untyped_start_takes_the_last_type_of_the_line() {
        let samples = read("<START> a <END> <START:loc> b <END> <START> c <END>\n<START> d <END>");
        let s = samples[0].as_ref().unwrap();
        assert_eq!(names(s), [(0, 1, Some("default")), (1, 2, Some("loc")), (2, 3, Some("loc"))]);
        assert_eq!(names(samples[1].as_ref().unwrap()), [(0, 1, Some("default"))]);
    }

    #[test]
    fn open_name_at_the_end_is_dropped() {
        let samples = read("a <START:x> b c");
        assert!(samples[0].as_ref().unwrap().names().is_empty());
    }

    #[test]
    fn blank_lines_clear_adaptive_data() {
        let samples = read("\n \t\na\nb\n\n\nc\n\n");
        let flags: Vec<bool> = samples.iter().map(|s| s.as_ref().unwrap().clear_adaptive_data()).collect();
        assert_eq!(flags, [true, false, true]);
        assert_eq!(samples[0].as_ref().unwrap().to_string(), "\na ");
    }

    #[test]
    fn errors_name_their_context_and_the_stream_goes_on() {
        let samples = read("a b <START:x> c <START:y> d e f\n<END> x\n<START:> y\nok");
        let messages: Vec<String> = samples.iter().filter_map(|s| s.as_ref().err()).map(|e| e.to_string()).collect();
        assert_eq!(
            messages,
            [
                "Found unexpected annotation while handling a name sequence: <START:x> c ###<START:y>### d e",
                "Found unexpected annotation: ###<END>### x ",
                "Missing a name type: ###<START:>### y ",
            ]
        );
        assert_eq!(samples[0].as_ref().unwrap_err().line(), Some(1));
        assert!(samples[3].is_ok());
    }

    #[test]
    fn an_empty_name_before_a_longer_one_overlaps() {
        let samples = read("<START:a> <END> <START:b> x <END>");
        let e = samples[0].as_ref().unwrap_err();
        assert_eq!(e.java_class(), Some("java.lang.RuntimeException"));
        assert_eq!(e.to_string(), "name spans [0..1) b and [0..0) a are overlapped in file: null, sentence: [x]");
    }

    #[test]
    fn writes_one_token_samples_with_their_trailing_space() {
        let samples = read("x\n<START:t> x <END>\n<START:t> <END>\n<START:a> <END> <START:b> <END> x");
        let written: Vec<String> = samples.iter().map(|s| s.as_ref().unwrap().to_string()).collect();
        assert_eq!(written, ["x ", "<START:t> x  <END>", " <END>", "<START:a> <END> <START:b> <END> x "]);
    }
}
