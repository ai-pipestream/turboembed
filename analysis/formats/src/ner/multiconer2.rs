//! The MultiCoNER v2 kept-corpus reader (issue #1533).
//!
//! MultiCoNER v2, the multilingual complex named entity recognition corpus
//! (12 languages), is a kept corpus here under CC BY 4.0
//! (docs/licences.md). Its files hold one sentence per line, a JSON object
//! per sentence, in the pinned form
//!
//! ```text
//! {"tokens": ["John", "works", "at", "Siemens"], "ner_tags": ["B-PER", "O", "O", "B-ORG"]}
//! ```
//!
//! Both values are arrays of JSON strings of equal length; a token's tag is
//! `O`, or `B-` or `I-` with the name type after the prefix, kept as
//! written. The parser below is a small checked reader for exactly this
//! form: any other key, a duplicate key, a non-string value, a structural
//! character out of place, a bad escape or a cut inside a surrogate pair is
//! refused with an error naming the line. It is not a general JSON parser.
//!
//! IOB2 is read as `Conll03NameSampleStream` reads it: a name starts at a
//! `B-` tag, or at an `I-` tag with no name open or whose type differs from
//! the open name's; the type of a name is its first tag after the prefix.
//! There is no OpenNLP stream for MultiCoNER v2: the reader follows the
//! conventions of the ported readers of this module, but its errors are its
//! own and carry no JVM message.
//!
//! # What is pinned here
//!
//! - A row decodes independently, so a refused row ends only its own read:
//!   the reader yields the error and goes on with the next line.
//! - A `ner_tags` value is a plain IOB2 tag of the corpus, without escapes,
//!   so a name's type is the source's own bytes and a tag with an escape is
//!   refused.
//! - A token's span is the byte span of its JSON string's content in the
//!   source line. A token without escapes is the source's own bytes there;
//!   a token with escapes holds its decoded text and the span of the raw
//!   content it was decoded from.
//! - The sample's source is the whole line, and no sample clears adaptive
//!   data: the corpus marks no document starts.

use std::fmt;

use turbo_analysis::Span;

use crate::error::FormatError;
use crate::sample::{Name, NameSample};
use crate::source::{Field, Lines};

/// What a MultiCoNER v2 reader refuses. `Display` is the message the error
/// gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MulticonerError {
    /// A line that is not the pinned object form: `at` is the byte offset
    /// where the parser stopped and `message` what it expected there.
    Syntax { line: u32, at: usize, message: &'static str },
    /// A row whose `tokens` and `ner_tags` arrays differ in length.
    Lengths { line: u32, tokens: usize, tags: usize },
    /// A row with an empty `tokens` array: the corpus has a token per row.
    Empty { line: u32 },
    /// A name tag that is not `O`, or `B-` or `I-` with a non-empty type.
    InvalidTag { line: u32, tag: String },
    /// A name or sample the shared model refused; never reached by a
    /// well-formed tag sequence.
    Sample { line: u32, source: FormatError },
}

impl fmt::Display for MulticonerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MulticonerError::Syntax { line, at, message } => {
                write!(f, "line {line}: expected {message} at byte {at}")
            }
            MulticonerError::Lengths { line, tokens, tags } => {
                write!(f, "line {line}: tokens has {tokens} values, ner_tags has {tags}")
            }
            MulticonerError::Empty { line } => write!(f, "line {line}: the tokens array is empty"),
            MulticonerError::InvalidTag { tag, .. } => write!(f, "Invalid tag: {tag}"),
            MulticonerError::Sample { source, .. } => source.fmt(f),
        }
    }
}

impl std::error::Error for MulticonerError {}

/// The parser's refusal of a row: each variant is the byte offset of what
/// went wrong with what was expected there.
enum Syntax {
    /// A structural character or value the pinned form does not have.
    Expected(usize, &'static str),
    /// A backslash cut that is not one of JSON's escapes.
    Escape(usize),
    /// A `\u` cut that is not four hexadecimal digits.
    Hex(usize),
    /// Half of a surrogate pair without its other half.
    Surrogate(usize),
    /// An unescaped control character in a string.
    Control(usize),
}

impl Syntax {
    fn at(&self) -> usize {
        match self {
            Syntax::Expected(at, _)
            | Syntax::Escape(at)
            | Syntax::Hex(at)
            | Syntax::Surrogate(at)
            | Syntax::Control(at) => *at,
        }
    }

    fn message(&self) -> &'static str {
        match self {
            Syntax::Expected(_, m) => m,
            Syntax::Escape(_) => "one of the escapes \\\" \\\\ \\/ \\b \\f \\n \\r \\t \\u",
            Syntax::Hex(_) => "four hexadecimal digits after \\u",
            Syntax::Surrogate(_) => "a whole surrogate pair after \\u",
            Syntax::Control(_) => "no unescaped control character in a string",
        }
    }
}

/// A JSON string of a row: its decoded text and the span of its raw
/// content between the quotes.
struct JsonString {
    text: String,
    span: Span,
}

/// A checked parser for one pinned row of a MultiCoNER v2 file, reading
/// `text` (one line of the source) with `pos` always on a character
/// boundary: it advances past ASCII punctuation a byte at a time and past
/// other characters whole.
struct Row<'p> {
    text: &'p str,
    pos: usize,
}

impl<'p> Row<'p> {
    fn new(text: &'p str) -> Row<'p> {
        Row { text, pos: 0 }
    }

    fn byte(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.byte(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, b: u8, what: &'static str) -> Result<(), Syntax> {
        if self.byte() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(Syntax::Expected(self.pos, what))
        }
    }

    /// The four hexadecimal digits after a `\u` cut.
    fn hex4(&mut self) -> Result<u32, Syntax> {
        let mut v = 0;
        for _ in 0..4 {
            let digit = match self.byte() {
                Some(b @ b'0'..=b'9') => b - b'0',
                Some(b @ b'a'..=b'f') => b - b'a' + 10,
                Some(b @ b'A'..=b'F') => b - b'A' + 10,
                _ => return Err(Syntax::Hex(self.pos)),
            };
            v = v * 16 + u32::from(digit);
            self.pos += 1;
        }
        Ok(v)
    }

    /// A `\u` cut: a scalar value, or half of a surrogate pair, which must
    /// be followed by its other half in the same string.
    fn unicode(&mut self, out: &mut String) -> Result<(), Syntax> {
        let u = self.hex4()?;
        if (0xD800..0xDC00).contains(&u) {
            if self.byte() == Some(b'\\') && self.text.as_bytes().get(self.pos + 1) == Some(&b'u') {
                self.pos += 2;
                let low = self.hex4()?;
                if !(0xDC00..0xE000).contains(&low) {
                    return Err(Syntax::Surrogate(self.pos));
                }
                let c = 0x10000 + ((u - 0xD800) << 10) + (low - 0xDC00);
                out.push(char::from_u32(c).expect("a combined surrogate pair"));
                Ok(())
            } else {
                Err(Syntax::Surrogate(self.pos))
            }
        } else if (0xDC00..0xE000).contains(&u) {
            Err(Syntax::Surrogate(self.pos))
        } else {
            out.push(char::from_u32(u).expect("a non-surrogate scalar"));
            Ok(())
        }
    }

    /// A JSON string: its decoded text and the span of its raw content
    /// bytes. With no escape the decoded text is the content itself.
    fn string(&mut self) -> Result<JsonString, Syntax> {
        self.expect(b'"', "a string")?;
        let start = self.pos;
        let mut escaped = false;
        let mut decoded = String::new();
        loop {
            match self.byte().ok_or(Syntax::Expected(self.pos, "a closing quote"))? {
                b'"' => {
                    let end = self.pos;
                    self.pos += 1;
                    let text = if escaped { decoded } else { self.text[start..end].to_owned() };
                    return Ok(JsonString { text, span: Span::new(start as u32, end as u32) });
                }
                b'\\' => {
                    let cut = self.pos;
                    self.pos += 1;
                    let escape = self.byte().ok_or(Syntax::Escape(cut))?;
                    self.pos += 1;
                    if !escaped {
                        escaped = true;
                        decoded = self.text[start..cut].to_owned();
                    }
                    match escape {
                        b'"' => decoded.push('"'),
                        b'\\' => decoded.push('\\'),
                        b'/' => decoded.push('/'),
                        b'b' => decoded.push('\u{8}'),
                        b'f' => decoded.push('\u{c}'),
                        b'n' => decoded.push('\n'),
                        b'r' => decoded.push('\r'),
                        b't' => decoded.push('\t'),
                        b'u' => self.unicode(&mut decoded)?,
                        _ => return Err(Syntax::Escape(cut)),
                    }
                }
                b if b < 0x20 => return Err(Syntax::Control(self.pos)),
                _ if !escaped => self.pos += 1,
                _ => {
                    let c = self.text[self.pos..].chars().next().expect("a character");
                    decoded.push(c);
                    self.pos += c.len_utf8();
                }
            }
        }
    }

    /// An array of JSON strings.
    fn strings(&mut self) -> Result<Vec<JsonString>, Syntax> {
        self.expect(b'[', "an array")?;
        let mut out = Vec::new();
        self.skip_ws();
        if self.byte() == Some(b']') {
            self.pos += 1;
            return Ok(out);
        }
        loop {
            self.skip_ws();
            out.push(self.string()?);
            self.skip_ws();
            match self.byte() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(out);
                }
                _ => return Err(Syntax::Expected(self.pos, "a comma or ]")),
            }
        }
    }

    /// The tokens and tags of one row: the keys `tokens` and `ner_tags` in
    /// either order, each once, then the end of the line.
    fn row(&mut self) -> Result<(Vec<JsonString>, Vec<JsonString>), Syntax> {
        self.skip_ws();
        self.expect(b'{', "an object")?;
        let mut tokens = None;
        let mut tags = None;
        loop {
            self.skip_ws();
            let key = self.string()?;
            match (key.text.as_str(), tokens.is_some(), tags.is_some()) {
                ("tokens", false, _) | ("ner_tags", _, false) => {}
                ("tokens" | "ner_tags", _, _) => {
                    return Err(Syntax::Expected(key.span.start as usize, "each key once"));
                }
                _ => return Err(Syntax::Expected(key.span.start as usize, "the key tokens or ner_tags")),
            }
            self.skip_ws();
            self.expect(b':', "a colon")?;
            self.skip_ws();
            let value = self.strings()?;
            if key.text == "tokens" {
                tokens = Some(value);
            } else {
                tags = Some(value);
            }
            self.skip_ws();
            match self.byte() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(Syntax::Expected(self.pos, "a comma or }")),
            }
        }
        self.skip_ws();
        if self.pos != self.text.len() {
            return Err(Syntax::Expected(self.pos, "the end of the line"));
        }
        match (tokens, tags) {
            (Some(t), Some(n)) => Ok((t, n)),
            _ => Err(Syntax::Expected(self.pos, "the keys tokens and ner_tags")),
        }
    }
}

/// A name tag of the source line and the line's number.
#[derive(Copy, Clone, Debug)]
struct Tag<'a> {
    text: &'a str,
    line: u32,
}

/// The name of tokens `begin..end` whose first tag is `tags[begin]`: the
/// type is the tag after its two-character prefix, kept as written.
fn extract<'a>(tags: &[Tag<'a>], begin: usize, end: i64) -> Result<Name<'static>, MulticonerError> {
    let tag = tags[begin];
    let kind = &tag.text[2..];
    let kind = ["PER", "ORG", "LOC", "MISC"]
        .iter()
        .copied()
        .find(|known| *known == kind)
        .ok_or_else(|| MulticonerError::InvalidTag { line: tag.line, tag: tag.text.to_owned() })?;
    let end = u32::try_from(end).expect("an open name ends after its first token");
    Name::new(begin as u32, end, Some(kind)).map_err(|e| MulticonerError::Sample { line: tag.line, source: e })
}

/// The names of a row's tags: a name starts at a `B-` tag, or at an `I-`
/// tag with no name open or whose type differs from the open name's.
fn multiconer_names(tags: &[Tag<'_>]) -> Result<Vec<Name<'static>>, MulticonerError> {
    let mut names = Vec::new();
    let mut begin: Option<usize> = None;
    let mut end: i64 = -1;
    for (i, t) in tags.iter().enumerate() {
        if t.text == "O" {
            if let Some(b) = begin.take() {
                names.push(extract(tags, b, end)?);
                end = -1;
            }
            continue;
        }
        let same_as_open = begin.is_some_and(|b| t.text[1..] == tags[b].text[1..]);
        let known = t.text.strip_prefix("B-").or_else(|| t.text.strip_prefix("I-")).is_some_and(|k| !k.is_empty());
        match (t.text.starts_with("B-"), known, same_as_open) {
            (_, false, _) => return Err(MulticonerError::InvalidTag { line: t.line, tag: t.text.to_owned() }),
            (false, true, true) => end += 1,
            _ => {
                if let Some(b) = begin {
                    names.push(extract(tags, b, end)?);
                }
                begin = Some(i);
                end = i as i64 + 1;
            }
        }
    }
    if let Some(b) = begin {
        names.push(extract(tags, b, end)?);
    }
    Ok(names)
}

/// Reads name samples from a MultiCoNER v2 file: one JSON object per line
/// in the pinned `{"tokens", "ner_tags"}` form (see the module
/// documentation).
#[derive(Clone, Debug)]
pub struct MulticonerReader<'a> {
    lines: Lines<'a>,
}

impl<'a> MulticonerReader<'a> {
    /// A reader of the MultiCoNER v2 text `source`.
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`] for a source longer than spans can index.
    pub fn new(source: &'a str) -> Result<MulticonerReader<'a>, FormatError> {
        Ok(MulticonerReader { lines: Lines::new(source)? })
    }
}

impl<'a> Iterator for MulticonerReader<'a> {
    type Item = Result<NameSample<'a>, MulticonerError>;

    fn next(&mut self) -> Option<Self::Item> {
        let line = self.lines.next()?;
        let number = line.number;
        let (tokens, tags) = match Row::new(line.text()).row() {
            Ok(row) => row,
            Err(e) => {
                let at = e.at();
                return Some(Err(MulticonerError::Syntax { line: number, at, message: e.message() }));
            }
        };
        if tokens.is_empty() {
            return Some(Err(MulticonerError::Empty { line: number }));
        }
        if tokens.len() != tags.len() {
            return Some(Err(MulticonerError::Lengths { line: number, tokens: tokens.len(), tags: tags.len() }));
        }
        if tags.iter().any(|t| &line.text()[t.span.range()] != t.text.as_str()) {
            let at = tags.iter().find(|t| &line.text()[t.span.range()] != t.text.as_str()).unwrap().span.start;
            return Some(Err(MulticonerError::Syntax { line: number, at: at as usize, message: "a plain name tag" }));
        }
        let fields = tokens
            .into_iter()
            .map(|t| {
                if &line.text()[t.span.range()] == t.text.as_str() {
                    Field::of(line.text(), t.span)
                } else {
                    Field::made(t.text, t.span)
                }
            })
            .collect();
        let tags: Vec<Tag<'_>> =
            tags.iter().map(|t| Tag { text: &line.text()[t.span.range()], line: number }).collect();
        let names = match multiconer_names(&tags) {
            Ok(n) => n,
            Err(e) => return Some(Err(e)),
        };
        Some(
            NameSample::new(fields, names, false, line.span(), number)
                .map_err(|e| MulticonerError::Sample { line: number, source: e }),
        )
    }
}
