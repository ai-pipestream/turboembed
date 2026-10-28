//! The WNUT-17 kept-corpus reader (issue #1533).
//!
//! WNUT-17, the shared task on Emerging and Rare entities (noisy-text 2017),
//! is a kept corpus here under CC BY 4.0 (docs/licences.md). Its files hold
//! one sentence per block of lines, an empty line between sentences, and a
//! line per token: the token and its IOB name tag in two tab-separated
//! fields, as in the task's training release. With the tabs shown as
//! spaces:
//!
//! ```text
//! Sarah      O
//! loves      O
//! Manchester B-location
//! United     I-location
//! ```
//!
//! A tag is `O`, or `B-` or `I-` with one of the task's six types
//! ([`WNUT17_TYPES`]), kept as written as the name's type. There is no
//! OpenNLP stream for WNUT-17: the reader follows the conventions of the
//! ported readers of this module, but its errors are its own and carry no
//! JVM message.
//!
//! # What is pinned here
//!
//! - A row has exactly two tab-separated fields; a row of any other count
//!   is refused, and the read ends at its line with the tokens before it
//!   lost, as a bad line ends a read in the ported readers.
//! - IOB is read as the shared `bio_names` helper of this module reads it:
//!   a name starts at a `B-` tag and grows with each `I-` tag, whatever its
//!   type, until an `O` or the next `B-`; an `I-` with no name open is read
//!   as `O`. The type of a name is its `B-` tag after the prefix; the types
//!   of `I-` tags are never checked.
//! - The corpus marks no document starts, so no sample clears adaptive
//!   data.
//! - Every offset is a UTF-8 byte offset into the source. A token is the
//!   source's own bytes at its span and is never re-cut, so tokens with
//!   multi-byte or supplementary characters keep their exact text.

use std::fmt;

use turbo_analysis::Span;

use crate::error::FormatError;
use crate::sample::{Name, NameSample};
use crate::source::{Field, Line, Lines, Piece};

/// The six name types of the WNUT-17 task: `corporation`, `creative-work`,
/// `group`, `location`, `person` and `product`.
pub const WNUT17_TYPES: [&str; 6] = ["corporation", "creative-work", "group", "location", "person", "product"];

/// What a WNUT-17 reader refuses. `Display` is the message the error gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WnutError {
    /// A line without exactly two tab-separated fields. `count` is the
    /// number of fields and `text` the line.
    Fields { line: u32, count: usize, text: String },
    /// A name tag that is not `O` and does not start with `B-` or `I-`.
    InvalidTag { line: u32, tag: String },
    /// A `B-` tag whose type is not one of [`WNUT17_TYPES`], found when its
    /// name ends: `kind` is the tag after its prefix.
    UnknownType { line: u32, kind: String },
    /// A name or sample the shared model refused; never reached by a
    /// well-formed tag sequence.
    Sample { line: u32, source: FormatError },
}

impl fmt::Display for WnutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WnutError::Fields { count, text, .. } => {
                write!(f, "Expected two tab-separated fields per line in WNUT-17 data, got {count} for line '{text}'!")
            }
            WnutError::InvalidTag { tag, .. } => write!(f, "Invalid tag: {tag}"),
            WnutError::UnknownType { kind, .. } => write!(f, "Unknown type: {kind}"),
            WnutError::Sample { source, .. } => source.fmt(f),
        }
    }
}

impl std::error::Error for WnutError {}

/// `StringUtil.split` with limit 0: the pieces between every `separator`,
/// with the empty pieces at the end removed.
fn split(field: Piece<'_>, separator: char) -> Vec<Piece<'_>> {
    if field.text.is_empty() {
        return vec![field];
    }
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in field.text.char_indices() {
        if c == separator {
            out.push(field.slice(start, i));
            start = i + c.len_utf8();
        }
    }
    out.push(field.slice(start, field.text.len()));
    while out.last().is_some_and(|f| f.text.is_empty()) {
        out.pop();
    }
    out
}

/// A name tag and the line it is on.
#[derive(Copy, Clone, Debug)]
struct Tag<'a> {
    text: &'a str,
    line: u32,
}

/// The lines of a sentence read so far.
#[derive(Default)]
struct Sentence<'a> {
    tokens: Vec<Field<'a>>,
    tags: Vec<Tag<'a>>,
    /// From the start of the first line read to the end of the last.
    source: Option<Span>,
    first_line: u32,
}

impl<'a> Sentence<'a> {
    fn cover(&mut self, line: &Line<'a>) {
        let span = line.span();
        match self.source {
            None => {
                self.source = Some(span);
                self.first_line = line.number;
            }
            Some(s) => self.source = Some(Span::new(s.start, span.end)),
        }
    }

    fn sample(&self, names: Result<Vec<Name<'a>>, WnutError>) -> Result<NameSample<'a>, WnutError> {
        let source = self.source.expect("a sentence with tokens has lines");
        let names = names?;
        NameSample::new(self.tokens.clone(), names, false, source, self.first_line)
            .map_err(|e| WnutError::Sample { line: self.first_line, source: e })
    }
}

/// The name of tokens `begin..end` whose first tag is `tags[begin]`: the
/// type is the tag after its two-character prefix, which must be one of
/// [`WNUT17_TYPES`].
fn extract(tags: &[Tag<'_>], begin: usize, end: i64) -> Result<Name<'static>, WnutError> {
    let tag = tags[begin];
    let kind = &tag.text[2..];
    let kind = WNUT17_TYPES
        .iter()
        .copied()
        .find(|known| *known == kind)
        .ok_or_else(|| WnutError::UnknownType { line: tag.line, kind: kind.to_owned() })?;
    let end = u32::try_from(end).expect("an open name ends after its first token");
    Name::new(begin as u32, end, Some(kind)).map_err(|e| WnutError::Sample { line: tag.line, source: e })
}

/// The names of a WNUT-17 sentence: a name starts at a `B-` tag and grows
/// with each `I-` tag, whatever its type, until `O` or the next `B-`; an
/// `I-` with no name open is read as `O`.
fn wnut_names(tags: &[Tag<'_>]) -> Result<Vec<Name<'static>>, WnutError> {
    let mut names = Vec::new();
    let mut begin: Option<usize> = None;
    // an `I-` with no name open moves this on from -1; the next `B-` sets
    // it again, so that does not show
    let mut end: i64 = -1;
    for (i, t) in tags.iter().enumerate() {
        if t.text.starts_with("B-") {
            if let Some(b) = begin {
                names.push(extract(tags, b, end)?);
            }
            begin = Some(i);
            end = i as i64 + 1;
        } else if t.text.starts_with("I-") {
            end += 1;
        } else if t.text == "O" {
            if let Some(b) = begin.take() {
                names.push(extract(tags, b, end)?);
                end = -1;
            }
        } else {
            return Err(WnutError::InvalidTag { line: t.line, tag: t.text.to_owned() });
        }
    }
    if let Some(b) = begin {
        names.push(extract(tags, b, end)?);
    }
    Ok(names)
}

/// Reads name samples from a WNUT-17 file: a sentence per block of lines,
/// ended by an empty line, a token and its IOB tag per line in two
/// tab-separated fields (see the module documentation for the pinned form).
#[derive(Clone, Debug)]
pub struct WnutReader<'a> {
    lines: Lines<'a>,
}

impl<'a> WnutReader<'a> {
    /// A reader of the WNUT-17 text `source`.
    ///
    /// # Errors
    ///
    /// [`FormatError::TooLong`] for a source longer than spans can index.
    pub fn new(source: &'a str) -> Result<WnutReader<'a>, FormatError> {
        Ok(WnutReader { lines: Lines::new(source)? })
    }
}

impl<'a> Iterator for WnutReader<'a> {
    type Item = Result<NameSample<'a>, WnutError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut sentence = Sentence::default();
            let mut at_end = false;
            loop {
                let Some(line) = self.lines.next() else {
                    at_end = true;
                    break;
                };
                if line.text().is_empty() {
                    break;
                }
                sentence.cover(&line);
                let fields = split(line.piece, '\t');
                if fields.len() != 2 {
                    let text = line.text().to_owned();
                    return Some(Err(WnutError::Fields { line: line.number, count: fields.len(), text }));
                }
                sentence.tokens.push(fields[0].field());
                sentence.tags.push(Tag { text: fields[1].text, line: line.number });
            }
            if !sentence.tokens.is_empty() {
                let names = wnut_names(&sentence.tags);
                return Some(sentence.sample(names));
            }
            if at_end {
                return None;
            }
        }
    }
}
