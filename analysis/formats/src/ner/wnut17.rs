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
//! - Whitespace-only lines end sentences. Every other row has exactly two
//!   tab-separated fields; any other count is refused at its line.
//! - IOB is strict: a name starts at `B-` and continues only through `I-`
//!   tags of the same type. An orphaned or mismatched continuation is an
//!   error at its line.
//! - The corpus marks no document starts, so no sample clears adaptive
//!   data.
//! - Every offset is a UTF-8 byte offset into the source. A token is the
//!   source's own bytes at its span and is never re-cut, so tokens with
//!   multi-byte or supplementary characters keep their exact text.

use std::fmt;

use turbo_analysis::Span;

use super::{Tag, bio_names};
use crate::error::FormatError;
use crate::sample::{Name, NameSample};
use crate::source::{Field, Line, Lines, split_fields};

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
                if line.text().trim().is_empty() {
                    break;
                }
                sentence.cover(&line);
                let fields = split_fields(line.piece, '\t');
                if fields.len() != 2 {
                    let text = line.text().to_owned();
                    return Some(Err(WnutError::Fields { line: line.number, count: fields.len(), text }));
                }
                sentence.tokens.push(fields[0].field());
                sentence.tags.push(Tag { text: fields[1].text, line: line.number });
            }
            if !sentence.tokens.is_empty() {
                let names = bio_names(
                    &sentence.tags,
                    |tag| tag,
                    |text, line| {
                        WNUT17_TYPES
                            .iter()
                            .copied()
                            .find(|known| *known == text)
                            .ok_or_else(|| FormatError::UnknownNameType { line, kind: text.to_owned() })
                    },
                );
                let names = names.map_err(|e| match e {
                    FormatError::UnknownNameType { line, kind } => WnutError::UnknownType { line, kind },
                    FormatError::InvalidNameTag { line, tag } => WnutError::InvalidTag { line, tag },
                    other => WnutError::Sample { line: sentence.first_line, source: other },
                });
                return Some(sentence.sample(names));
            }
            if at_end {
                return None;
            }
        }
    }
}
