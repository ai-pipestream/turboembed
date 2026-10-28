//! The LitBank kept-corpus reader (issue #1533).
//!
//! LitBank, the corpus of 210 English-language novels with named entity
//! annotations, keeps its documents as pairs of files: the document's text
//! and a standoff annotation file in the pinned brat form, one text-bound
//! annotation per line (with the tabs shown as spaces):
//!
//! ```text
//! T1  PER 0 3  Ada
//! T2  GPE 12 18 London
//! ```
//!
//! The id and the `TYPE START END` field are separated from the covered
//! text by tabs. Offsets count UTF-16 code units, as Java reads them and as
//! the house brat reader of this module reads them, and are converted to
//! UTF-8 byte spans here; a boundary inside a surrogate pair, or outside
//! the text, is refused rather than guessed. A covered text must be exactly
//! the text's bytes at the span, with no leading or trailing whitespace,
//! and a name must start and end where tokens of its sentence start and
//! end. There is no OpenNLP stream for LitBank: the reader follows the
//! conventions of the ported readers of this module, but its errors are its
//! own and carry no JVM message.
//!
//! # What is pinned here
//!
//! - Only text-bound annotations are names. Lines whose id starts with
//!   another letter, and notes, are passed over.
//! - A text-bound annotation has exactly three tab-separated fields and a
//!   single `START END` pair: the `;`-separated fragments of general brat
//!   are refused, as are relations, attributes, events and equivalences,
//!   which LitBank's entity files do not use.
//! - The text is split into sentences and tokens by the given
//!   [`SentenceSpans`] and [`TokenSpans`], as the brat reader of this
//!   module takes them. A name that crosses a sentence boundary, or that no
//!   sentence holds, is refused or passed over rather than re-cut.
//! - The first sample of a document clears adaptive data, as the first
//!   sample of a brat document does.

use std::collections::BTreeSet;
use std::fmt;

use turbo_analysis::Span;

use super::{SentenceSpans, TokenSpans};
use crate::error::FormatError;
use crate::sample::{Name, NameSample};
use crate::source::{Field, Lines, Piece};

/// The six entity types of LitBank: `PER`, `FAC`, `GPE`, `LOC`, `ORG` and
/// `VEH`. The reader keeps every type as written when it is given `None`
/// for `name_types`; the constant is the set to pass for LitBank itself.
pub const LITBANK_TYPES: [&str; 6] = ["PER", "FAC", "GPE", "LOC", "ORG", "VEH"];

/// What a LitBank reader refuses. `Display` is the message the error gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LitBankError {
    /// A source text longer than `u32` spans can index.
    TooLong { len: usize },
    /// An empty set of name types.
    NoNameTypes,
    /// A text-bound line without exactly three tab-separated fields.
    /// `text` is the line.
    Fields { line: u32, text: String },
    /// A text-bound line whose second field is not `TYPE START END`.
    /// `text` is the line.
    SpanFields { line: u32, text: String },
    /// An offset that is not an integer. `value` is the field.
    Offset { line: u32, value: String },
    /// Offsets that are negative, backwards, or past the text's end;
    /// `len` is the text's length in UTF-16 code units.
    OutOfBounds { line: u32, start: i64, end: i64, len: usize },
    /// An offset that cuts between the two halves of a surrogate pair.
    SplitsPair { line: u32, offset: i64 },
    /// A text-bound annotation of `;`-separated fragments, which the
    /// pinned LitBank form does not have. `text` is the line.
    Fragments { line: u32, text: String },
    /// A covered text that is not the text's bytes at the span.
    CoveredText { line: u32, id: String },
    /// A span with leading or trailing whitespace.
    Padded { line: u32, id: String },
    /// A span that starts or ends inside a token.
    OffTokens { line: u32, id: String },
    /// A span that crosses a sentence boundary.
    CrossesSentence { line: u32, id: String },
    /// A name or sample the shared model refused.
    Sample { line: u32, source: FormatError },
}

impl fmt::Display for LitBankError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LitBankError::TooLong { len } => write!(f, "a source of {len} bytes is longer than spans can index"),
            LitBankError::NoNameTypes => f.write_str("nameTypes should be null or have one or more elements"),
            LitBankError::Fields { text, .. } => {
                write!(f, "Expected three tab-separated fields per text-bound line in LitBank data, got line '{text}'!")
            }
            LitBankError::SpanFields { text, .. } => {
                write!(
                    f,
                    "Expected TYPE START END in the second field of a LitBank text-bound line, got line '{text}'!"
                )
            }
            LitBankError::Offset { value, .. } => write!(f, "For input string: \"{value}\""),
            LitBankError::OutOfBounds { start, end, len, .. } => {
                if *start < 0 || *end < 0 {
                    write!(f, "start index must be zero or greater: start={start}, end={end}")
                } else if start > end {
                    write!(f, "start index must not be larger than end index: start={start}, end={end}")
                } else {
                    write!(f, "the span {start}..{end} is outside a text of {len} UTF-16 code units")
                }
            }
            LitBankError::SplitsPair { offset, .. } => {
                write!(f, "the offset {offset} cuts between the two halves of a surrogate pair")
            }
            LitBankError::Fragments { text, .. } => {
                write!(f, "Expected a single START END pair in a LitBank text-bound line, got line '{text}'!")
            }
            LitBankError::CoveredText { id, .. } => {
                write!(f, "the covered text of {id} is not the text's bytes at its span")
            }
            LitBankError::Padded { id, .. } => write!(f, "the span of {id} has leading or trailing whitespace"),
            LitBankError::OffTokens { id, .. } => {
                write!(f, "the span of {id} starts or ends inside a token")
            }
            LitBankError::CrossesSentence { id, .. } => write!(f, "the span of {id} crosses a sentence boundary"),
            LitBankError::Sample { source, .. } => source.fmt(f),
        }
    }
}

impl std::error::Error for LitBankError {}

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

/// The byte offset of each UTF-16 code unit of the text: `None` between
/// the two halves of a surrogate pair. Standoff offsets count UTF-16 code
/// units, so a boundary inside a pair maps to no byte offset.
struct Units {
    bytes: Vec<Option<u32>>,
}

impl Units {
    fn new(text: &str) -> Units {
        let mut bytes = Vec::with_capacity(text.len() + 1);
        for (i, c) in text.char_indices() {
            bytes.push(Some(i as u32));
            if c.len_utf16() == 2 {
                bytes.push(None);
            }
        }
        bytes.push(Some(text.len() as u32));
        Units { bytes }
    }

    /// The byte offset of the character boundary at `unit`: `None` past
    /// the text's end, and `Some(None)` inside a surrogate pair.
    fn byte(&self, unit: i64) -> Option<Option<u32>> {
        usize::try_from(unit).ok().and_then(|u| self.bytes.get(u).copied())
    }

    /// The text's length in UTF-16 code units.
    fn len(&self) -> usize {
        self.bytes.len() - 1
    }
}

/// A text-bound annotation of the document: UTF-8 bytes `start..end` of
/// the text and its type, as read from the annotation file.
#[derive(Clone, Debug)]
struct Entity<'a> {
    start: u32,
    end: u32,
    kind: &'a str,
    id: &'a str,
    line: u32,
}

/// Reads name samples from one LitBank document: the document's `text` and
/// its standoff `ann`otation (see the module documentation for the pinned
/// form), split into sentences by `sentences` and tokens by `tokenizer`.
/// The names are the text-bound annotations of `name_types` (every type
/// when `None`).
#[derive(Debug)]
pub struct LitBankReader<'a, T> {
    text: &'a str,
    sentences: Vec<Span>,
    entities: Vec<Entity<'a>>,
    tokenizer: T,
    next: usize,
}

impl<'a, T: TokenSpans> LitBankReader<'a, T> {
    /// A reader of the document `text` with its annotation file `ann`,
    /// keeping the names of `name_types` (every type when `None`).
    ///
    /// # Errors
    ///
    /// [`LitBankError::TooLong`] for a text longer than spans can index,
    /// [`LitBankError::NoNameTypes`] for an empty set of name types, and
    /// the other variants for a line of `ann` the pinned form refuses.
    pub fn new<S: SentenceSpans>(
        text: &'a str,
        ann: &'a str,
        sentences: S,
        tokenizer: T,
        name_types: Option<BTreeSet<String>>,
    ) -> Result<LitBankReader<'a, T>, LitBankError> {
        let len = text.len();
        if u32::try_from(len).is_err() {
            return Err(LitBankError::TooLong { len });
        }
        if name_types.as_ref().is_some_and(BTreeSet::is_empty) {
            return Err(LitBankError::NoNameTypes);
        }
        let units = Units::new(text);
        let entities = entities(text, ann, &units, name_types)?;
        let mut sentence_spans = Vec::new();
        sentences.sentence_spans(text, &mut sentence_spans);
        Ok(LitBankReader { text, sentences: sentence_spans, entities, tokenizer, next: 0 })
    }

    /// The line of the text `at` is on, counting from 1.
    fn line_of(&self, at: u32) -> u32 {
        self.text[..at as usize].bytes().filter(|&b| b == b'\n').count() as u32 + 1
    }

    /// The sample of the sentence `sentence`: its tokens and the names of
    /// the entities fully inside it.
    fn sample(&self, sentence: Span, first: bool) -> Result<NameSample<'a>, LitBankError> {
        let from = sentence.start as usize;
        let to = sentence.end as usize;
        let mut spans = Vec::new();
        self.tokenizer.token_spans(&self.text[from..to], &mut spans);
        let tokens: Vec<Field<'a>> =
            spans.iter().map(|s| Field::of(self.text, Span::new(from as u32 + s.start, from as u32 + s.end))).collect();
        let mut names = Vec::new();
        for e in &self.entities {
            if e.end <= sentence.start || e.start >= sentence.end {
                continue;
            }
            if e.start < sentence.start || e.end > sentence.end {
                return Err(LitBankError::CrossesSentence { line: e.line, id: e.id.to_owned() });
            }
            let start = tokens.iter().position(|t| t.span.start == e.start);
            let end = tokens.iter().position(|t| t.span.end == e.end);
            match (start, end) {
                (Some(s), Some(en)) if s <= en => names.push(
                    Name::new(s as u32, en as u32 + 1, Some(e.kind))
                        .map_err(|err| LitBankError::Sample { line: e.line, source: err })?,
                ),
                _ => return Err(LitBankError::OffTokens { line: e.line, id: e.id.to_owned() }),
            }
        }
        let line = self.line_of(sentence.start);
        NameSample::new(tokens, names, first, sentence, line).map_err(|e| LitBankError::Sample { line, source: e })
    }
}

/// The entities of the annotation file `ann` of a document `text`:
/// the text-bound annotations of a kept type, in the file's order.
fn entities<'a>(
    text: &'a str,
    ann: &'a str,
    units: &Units,
    name_types: Option<BTreeSet<String>>,
) -> Result<Vec<Entity<'a>>, LitBankError> {
    let mut out = Vec::new();
    let total = units.len() as i64;
    let lines = Lines::new(ann).map_err(|e| match e {
        FormatError::TooLong { len } => LitBankError::TooLong { len },
        e => LitBankError::Sample { line: 0, source: e },
    })?;
    for line in lines {
        let fields = split(line.piece, '\t');
        let id = fields.first().map_or("", |f| f.text);
        if !id.starts_with('T') {
            continue;
        }
        let line_no = line.number;
        if fields.len() != 3 {
            return Err(LitBankError::Fields { line: line_no, text: line.text().to_owned() });
        }
        let offsets = fields[1].text;
        if offsets.contains(';') {
            return Err(LitBankError::Fragments { line: line_no, text: line.text().to_owned() });
        }
        let pieces: Vec<&str> = offsets.split_whitespace().collect();
        if pieces.len() != 3 {
            return Err(LitBankError::SpanFields { line: line_no, text: line.text().to_owned() });
        }
        let kind = pieces[0];
        if name_types.as_ref().is_some_and(|t| !t.contains(kind)) {
            continue;
        }
        let parse = |value: &str| {
            value.parse::<i64>().map_err(|_| LitBankError::Offset { line: line_no, value: value.to_owned() })
        };
        let (start, end) = (parse(pieces[1])?, parse(pieces[2])?);
        if start < 0 || end < 0 || start > end || end > total {
            return Err(LitBankError::OutOfBounds { line: line_no, start, end, len: total as usize });
        }
        let at = |unit: i64| match units.byte(unit) {
            Some(Some(b)) => Ok(b),
            Some(None) => Err(LitBankError::SplitsPair { line: line_no, offset: unit }),
            None => Err(LitBankError::OutOfBounds { line: line_no, start, end, len: total as usize }),
        };
        let (from, to) = (at(start)?, at(end)?);
        let covered = fields[2].text;
        if text[from as usize..to as usize] != *covered {
            return Err(LitBankError::CoveredText { line: line_no, id: id.to_owned() });
        }
        if covered.trim_matches(char::is_whitespace) != covered {
            return Err(LitBankError::Padded { line: line_no, id: id.to_owned() });
        }
        out.push(Entity { start: from, end: to, kind, id, line: line_no });
    }
    Ok(out)
}

impl<'a, T: TokenSpans> Iterator for LitBankReader<'a, T> {
    type Item = Result<NameSample<'a>, LitBankError>;

    fn next(&mut self) -> Option<Self::Item> {
        let sentence = *self.sentences.get(self.next)?;
        let first = self.next == 0;
        self.next += 1;
        Some(self.sample(sentence, first))
    }
}
