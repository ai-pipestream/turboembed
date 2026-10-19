//! Documents with gold coreference from CoNLL-U with the CorefUD `Entity`
//! attribute: `ConlluCorefDocumentStream` of the OpenNLP addons'
//! `ready/coref` branch.
//!
//! A document starts at each line that starts with `# newdoc`; a text
//! without one is one document. A document none of whose lines is a word
//! line (a line that is not empty and does not start with `#`) is passed
//! over. Within a document:
//!
//! - A sentence ends at a line of length zero. A line that starts with
//!   `#` is a comment; `# speaker = <label>` gives the sentence its
//!   speaker, the label trimmed of Unicode whitespace.
//! - Every other line must have ten tab-separated columns, none of them
//!   empty or all whitespace. An id that is a positive decimal without a
//!   leading zero is a word; ids `N-M` (multiword tokens, `N` < `M`) and
//!   `N.M` (empty nodes) are passed over; any other id is an error. The
//!   words of a sentence must be numbered `1`, `2`, ... in order.
//! - The text is the word forms, one space between two words unless the
//!   first has `SpaceAfter=No`, and a line feed between two sentences.
//! - A word's `Entity` attribute opens and closes mentions
//!   ([`entity`]). A mention must close in the sentence it opens in.
//!
//! Each mention belongs to a chain, its entity's; the mentions are in
//! text order (by start, the longer first, and otherwise in the order
//! their entities first appear and then the order they closed), and the
//! chains are numbered by their first mention in that order.
//!
//! [`CorefDocument`] keeps what Java's `Document` keeps (the text, the
//! sentences, the tokens with their tags, the mentions with their chains,
//! the speakers) and the source bytes of each: every token and tag is a
//! field of the source, and every mention has the brackets that made it.
//!
//! # What OpenNLP does here
//!
//! - A byte order mark is not removed, so a text that starts with one
//!   reads its first line as a word line.
//! - The parts of a discontinuous mention are mentions of their own, and
//!   a closing bracket closes the innermost open mention of its entity,
//!   whatever part either names.
//! - The words of a multiword token are the tokens, and the text has a
//!   space between them; `SpaceAfter=No` on the multiword token's line is
//!   not read, and the word after it always follows a space.
//! - An empty node's `Entity` attribute is not read, so a mention that
//!   starts or ends on one is lost or leaves a bracket unbalanced. The
//!   reader keeps a [`CorefWarning`] for each of these two lines.
//! - A `# speaker` comment in a run of comments with no words after it
//!   gives its speaker to the next sentence.
//! - When several mentions are open at a boundary, the error names the
//!   first open entity in `HashMap` order, the map being filled with
//!   `computeIfAbsent` (a new key goes to the head of its bin).

mod builder;
pub mod entity;

use std::fmt;
use std::ops::Range;

use turbo_analysis::Span;

use self::builder::Builder;
use super::Tagset;
use crate::error::FormatError;
use crate::source::{Field, Line, Lines};
use entity::{Bracket, EntityError};

/// The comment that starts a document.
const NEWDOC: &str = "# newdoc";
/// The comment that names a sentence's speaker.
const SPEAKER: &str = "# speaker =";
const COLUMNS: usize = 10;

/// What `ConlluCorefDocumentStream` refuses in a document; each is its
/// `InvalidFormatException`, with the message as `Display`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CorefError {
    /// A `# speaker =` comment with a blank label.
    BlankSpeaker { line: u32 },
    /// A word line without ten columns; `text` is the line.
    Columns { line: u32, text: String },
    /// A column that is empty or all Unicode whitespace; `column` counts
    /// from 1.
    BlankColumn { line: u32, column: usize, text: String },
    /// An id that is neither a word, a multiword token nor an empty node.
    InvalidId { line: u32, id: String },
    /// A word whose id is not the next number of its sentence.
    UnexpectedId { line: u32, expected: u32, found: String },
    /// An `Entity` value that is not a run of brackets.
    Entity { line: u32, error: EntityError },
    /// A closing bracket of an entity with no open mention.
    NotOpen { line: u32, entity: String },
    /// A mention still open at the end of its sentence.
    CrossesSentence { line: u32, entity: String },
    /// A mention still open at the end of the document.
    NeverClosed { line: u32, entity: String },
}

impl CorefError {
    /// The line the error is on, or for a mention left open, the line
    /// that ends its sentence or the document's last line.
    pub fn line(&self) -> u32 {
        use CorefError::*;
        match *self {
            BlankSpeaker { line }
            | Columns { line, .. }
            | BlankColumn { line, .. }
            | InvalidId { line, .. }
            | UnexpectedId { line, .. }
            | Entity { line, .. }
            | NotOpen { line, .. }
            | CrossesSentence { line, .. }
            | NeverClosed { line, .. } => line,
        }
    }
}

impl fmt::Display for CorefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use CorefError::*;
        match self {
            BlankSpeaker { .. } => f.write_str("speaker label must not be blank"),
            Columns { text, .. } => write!(f, "expected 10 columns: {text}"),
            BlankColumn { column, text, .. } => write!(f, "column {column} must not be blank: {text}"),
            InvalidId { id, .. } => write!(f, "invalid token id: {id}"),
            UnexpectedId { expected, found, .. } => write!(f, "expected token id {expected} but found {found}"),
            Entity { error, .. } => error.fmt(f),
            NotOpen { entity, .. } => write!(f, "entity {entity} is closed but not open"),
            CrossesSentence { entity, .. } => write!(f, "entity {entity} crosses a sentence boundary"),
            NeverClosed { entity, .. } => write!(f, "entity {entity} is never closed"),
        }
    }
}

impl std::error::Error for CorefError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CorefError::Entity { error, .. } => error.source(),
            _ => None,
        }
    }
}

/// Something in a document Java reads past without a word, which a
/// caller may want to know about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CorefWarning {
    /// An empty node with an `Entity` attribute, which is not read.
    EmptyNodeEntity { line: u32, value: Span },
    /// A multiword token with `SpaceAfter=No`, which is not read.
    MultiwordSpaceAfter { line: u32 },
}

/// A word of a document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorefToken<'a> {
    /// The form, a field of the source.
    pub form: Field<'a>,
    /// The tag of the selected tag set.
    pub tag: Field<'a>,
    /// The form's bytes in the document's text.
    pub text: Span,
    /// The word's line, counting from 1.
    pub line: u32,
}

/// A sentence of a document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorefSentence<'a> {
    /// The sentence's bytes in the document's text.
    pub text: Span,
    /// Its tokens, by index.
    pub tokens: Range<u32>,
    /// The label of its `# speaker` comment.
    pub speaker: Option<Field<'a>>,
}

/// A mention of an entity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mention<'a> {
    /// The chain, the index of its cluster.
    pub chain: u32,
    /// Its tokens, by index.
    pub tokens: Range<u32>,
    /// Its bytes in the document's text, from its first token's start to
    /// its last token's end.
    pub text: Span,
    /// Its bytes in the source, from its first token's form to its last
    /// token's form.
    pub source: Span,
    /// The bracket that opened it, or the one-word bracket that is all of
    /// it; its entity id is the mention's.
    pub open: Bracket<'a>,
    /// The bracket that closed it, none for a one-word bracket.
    pub close: Option<Bracket<'a>>,
}

impl<'a> Mention<'a> {
    /// The entity id, as the opening bracket writes it.
    pub fn entity(&self) -> &Field<'a> {
        &self.open.entity
    }
}

/// The mentions of one entity: a chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cluster<'a> {
    /// The entity id, from the bracket of its first mention.
    pub entity: Field<'a>,
    /// Its mentions, by index, in text order.
    pub mentions: Vec<u32>,
}

/// A document with gold coreference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CorefDocument<'a> {
    text: String,
    sentences: Vec<CorefSentence<'a>>,
    tokens: Vec<CorefToken<'a>>,
    mentions: Vec<Mention<'a>>,
    clusters: Vec<Cluster<'a>>,
    warnings: Vec<CorefWarning>,
    source: Span,
}

impl<'a> CorefDocument<'a> {
    /// The text rebuilt from the word forms.
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn sentences(&self) -> &[CorefSentence<'a>] {
        &self.sentences
    }

    pub fn tokens(&self) -> &[CorefToken<'a>] {
        &self.tokens
    }

    /// The mentions in text order, as Java's gold chain layer has them.
    pub fn mentions(&self) -> &[Mention<'a>] {
        &self.mentions
    }

    /// The chains, by number.
    pub fn clusters(&self) -> &[Cluster<'a>] {
        &self.clusters
    }

    /// What the document has that Java does not read.
    pub fn warnings(&self) -> &[CorefWarning] {
        &self.warnings
    }

    /// The bytes of the document's lines in the source.
    pub fn source(&self) -> Span {
        self.source
    }
}

/// Reads documents with gold coreference from CoNLL-U:
/// `ConlluCorefDocumentStream`.
#[derive(Clone, Debug)]
pub struct ConlluCorefReader<'a> {
    lines: Lines<'a>,
    pending: Option<Line<'a>>,
    tagset: Tagset,
}

impl<'a> ConlluCorefReader<'a> {
    /// A reader of `source` that takes its tags from `tagset`.
    pub fn new(source: &'a str, tagset: Tagset) -> Result<Self, FormatError> {
        Ok(ConlluCorefReader { lines: Lines::new(source)?, pending: None, tagset })
    }
}

impl<'a> Iterator for ConlluCorefReader<'a> {
    type Item = Result<CorefDocument<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut lines = Vec::new();
            let mut line = self.pending.take().or_else(|| self.lines.next());
            while let Some(l) = line {
                if l.text().starts_with(NEWDOC) && !lines.is_empty() {
                    self.pending = Some(l);
                    break;
                }
                lines.push(l);
                line = self.lines.next();
            }
            if lines.iter().any(|l| !l.text().is_empty() && !l.text().starts_with('#')) {
                return Some(Builder::new(self.tagset).read(&lines).map_err(FormatError::Coref));
            }
            self.pending?;
        }
    }
}

#[cfg(test)]
mod tests;
