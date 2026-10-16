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

pub mod entity;

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;

use turbo_analysis::Span;
use turbo_jvm::hash_map::JavaHashMap;
use turbo_unicode::WhitespaceMode;

use super::Tagset;
use crate::error::FormatError;
use crate::source::{Field, Line, Lines, Piece};
use entity::{Bracket, BracketKind, EntityError, entity_brackets, misc_attribute};

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

impl std::error::Error for CorefError {}

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

/// A mention while it is read: its tokens and brackets.
struct Pending<'a> {
    first: u32,
    last: u32,
    open: Bracket<'a>,
    close: Option<Bracket<'a>>,
}

/// The state of `ConlluCorefDocumentStream.parse` over one document.
struct Builder<'a> {
    tagset: Tagset,
    text: String,
    sentences: Vec<CorefSentence<'a>>,
    tokens: Vec<CorefToken<'a>>,
    warnings: Vec<CorefWarning>,
    /// The open mentions of each entity, innermost last (`openStarts`). It
    /// is a model of the Java `HashMap` because the error for a mention
    /// left open names the first entity in that map's order.
    open: JavaHashMap<&'a str, Vec<(u32, Bracket<'a>)>>,
    /// The finished mentions of each entity, entities in the order they
    /// first appeared (`entities`, a `LinkedHashMap`).
    entities: Vec<(&'a str, Vec<Pending<'a>>)>,
    entity_index: HashMap<&'a str, usize>,
    /// The first token of the sentence being read, its speaker, the next
    /// word id and whether the next token follows without a space.
    sentence: Option<u32>,
    sentence_start: usize,
    speaker: Option<Field<'a>>,
    expected: u32,
    glue: bool,
}

impl<'a> Builder<'a> {
    fn new(tagset: Tagset) -> Builder<'a> {
        Builder {
            tagset,
            text: String::new(),
            sentences: Vec::new(),
            tokens: Vec::new(),
            warnings: Vec::new(),
            open: JavaHashMap::new(),
            entities: Vec::new(),
            entity_index: HashMap::new(),
            sentence: None,
            sentence_start: 0,
            speaker: None,
            expected: 1,
            glue: false,
        }
    }

    fn read(mut self, lines: &[Line<'a>]) -> Result<CorefDocument<'a>, CorefError> {
        for &line in lines {
            if line.text().is_empty() {
                if self.sentence.is_some() {
                    self.reject_open(|entity| CorefError::CrossesSentence { line: line.number, entity })?;
                    self.end_sentence();
                }
                self.expected = 1;
            } else if line.text().starts_with('#') {
                if line.text().starts_with(SPEAKER) {
                    let label = unicode_trim(line.piece.slice(SPEAKER.len(), line.text().len()));
                    if label.text.is_empty() {
                        return Err(CorefError::BlankSpeaker { line: line.number });
                    }
                    self.speaker = Some(label.field());
                }
            } else {
                self.word(line)?;
            }
        }
        if self.sentence.is_some() {
            self.end_sentence();
        }
        let last = lines.last().map_or(0, |l| l.number);
        self.reject_open(|entity| CorefError::NeverClosed { line: last, entity })?;
        let source = match (lines.first(), lines.last()) {
            (Some(a), Some(b)) => Span::new(a.span().start, b.span().end),
            _ => Span::new(0, 0),
        };
        Ok(self.finish(source))
    }

    fn end_sentence(&mut self) {
        let first = self.sentence.take().expect("a sentence is open");
        let end = self.text.len();
        self.sentences.push(CorefSentence {
            text: span(self.sentence_start, end),
            tokens: first..self.tokens.len() as u32,
            speaker: self.speaker.take(),
        });
    }

    /// `rejectOpenMentions`: the first entity in `HashMap` order that has a
    /// mention open.
    fn reject_open(&self, error: impl FnOnce(String) -> CorefError) -> Result<(), CorefError> {
        match self.open.iter().find(|(_, open)| !open.is_empty()) {
            Some((entity, _)) => Err(error((*entity).to_owned())),
            None => Ok(()),
        }
    }

    fn word(&mut self, line: Line<'a>) -> Result<(), CorefError> {
        let number = line.number;
        let columns = split_all(line.piece);
        let Ok(columns) = <[Piece<'a>; COLUMNS]>::try_from(columns) else {
            return Err(CorefError::Columns { line: number, text: line.text().to_owned() });
        };
        if let Some(blank) = columns.iter().position(|c| unicode_trim(*c).text.is_empty()) {
            return Err(CorefError::BlankColumn { line: number, column: blank + 1, text: line.text().to_owned() });
        }
        let [id, form, _, upos, xpos, _, _, _, _, misc] = columns;
        match word_id(id.text) {
            Id::Word => {}
            Id::Multiword => {
                if misc_attribute(misc, "SpaceAfter").is_some_and(|v| v.text == "No") {
                    self.warnings.push(CorefWarning::MultiwordSpaceAfter { line: number });
                }
                return Ok(());
            }
            Id::EmptyNode => {
                if let Some(value) = misc_attribute(misc, "Entity") {
                    self.warnings.push(CorefWarning::EmptyNodeEntity { line: number, value: value.span });
                }
                return Ok(());
            }
            Id::Invalid => return Err(CorefError::InvalidId { line: number, id: id.text.to_owned() }),
        }
        if id.text != self.expected.to_string() {
            return Err(CorefError::UnexpectedId { line: number, expected: self.expected, found: id.text.to_owned() });
        }
        self.expected = self.expected.wrapping_add(1);
        if self.sentence.is_none() {
            if !self.text.is_empty() {
                self.text.push('\n');
            }
            self.sentence = Some(self.tokens.len() as u32);
            self.sentence_start = self.text.len();
            self.glue = true;
        }
        if !self.glue {
            self.text.push(' ');
        }
        let start = self.text.len();
        self.text.push_str(form.text);
        let token = self.tokens.len() as u32;
        let tag = match self.tagset {
            Tagset::U => upos,
            Tagset::X => xpos,
        };
        self.tokens.push(CorefToken {
            form: form.field(),
            tag: tag.field(),
            text: span(start, self.text.len()),
            line: number,
        });
        self.glue = misc_attribute(misc, "SpaceAfter").is_some_and(|v| v.text == "No");
        if let Some(value) = misc_attribute(misc, "Entity") {
            // Each bracket is applied before the next is read, so an
            // earlier bracket's error comes first, as in Java.
            for b in entity_brackets(value) {
                let b = b.map_err(|error| CorefError::Entity { line: number, error })?;
                self.bracket(b, token, number)?;
            }
        }
        Ok(())
    }

    /// One bracket of `brackets`, at token `token`.
    fn bracket(&mut self, b: Bracket<'a>, token: u32, line: u32) -> Result<(), CorefError> {
        let key = piece_text(&b.entity);
        match b.kind {
            BracketKind::Single => {
                self.entity(key).push(Pending { first: token, last: token, open: b, close: None });
            }
            BracketKind::Open => {
                self.entity(key);
                // Java's `computeIfAbsent(id, ...).add(start)`: resizes and
                // inserts at the head of a bin as `merge` does.
                self.open.merge(key, vec![(token, b)], |open, new| open.extend(new));
            }
            BracketKind::Close => {
                let Some((first, open)) = self.open.get_mut(key).and_then(Vec::pop) else {
                    return Err(CorefError::NotOpen { line, entity: key.to_owned() });
                };
                self.entity(key).push(Pending { first, last: token, open, close: Some(b) });
            }
        }
        Ok(())
    }

    /// The finished mentions of an entity, added in first-seen order.
    fn entity(&mut self, key: &'a str) -> &mut Vec<Pending<'a>> {
        let at = *self.entity_index.entry(key).or_insert_with(|| {
            self.entities.push((key, Vec::new()));
            self.entities.len() - 1
        });
        &mut self.entities[at].1
    }

    /// `chains`: the mentions in text order, the longer first at one start
    /// and otherwise in the order collected, and the chains numbered by
    /// their first mention.
    fn finish(self, source: Span) -> CorefDocument<'a> {
        let tokens = self.tokens;
        let mut pending: Vec<(&'a str, Pending<'a>)> =
            self.entities.into_iter().flat_map(|(key, ms)| ms.into_iter().map(move |m| (key, m))).collect();
        let text_of =
            |p: &Pending<'_>| Span::new(tokens[p.first as usize].text.start, tokens[p.last as usize].text.end);
        pending.sort_by(|(_, a), (_, b)| {
            let (a, b) = (text_of(a), text_of(b));
            a.start.cmp(&b.start).then(b.end.cmp(&a.end))
        });
        let mut chains: HashMap<&'a str, u32> = HashMap::new();
        let mut clusters: Vec<Cluster<'a>> = Vec::new();
        let mut mentions = Vec::with_capacity(pending.len());
        for (i, (key, p)) in pending.into_iter().enumerate() {
            let chain = *chains.entry(key).or_insert_with(|| {
                clusters.push(Cluster { entity: p.open.entity.clone(), mentions: Vec::new() });
                clusters.len() as u32 - 1
            });
            clusters[chain as usize].mentions.push(i as u32);
            let (first, last) = (&tokens[p.first as usize], &tokens[p.last as usize]);
            mentions.push(Mention {
                chain,
                tokens: p.first..p.last + 1,
                text: Span::new(first.text.start, last.text.end),
                source: Span::new(first.form.span.start, last.form.span.end),
                open: p.open,
                close: p.close,
            });
        }
        CorefDocument {
            text: self.text,
            sentences: self.sentences,
            tokens,
            mentions,
            clusters,
            warnings: self.warnings,
            source,
        }
    }
}

/// The text of a field read from the source, borrowed for as long as the
/// source.
fn piece_text<'a>(field: &Field<'a>) -> &'a str {
    match field.text {
        std::borrow::Cow::Borrowed(s) => s,
        std::borrow::Cow::Owned(_) => unreachable!("a bracket's fields are the source's bytes"),
    }
}

fn span(start: usize, end: usize) -> Span {
    Span::new(start as u32, end as u32)
}

/// What kind of row an id names.
enum Id {
    Word,
    Multiword,
    EmptyNode,
    Invalid,
}

/// `plainId`: a positive decimal is a word; `N-M` with `N` < `M` both
/// positive is a multiword token and `N.M` with `N` not negative and `M`
/// positive an empty node; every decimal without a leading zero.
fn word_id(id: &str) -> Id {
    if positive(id) {
        return Id::Word;
    }
    let compound = |separator: char| {
        let Some((first, second)) = id.split_once(separator) else {
            return false;
        };
        if first.is_empty() || second.contains(separator) || !positive(second) {
            return false;
        }
        if separator == '.' {
            first == "0" || positive(first)
        } else {
            positive(first) && (first.len(), first) < (second.len(), second)
        }
    };
    if compound('-') {
        Id::Multiword
    } else if compound('.') {
        Id::EmptyNode
    } else {
        Id::Invalid
    }
}

/// `positiveInteger`: ASCII digits without a leading zero.
fn positive(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty() && (b'1'..=b'9').contains(&b[0]) && b[1..].iter().all(u8::is_ascii_digit)
}

/// The columns between every tab, the empty ones at the end included.
fn split_all(line: Piece<'_>) -> Vec<Piece<'_>> {
    let mut out = Vec::with_capacity(COLUMNS);
    let mut start = 0;
    for end in line.text.match_indices('\t').map(|(i, _)| i).chain([line.text.len()]) {
        out.push(line.slice(start, end));
        start = end + 1;
    }
    out
}

/// `StringUtil.trimUnicodeWhitespace`: the piece without the Unicode
/// whitespace at its ends.
fn unicode_trim(piece: Piece<'_>) -> Piece<'_> {
    let ws = |c: char| WhitespaceMode::Unicode.is_whitespace(c as u32);
    let t = piece.text;
    let start = t.len() - t.trim_start_matches(ws).len();
    let end = t.trim_end_matches(ws).len().max(start);
    piece.slice(start, end)
}

#[cfg(test)]
mod tests;
