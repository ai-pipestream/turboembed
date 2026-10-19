use std::collections::HashMap;

use turbo_analysis::Span;
use turbo_jvm::hash_map::JavaHashMap;
use turbo_unicode::WhitespaceMode;

use super::super::Tagset;
use super::entity::{Bracket, BracketKind, entity_brackets, misc_attribute};
use super::{COLUMNS, Cluster, CorefDocument, CorefError, CorefSentence, CorefToken, CorefWarning, Mention, SPEAKER};
use crate::source::{Field, Line, Piece};

/// A mention while it is read: its tokens and brackets.
struct Pending<'a> {
    first: u32,
    last: u32,
    open: Bracket<'a>,
    close: Option<Bracket<'a>>,
}

/// The state of `ConlluCorefDocumentStream.parse` over one document.
pub(super) struct Builder<'a> {
    tagset: Tagset,
    text: String,
    sentences: Vec<CorefSentence<'a>>,
    tokens: Vec<CorefToken<'a>>,
    warnings: Vec<CorefWarning>,
    /// The open mentions of each entity, innermost last (`openStarts`). It
    /// is a model of the Java `HashMap` because the error for a mention
    /// left open names the first entity in that map's order.
    open: JavaHashMap<&'a str, Vec<(u32, Bracket<'a>)>>,
    /// The number of mentions open in `open`, so a boundary with none open
    /// does not walk the map, whose keys are every entity opened so far.
    open_mentions: usize,
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
    pub(super) fn new(tagset: Tagset) -> Builder<'a> {
        Builder {
            tagset,
            text: String::new(),
            sentences: Vec::new(),
            tokens: Vec::new(),
            warnings: Vec::new(),
            open: JavaHashMap::new(),
            open_mentions: 0,
            entities: Vec::new(),
            entity_index: HashMap::new(),
            sentence: None,
            sentence_start: 0,
            speaker: None,
            expected: 1,
            glue: false,
        }
    }

    pub(super) fn read(mut self, lines: &[Line<'a>]) -> Result<CorefDocument<'a>, CorefError> {
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
        if self.open_mentions == 0 {
            return Ok(());
        }
        match self.open.iter().find(|(_, open)| !open.is_empty()) {
            Some((entity, _)) => Err(error((*entity).to_owned())),
            None => Ok(()),
        }
    }

    fn word(&mut self, line: Line<'a>) -> Result<(), CorefError> {
        let number = line.number;
        let Some(columns) = split_columns(line.piece) else {
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
        // `fields[0].equals(Integer.toString(expectedWordId))`: the id is a
        // decimal without a leading zero, so it is equal when its value is.
        if id.text.parse::<u32>() != Ok(self.expected) {
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
                self.open_mentions += 1;
            }
            BracketKind::Close => {
                let Some((first, open)) = self.open.get_mut(key).and_then(Vec::pop) else {
                    return Err(CorefError::NotOpen { line, entity: key.to_owned() });
                };
                self.open_mentions -= 1;
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
pub(super) enum Id {
    Word,
    Multiword,
    EmptyNode,
    Invalid,
}

/// `plainId`: a positive decimal is a word; `N-M` with `N` < `M` both
/// positive is a multiword token and `N.M` with `N` not negative and `M`
/// positive an empty node; every decimal without a leading zero.
pub(super) fn word_id(id: &str) -> Id {
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

/// The columns between every tab, the empty ones at the end included,
/// when there are ten of them.
fn split_columns(line: Piece<'_>) -> Option<[Piece<'_>; COLUMNS]> {
    let mut columns = [line.slice(0, 0); COLUMNS];
    let mut count = 0;
    let mut start = 0;
    for end in line.text.match_indices('\t').map(|(i, _)| i).chain([line.text.len()]) {
        *columns.get_mut(count)? = line.slice(start, end);
        count += 1;
        start = end + 1;
    }
    (count == COLUMNS).then_some(columns)
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
