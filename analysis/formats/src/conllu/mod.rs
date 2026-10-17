//! The CoNLL-U format of Universal Dependencies: `ConlluStream` and the
//! sample streams OpenNLP reads from it.
//!
//! A sentence is a block of lines ended by a line of length zero (a line
//! of spaces does not end one). A line that starts with `#` after
//! `String.trim` is a comment; every other line is a word line of ten
//! tab-separated columns (shown here with spaces):
//!
//! ```text
//! # sent_id = 1
//! # text = Vamos al mar.
//! 1  Vamos  ir  VERB  _  _  0  root  _  _
//! 2-3  al  _  _  _  _  _  _  _  _
//! 2  a  a  ADP  _  _  4  case  _  _
//! 3  el  el  DET  _  _  4  det  _  _
//! 4  mar  mar  NOUN  _  _  1  obl  _  SpaceAfter=No
//! 5  .  .  PUNCT  _  _  1  punct  _  _
//! ```
//!
//! [`ConlluReader`] reads the sentences. A multiword token line (an id
//! `start-end`) replaces the word lines of its range: its lemma, both tags
//! and its features are those of its words joined with `+`, leaving out
//! the words whose value is `_`, so the line above reads as the token
//! `al` with the lemma `a+el` and the universal tag `ADP+DET`. A joined
//! value is a [`Field`] with a text of its own and a span from its first
//! piece to its last; a value taken from one word is that word's field.
//! The other columns are the multiword line's own. Empty nodes (ids like
//! `8.1`) are word lines like any other.
//!
//! The sample readers are in [`samples`]. [`dependency`] reads
//! dependency samples and [`coref`] documents with CorefUD coreference;
//! neither goes through [`ConlluReader`], as their Java streams read the
//! lines themselves.
//!
//! # What OpenNLP does here
//!
//! - A comment's key and value are trimmed with `String.trim`, and a
//!   comment with an empty value is ignored, except a `text_<language>`
//!   key, which is kept with an empty text. A `text_` key without two or
//!   three lowercase ASCII letters after a `text_` is an error.
//! - A multiword range that names a word id no line has is an error, and
//!   ids are compared as strings: `01` is not `1`.
//! - `ConlluStream` removes the words of every multiword token by their
//!   places in the sentence before any removal, the last first. Ranges
//!   that are out of order remove other lines than their words, and
//!   ranges that overlap or repeat remove too many: that removes a merged
//!   line or other words, or throws an `IndexOutOfBoundsException`
//!   ([`FormatError::MultiwordRemoval`]). This reader removes the same
//!   lines and fails the same way.
//! - A repeated multiword id merges only its last line; the first is kept
//!   as it is, and the words are removed twice.
//! - An error in a sentence discards the whole sentence, and reading goes
//!   on with the next.

pub mod coref;
pub mod dependency;
pub mod document;
pub mod node_ids;
pub mod samples;
pub mod words;

use std::collections::HashMap;

use turbo_analysis::Span;

use crate::error::FormatError;
use crate::source::{Field, Line, Lines, Piece, java_trim, split_fields};

pub use coref::{ConlluCorefReader, CorefDocument};
pub use dependency::ConlluDependencyReader;
pub use document::write_document_conllu;
pub use samples::{ConlluLemmaReader, ConlluPosReader, ConlluSentenceReader, ConlluTokenReader};
pub use words::{
    ConlluWordReader, MultiwordToken, SyntacticSentence, misc_spaces_in_token, misc_trailing_space,
    write_conllu_sentences,
};

/// The tag column a sample takes its tags from: `ConlluTagset`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Tagset {
    /// The universal part-of-speech tag, column 4 (`UPOS`). OpenNLP's
    /// default.
    #[default]
    U,
    /// The language-specific tag, column 5 (`XPOS`).
    X,
}

impl Tagset {
    /// `ConlluTagset.fromParameter`: `u` or `x`, the value of the format
    /// factories' `tagset` parameter.
    pub fn from_parameter(parameter: &str) -> Option<Tagset> {
        match parameter {
            "u" => Some(Tagset::U),
            "x" => Some(Tagset::X),
            _ => None,
        }
    }
}

/// One word line of a sentence, or the merged line of a multiword token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WordLine<'a> {
    pub id: Field<'a>,
    pub form: Field<'a>,
    pub lemma: Field<'a>,
    pub upos: Field<'a>,
    pub xpos: Field<'a>,
    pub feats: Field<'a>,
    pub head: Field<'a>,
    pub deprel: Field<'a>,
    pub deps: Field<'a>,
    pub misc: Field<'a>,
    /// The line's number, counting from 1; a merged line's is the
    /// multiword line's.
    pub line: u32,
    /// The bytes of the line.
    pub source: Span,
}

impl<'a> WordLine<'a> {
    /// `new ConlluWordLine(line)`.
    fn parse(line: Line<'a>) -> Result<WordLine<'a>, FormatError> {
        let columns = split_fields(line.piece, '\t');
        let Ok([id, form, lemma, upos, xpos, feats, head, deprel, deps, misc]) = <[Piece<'a>; 10]>::try_from(columns)
        else {
            return Err(FormatError::ConlluColumns { line: line.number, text: line.text().to_owned() });
        };
        Ok(WordLine {
            id: id.field(),
            form: form.field(),
            lemma: lemma.field(),
            upos: upos.field(),
            xpos: xpos.field(),
            feats: feats.field(),
            head: head.field(),
            deprel: deprel.field(),
            deps: deps.field(),
            misc: misc.field(),
            line: line.number,
            source: line.span(),
        })
    }

    /// The tag in `tagset`'s column.
    pub fn tag(&self, tagset: Tagset) -> &Field<'a> {
        match tagset {
            Tagset::U => &self.upos,
            Tagset::X => &self.xpos,
        }
    }
}

/// A `# text_<language> = ...` comment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextLang<'a> {
    /// The two or three lowercase ASCII letters after `text_`.
    pub language: Field<'a>,
    pub text: Field<'a>,
}

/// One sentence: `ConlluSentence`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sentence<'a> {
    /// The word lines, each multiword token's words merged into its line.
    pub word_lines: Vec<WordLine<'a>>,
    /// The `sent_id` comment.
    pub sentence_id: Option<Field<'a>>,
    /// The `text` comment.
    pub text: Option<Field<'a>>,
    /// Whether a `newdoc` comment starts a document here.
    pub new_document: bool,
    pub document_id: Option<Field<'a>>,
    /// Whether a `newpar` comment starts a paragraph here.
    pub new_paragraph: bool,
    pub paragraph_id: Option<Field<'a>>,
    /// The `text_<language>` comments, in the order their languages first
    /// appear; a later comment for a language replaces the earlier one.
    pub text_lang: Vec<TextLang<'a>>,
    /// The `translit` comment.
    pub translit: Option<Field<'a>>,
    /// The number of the sentence's first line.
    pub line: u32,
    /// The bytes of the sentence's lines.
    pub source: Span,
}

/// Reads the sentences of a CoNLL-U text: `ConlluStream`.
#[derive(Clone, Debug)]
pub struct ConlluReader<'a> {
    lines: Lines<'a>,
}

impl<'a> ConlluReader<'a> {
    pub fn new(source: &'a str) -> Result<ConlluReader<'a>, FormatError> {
        Ok(ConlluReader { lines: Lines::new(source)? })
    }

    /// `ParagraphStream.read`: the next run of lines that are not of
    /// length zero.
    fn block(&mut self) -> Vec<Line<'a>> {
        let mut block = Vec::new();
        for line in self.lines.by_ref() {
            if !line.text().is_empty() {
                block.push(line);
            } else if !block.is_empty() {
                break;
            }
        }
        block
    }
}

impl<'a> Iterator for ConlluReader<'a> {
    type Item = Result<Sentence<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let block = self.block();
        let (first, last) = (block.first()?, block.last()?);
        let source = Span::new(first.span().start, last.span().end);
        Some(read_sentence(&block, first.number, source))
    }
}

fn read_sentence<'a>(block: &[Line<'a>], first: u32, source: Span) -> Result<Sentence<'a>, FormatError> {
    let mut sentence = Sentence {
        word_lines: Vec::new(),
        sentence_id: None,
        text: None,
        new_document: false,
        document_id: None,
        new_paragraph: false,
        paragraph_id: None,
        text_lang: Vec::new(),
        translit: None,
        line: first,
        source,
    };
    for &line in block {
        let trimmed = java_trim(line.piece);
        if !trimmed.text.starts_with('#') {
            sentence.word_lines.push(WordLine::parse(line)?);
            continue;
        }
        let comment = trimmed.slice(1, trimmed.text.len());
        let Some(eq) = comment.text.find('=') else {
            match java_trim(comment).text {
                "newdoc" => sentence.new_document = true,
                "newpar" => sentence.new_paragraph = true,
                _ => {}
            }
            continue;
        };
        let key = java_trim(comment.slice(0, eq));
        let value = java_trim(comment.slice(eq + 1, comment.text.len()));
        if !value.text.is_empty() {
            match key.text {
                "newdoc id" => {
                    sentence.new_document = true;
                    sentence.document_id = Some(value.field());
                }
                "newpar id" => {
                    sentence.new_paragraph = true;
                    sentence.paragraph_id = Some(value.field());
                }
                "sent_id" => sentence.sentence_id = Some(value.field()),
                "text" => sentence.text = Some(value.field()),
                "translit" => sentence.translit = Some(value.field()),
                _ => {}
            }
        }
        if key.text.starts_with(TEXT_LANG_PREFIX) {
            let Some(language) = text_language(key) else {
                return Err(FormatError::InvalidLanguage { line: line.number, key: key.text.to_owned() });
            };
            let entry = TextLang { language: language.field(), text: value.field() };
            match sentence.text_lang.iter_mut().find(|t| t.language.as_str() == language.text) {
                Some(old) => *old = entry,
                None => sentence.text_lang.push(entry),
            }
        }
    }
    merge_multiwords(&mut sentence.word_lines, first)?;
    Ok(sentence)
}

const TEXT_LANG_PREFIX: &str = "text_";

/// `extractTextLang`: the two or three lowercase ASCII letters after the
/// first `text_` that has at least two.
fn text_language(key: Piece<'_>) -> Option<Piece<'_>> {
    let mut from = 0;
    while let Some(at) = key.text[from..].find(TEXT_LANG_PREFIX).map(|i| from + i) {
        let start = at + TEXT_LANG_PREFIX.len();
        let len = key.text[start..].bytes().take(3).take_while(u8::is_ascii_lowercase).count();
        if len >= 2 {
            return Some(key.slice(start, start + len));
        }
        from = at + 1;
    }
    None
}

/// Why a multiword token id is refused.
enum RangeError {
    Invalid,
    Backwards,
}

/// `parseContractionRange` of an id that holds a hyphen: the first and
/// last word id.
fn multiword_range(id: &str) -> Result<(u32, u32), RangeError> {
    let bytes = id.as_bytes();
    let hyphen = id.find('-').expect("a multiword id holds a hyphen");
    let digits_end = |from: usize| from + bytes[from..].iter().take_while(|b| b.is_ascii_digit()).count();
    let start_end = digits_end(0);
    let end_end = digits_end(hyphen + 1);
    if hyphen < 1
        || start_end != hyphen
        || end_end == hyphen + 1
        || end_end != id.len()
        || bytes[0] == b'0'
        || bytes[hyphen + 1] == b'0'
    {
        return Err(RangeError::Invalid);
    }
    let parse = |digits: &str| digits.parse::<i32>().map_err(|_| RangeError::Invalid);
    let (start, end) = (parse(&id[..hyphen])?, parse(&id[hyphen + 1..])?);
    if end < start {
        return Err(RangeError::Backwards);
    }
    // Both are digit runs, so neither is negative.
    Ok((start as u32, end as u32))
}

/// `postProcessContractions`: each multiword token's words merged into
/// its line, and the words removed as Java removes them.
fn merge_multiwords(lines: &mut Vec<WordLine<'_>>, first: u32) -> Result<(), FormatError> {
    // The place of each id; a repeated id has the place of its last line.
    let index: HashMap<String, usize> = lines.iter().enumerate().map(|(i, l)| (l.id.as_str().to_owned(), i)).collect();
    // Each multiword id with the places of its words, in the order the ids
    // first appear, the words of a repeated id from its last line.
    let mut multiwords: Vec<(&str, Vec<usize>)> = Vec::new();
    let mut to_remove = Vec::new();
    for line in lines.iter() {
        let id = line.id.as_str();
        if !id.contains('-') {
            continue;
        }
        let (start, end) = multiword_range(id).map_err(|e| match e {
            RangeError::Invalid => FormatError::InvalidMultiwordId { line: line.line, id: id.to_owned() },
            RangeError::Backwards => FormatError::BackwardsMultiwordId { line: line.line, id: id.to_owned() },
        })?;
        let mut words = Vec::new();
        for j in u64::from(start)..=u64::from(end) {
            let Some(&at) = index.get(&j.to_string()) else {
                return Err(FormatError::MissingMultiwordLine { line: line.line, id: id.to_owned(), missing: j });
            };
            words.push(at);
            to_remove.push(at);
        }
        match multiwords.iter_mut().find(|(m, _)| *m == id) {
            Some(old) => old.1 = words,
            None => multiwords.push((id, words)),
        }
    }
    let merged: Vec<(usize, WordLine<'_>)> =
        multiwords.iter().map(|(id, words)| (index[*id], merge(&lines[index[*id]], words, lines))).collect();
    for (at, line) in merged {
        lines[at] = line;
    }
    for &at in to_remove.iter().rev() {
        if at >= lines.len() {
            return Err(FormatError::MultiwordRemoval { line: first, index: at, len: lines.len() });
        }
        lines.remove(at);
    }
    Ok(())
}

/// `mergeAnnotation`: the multiword line with the lemma, tags and
/// features of its words.
fn merge<'a>(multiword: &WordLine<'a>, words: &[usize], lines: &[WordLine<'a>]) -> WordLine<'a> {
    let column = |get: for<'l> fn(&'l WordLine<'a>) -> &'l Field<'a>| {
        join(words.iter().map(|&w| get(&lines[w])), multiword.source)
    };
    WordLine {
        lemma: column(|l| &l.lemma),
        upos: column(|l| &l.upos),
        xpos: column(|l| &l.xpos),
        feats: column(|l| &l.feats),
        ..multiword.clone()
    }
}

/// The values that are not `_`, joined with `+`: the value itself when
/// there is one, a field with a text of its own and a span from the
/// first value to the last when there are more, and an empty field at the
/// end of the multiword line `line` when there is none.
fn join<'f, 'a: 'f>(values: impl Iterator<Item = &'f Field<'a>>, line: Span) -> Field<'a> {
    let kept: Vec<&Field<'a>> = values.filter(|v| v.as_str() != "_").collect();
    match kept.as_slice() {
        [one] => (*one).clone(),
        [] => Field::made(String::new(), Span::new(line.end, line.end)),
        _ => {
            let start = kept.iter().map(|v| v.span.start).min().expect("not empty");
            let end = kept.iter().map(|v| v.span.end).max().expect("not empty");
            let text: Vec<&str> = kept.iter().map(|v| v.as_str()).collect();
            Field::made(text.join("+"), Span::new(start, end))
        }
    }
}

#[cfg(test)]
mod tests;
