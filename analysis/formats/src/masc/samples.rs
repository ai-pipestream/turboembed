//! Streaming typed part-of-speech, token, sentence and name samples from MASC.
//!
//! Every reader supports explicit reads, reset and iteration. Iterators stop
//! cleanly at the end of the corpus; direct reads retain their typed errors.

use std::borrow::Cow;

use turbo_analysis::Span;

use super::corpus::MascCorpus;
use super::document::{MascDocument, MascSentence};
use super::error::{MascError, MascStream};
use crate::sample::{PosSample, SentenceSample, TokenSample};
use crate::source::Field;

mod readers;
#[cfg(test)]
mod tests;

pub use self::readers::{MascNameReader, MascSentenceReader};

/// A sample of one sentence of a corpus, with the place of the document
/// and of the sentence it was read from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascSample<S> {
    /// The document, a place in [`MascCorpus::documents`].
    pub document: usize,
    /// The sentence, a place in [`MascDocument::sentences`].
    pub sentence: usize,
    pub sample: S,
}

/// A sentence sample, whose sentences may come from several documents:
/// the sample's sources are byte spans of the documents' texts, and
/// `documents` gives the document of each sentence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascSentenceSample<'c> {
    pub sample: SentenceSample<'c>,
    /// The document of each sentence, places in [`MascCorpus::documents`].
    pub documents: Vec<usize>,
}

/// Where a reader is in the corpus: the document it reads (`None` once
/// the corpus is used up, Java's null `buffer`), the next document of the
/// document stream, and the next sentence of each document.
#[derive(Clone, Debug)]
struct Cursor<'c> {
    corpus: &'c MascCorpus,
    buffer: Option<usize>,
    next_document: usize,
    next_sentence: Vec<usize>,
}

impl<'c> Cursor<'c> {
    fn new(corpus: &'c MascCorpus) -> Cursor<'c> {
        Cursor { corpus, buffer: None, next_document: 0, next_sentence: vec![0; corpus.documents().len()] }
    }

    fn doc(&self, d: usize) -> &'c MascDocument {
        &self.corpus.documents()[d]
    }

    /// `MascDocumentStream.read`.
    fn read_document(&mut self) -> Option<usize> {
        let d = self.next_document;
        if d < self.corpus.documents().len() {
            self.next_document += 1;
            Some(d)
        } else {
            None
        }
    }

    /// `MascDocument.read` on document `d`.
    fn read_sentence(&mut self, d: usize) -> Option<usize> {
        let s = self.next_sentence[d];
        if s < self.doc(d).sentences().len() {
            self.next_sentence[d] += 1;
            Some(s)
        } else {
            None
        }
    }

    /// `buffer.read()`, which throws on a null buffer.
    fn read_buffer(&mut self) -> Result<Option<usize>, MascError> {
        let d = self.buffer.ok_or(MascError::NullDocument)?;
        Ok(self.read_sentence(d))
    }

    /// The document stream's `reset`, then `buffer = samples.read()`.
    fn reset(&mut self) {
        self.next_document = 0;
        self.next_sentence.iter_mut().for_each(|s| *s = 0);
        self.buffer = self.read_document();
    }

    /// The next sentence of the buffer, moving on to the next document
    /// that `wanted` accepts when the buffer has no more: the read the
    /// part-of-speech, token and name streams share. `None` at the end.
    fn next_sentence_where(
        &mut self,
        wanted: impl Fn(&MascDocument) -> bool,
    ) -> Result<Option<(usize, usize)>, MascError> {
        let mut sentence = self.read_buffer()?;
        loop {
            if let (Some(s), Some(d)) = (sentence, self.buffer) {
                return Ok(Some((d, s)));
            }
            self.buffer = self.read_document();
            let Some(d) = self.buffer else {
                return Ok(None);
            };
            if wanted(self.doc(d)) {
                sentence = self.read_sentence(d);
            }
        }
    }

    /// The constructors of the part-of-speech, token and name streams:
    /// the first document `wanted` accepts.
    fn first_where(&mut self, wanted: impl Fn(&MascDocument) -> bool, stream: MascStream) -> Result<(), MascError> {
        loop {
            self.buffer = self.read_document();
            match self.buffer {
                None => return Err(MascError::NoDocuments { stream }),
                Some(d) if wanted(self.doc(d)) => return Ok(()),
                Some(_) => {}
            }
        }
    }
}

/// The token fields of a sentence, borrowed from its document's text.
pub(super) fn token_fields<'c>(doc: &'c MascDocument, sentence: &MascSentence) -> Result<Vec<Field<'c>>, MascError> {
    Ok(doc.token_spans(sentence)?.into_iter().map(|span| Field::of(doc.text(), span)).collect())
}

/// The bytes from the first token field's start to the furthest end, or
/// an empty span at the start of the sentence's region (or of the text).
pub(super) fn cover(doc: &MascDocument, sentence: &MascSentence, fields: &[Field<'_>]) -> Span {
    match (fields.iter().map(|f| f.span.start).min(), fields.iter().map(|f| f.span.end).max()) {
        (Some(start), Some(end)) if start <= end => Span::new(start, end),
        (Some(start), _) => Span::new(start, start),
        _ => {
            let at = doc.sentence_text(sentence).map_or(0, |s| s.start);
            Span::new(at, at)
        }
    }
}

/// Fuses an iterator over reads at the end and at the null document.
fn next_of<T>(done: &mut bool, read: Result<Option<T>, MascError>) -> Option<Result<T, MascError>> {
    if *done {
        return None;
    }
    match read {
        Ok(Some(s)) => Some(Ok(s)),
        Ok(None) | Err(MascError::NullDocument) => {
            *done = true;
            None
        }
        Err(e) => Some(Err(e)),
    }
}

/// `MascPOSSampleStream`: a part-of-speech sample per sentence of the
/// documents with Penn tags.
///
/// A sample's tokens are fields of the document's text; its tags are the
/// Penn file's `msd` values, borrowed from the document, with the span of
/// their token. A token without a tag fails the sample, as `POSSample`
/// refuses a null tag.
#[derive(Clone, Debug)]
pub struct MascPosReader<'c> {
    cursor: Cursor<'c>,
    done: bool,
}

impl<'c> MascPosReader<'c> {
    /// # Errors
    ///
    /// [`MascError::NoDocuments`] when no document has Penn tags.
    pub fn new(corpus: &'c MascCorpus) -> Result<MascPosReader<'c>, MascError> {
        let mut cursor = Cursor::new(corpus);
        cursor.first_where(MascDocument::has_penn_tags, MascStream::PartOfSpeech)?;
        Ok(MascPosReader { cursor, done: false })
    }

    /// `read`.
    ///
    /// # Errors
    ///
    /// What `read` throws: a token outside the text, a token without a
    /// tag, a document without tokens after a reset, or a read after the
    /// end.
    pub fn read(&mut self) -> Result<Option<MascSample<PosSample<'c>>>, MascError> {
        let Some((d, s)) = self.cursor.next_sentence_where(MascDocument::has_penn_tags)? else {
            return Ok(None);
        };
        let doc = self.cursor.doc(d);
        let sentence = &doc.sentences()[s];
        let tokens = token_fields(doc, sentence)?;
        let list = sentence.tokens().unwrap_or_default();
        let mut tags = Vec::with_capacity(list.len());
        for (t, field) in list.iter().zip(&tokens) {
            let tag = t.tag.as_deref().ok_or(MascError::NullTag)?;
            tags.push(Field { text: Cow::Borrowed(tag), span: field.span });
        }
        let source = cover(doc, sentence, &tokens);
        Ok(Some(MascSample { document: d, sentence: s, sample: PosSample::new(tokens, tags, source)? }))
    }

    /// `reset`: back to the first document, whether it has Penn tags or
    /// not (OpenNLP's reset does not look), so the next read of a corpus
    /// whose first document has none fails.
    pub fn reset(&mut self) {
        self.cursor.reset();
        self.done = false;
    }
}

impl<'c> Iterator for MascPosReader<'c> {
    type Item = Result<MascSample<PosSample<'c>>, MascError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let read = self.read();
        next_of(&mut self.done, read)
    }
}

/// A token sample per nonempty Penn-tokenized sentence.
///
/// The sample covers the first token through the last, and each token also
/// retains its exact source span. Empty sentences and empty tokens are skipped
/// and counted without discarding the rest of the corpus.
#[derive(Clone, Debug)]
pub struct MascTokenReader<'c> {
    cursor: Cursor<'c>,
    done: bool,
    skipped: usize,
}

impl<'c> MascTokenReader<'c> {
    /// # Errors
    ///
    /// [`MascError::NoDocuments`] when no document has Penn tags.
    pub fn new(corpus: &'c MascCorpus) -> Result<MascTokenReader<'c>, MascError> {
        let mut cursor = Cursor::new(corpus);
        cursor.first_where(MascDocument::has_penn_tags, MascStream::Tokens)?;
        Ok(MascTokenReader { cursor, done: false, skipped: 0 })
    }

    /// The empty sentences skipped so far.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// Reads the next valid token sample.
    ///
    /// # Errors
    ///
    /// Returns an error for backwards or out-of-bounds token spans, an
    /// untokenized document after a reset, or a read after the end.
    pub fn read(&mut self) -> Result<Option<MascSample<TokenSample<'c>>>, MascError> {
        loop {
            let Some((d, s)) = self.cursor.next_sentence_where(MascDocument::has_penn_tags)? else {
                return Ok(None);
            };
            let doc = self.cursor.doc(d);
            let sentence = &doc.sentences()[s];
            let text = doc.token_text(sentence)?;
            let tokens = sentence.tokens().unwrap_or_default();
            let offset = text.map_or(0, |(_, first)| first);
            let mut units = Vec::with_capacity(tokens.len());
            for t in tokens {
                let start = i64::from(t.start) - i64::from(offset);
                if start < 0 {
                    return Err(MascError::NegativeTokenStart { start });
                }
                units.push((start, i64::from(t.end) - i64::from(offset)));
            }
            if text.is_none_or(|(span, _)| span.is_empty()) || units.iter().any(|&(start, end)| end - start == 0) {
                self.skipped += 1;
                continue;
            }
            let Some((text_span, _)) = text else {
                continue;
            };
            let len = doc.text()[text_span.range()].encode_utf16().count();
            if let Some(&(start, end)) = units.iter().find(|&&(start, end)| start > len as i64 || end > len as i64) {
                return Err(MascError::TokenOutOfBounds { start, end, len });
            }
            let mut relative = Vec::with_capacity(tokens.len());
            let mut sources = Vec::with_capacity(tokens.len());
            for t in tokens {
                let span = doc.span(i64::from(t.start), i64::from(t.end))?;
                relative.push(Span::new(span.start - text_span.start, span.end - text_span.start));
                sources.push(span);
            }
            let sample = TokenSample::new(Cow::Borrowed(&doc.text()[text_span.range()]), relative, sources, text_span)?;
            return Ok(Some(MascSample { document: d, sentence: s, sample }));
        }
    }

    /// Returns to the first document and clears the skipped-sentence count.
    pub fn reset(&mut self) {
        self.cursor.reset();
        self.done = false;
        self.skipped = 0;
    }
}

impl<'c> Iterator for MascTokenReader<'c> {
    type Item = Result<MascSample<TokenSample<'c>>, MascError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let read = self.read();
        next_of(&mut self.done, read)
    }
}
