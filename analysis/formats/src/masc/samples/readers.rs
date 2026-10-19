use std::borrow::Cow;

use turbo_analysis::Span;

use super::super::corpus::MascCorpus;
use super::super::document::MascDocument;
use super::super::error::{MascError, MascStream};
use super::{Cursor, MascSample, MascSentenceSample, cover, next_of, token_fields};
use crate::sample::{Name, NameSample, SentenceSample};

/// `MascNamedEntitySampleStream`: a name sample per sentence of the
/// documents with named entities, each with `clearAdaptiveData` set.
#[derive(Clone, Debug)]
pub struct MascNameReader<'c> {
    cursor: Cursor<'c>,
    done: bool,
}

impl<'c> MascNameReader<'c> {
    /// # Errors
    ///
    /// [`MascError::NoDocuments`] when no document has named entities.
    pub fn new(corpus: &'c MascCorpus) -> Result<MascNameReader<'c>, MascError> {
        let mut cursor = Cursor::new(corpus);
        cursor.first_where(MascDocument::has_named_entities, MascStream::NamedEntities)?;
        Ok(MascNameReader { cursor, done: false })
    }

    /// `read`.
    ///
    /// # Errors
    ///
    /// What `read` throws: a token outside the text, names that overlap,
    /// a document without tokens after a reset, or a read after the end.
    pub fn read(&mut self) -> Result<Option<MascSample<NameSample<'c>>>, MascError> {
        let Some((d, s)) = self.cursor.next_sentence_where(MascDocument::has_named_entities)? else {
            return Ok(None);
        };
        let doc = self.cursor.doc(d);
        let sentence = &doc.sentences()[s];
        let tokens = token_fields(doc, sentence)?;
        let mut names = Vec::with_capacity(sentence.names().len());
        for n in sentence.names() {
            names.push(Name::new(n.start, n.end, n.kind.as_deref())?);
        }
        let source = cover(doc, sentence, &tokens);
        let sample = NameSample::new(tokens, names, true, source, 0)?;
        Ok(Some(MascSample { document: d, sentence: s, sample }))
    }

    /// `reset`: back to the first document, whether it has named entities
    /// or not.
    pub fn reset(&mut self) {
        self.cursor.reset();
        self.done = false;
    }
}

impl<'c> Iterator for MascNameReader<'c> {
    type Item = Result<MascSample<NameSample<'c>>, MascError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let read = self.read();
        next_of(&mut self.done, read)
    }
}

/// `MascSentenceSampleStream`: samples of `per_sample` sentences each,
/// from every document, Penn tags or not.
///
/// A sample's text is its sentences' regions with a space after each,
/// and a line feed where a document ends, all but the last character of
/// that: so a sample that starts in a new document starts with the line
/// feed, and the sentence before a document's end keeps its space. A
/// document's end does not count as a sentence.
#[derive(Clone, Debug)]
pub struct MascSentenceReader<'c> {
    cursor: Cursor<'c>,
    per_sample: usize,
    done: bool,
}

impl<'c> MascSentenceReader<'c> {
    pub fn new(corpus: &'c MascCorpus, per_sample: usize) -> MascSentenceReader<'c> {
        let mut cursor = Cursor::new(corpus);
        cursor.buffer = cursor.read_document();
        MascSentenceReader { cursor, per_sample, done: false }
    }

    /// `read`.
    ///
    /// # Errors
    ///
    /// What `read` throws: a region outside the text, or a read after the
    /// end, which follows the last sample unless it was full, and is the
    /// first read of an empty corpus.
    pub fn read(&mut self) -> Result<Option<MascSentenceSample<'c>>, MascError> {
        if self.per_sample == 0 {
            self.cursor.buffer.ok_or(MascError::NullDocument)?;
            return Ok(None);
        }
        let mut text = String::new();
        let mut spans = Vec::new();
        let mut sources = Vec::new();
        let mut documents = Vec::new();
        let mut i = 0;
        while i < self.per_sample {
            let d = self.cursor.buffer.ok_or(MascError::NullDocument)?;
            match self.cursor.read_sentence(d) {
                Some(s) => {
                    let doc = self.cursor.doc(d);
                    let source = doc.sentence_text(&doc.sentences()[s])?;
                    let start = text.len();
                    text.push_str(&doc.text()[source.range()]);
                    spans.push(Span::new(start as u32, text.len() as u32));
                    text.push(' ');
                    sources.push(source);
                    documents.push(d);
                    i += 1;
                }
                None => {
                    self.cursor.buffer = self.cursor.read_document();
                    if self.cursor.buffer.is_none() {
                        break;
                    }
                    text.push('\n');
                }
            }
        }
        if text.is_empty() {
            return Ok(None);
        }
        text.pop();
        if u32::try_from(text.len()).is_err() {
            return Err(MascError::TooLong { len: text.len() });
        }
        let sample = SentenceSample::new(Cow::Owned(text), spans, sources, Span::default())?;
        Ok(Some(MascSentenceSample { sample, documents }))
    }

    /// `reset`.
    pub fn reset(&mut self) {
        self.cursor.reset();
        self.done = false;
    }
}

impl<'c> Iterator for MascSentenceReader<'c> {
    type Item = Result<MascSentenceSample<'c>, MascError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let read = self.read();
        next_of(&mut self.done, read)
    }
}
