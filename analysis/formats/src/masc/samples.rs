//! The sample streams OpenNLP reads from a MASC corpus: part-of-speech,
//! token, sentence and name samples.
//!
//! Each reader has OpenNLP's `read`, which returns `Ok(None)` at the end
//! and fails where the Java stream throws, `reset`, and an iterator over
//! the same reads that ends at the end of the corpus. OpenNLP's streams
//! leave their document null at the end, so a `read` after the end throws
//! `NullPointerException` ([`MascError::NullDocument`]); the sentence
//! stream ends that way after its last sample unless that sample was
//! full. The iterators end there too, rather than yield that error.

use std::borrow::Cow;

use turbo_analysis::Span;

use super::corpus::MascCorpus;
use super::document::{MascDocument, MascSentence};
use super::error::{MascError, MascStream};
use crate::sample::{Name, NameSample, PosSample, SentenceSample, TokenSample};
use crate::source::Field;

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
fn token_fields<'c>(doc: &'c MascDocument, sentence: &MascSentence) -> Result<Vec<Field<'c>>, MascError> {
    Ok(doc.token_spans(sentence)?.into_iter().map(|span| Field::of(doc.text(), span)).collect())
}

/// The bytes from the first token field's start to the furthest end, or
/// an empty span at the start of the sentence's region (or of the text).
fn cover(doc: &MascDocument, sentence: &MascSentence, fields: &[Field<'_>]) -> Span {
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

/// What the token reader does with a sentence it cannot make a sample
/// of: one whose text is empty or that has an empty token.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum EmptySentence {
    /// As OpenNLP: the sentence and every sentence after it in the same
    /// `read` are skipped, so the read ends the corpus.
    #[default]
    EndsCorpus,
    /// The sentence alone is skipped and counted
    /// ([`MascTokenReader::skipped`]); the read goes on with the next
    /// sentence. Not what OpenNLP does: a training recipe that should not
    /// lose the rest of a corpus to one such sentence uses it.
    Skipped,
}

/// `MascTokenSampleStream`: a token sample per sentence of the documents
/// with Penn tags, of the text from the start of the sentence's first
/// token to the end of its last.
///
/// A sentence whose text is empty (it has no tokens) or that has an
/// empty token is skipped, and so is every sentence after it in the same
/// `read`: the stream's flag for a sentence found is set once before its
/// loop and never again, so the read goes on to the end of the corpus and
/// returns the end. [`MascTokenReader::with_empty_sentences`] can skip
/// such a sentence alone instead.
#[derive(Clone, Debug)]
pub struct MascTokenReader<'c> {
    cursor: Cursor<'c>,
    done: bool,
    empty: EmptySentence,
    skipped: usize,
}

impl<'c> MascTokenReader<'c> {
    /// # Errors
    ///
    /// [`MascError::NoDocuments`] when no document has Penn tags.
    pub fn new(corpus: &'c MascCorpus) -> Result<MascTokenReader<'c>, MascError> {
        MascTokenReader::with_empty_sentences(corpus, EmptySentence::EndsCorpus)
    }

    /// A reader that treats a sentence without a sample as `empty` says.
    ///
    /// # Errors
    ///
    /// [`MascError::NoDocuments`] when no document has Penn tags.
    pub fn with_empty_sentences(
        corpus: &'c MascCorpus,
        empty: EmptySentence,
    ) -> Result<MascTokenReader<'c>, MascError> {
        let mut cursor = Cursor::new(corpus);
        cursor.first_where(MascDocument::has_penn_tags, MascStream::Tokens)?;
        Ok(MascTokenReader { cursor, done: false, empty, skipped: 0 })
    }

    /// The sentences skipped alone so far under
    /// [`EmptySentence::Skipped`]; always 0 under
    /// [`EmptySentence::EndsCorpus`].
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    /// `read`.
    ///
    /// # Errors
    ///
    /// What `read` throws: tokens out of order so that the text runs
    /// backwards, a token starts before the first or ends after the last,
    /// a document without tokens after a reset, or a read after the end.
    pub fn read(&mut self) -> Result<Option<MascSample<TokenSample<'c>>>, MascError> {
        let mut found = true;
        loop {
            let Some((d, s)) = self.cursor.next_sentence_where(MascDocument::has_penn_tags)? else {
                return Ok(None);
            };
            let doc = self.cursor.doc(d);
            let sentence = &doc.sentences()[s];
            let text = doc.token_text(sentence)?;
            let tokens = sentence.tokens().unwrap_or_default();
            let offset = text.map_or(0, |(_, first)| first);
            // getTokensSpans: each token less the first token's start
            let mut units = Vec::with_capacity(tokens.len());
            for t in tokens {
                let start = i64::from(t.start) - i64::from(offset);
                if start < 0 {
                    return Err(MascError::NegativeTokenStart { start });
                }
                units.push((start, i64::from(t.end) - i64::from(offset)));
            }
            if text.is_none_or(|(span, _)| span.is_empty()) || units.iter().any(|&(start, end)| end - start == 0) {
                if self.empty == EmptySentence::Skipped {
                    self.skipped += 1;
                    continue;
                }
                found = false;
            }
            if !found {
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

    /// `reset`: back to the first document, whether it has Penn tags or
    /// not.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masc::document::MascFiles;

    /// A document of three sentences, `One two.`, `Three four.` and
    /// `Five six.`, whose middle sentence has no Penn tokens.
    fn corpus_with_an_empty_sentence() -> MascCorpus {
        let text = "One two. Three four. Five six.";
        let quarks = [(0, 3), (4, 7), (7, 8), (9, 14), (15, 19), (19, 20), (21, 25), (26, 29), (29, 30)];
        let mut seg = String::from("<graph>");
        for (i, (a, b)) in quarks.iter().enumerate() {
            seg.push_str(&format!("<region xml:id=\"seg-r{i}\" anchors=\"{a} {b}\"/>"));
        }
        seg.push_str("</graph>");
        let mut s = String::from("<graph>");
        for (i, (a, b)) in [(0, 8), (9, 20), (21, 30)].iter().enumerate() {
            s.push_str(&format!(
                "<region xml:id=\"s-r{i}\" anchors=\"{a} {b}\"/><node xml:id=\"s-n{i}\"><link targets=\"s-r{i}\"/></node>"
            ));
        }
        s.push_str("</graph>");
        let mut penn = String::from("<graph>");
        for i in [0, 1, 2, 6, 7, 8] {
            penn.push_str(&format!(
                "<node xml:id=\"penn-n{i}\"><link targets=\"seg-r{i}\"/></node>\
                 <a label=\"tok\" ref=\"penn-n{i}\"><fs><f name=\"msd\" value=\"NN\"/></fs></a>"
            ));
        }
        penn.push_str("</graph>");
        let files = MascFiles {
            text: text.as_bytes(),
            segments: Some(seg.as_bytes()),
            sentences: Some(s.as_bytes()),
            penn: Some(penn.as_bytes()),
            named_entities: None,
        };
        let doc = MascDocument::parse("/corpus/doc.hdr", files).unwrap();
        // The same document twice, so the sentence after the empty one in
        // the second document shows where the read stops.
        MascCorpus::from_documents(vec![doc.clone(), doc])
    }

    fn texts(reader: MascTokenReader<'_>) -> Vec<(usize, String)> {
        reader.map(|s| s.unwrap()).map(|s| (s.document, s.sample.text().to_owned())).collect()
    }

    #[test]
    fn an_empty_sentence_ends_the_corpus_as_in_opennlp_or_is_skipped_alone() {
        let corpus = corpus_with_an_empty_sentence();
        let opennlp = MascTokenReader::new(&corpus).unwrap();
        assert_eq!(texts(opennlp), [(0, "One two.".to_owned())]);

        let mut reader = MascTokenReader::with_empty_sentences(&corpus, EmptySentence::Skipped).unwrap();
        let mut read = Vec::new();
        while let Some(s) = reader.read().unwrap() {
            read.push((s.document, s.sample.text().to_owned()));
        }
        assert_eq!(
            read,
            [(0, "One two.".into()), (0, "Five six.".into()), (1, "One two.".into()), (1, "Five six.".into())]
        );
        assert_eq!(reader.skipped(), 2);
        reader.reset();
        assert_eq!(reader.skipped(), 0);
        assert_eq!(texts(reader).len(), 4);
    }
}
