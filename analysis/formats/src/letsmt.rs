//! LetsMT XML sentence reader and detokenizing sentence stream
//! (`opennlp.tools.formats.letsmt`).
//!
//! Port of `LetsmtDocument`, `LetsmtSentenceStream`, `LetsmtSentenceStreamFactory`,
//! and `DetokenizeSentenceSampleStream` from OpenNLP format utilities.

use std::borrow::Cow;

use turbo_analysis::Span;
use turbo_maxent::xml::{self, Attribute, Handler, Names};
use turbo_tokenize::{Detokenizer, TokenizerOptions, WhitespaceMode, whitespace_spans};

use crate::error::FormatError;
use crate::sample::SentenceSample;

/// Number of sentences grouped into each [`SentenceSample`] by [`LetsmtSentenceReader`].
pub const SENTENCES_PER_SAMPLE: usize = 25;

/// Java's `String.trim()` on a `&str` (strips leading and trailing characters `<= '\u{0020}'`).
#[inline]
pub(crate) fn trim_java_str(s: &str) -> &str {
    s.trim_matches(|c: char| c <= ' ')
}

/// A single `<s ...>` entry in a [`LetsmtDocument`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LetsmtSentence {
    /// Non-tokenized sentence text when the `<s>` element contained no `<w>` children.
    pub non_tokenized_text: Option<String>,
    /// Token strings when the `<s>` element contained `<w>` children.
    pub tokens: Vec<String>,
}

impl LetsmtSentence {
    /// Returns the non-tokenized text if this sentence had no `<w>` elements.
    #[inline]
    pub fn non_tokenized_text(&self) -> Option<&str> {
        self.non_tokenized_text.as_deref()
    }

    /// Returns the token slice for this sentence.
    #[inline]
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }
}

/// Parsed LetsMT XML corpus document (`<letsmt>`).
///
/// Mirrors `opennlp.tools.formats.letsmt.LetsmtDocument`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LetsmtDocument {
    sentences: Vec<LetsmtSentence>,
}

impl LetsmtDocument {
    /// Parses a LetsMT XML document from raw bytes.
    ///
    /// Rejects `DOCTYPE` declarations (`disallow-doctype-decl`), matching OpenNLP's
    /// `XmlUtil.createSaxParser()`.
    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        let text = xml::decode(bytes).map_err(|err| FormatError::Xml { format: "LetsMT", message: err.to_string() })?;
        Self::parse_str(&text)
    }

    /// Parses a LetsMT XML document from decoded UTF-8 text.
    pub fn parse_str(text: &str) -> Result<Self, FormatError> {
        let mut handler = LetsmtHandler::default();
        xml::parse_with(text, Names::Qualified, &mut handler)
            .map_err(|err| FormatError::Xml { format: "LetsMT", message: err.to_string() })?;
        Ok(Self { sentences: handler.sentences })
    }

    /// Returns the parsed sentences in document order.
    #[inline]
    pub fn sentences(&self) -> &[LetsmtSentence] {
        &self.sentences
    }

    /// Consumes the document and returns the parsed sentences.
    #[inline]
    pub fn into_sentences(self) -> Vec<LetsmtSentence> {
        self.sentences
    }
}

#[derive(Default)]
struct LetsmtHandler {
    chars: String,
    tokens: Vec<String>,
    sentences: Vec<LetsmtSentence>,
}

impl Handler for LetsmtHandler {
    fn start_element(
        &mut self,
        _name: &str,
        _attributes: &[Attribute<'_>],
        _offset: usize,
    ) -> turbo_maxent::Result<()> {
        Ok(())
    }

    fn characters(&mut self, text: &str) {
        self.chars.push_str(text);
    }

    fn end_element(&mut self, name: &str, _offset: usize) -> turbo_maxent::Result<()> {
        match name {
            "w" => {
                self.tokens.push(trim_java_str(&self.chars).to_owned());
                self.chars.clear();
            }
            "s" => {
                if !self.tokens.is_empty() {
                    self.sentences
                        .push(LetsmtSentence { non_tokenized_text: None, tokens: std::mem::take(&mut self.tokens) });
                } else {
                    self.sentences.push(LetsmtSentence {
                        non_tokenized_text: Some(trim_java_str(&self.chars).to_owned()),
                        tokens: Vec::new(),
                    });
                }
                self.chars.clear();
            }
            _ => {}
        }
        Ok(())
    }
}

/// Stream of [`SentenceSample`]s produced from a [`LetsmtDocument`].
///
/// Mirrors `opennlp.tools.formats.letsmt.LetsmtSentenceStream`: groups up to 25
/// sentences per [`SentenceSample`], joining `<w>` tokens with single spaces (or
/// appending `non_tokenized_text`) and appending a trailing `' '` after every sentence.
#[derive(Debug, Clone)]
pub struct LetsmtSentenceReader {
    sentences: Vec<LetsmtSentence>,
    index: usize,
}

impl LetsmtSentenceReader {
    /// Creates a sentence reader from an already-parsed [`LetsmtDocument`].
    #[inline]
    pub fn from_document(doc: LetsmtDocument) -> Self {
        Self { sentences: doc.sentences, index: 0 }
    }

    /// Parses a LetsMT XML document from bytes and returns a sentence reader over it.
    pub fn from_xml(bytes: &[u8]) -> Result<Self, FormatError> {
        Ok(Self::from_document(LetsmtDocument::parse(bytes)?))
    }

    /// Parses a LetsMT XML document from `&str` and returns a sentence reader over it.
    pub fn from_xml_str(text: &str) -> Result<Self, FormatError> {
        Ok(Self::from_document(LetsmtDocument::parse_str(text)?))
    }

    /// Reads the next [`SentenceSample`] (up to 25 sentences), or `None` when exhausted.
    pub fn next_sample(&mut self) -> Option<Result<SentenceSample<'static>, FormatError>> {
        if self.index >= self.sentences.len() {
            return None;
        }

        let mut sentences_string = String::new();
        let mut sentence_spans = Vec::new();
        let mut count = 0;

        while count < SENTENCES_PER_SAMPLE && self.index < self.sentences.len() {
            let sentence = &self.sentences[self.index];
            self.index += 1;
            count += 1;

            let begin = sentences_string.len() as u32;
            if let Some(non_tok) = &sentence.non_tokenized_text {
                sentences_string.push_str(non_tok);
            } else if !sentence.tokens.is_empty() {
                for (i, tok) in sentence.tokens.iter().enumerate() {
                    if i > 0 {
                        sentences_string.push(' ');
                    }
                    sentences_string.push_str(tok);
                }
            }
            let end = sentences_string.len() as u32;
            sentence_spans.push(Span::new(begin, end));
            sentences_string.push(' ');
        }

        if sentence_spans.is_empty() {
            None
        } else {
            let sources = sentence_spans.clone();
            let source = Span::new(0, sentences_string.len() as u32);
            Some(SentenceSample::new(Cow::Owned(sentences_string), sentence_spans, sources, source))
        }
    }
}

impl Iterator for LetsmtSentenceReader {
    type Item = Result<SentenceSample<'static>, FormatError>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_sample()
    }
}

/// Adapter stream that detokenizes each sentence in a [`SentenceSample`] stream
/// using a [`Detokenizer`].
///
/// Mirrors `opennlp.tools.formats.DetokenizeSentenceSampleStream`: each sentence
/// span is split with `WhitespaceTokenizer` (`WhitespaceMode::Legacy` by default)
/// and rebuilt with `detokenizer.detokenize(tokens, null)`, followed by a trailing `' '`.
#[derive(Debug, Clone)]
pub struct DetokenizeSentenceReader<'d, I> {
    samples: I,
    detokenizer: Detokenizer<'d>,
    whitespace_mode: WhitespaceMode,
}

impl<'d, 'a, I> DetokenizeSentenceReader<'d, I>
where
    I: Iterator<Item = Result<SentenceSample<'a>, FormatError>>,
{
    /// Wraps a [`SentenceSample`] iterator with `detokenizer` using legacy Java whitespace.
    #[inline]
    pub fn new(samples: I, detokenizer: Detokenizer<'d>) -> Self {
        Self { samples, detokenizer, whitespace_mode: WhitespaceMode::Legacy }
    }

    /// Sets the whitespace mode used to split pre-tokenized sentences before detokenizing.
    #[inline]
    pub fn with_whitespace_mode(mut self, mode: WhitespaceMode) -> Self {
        self.whitespace_mode = mode;
        self
    }

    /// Reads and detokenizes the next [`SentenceSample`].
    pub fn next_sample(&mut self) -> Option<Result<SentenceSample<'a>, FormatError>> {
        let sample = match self.samples.next()? {
            Ok(s) => s,
            Err(e) => return Some(Err(e)),
        };
        let doc_text = sample.document();
        let mut sentence_string = String::with_capacity(doc_text.len());
        let mut new_spans = Vec::with_capacity(sample.sentences().len());
        let mut token_spans = Vec::new();
        let mut token_slices = Vec::new();
        let options = TokenizerOptions { whitespace: self.whitespace_mode, keep_newlines: false };

        for span in sample.sentences() {
            let sentence = &doc_text[span.range()];
            token_spans.clear();
            whitespace_spans(sentence, options, &mut token_spans);
            token_slices.clear();
            token_slices.extend(token_spans.iter().map(|s| &sentence[s.range()]));

            let begin = sentence_string.len() as u32;
            self.detokenizer.detokenize(&token_slices, None, &mut sentence_string);
            let end = sentence_string.len() as u32;
            sentence_string.push(' ');
            new_spans.push(Span::new(begin, end));
        }

        let sources = sample.sources().to_vec();
        let source = sample.source();
        Some(SentenceSample::new(Cow::Owned(sentence_string), new_spans, sources, source))
    }
}

impl<'d, 'a, I> Iterator for DetokenizeSentenceReader<'d, I>
where
    I: Iterator<Item = Result<SentenceSample<'a>, FormatError>>,
{
    type Item = Result<SentenceSample<'a>, FormatError>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_tokenize::{DetokenizerDictionary, DictionaryOperation};

    const LETSMT_WITH_WORDS: &[u8] = br#"<?xml version="1.0" encoding="utf-8"?>
<letsmt version="1.0">
    <head></head>
    <body>
        <s id="1">
            <w>The</w>
            <w>Apache</w>
            <w>Software</w>
            <w>Foundation</w>
            <w>uses</w>
            <w>various</w>
            <w>licenses</w>
            <w>to</w>
            <w>distribute</w>
            <w>software</w>
            <w>and</w>
            <w>documentation</w>
            <w>,</w>
            <w>to</w>
            <w>accept</w>
            <w>regular</w>
            <w>contributions</w>
            <w>from</w>
            <w>individuals</w>
            <w>and</w>
            <w>corporations</w>
            <w>,</w>
            <w>and</w>
            <w>to</w>
            <w>accept</w>
            <w>larger</w>
            <w>grants</w>
            <w>of</w>
            <w>existing</w>
            <w>software</w>
            <w>products</w>
            <w>.</w>
        </s>
        <s id="2">
            <w>All</w>
            <w>software</w>
            <w>produced</w>
            <w>by</w>
            <w>The</w>
            <w>Apache</w>
            <w>Software</w>
            <w>Foundation</w>
            <w>or</w>
            <w>any</w>
            <w>of</w>
            <w>its</w>
            <w>projects</w>
            <w>or</w>
            <w>subjects</w>
            <w>is</w>
            <w>licensed</w>
            <w>according</w>
            <w>to</w>
            <w>the</w>
            <w>terms</w>
            <w>of</w>
            <w>the</w>
            <w>documents</w>
            <w>listed</w>
            <w>below</w>
            <w>.</w>
        </s>
    </body>
</letsmt>"#;

    #[test]
    fn parse_letsmt_with_words_matches_opennlp() {
        let doc = LetsmtDocument::parse(LETSMT_WITH_WORDS).unwrap();
        let sents = doc.sentences();
        assert_eq!(sents.len(), 2);
        assert!(sents[0].non_tokenized_text().is_none());
        assert_eq!(
            sents[0].tokens(),
            &[
                "The",
                "Apache",
                "Software",
                "Foundation",
                "uses",
                "various",
                "licenses",
                "to",
                "distribute",
                "software",
                "and",
                "documentation",
                ",",
                "to",
                "accept",
                "regular",
                "contributions",
                "from",
                "individuals",
                "and",
                "corporations",
                ",",
                "and",
                "to",
                "accept",
                "larger",
                "grants",
                "of",
                "existing",
                "software",
                "products",
                "."
            ]
        );
        assert!(sents[1].non_tokenized_text().is_none());
        assert_eq!(
            sents[1].tokens(),
            &[
                "All",
                "software",
                "produced",
                "by",
                "The",
                "Apache",
                "Software",
                "Foundation",
                "or",
                "any",
                "of",
                "its",
                "projects",
                "or",
                "subjects",
                "is",
                "licensed",
                "according",
                "to",
                "the",
                "terms",
                "of",
                "the",
                "documents",
                "listed",
                "below",
                "."
            ]
        );
    }

    #[test]
    fn letsmt_sentence_reader_and_detokenizer() {
        let reader = LetsmtSentenceReader::from_xml(LETSMT_WITH_WORDS).unwrap();
        let mut dict = DetokenizerDictionary::new();
        dict.insert(",", DictionaryOperation::MoveLeft);
        dict.insert(".", DictionaryOperation::MoveLeft);
        let detok = Detokenizer::new(&dict);
        let mut detok_reader = DetokenizeSentenceReader::new(reader, detok);
        let sample = detok_reader.next().expect("one sample").unwrap();
        assert!(detok_reader.next().is_none());
        assert_eq!(sample.sentences().len(), 2);
        assert_eq!(
            sample.sentence(0),
            "The Apache Software Foundation uses various licenses to distribute software and documentation, to accept regular contributions from individuals and corporations, and to accept larger grants of existing software products."
        );
        assert_eq!(
            sample.sentence(1),
            "All software produced by The Apache Software Foundation or any of its projects or subjects is licensed according to the terms of the documents listed below."
        );
    }

    #[test]
    fn letsmt_non_tokenized_sentence_and_doctype_rejected() {
        let xml = br#"<letsmt version="1.0"><body><s id="1">  Plain sentence without words.  </s></body></letsmt>"#;
        let mut reader = LetsmtSentenceReader::from_xml(xml).unwrap();
        let sample = reader.next().unwrap().unwrap();
        assert_eq!(sample.sentences().len(), 1);
        assert_eq!(sample.sentence(0), "Plain sentence without words.");

        let xxe = br#"<!DOCTYPE letsmt [<!ENTITY xxe "boom">]><letsmt/>"#;
        assert!(LetsmtDocument::parse(xxe).is_err());
    }
}
