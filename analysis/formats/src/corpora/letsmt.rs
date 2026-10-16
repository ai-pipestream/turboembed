//! LetsMT XML: `LetsmtDocument`, `LetsmtSentenceStream` and
//! `DetokenizeSentenceSampleStream`.

use std::borrow::Cow;

use turbo_analysis::Span;
use turbo_maxent::xml::{self, Attribute, Handler, Names};
use turbo_tokenize::detok::Detokenizer;
use turbo_unicode::WhitespaceMode;

use super::error::CorpusError;
use super::moses::SENTENCES_PER_SAMPLE;
use crate::error::FormatError;
use crate::sample::SentenceSample;
use crate::source::{Piece, check_len, java_trim, whitespace_tokens};

/// One sentence of a LetsMT document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LetsmtSentence {
    /// The words of the sentence, when it has `w` elements; the
    /// sentence's text is then the words joined by single spaces.
    pub tokens: Option<Vec<String>>,
    /// The sentence's text when it has no words.
    pub text: Option<String>,
    /// The bytes the sentence was read from: from the end of the previous
    /// sentence's end tag (or the start of the document) to the end of
    /// this one's.
    pub source: Span,
}

impl LetsmtSentence {
    /// The sentence as `LetsmtSentenceStream` writes it: the words joined
    /// by spaces, or the text.
    pub fn push_text(&self, out: &mut String) {
        match (&self.tokens, &self.text) {
            (Some(tokens), _) => {
                for (i, t) in tokens.iter().enumerate() {
                    if i > 0 {
                        out.push(' ');
                    }
                    out.push_str(t);
                }
            }
            (None, Some(text)) => out.push_str(text),
            (None, None) => {}
        }
    }
}

/// The sentences of a LetsMT document, as `LetsmtDocument.parse` reads
/// them with a SAX handler.
///
/// The handler keeps one buffer of character data, which only the end of
/// a `w` or `s` element (by qualified name) empties. So a word is all the
/// text since the last such end tag, trimmed with `String.trim`: text
/// between two words, or between `<s>` and the first word, is part of the
/// next word. At the end of an `s` element the sentence takes the words
/// read since the last sentence, or, if there are none, the buffer
/// trimmed, which holds the text of everything since the last end of a
/// `w` or `s`, such as a header before the first sentence. Words outside
/// any `s` element go to the next sentence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LetsmtDocument {
    pub sentences: Vec<LetsmtSentence>,
}

impl LetsmtDocument {
    /// Reads `text`, a decoded document (a byte order mark at its start is
    /// skipped).
    ///
    /// # Errors
    ///
    /// [`CorpusError::LetsmtXml`] when it is not well-formed XML. The JDK's
    /// parser is namespace aware and refuses a prefix that is not
    /// declared; this one does not check prefixes.
    pub fn parse(text: &str) -> Result<LetsmtDocument, FormatError> {
        check_len(text)?;
        let bom = if text.starts_with('\u{FEFF}') { 3 } else { 0 };
        let mut h = Sentences {
            sentences: Vec::new(),
            chars: String::new(),
            tokens: Vec::new(),
            from: bom,
            sentence_ended: false,
            bom,
        };
        xml::parse_with(&text[bom..], Names::Qualified, &mut h)
            .map_err(|e| FormatError::Corpus(CorpusError::LetsmtXml { detail: e.to_string() }))?;
        Ok(LetsmtDocument { sentences: h.sentences })
    }
}

struct Sentences {
    sentences: Vec<LetsmtSentence>,
    chars: String,
    tokens: Vec<String>,
    /// Where the bytes of the next sentence start: the end of the last
    /// sentence's end tag, or the start of the document.
    from: usize,
    /// Whether the element that just ended was a sentence, whose source
    /// ends with its end tag.
    sentence_ended: bool,
    /// The length of the byte order mark the parser was not given.
    bom: usize,
}

impl Handler for Sentences {
    fn start_element(&mut self, _: &str, _: &[Attribute<'_>], _: usize) -> turbo_maxent::Result<()> {
        Ok(())
    }

    fn end_element(&mut self, name: &str, _: usize) -> turbo_maxent::Result<()> {
        match name {
            "w" => {
                let token = trimmed(&self.chars).to_owned();
                self.tokens.push(token);
                self.chars.clear();
            }
            "s" => {
                let (tokens, text) = if self.tokens.is_empty() {
                    (None, Some(trimmed(&self.chars).to_owned()))
                } else {
                    (Some(std::mem::take(&mut self.tokens)), None)
                };
                // `element_closed` gives the end of the source
                self.sentences.push(LetsmtSentence {
                    tokens,
                    text,
                    source: Span::new(self.from as u32, self.from as u32),
                });
                self.chars.clear();
                self.sentence_ended = true;
            }
            _ => {}
        }
        Ok(())
    }

    fn element_closed(&mut self, end: usize) {
        if std::mem::take(&mut self.sentence_ended) {
            self.from = end + self.bom;
            let last = self.sentences.last_mut().expect("a sentence just ended");
            last.source = Span::new(last.source.start, self.from as u32);
        }
    }

    fn characters(&mut self, text: &str) {
        self.chars.push_str(text);
    }
}

/// `String.trim`.
fn trimmed(s: &str) -> &str {
    java_trim(Piece { text: s, span: Span::new(0, s.len() as u32) }).text
}

/// Reads sentence samples from a LetsMT document: `LetsmtSentenceStream`,
/// 25 sentences per sample, each followed by a space in the document,
/// and with a detokenizer `DetokenizeSentenceSampleStream` after it, which
/// splits each sentence with the whitespace tokenizer and joins the
/// tokens again with the detokenizer.
#[derive(Clone, Debug)]
pub struct LetsmtReader<'d> {
    sentences: std::vec::IntoIter<LetsmtSentence>,
    detokenizer: Option<(Detokenizer<'d>, WhitespaceMode)>,
}

impl<'d> LetsmtReader<'d> {
    /// The samples of `document`, detokenized with `detokenizer` when
    /// one is given, its tokens split by whitespace in `mode`.
    pub fn new(document: LetsmtDocument, detokenizer: Option<(Detokenizer<'d>, WhitespaceMode)>) -> Self {
        LetsmtReader { sentences: document.sentences.into_iter(), detokenizer }
    }
}

impl<'d> Iterator for LetsmtReader<'d> {
    type Item = Result<SentenceSample<'d>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut document = String::new();
        let mut spans = Vec::new();
        let mut sources = Vec::new();
        let mut sentence = String::new();
        for s in self.sentences.by_ref().take(SENTENCES_PER_SAMPLE) {
            let start = document.len() as u32;
            match &self.detokenizer {
                None => s.push_text(&mut document),
                Some((detokenizer, mode)) => {
                    sentence.clear();
                    s.push_text(&mut sentence);
                    let whole = Piece { text: &sentence, span: Span::new(0, sentence.len() as u32) };
                    let tokens: Vec<&str> = whitespace_tokens(whole, *mode).map(|p| p.text).collect();
                    detokenizer.detokenize(&tokens, None, &mut document);
                }
            }
            spans.push(Span::new(start, document.len() as u32));
            sources.push(s.source);
            document.push(' ');
        }
        let first = sources.first()?;
        let source = Span::new(first.start, sources.last().map_or(first.end, |s| s.end));
        Some(SentenceSample::new(Cow::Owned(document), spans, sources, source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_buffer_empties_only_at_the_end_of_a_word_or_sentence() {
        let src = "<d><h>Title</h>\n<s> Hello there </s><s><w>a</w> x <w>b </w>tail</s><w>c</w></d>";
        let d = LetsmtDocument::parse(src).unwrap();
        assert_eq!(d.sentences.len(), 2);
        assert_eq!(d.sentences[0].text.as_deref(), Some("Title\n Hello there"));
        assert_eq!(&src[d.sentences[0].source.range()], "<d><h>Title</h>\n<s> Hello there </s>");
        assert_eq!(d.sentences[1].tokens.as_deref(), Some(&["a".to_owned(), "x b".to_owned()][..]));
        assert_eq!(&src[d.sentences[1].source.range()], "<s><w>a</w> x <w>b </w>tail</s>");
        let samples: Vec<_> = LetsmtReader::new(d.clone(), None).map(Result::unwrap).collect();
        assert_eq!(samples[0].document(), "Title\n Hello there a x b ");
    }

    #[test]
    fn malformed_xml_is_refused() {
        let e = LetsmtDocument::parse("<d><s></d>").unwrap_err();
        assert_eq!(e.java_class(), Some("java.io.IOException"));
        assert_eq!(e.to_string(), "Failed to parse letsmt xml!");
    }
}
