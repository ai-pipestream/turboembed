//! Irish Sentence Bank XML corpus readers (`opennlp.tools.formats.irishsentencebank`).
//!
//! Port of `IrishSentenceBankDocument`, `IrishSentenceBankSentenceStream`,
//! `IrishSentenceBankTokenSampleStream`, and their stream factories from OpenNLP
//! format utilities.

use std::borrow::Cow;
use std::collections::HashMap;

use turbo_analysis::Span;

use crate::error::FormatError;
use crate::nkjp::{XmlElement, XmlNode, parse_xml_tree, parse_xml_tree_str};
use crate::sample::{SentenceSample, TokenSample};

/// Surface form and inflection/lemma list for one numbered slot in an
/// [`IrishSentenceBankSentence`].
///
/// Mirrors `opennlp.tools.formats.irishsentencebank.IrishSentenceBankDocument.IrishSentenceBankFlex`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrishSentenceBankFlex {
    surface: String,
    flex: Vec<String>,
}

impl IrishSentenceBankFlex {
    /// Creates a new flex entry with a surface token and its lemma/inflection strings.
    #[inline]
    pub fn new(surface: String, flex: Vec<String>) -> Self {
        Self { surface, flex }
    }

    /// Returns the surface form from `<token slot="...">`.
    #[inline]
    pub fn surface(&self) -> &str {
        &self.surface
    }

    /// Returns the lemma/inflection strings from `<flex slot="..." lemma="..."/>`.
    #[inline]
    pub fn flex(&self) -> &[String] {
        &self.flex
    }
}

/// One parsed `<sentence>` entry in an [`IrishSentenceBankDocument`].
///
/// Mirrors `opennlp.tools.formats.irishsentencebank.IrishSentenceBankDocument.IrishSentenceBankSentence`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrishSentenceBankSentence {
    source: String,
    translation: String,
    original: String,
    /// Token spans as UTF-8 byte offsets into `original` (for [`TokenSample`]).
    tokens: Vec<Span>,
    /// Token spans as UTF-16 code-unit offsets into `original` (matching Java's `Span[]`).
    utf16_tokens: Vec<Span>,
    flex: Option<Vec<IrishSentenceBankFlex>>,
}

impl IrishSentenceBankSentence {
    /// Returns the `source` attribute of `<sentence>`.
    #[inline]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Returns the `<translation>` text.
    #[inline]
    pub fn translation(&self) -> &str {
        &self.translation
    }

    /// Returns the reconstructed `<original>` sentence text.
    #[inline]
    pub fn original(&self) -> &str {
        &self.original
    }

    /// Returns the UTF-8 byte spans of the tokens in [`Self::original`].
    #[inline]
    pub fn tokens(&self) -> &[Span] {
        &self.tokens
    }

    /// Returns the UTF-16 code-unit spans of the tokens in [`Self::original`],
    /// matching Java's `IrishSentenceBankSentence.getTokens()`.
    #[inline]
    pub fn utf16_tokens(&self) -> &[Span] {
        &self.utf16_tokens
    }

    /// Returns the flex array if every token slot had at least one `<flex>` entry.
    #[inline]
    pub fn flex(&self) -> Option<&[IrishSentenceBankFlex]> {
        self.flex.as_deref()
    }

    /// Builds a [`TokenSample`] borrowing this sentence's `original` text.
    pub fn token_sample(&self) -> Result<TokenSample<'_>, FormatError> {
        let tokens = self.tokens.clone();
        let sources = tokens.clone();
        let source = Span::new(0, self.original.len() as u32);
        TokenSample::new(Cow::Borrowed(&self.original), tokens, sources, source)
    }

    /// Builds an owned [`TokenSample`].
    pub fn to_token_sample(&self) -> Result<TokenSample<'static>, FormatError> {
        let tokens = self.tokens.clone();
        let sources = tokens.clone();
        let source = Span::new(0, self.original.len() as u32);
        TokenSample::new(Cow::Owned(self.original.clone()), tokens, sources, source)
    }
}

/// Adjusts the start offset of a punctuation text node by skipping leading `' '` bytes.
///
/// Mirrors `IrishSentenceBankDocument.advanceLeft`.
fn advance_left_bytes(s: &str, start: u32) -> u32 {
    let mut ret = start;
    for b in s.bytes() {
        if b == b' ' {
            ret += 1;
        } else {
            return ret;
        }
    }
    ret
}

/// Adjusts the end offset of a punctuation text node by skipping trailing `' '` bytes
/// after index 0.
///
/// Mirrors `IrishSentenceBankDocument.advanceRight`.
fn advance_right_bytes(s: &str, start: u32) -> u32 {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return start;
    }
    let end = bytes.len() - 1;
    let mut ret = start + bytes.len() as u32;
    for i in (1..=end).rev() {
        if bytes[i] == b' ' {
            ret -= 1;
        } else {
            return ret;
        }
    }
    ret
}

fn utf16_len(s: &str) -> u32 {
    s.chars().map(|c| c.len_utf16() as u32).sum()
}

fn advance_left_utf16(s: &str, start: u32) -> u32 {
    let mut ret = start;
    for c in s.chars() {
        if c == ' ' {
            ret += 1;
        } else {
            return ret;
        }
    }
    ret
}

fn advance_right_utf16(s: &str, start: u32) -> u32 {
    let units: Vec<u16> = s.encode_utf16().collect();
    if units.is_empty() {
        return start;
    }
    let end = units.len() - 1;
    let mut ret = start + units.len() as u32;
    for i in (1..=end).rev() {
        if units[i] == u16::from(b' ') {
            ret -= 1;
        } else {
            return ret;
        }
    }
    ret
}

/// Parsed Irish Sentence Bank XML document (`<sentences>`).
///
/// Mirrors `opennlp.tools.formats.irishsentencebank.IrishSentenceBankDocument`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrishSentenceBankDocument {
    sentences: Vec<IrishSentenceBankSentence>,
}

impl IrishSentenceBankDocument {
    /// Parses an Irish Sentence Bank XML document from raw bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        let root = parse_xml_tree(bytes, "IrishSentenceBank")?;
        Self::from_root(&root)
    }

    /// Parses an Irish Sentence Bank XML document from decoded UTF-8 text.
    pub fn parse_str(text: &str) -> Result<Self, FormatError> {
        let root = parse_xml_tree_str(text, "IrishSentenceBank")?;
        Self::from_root(&root)
    }

    fn from_root(root: &XmlElement) -> Result<Self, FormatError> {
        if !root.name.eq_ignore_ascii_case("sentences") {
            return Err(FormatError::Xml {
                format: "IrishSentenceBank",
                message: format!("Expected root node {}", root.name),
            });
        }

        let mut sentences = Vec::new();
        for node in &root.children {
            let XmlNode::Element(sent_elem) = node else {
                continue;
            };
            if sent_elem.name != "sentence" {
                return Err(FormatError::Xml {
                    format: "IrishSentenceBank",
                    message: format!("Unexpected node: {}", sent_elem.name),
                });
            }

            let src = sent_elem
                .attr("source")
                .ok_or_else(|| FormatError::Xml {
                    format: "IrishSentenceBank",
                    message: "missing source attribute on <sentence>".to_owned(),
                })?
                .to_owned();
            let mut trans = String::new();
            let mut toks: HashMap<usize, String> = HashMap::new();
            let mut flx: HashMap<usize, Vec<String>> = HashMap::new();
            let mut spans: Vec<Span> = Vec::new();
            let mut utf16_spans: Vec<Span> = Vec::new();
            let mut flexes: usize = 1;
            let mut orig = String::new();

            for child in &sent_elem.children {
                let XmlNode::Element(child_elem) = child else {
                    continue;
                };
                match child_elem.name.as_str() {
                    "flex" => {
                        let slot_str = child_elem.attr("slot").ok_or_else(|| FormatError::Xml {
                            format: "IrishSentenceBank",
                            message: "missing slot attribute on <flex>".to_owned(),
                        })?;
                        let flexslot = slot_str.parse::<usize>().map_err(|_| FormatError::Xml {
                            format: "IrishSentenceBank",
                            message: format!("invalid slot number on <flex>: {slot_str}"),
                        })?;
                        if flexslot == 0 {
                            return Err(FormatError::Xml {
                                format: "IrishSentenceBank",
                                message: "slot index must be >= 1".to_owned(),
                            });
                        }
                        if flexslot > flexes {
                            flexes = flexslot;
                        }
                        let lemma = child_elem
                            .attr("lemma")
                            .ok_or_else(|| FormatError::Xml {
                                format: "IrishSentenceBank",
                                message: "missing lemma attribute on <flex>".to_owned(),
                            })?
                            .to_owned();
                        flx.entry(flexslot).or_default().push(lemma);
                    }
                    "translation" => {
                        trans = child_elem.text_content();
                    }
                    "original" => {
                        let mut last_byte = 0u32;
                        let mut last_utf16 = 0u32;
                        for orig_child in &child_elem.children {
                            match orig_child {
                                XmlNode::Element(tok_elem) if tok_elem.name == "token" => {
                                    let tmptok = tok_elem.text_content();
                                    let tok_byte_len = tmptok.len() as u32;
                                    let tok_utf16_len = utf16_len(&tmptok);
                                    spans.push(Span::new(last_byte, last_byte + tok_byte_len));
                                    utf16_spans.push(Span::new(last_utf16, last_utf16 + tok_utf16_len));

                                    let slot_str = tok_elem.attr("slot").ok_or_else(|| FormatError::Xml {
                                        format: "IrishSentenceBank",
                                        message: "missing slot attribute on <token>".to_owned(),
                                    })?;
                                    let tokslot = slot_str.parse::<usize>().map_err(|_| FormatError::Xml {
                                        format: "IrishSentenceBank",
                                        message: format!("invalid slot number on <token>: {slot_str}"),
                                    })?;
                                    if tokslot == 0 {
                                        return Err(FormatError::Xml {
                                            format: "IrishSentenceBank",
                                            message: "slot index must be >= 1".to_owned(),
                                        });
                                    }
                                    if tokslot > flexes {
                                        flexes = tokslot;
                                    }
                                    toks.insert(tokslot, tmptok.clone());
                                    orig.push_str(&tmptok);
                                    last_byte += tok_byte_len;
                                    last_utf16 += tok_utf16_len;
                                }
                                XmlNode::Element(other) => {
                                    return Err(FormatError::Xml {
                                        format: "IrishSentenceBank",
                                        message: format!("Unexpected node: {}", other.name),
                                    });
                                }
                                XmlNode::Text(tmptxt) => {
                                    orig.push_str(tmptxt);
                                    let txt_byte_len = tmptxt.len() as u32;
                                    let txt_utf16_len = utf16_len(tmptxt);
                                    if tmptxt != " " {
                                        spans.push(Span::new(
                                            advance_left_bytes(tmptxt, last_byte),
                                            advance_right_bytes(tmptxt, last_byte),
                                        ));
                                        utf16_spans.push(Span::new(
                                            advance_left_utf16(tmptxt, last_utf16),
                                            advance_right_utf16(tmptxt, last_utf16),
                                        ));
                                    }
                                    last_byte += txt_byte_len;
                                    last_utf16 += txt_utf16_len;
                                }
                            }
                        }
                    }
                    other => {
                        return Err(FormatError::Xml {
                            format: "IrishSentenceBank",
                            message: format!("Unexpected node: {other}"),
                        });
                    }
                }
            }

            let mut flexa: Option<Vec<IrishSentenceBankFlex>> = if toks.is_empty() {
                Some(Vec::new())
            } else {
                let mut vec = vec![IrishSentenceBankFlex::new(String::new(), Vec::new()); flexes];
                let mut complete = true;
                for (&flexidx, left) in &toks {
                    let Some(right) = flx.get(&flexidx) else {
                        complete = false;
                        break;
                    };
                    vec[flexidx - 1] = IrishSentenceBankFlex::new(left.clone(), right.clone());
                }
                if complete { Some(vec) } else { None }
            };
            if toks.is_empty() && flx.is_empty() {
                flexa = Some(Vec::new());
            }

            sentences.push(IrishSentenceBankSentence {
                source: src,
                translation: trans,
                original: orig,
                tokens: spans,
                utf16_tokens: utf16_spans,
                flex: flexa,
            });
        }

        Ok(Self { sentences })
    }

    /// Returns the parsed sentences in document order.
    #[inline]
    pub fn sentences(&self) -> &[IrishSentenceBankSentence] {
        &self.sentences
    }

    /// Consumes the document and returns the parsed sentences.
    #[inline]
    pub fn into_sentences(self) -> Vec<IrishSentenceBankSentence> {
        self.sentences
    }
}

/// Stream of [`SentenceSample`]s produced from an [`IrishSentenceBankDocument`].
///
/// Mirrors `opennlp.tools.formats.irishsentencebank.IrishSentenceBankSentenceStream`:
/// emits a single [`SentenceSample`] joining all sentences in the document with `' '`
/// (including a trailing `' '`).
#[derive(Debug, Clone)]
pub struct IrishSentenceBankSentenceReader {
    sentences: Vec<IrishSentenceBankSentence>,
    emitted: bool,
}

impl IrishSentenceBankSentenceReader {
    /// Creates a sentence reader from an already-parsed [`IrishSentenceBankDocument`].
    #[inline]
    pub fn from_document(doc: IrishSentenceBankDocument) -> Self {
        Self { sentences: doc.sentences, emitted: false }
    }

    /// Parses an Irish Sentence Bank XML document from bytes and returns a sentence reader.
    pub fn from_xml(bytes: &[u8]) -> Result<Self, FormatError> {
        Ok(Self::from_document(IrishSentenceBankDocument::parse(bytes)?))
    }

    /// Parses an Irish Sentence Bank XML document from `&str` and returns a sentence reader.
    pub fn from_xml_str(text: &str) -> Result<Self, FormatError> {
        Ok(Self::from_document(IrishSentenceBankDocument::parse_str(text)?))
    }

    /// Reads the document's [`SentenceSample`], or `None` once already emitted.
    pub fn next_sample(&mut self) -> Option<Result<SentenceSample<'static>, FormatError>> {
        if self.emitted || self.sentences.is_empty() {
            return None;
        }
        self.emitted = true;

        let mut sentence_string = String::new();
        let mut sentence_spans = Vec::with_capacity(self.sentences.len());

        for sent in &self.sentences {
            let begin = sentence_string.len() as u32;
            sentence_string.push_str(&sent.original);
            let end = sentence_string.len() as u32;
            sentence_string.push(' ');
            sentence_spans.push(Span::new(begin, end));
        }

        let sources = sentence_spans.clone();
        let source = Span::new(0, sentence_string.len() as u32);
        Some(SentenceSample::new(Cow::Owned(sentence_string), sentence_spans, sources, source))
    }
}

impl Iterator for IrishSentenceBankSentenceReader {
    type Item = Result<SentenceSample<'static>, FormatError>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_sample()
    }
}

/// Stream of [`TokenSample`]s produced from an [`IrishSentenceBankDocument`].
///
/// Mirrors `opennlp.tools.formats.irishsentencebank.IrishSentenceBankTokenSampleStream`:
/// emits one [`TokenSample`] per sentence in the document.
#[derive(Debug, Clone)]
pub struct IrishSentenceBankTokenReader {
    sentences: Vec<IrishSentenceBankSentence>,
    index: usize,
}

impl IrishSentenceBankTokenReader {
    /// Creates a token sample reader from an already-parsed [`IrishSentenceBankDocument`].
    #[inline]
    pub fn from_document(doc: IrishSentenceBankDocument) -> Self {
        Self { sentences: doc.sentences, index: 0 }
    }

    /// Parses an Irish Sentence Bank XML document from bytes and returns a token reader.
    pub fn from_xml(bytes: &[u8]) -> Result<Self, FormatError> {
        Ok(Self::from_document(IrishSentenceBankDocument::parse(bytes)?))
    }

    /// Parses an Irish Sentence Bank XML document from `&str` and returns a token reader.
    pub fn from_xml_str(text: &str) -> Result<Self, FormatError> {
        Ok(Self::from_document(IrishSentenceBankDocument::parse_str(text)?))
    }

    /// Reads the next sentence's [`TokenSample`], or `None` when exhausted.
    pub fn next_sample(&mut self) -> Option<Result<TokenSample<'static>, FormatError>> {
        let sent = self.sentences.get(self.index)?;
        self.index += 1;
        Some(sent.to_token_sample())
    }
}

impl Iterator for IrishSentenceBankTokenReader {
    type Item = Result<TokenSample<'static>, FormatError>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IRISH_SB_SAMPLE: &[u8] = br#"<sentences datestamp="2015-03-10">
<sentence source='potaL'>
	<original xml:space="preserve"><token slot='1'>A</token> <token slot='2'>Dhia</token>, <token slot='3'>t&#225;</token> <token slot='4'>m&#233;</token> <token slot='5'>ag</token> <token slot='6'>iompar</token> <token slot='7'>clainne</token>!</original>
	<translation>Oh my God, I&apos;m pregnant!</translation>
	<flex slot='1' lemma='a'/>
	<flex slot='2' lemma='dia'/>
	<flex slot='3' lemma='b&#237;'/>
	<flex slot='4' lemma='m&#233;'/>
	<flex slot='5' lemma='ag'/>
	<flex slot='6' lemma='iompair'/>
	<flex slot='7' lemma='clann'/>
</sentence>
<sentence source='potaL'>
	<original xml:space="preserve"><token slot='1'>Gabh</token> <token slot='2'>mo</token> <token slot='3'>leithsc&#233;al</token>, <token slot='4'>an</token> <token slot='5'>&#243;n</token> <token slot='6'>chlochaois</token> <token slot='7'>th&#250;</token>?</original>
	<translation>Excuse me, are you from the stone age?</translation>
	<flex slot='1' lemma='gabh'/>
	<flex slot='2' lemma='mo'/>
	<flex slot='3' lemma='leithsc&#233;al'/>
	<flex slot='4' lemma='an'/>
	<flex slot='5' lemma='&#243;'/>
	<flex slot='5' lemma='an'/>
	<flex slot='6' lemma='clochaois'/>
	<flex slot='7' lemma='th&#250;'/>
</sentence>
</sentences>"#;

    #[test]
    fn parse_irish_sentence_bank_matches_opennlp() {
        let doc = IrishSentenceBankDocument::parse(IRISH_SB_SAMPLE).unwrap();
        let sents = doc.sentences();
        assert_eq!(sents.len(), 2);

        let sent1 = &sents[0];
        let sent2 = &sents[1];

        assert_eq!(sent1.original(), "A Dhia, tá mé ag iompar clainne!");

        let flex1 = sent1.flex().unwrap();
        assert_eq!(flex1.len(), 7);
        assert_eq!(flex1[0].surface(), "A");
        assert_eq!(flex1[0].flex(), &["a"]);

        let flex2 = sent2.flex().unwrap();
        assert_eq!(flex2[4].surface(), "ón");
        assert_eq!(flex2[4].flex(), &["ó", "an"]);
        assert_eq!(sent2.translation(), "Excuse me, are you from the stone age?");

        let ts = sent1.token_sample().unwrap();
        assert_eq!(ts.tokens().len(), 9);
        assert_eq!(ts.token(7), "clainne");
        // Verify UTF-16 span matches Java's Span(24, 31).
        assert_eq!(sent1.utf16_tokens()[7], Span::new(24, 31));
    }

    #[test]
    fn irish_sentence_bank_sentence_and_token_streams() {
        let mut sent_reader = IrishSentenceBankSentenceReader::from_xml(IRISH_SB_SAMPLE).unwrap();
        let sent_sample = sent_reader.next().expect("one sample").unwrap();
        assert!(sent_reader.next().is_none());
        assert_eq!(sent_sample.sentences().len(), 2);
        assert_eq!(sent_sample.sentence(0), "A Dhia, tá mé ag iompar clainne!");
        assert_eq!(sent_sample.sentence(1), "Gabh mo leithscéal, an ón chlochaois thú?");

        let tok_samples: Vec<TokenSample<'static>> =
            IrishSentenceBankTokenReader::from_xml(IRISH_SB_SAMPLE).unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(tok_samples.len(), 2);
        assert_eq!(tok_samples[0].tokens().len(), 9);
        assert_eq!(tok_samples[1].tokens().len(), 9);
    }
}
