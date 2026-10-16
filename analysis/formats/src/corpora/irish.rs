//! The Irish Sentence Bank: `IrishSentenceBankDocument`,
//! `IrishSentenceBankSentenceStream` and
//! `IrishSentenceBankTokenSampleStream`.
//!
//! A document is a `sentences` element of `sentence` elements, each with
//! an `original` (the sentence as `token` elements and the text between
//! them), a `translation` and `flex` elements that give the lemmas of a
//! token slot.

use std::borrow::Cow;

use turbo_analysis::Span;
use turbo_jvm::chars::utf16_len;
use turbo_jvm::hash_map::JavaHashMap;

use super::dom::{self, Element, Node};
use super::error::CorpusError;
use super::nkjp::shift;
use crate::error::FormatError;
use crate::masc::java::{equals_ignore_case_ascii, parse_int};
use crate::sample::{SentenceSample, TokenSample};
use crate::source::{check_len, utf16_cut};

/// The lemmas of a token slot, with the token's text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrishFlex {
    pub surface: String,
    pub lemmas: Vec<String>,
}

/// One sentence of the bank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrishSentence {
    /// The `source` attribute of the sentence.
    pub source_name: String,
    /// The text content of the first node of the last `translation`, or
    /// empty.
    pub translation: String,
    /// The text of the sentence: each `token`'s text content and the text
    /// between the tokens, of every `original` element in order.
    pub original: String,
    /// The tokens' spans in [`IrishSentence::original`]. A token element
    /// is a token, and so is the text between two tokens unless it is
    /// exactly one space; that text is cut at its first and last character
    /// other than a space (U+0020 only), so a line break in it is part of
    /// a token. A second `original` element starts its offsets again from
    /// zero, so its spans point into the text of the first.
    pub tokens: Vec<Span>,
    /// For each token, the bytes of its `token` element or of its text.
    pub token_sources: Vec<Span>,
    /// The flex of each token slot, by slot minus one, for the slots that
    /// have a token, with the number of slots (the largest slot, at least
    /// one); `None` when a token's slot has no `flex` element, as Java
    /// gives up on the whole array there.
    pub flex: Option<(Vec<(usize, IrishFlex)>, usize)>,
    /// The bytes of the `sentence` element.
    pub source: Span,
}

/// The sentences of an Irish Sentence Bank document,
/// `IrishSentenceBankDocument.parse`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IrishSentenceBank {
    pub sentences: Vec<IrishSentence>,
}

impl IrishSentenceBank {
    /// Reads a decoded document (a byte order mark at its start is
    /// skipped).
    ///
    /// # Errors
    ///
    /// What [`CorpusError`] lists for the Irish Sentence Bank, where
    /// OpenNLP throws. Java allocates the flex array of the largest slot,
    /// so a slot near `Integer.MAX_VALUE` runs it out of memory; this port
    /// keeps the flexes in a list and reads such a sentence.
    pub fn parse(text: &str) -> Result<IrishSentenceBank, FormatError> {
        check_len(text)?;
        let bom = if text.starts_with('\u{FEFF}') { 3 } else { 0 };
        let mut root = dom::parse(&text[bom..])
            .map_err(|e| FormatError::Corpus(CorpusError::IrishXml { detail: e.to_string() }))?;
        if bom > 0 {
            shift(&mut root, bom as u32);
        }
        if !equals_ignore_case_ascii(&root.name, "sentences") {
            return Err(corpus(CorpusError::IrishRoot { name: root.name }));
        }
        let mut sentences = Vec::new();
        for node in &root.children {
            match node.name() {
                "sentence" => sentences.push(sentence(attributes(node)?, node.span())?),
                "#text" | "#comment" => {}
                other => return Err(unexpected(other)),
            }
        }
        Ok(IrishSentenceBank { sentences })
    }

    /// The sentence sample of the document: every sentence's original,
    /// each followed by a space. There is none when the document has no
    /// sentence.
    pub fn sentence_sample<'a>(&self) -> Option<Result<SentenceSample<'a>, FormatError>> {
        let first = self.sentences.first()?;
        let last = self.sentences.last()?;
        let mut document = String::new();
        let mut spans = Vec::with_capacity(self.sentences.len());
        let mut sources = Vec::with_capacity(self.sentences.len());
        for s in &self.sentences {
            let start = document.len() as u32;
            document.push_str(&s.original);
            spans.push(Span::new(start, document.len() as u32));
            sources.push(s.source);
            document.push(' ');
        }
        let source = Span::new(first.source.start, last.source.end);
        Some(SentenceSample::new(Cow::Owned(document), spans, sources, source))
    }

    /// The token samples of the document, one per sentence.
    pub fn token_samples(&self) -> impl Iterator<Item = Result<TokenSample<'_>, FormatError>> + '_ {
        self.sentences.iter().map(|s| {
            TokenSample::new(Cow::Borrowed(s.original.as_str()), s.tokens.clone(), s.token_sources.clone(), s.source)
        })
    }

    /// [`IrishSentenceBank::token_samples`], the samples owning their
    /// text.
    pub fn into_token_samples<'a>(self) -> impl Iterator<Item = Result<TokenSample<'a>, FormatError>> {
        self.sentences
            .into_iter()
            .map(|s| TokenSample::new(Cow::Owned(s.original), s.tokens, s.token_sources, s.source))
    }
}

fn corpus(e: CorpusError) -> FormatError {
    FormatError::Corpus(e)
}

fn unexpected(name: &str) -> FormatError {
    corpus(CorpusError::IrishUnexpectedNode { name: name.to_owned() })
}

/// `node.getAttributes()`, which is null for a processing instruction
/// that has the name the reader looks for.
fn attributes(node: &Node) -> Result<&Element, FormatError> {
    node.as_element().ok_or(corpus(CorpusError::IrishNoAttributes))
}

/// `getAttributes().getNamedItem(name).getNodeValue()`.
fn attribute<'e>(e: &'e Element, name: &str) -> Result<&'e str, FormatError> {
    e.attribute(name).ok_or(corpus(CorpusError::IrishMissingAttribute))
}

/// `Integer.parseInt` of a slot.
fn slot(e: &Element) -> Result<i32, FormatError> {
    let s = attribute(e, "slot")?;
    parse_int(s).ok_or_else(|| corpus(CorpusError::NotANumber { text: s.into() }))
}

/// `getFirstChild().getTextContent()`.
fn first_child_text(node: &Node) -> Result<String, FormatError> {
    let first = match node {
        Node::Element(e) => e.children.first(),
        _ => None,
    };
    let first = first.ok_or(corpus(CorpusError::IrishNoContent))?;
    let mut out = String::new();
    first.push_text_content(&mut out);
    Ok(out)
}

/// One `sentence` element.
fn sentence(e: &Element, span: Span) -> Result<IrishSentence, FormatError> {
    let source_name = attribute(e, "source")?.to_owned();
    let mut translation = String::new();
    let mut tokens_by_slot: JavaHashMap<i32, String> = JavaHashMap::new();
    let mut flex_by_slot: std::collections::HashMap<i32, Vec<String>> = std::collections::HashMap::new();
    // token spans in UTF-16 code units, as Java counts them
    let mut units: Vec<(usize, usize)> = Vec::new();
    let mut token_sources = Vec::new();
    let mut flexes: i32 = 1;
    let mut original = String::new();
    for child in &e.children {
        match child.name() {
            "flex" => {
                let f = attributes(child)?;
                let slot = slot(f)?;
                flexes = flexes.max(slot);
                let lemma = attribute(f, "lemma")?;
                flex_by_slot.entry(slot).or_default().push(lemma.to_owned());
            }
            "translation" => translation = first_child_text(child)?,
            "original" => {
                let mut last = 0usize;
                let children = child.as_element().map_or(&[][..], |o| &o.children[..]);
                for part in children {
                    match part {
                        Node::Element(t) if t.name == "token" => {
                            let token = first_child_text(part)?;
                            let len = utf16_len(&token);
                            units.push((last, last + len));
                            token_sources.push(t.span);
                            let slot = slot(t)?;
                            flexes = flexes.max(slot);
                            original.push_str(&token);
                            tokens_by_slot.put(slot, token);
                            last += len;
                        }
                        // a processing instruction named token has no child
                        Node::Pi { target, .. } if target == "token" => {
                            return Err(corpus(CorpusError::IrishNoContent));
                        }
                        Node::Text(d) => {
                            original.push_str(&d.text);
                            let len = utf16_len(&d.text);
                            if d.text != " " {
                                let (start, end) = trimmed_spaces(&d.text, last);
                                if start > end {
                                    return Err(corpus(CorpusError::IrishBackwardsSpan { start, end }));
                                }
                                units.push((start, end));
                                token_sources.push(d.span);
                            }
                            last += len;
                        }
                        other => return Err(unexpected(other.name())),
                    }
                }
            }
            "#text" | "#comment" => {}
            other => return Err(unexpected(other)),
        }
    }
    let len = flexes as usize;
    let mut flex = Some(Vec::new());
    for (slot, surface) in tokens_by_slot.iter() {
        let Some(lemmas) = flex_by_slot.get(slot) else {
            flex = None;
            break;
        };
        let index = i64::from(*slot) - 1;
        if index < 0 {
            return Err(corpus(CorpusError::IrishFlexIndex { index, len }));
        }
        let list = flex.as_mut().expect("set above");
        let entry = (index as usize, IrishFlex { surface: surface.clone(), lemmas: lemmas.clone() });
        match list.iter_mut().find(|(i, _)| *i == entry.0) {
            Some(e) => *e = entry,
            None => list.push(entry),
        }
    }
    let mut tokens = Vec::with_capacity(units.len());
    for (start, end) in units {
        let byte = |u| match utf16_cut(&original, u) {
            Ok(Some(b)) => Ok(b as u32),
            Ok(None) => Err(corpus(CorpusError::IrishTokenSplitsSurrogatePair)),
            // every span ends within the text read so far
            Err(_) => unreachable!("a token span past the end of the original"),
        };
        tokens.push(Span::new(byte(start)?, byte(end)?));
    }
    Ok(IrishSentence {
        source_name,
        translation,
        original,
        tokens,
        token_sources,
        flex: flex.map(|f| (f, len)),
        source: span,
    })
}

/// `advanceLeft` and `advanceRight`: the span of `text`, which starts at
/// `last`, without its leading spaces and its trailing spaces, where the
/// first character is never taken for a trailing space. Both count
/// UTF-16 code units; two or more spaces give a span that ends before it
/// starts.
fn trimmed_spaces(text: &str, last: usize) -> (usize, usize) {
    let units: Vec<u16> = text.encode_utf16().collect();
    let left = last + units.iter().take_while(|&&u| u == u16::from(b' ')).count();
    let mut right = last + units.len();
    for &u in units.iter().skip(1).rev() {
        if u != u16::from(b' ') {
            break;
        }
        right -= 1;
    }
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_and_the_text_between_them() {
        let src = r#"<sentences><sentence source="s"><original>a &amp; b<token slot="1">x</token>  y  <token slot="2">z</token></original><flex slot="1" lemma="l"/><flex slot="2" lemma="m"/></sentence></sentences>"#;
        let bank = IrishSentenceBank::parse(src).unwrap();
        let s = &bank.sentences[0];
        assert_eq!(s.original, "a & bx  y  z");
        assert_eq!(s.tokens, [Span::new(0, 5), Span::new(5, 6), Span::new(8, 9), Span::new(11, 12)]);
        assert_eq!(&src[s.token_sources[0].range()], "a &amp; b");
        let sample = bank.token_samples().next().unwrap().unwrap();
        assert_eq!(sample.to_string(), "a & b<SPLIT>x y z");
        assert_eq!(bank.sentence_sample().unwrap().unwrap().document(), "a & bx  y  z ");
        assert_eq!(s.flex.as_ref().unwrap().0.len(), 2);
    }

    #[test]
    fn what_java_throws() {
        let err = |src: &str| {
            let e = IrishSentenceBank::parse(src).unwrap_err();
            (e.java_class().unwrap(), e.to_string())
        };
        let s = |original: &str| {
            format!(r#"<sentences><sentence source="s"><original>{original}</original></sentence></sentences>"#)
        };
        assert_eq!(err(&s("<![CDATA[b]]>")), ("java.io.IOException", "Unexpected node: #cdata-section".into()));
        assert_eq!(
            err(&s("  ")),
            (
                "java.lang.IllegalArgumentException",
                "start index must not be larger than end index: start=2, end=1".into()
            )
        );
        assert_eq!(err("<Sentence/>").0, "java.io.IOException");
        assert_eq!(err("<x/>").1, "Expected root node x");
        let flex = r#"<sentences><sentence source="s"><original><token slot="0">a</token></original><flex slot="0" lemma="l"/></sentence></sentences>"#;
        assert_eq!(
            err(flex),
            ("java.lang.ArrayIndexOutOfBoundsException", "Index -1 out of bounds for length 1".into())
        );
        assert_eq!(err(&s(r#"<token>a</token>"#)).0, "java.lang.NullPointerException");
    }

    #[test]
    fn a_token_without_flex_drops_the_flexes() {
        let src =
            r#"<sentences><sentence source="s"><original><token slot="1">a</token></original></sentence></sentences>"#;
        assert_eq!(IrishSentenceBank::parse(src).unwrap().sentences[0].flex, None);
    }
}
