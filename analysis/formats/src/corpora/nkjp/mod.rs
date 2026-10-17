//! The National Corpus of Polish (NKJP): `NKJPSegmentationDocument`,
//! `NKJPTextDocument` and `NKJPSentenceSampleStream`.
//!
//! A text's sentences are in its segmentation file (`ann_segmentation.xml`)
//! as lists of segments, each pointing at a stretch of a paragraph of the
//! text file (`text.xml`) by the paragraph's id, an offset and a length
//! in UTF-16 code units. A sentence is the paragraph's text from the
//! start of its first segment to the end of its last.

#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::collections::HashMap;

use turbo_analysis::Span;
use turbo_jvm::hash_map::JavaHashMap;

use super::dom::{self, Element, Node};
use super::error::CorpusError;
use crate::error::FormatError;
use crate::masc::java::{equals_ignore_case_ascii, parse_int};
use crate::sample::SentenceSample;
use crate::source::{check_len, utf16_cut};

/// Where a segment is: `corresp="text.xml#string-range(p-1,5,3)"`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NkjpPointer {
    /// The id of the paragraph in the text file.
    pub paragraph: String,
    /// The offset in UTF-16 code units. A pointer with three numbers
    /// gives it as the first two, `a * 1000 + b`, computed with `int` overflow.
    pub offset: i32,
    /// The length in UTF-16 code units.
    pub length: i32,
}

/// One sentence of a segmentation file: its id, its segments by id in the
/// order first seen (a segment id given twice keeps its first place and
/// its last pointer, as `LinkedHashMap.put` does), and the bytes of its
/// `s` element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NkjpSentence {
    pub id: Option<String>,
    pub segments: Vec<(String, NkjpPointer)>,
    pub source: Span,
}

/// The sentences of a segmentation file, `NKJPSegmentationDocument`: the
/// `s` elements at `/teiCorpus/TEI/text/body/p/s`, by id in the order
/// first seen (a sentence id given twice, or two sentences without one,
/// keep the first place and the last segments).
///
/// The segments of a sentence are its `seg` children and the segments of
/// its `choice` children: of a `choice`, each `nkjp:paren` whose `seg`
/// children are not all rejected (`nkjp:rejected="true"`) gives all its
/// `seg` children, rejected ones included, and each `seg` that is not
/// rejected is one. A `nkjp:paren` without `seg` children counts as
/// rejected unless it has no children at all.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NkjpSegmentation {
    pub sentences: Vec<NkjpSentence>,
}

impl NkjpSegmentation {
    /// Reads a decoded segmentation file (a byte order mark at its start
    /// is skipped).
    ///
    /// # Errors
    ///
    /// [`CorpusError::NkjpDocument`] when it is not well-formed XML, or a
    /// segment has no `xml:id` or `corresp`, or a `corresp` is not a
    /// `document#function(id,offset,length)` or
    /// `document#function(id,offset,offset,length)` pointer.
    /// [`CorpusError::NotANumber`] for a number `Integer.parseInt`
    /// refuses, which OpenNLP does not catch.
    pub fn parse(text: &str) -> Result<NkjpSegmentation, FormatError> {
        let (root, _) = parse_xml(text)?;
        let mut sentences: Vec<NkjpSentence> = Vec::new();
        let mut by_id: HashMap<Option<String>, usize> = HashMap::new();
        if root.name != "teiCorpus" {
            return Ok(NkjpSegmentation { sentences });
        }
        let paragraphs = root
            .children_named("TEI")
            .flat_map(|e| e.children_named("text"))
            .flat_map(|e| e.children_named("body"))
            .flat_map(|e| e.children_named("p"));
        for s in paragraphs.flat_map(|e| e.children_named("s")) {
            let id = s.attribute("xml:id").map(str::to_owned);
            let mut segments = Segments::default();
            for child in s.children.iter().filter_map(Node::as_element) {
                match child.name.as_str() {
                    "seg" => segments.put(child)?,
                    "choice" => {
                        for c in &child.children {
                            match (c.name(), c.as_element()) {
                                ("nkjp:paren", Some(paren)) => {
                                    if !rejected_paren(paren) {
                                        for seg in paren.children_named("seg") {
                                            segments.put(seg)?;
                                        }
                                    }
                                }
                                ("seg", Some(seg)) => {
                                    if !rejected(seg) {
                                        segments.put(seg)?;
                                    }
                                }
                                // a processing instruction named seg has
                                // no attributes to read an id from
                                ("seg", None) => return Err(document("a seg that is not an element")),
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
            }
            let sentence = NkjpSentence { id: id.clone(), segments: segments.list, source: s.span };
            match by_id.get(&id) {
                Some(&i) => {
                    sentences[i].segments = sentence.segments;
                    sentences[i].source = sentence.source;
                }
                None => {
                    by_id.insert(id, sentences.len());
                    sentences.push(sentence);
                }
            }
        }
        Ok(NkjpSegmentation { sentences })
    }
}

/// The segments of one sentence, as the `LinkedHashMap` keeps them.
#[derive(Default)]
struct Segments {
    list: Vec<(String, NkjpPointer)>,
    by_id: HashMap<String, usize>,
}

impl Segments {
    /// `segments.put(xmlID(seg), fromSeg(seg))`.
    fn put(&mut self, seg: &Element) -> Result<(), FormatError> {
        let id = seg.attribute("xml:id").ok_or_else(|| document("a segment has no xml:id"))?.to_owned();
        let pointer = pointer(seg)?;
        match self.by_id.get(&id) {
            Some(&i) => self.list[i].1 = pointer,
            None => {
                self.by_id.insert(id.clone(), self.list.len());
                self.list.push((id, pointer));
            }
        }
        Ok(())
    }
}

/// `checkRejected`.
fn rejected(seg: &Element) -> bool {
    seg.attribute("nkjp:rejected") == Some("true")
}

/// `checkRejectedParen`.
fn rejected_paren(paren: &Element) -> bool {
    if paren.children.is_empty() {
        return false;
    }
    paren.children.iter().filter_map(Node::as_element).filter(|e| e.name == "seg").all(rejected)
}

/// `fromSeg`: the pointer of a segment's `corresp`.
fn pointer(seg: &Element) -> Result<NkjpPointer, FormatError> {
    let ptr = seg.attribute("corresp").ok_or_else(|| document("a segment has no corresp"))?;
    let invalid = || document("a corresp is not a valid pointer");
    if !ptr.contains('#') || !ptr.contains('(') || !ptr.ends_with(')') {
        return Err(invalid());
    }
    let open = ptr.find('(').expect("checked above");
    let pieces = java_split(&ptr[open + 1..ptr.len() - 1], ',');
    let number = |s: &str| parse_int(s).ok_or_else(|| FormatError::Corpus(CorpusError::NotANumber { text: s.into() }));
    let (offset, length) = match pieces[..] {
        [_, offset, length] => (number(offset)?, number(length)?),
        [_, high, low, length] => {
            let (high, low) = (number(high)?, number(low)?);
            (high.wrapping_mul(1000).wrapping_add(low), number(length)?)
        }
        _ => return Err(invalid()),
    };
    Ok(NkjpPointer { paragraph: pieces[0].to_owned(), offset, length })
}

/// `StringUtil.split(s, separator)`: every separator splits, and the
/// empty fields at the end are removed (an empty string is one empty
/// field).
fn java_split(s: &str, separator: char) -> Vec<&str> {
    if s.is_empty() {
        return vec![""];
    }
    let mut fields: Vec<&str> = s.split(separator).collect();
    while fields.last().is_some_and(|f| f.is_empty()) {
        fields.pop();
    }
    fields
}

fn document(detail: &str) -> FormatError {
    FormatError::Corpus(CorpusError::NkjpDocument { detail: detail.into() })
}

/// Parses a decoded file, skipping a byte order mark, and gives the root
/// element with its spans moved past the mark.
fn parse_xml(text: &str) -> Result<(Element, usize), FormatError> {
    check_len(text)?;
    let bom = if text.starts_with('\u{FEFF}') { 3 } else { 0 };
    let mut root = dom::parse(&text[bom..]).map_err(|e| document(&e.to_string()))?;
    if bom > 0 {
        shift(&mut root, bom as u32);
    }
    Ok((root, bom))
}

/// Moves every span of `e` by `by` bytes.
pub(crate) fn shift(e: &mut Element, by: u32) {
    e.span = Span::new(e.span.start + by, e.span.end + by);
    for child in &mut e.children {
        match child {
            Node::Element(c) => shift(c, by),
            Node::Text(d) | Node::CData(d) | Node::Comment(d) | Node::Pi { data: d, .. } => {
                d.span = Span::new(d.span.start + by, d.span.end + by);
            }
        }
    }
}

/// A paragraph of a text file: its text and the bytes of its element.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NkjpParagraph {
    pub text: String,
    pub source: Span,
}

/// The paragraphs of a text file by id, `NKJPTextDocument.getParagraphs`.
///
/// The texts are the elements at `/teiCorpus/TEI/text/group/text`, or
/// at `/teiCorpus/TEI/text` when there are none; their paragraphs are
/// the `p` and `ab` children of their `body/div` elements. A paragraph's
/// text is its text content, which must be its only node or start with
/// a text node. OpenNLP keeps the paragraphs in a `HashMap` per division,
/// the divisions in one per text and the texts in one by id, and flattens
/// them in the maps' iteration order, so a paragraph id given twice keeps
/// the paragraph that comes last in that order; a division or text whose
/// id is given twice loses its earlier namesake's paragraphs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NkjpText {
    paragraphs: HashMap<String, NkjpParagraph>,
}

type Divisions = JavaHashMap<String, JavaHashMap<String, NkjpParagraph>>;

impl NkjpText {
    /// Reads a decoded text file (a byte order mark at its start is
    /// skipped).
    ///
    /// # Errors
    ///
    /// [`CorpusError::NkjpDocument`] when it is not well-formed XML, its
    /// root element is not `teiCorpus` in any case, a text, division or
    /// paragraph has no `xml:id`, or a paragraph has more than one node
    /// and the first is not text. [`CorpusError::NkjpEmptyParagraph`] for
    /// a paragraph with no content.
    pub fn parse(text: &str) -> Result<NkjpText, FormatError> {
        let (root, _) = parse_xml(text)?;
        if !equals_ignore_case_ascii(&root.name, "teiCorpus") {
            return Err(document("the root element is not teiCorpus"));
        }
        let mut texts: JavaHashMap<String, Divisions> = JavaHashMap::new();
        let teis = || root.children_named("TEI").filter(|_| root.name == "teiCorpus");
        let grouped: Vec<&Element> = teis()
            .flat_map(|e| e.children_named("text"))
            .flat_map(|e| e.children_named("group"))
            .flat_map(|e| e.children_named("text"))
            .collect();
        let nodes = if grouped.is_empty() { teis().flat_map(|e| e.children_named("text")).collect() } else { grouped };
        for node in nodes {
            let id = required(node, "xml:id")?;
            let mut divisions: Divisions = JavaHashMap::new();
            for div in node.children_named("body").flat_map(|e| e.children_named("div")) {
                let div_id = required(div, "xml:id")?;
                let mut paragraphs = JavaHashMap::new();
                for p in div.children.iter().filter_map(Node::as_element).filter(|e| e.name == "p" || e.name == "ab") {
                    let p_id = required(p, "xml:id")?;
                    if p.children.len() != 1 {
                        match p.children.first() {
                            None => return Err(FormatError::Corpus(CorpusError::NkjpEmptyParagraph)),
                            Some(Node::Text(_)) => {}
                            Some(_) => return Err(document("a paragraph does not start with text")),
                        }
                    }
                    paragraphs.put(p_id, NkjpParagraph { text: p.text_content(), source: p.span });
                }
                divisions.put(div_id, paragraphs);
            }
            texts.put(id, divisions);
        }
        let mut paragraphs = HashMap::new();
        for (_, divisions) in texts.into_ordered() {
            for (_, ps) in divisions.into_ordered() {
                for (id, p) in ps.into_ordered() {
                    paragraphs.insert(id, p);
                }
            }
        }
        Ok(NkjpText { paragraphs })
    }

    /// The paragraph with id `id`.
    pub fn paragraph(&self, id: &str) -> Option<&NkjpParagraph> {
        self.paragraphs.get(id)
    }
}

/// `attrib(n, name, true)`.
fn required(e: &Element, name: &str) -> Result<String, FormatError> {
    e.attribute(name).map(str::to_owned).ok_or_else(|| document("an element has no xml:id"))
}

/// Reads sentence samples from a segmentation file and its text file:
/// `NKJPSentenceSampleStream`.
///
/// One sample holds every sentence, each followed by a space. A sentence
/// is cut from the paragraph of its first segment, from that segment's
/// start to the end of the last segment before the paragraph changes;
/// segments in another paragraph make another sentence of the sample.
/// A sentence with no segments is empty. A sentence that throws (a
/// pointer that makes no span, a paragraph the text does not have, a
/// cut outside the paragraph) loses the sample read so far, and the next
/// sample starts at the sentence after it.
#[derive(Clone, Debug)]
pub struct NkjpReader<'d> {
    segmentation: NkjpSegmentation,
    text: &'d NkjpText,
    next: usize,
}

impl<'d> NkjpReader<'d> {
    pub fn new(segmentation: NkjpSegmentation, text: &'d NkjpText) -> NkjpReader<'d> {
        NkjpReader { segmentation, text, next: 0 }
    }
}

/// The empty paragraph a sentence without segments is cut from.
const NO_PARAGRAPH: NkjpParagraph = NkjpParagraph { text: String::new(), source: Span::new(0, 0) };

impl<'d> Iterator for NkjpReader<'d> {
    type Item = Result<SentenceSample<'d>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut document = String::new();
        let mut spans = Vec::new();
        let mut sources = Vec::new();
        while let Some(sentence) = self.segmentation.sentences.get(self.next) {
            self.next += 1;
            if let Err(e) = sentence_pieces(self.text, sentence, &mut document, &mut spans) {
                return Some(Err(e));
            }
            while sources.len() < spans.len() {
                sources.push(sentence.source);
            }
        }
        let start = sources.iter().map(|s| s.start).min()?;
        let end = sources.iter().map(|s| s.end).max()?;
        Some(SentenceSample::new(Cow::Owned(document), spans, sources, Span::new(start, end)))
    }
}

/// Appends the pieces of one sentence to `document`, each followed by a
/// space, with their spans.
fn sentence_pieces(
    text: &NkjpText,
    sentence: &NkjpSentence,
    document: &mut String,
    spans: &mut Vec<Span>,
) -> Result<(), FormatError> {
    let (mut start, mut end) = (0, 0);
    let mut current: Option<(&str, Option<&NkjpParagraph>)> = None;
    for (_, pointer) in &sentence.segments {
        let span = to_span(pointer)?;
        let paragraph = text.paragraph(&pointer.paragraph);
        match current {
            None => {
                start = span.0;
                end = span.1;
                current = Some((&pointer.paragraph, paragraph));
            }
            Some((id, p)) if id != pointer.paragraph => {
                cut(p, start, end, document, spans)?;
                start = span.0;
                end = span.1;
                current = Some((&pointer.paragraph, paragraph));
            }
            Some(_) => end = span.1,
        }
    }
    cut(current.map_or(Some(&NO_PARAGRAPH), |(_, p)| p), start, end, document, spans)
}

/// `Pointer.toSpan`: the offset and the offset plus the length, as
/// `Span`'s constructor checks them.
fn to_span(pointer: &NkjpPointer) -> Result<(i32, i32), FormatError> {
    let (start, end) = (pointer.offset, pointer.offset.wrapping_add(pointer.length));
    if start < 0 || end < 0 || start > end {
        return Err(FormatError::Corpus(CorpusError::NkjpSpan { start, end }));
    }
    Ok((start, end))
}

/// `currentParagraph.substring(start, end)` appended to the document,
/// followed by a space.
fn cut(
    paragraph: Option<&NkjpParagraph>,
    start: i32,
    end: i32,
    document: &mut String,
    spans: &mut Vec<Span>,
) -> Result<(), FormatError> {
    let p = paragraph.ok_or(FormatError::Corpus(CorpusError::NkjpMissingParagraph))?;
    let text = &p.text;
    let out_of_bounds = |length| FormatError::Corpus(CorpusError::NkjpSubstring { begin: start, end, length });
    let surrogate = FormatError::Corpus(CorpusError::SplitsSurrogatePair);
    let to_byte = |units: i32| match utf16_cut(text, units as usize) {
        Ok(Some(b)) => Ok(b),
        Ok(None) => Err(surrogate.clone()),
        Err(length) => Err(out_of_bounds(length)),
    };
    if start > end {
        return Err(out_of_bounds(turbo_jvm::chars::utf16_len(text)));
    }
    let to = to_byte(end)?;
    let from = to_byte(start)?;
    let begin = document.len();
    document.push_str(&text[from..to]);
    spans.push(Span::new(begin as u32, document.len() as u32));
    document.push(' ');
    Ok(())
}
