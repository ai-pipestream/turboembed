//! Polish National Corpus (NKJP) TEI XML readers (`opennlp.tools.formats.nkjp`).
//!
//! Port of `NKJPSegmentationDocument`, `NKJPTextDocument`, `NKJPSentenceSampleStream`,
//! and `NKJPSentenceSampleStreamFactory` from OpenNLP format utilities.

use std::borrow::Cow;
use std::collections::HashMap;

use turbo_analysis::Span;
use turbo_maxent::xml::{self, Attribute, Handler, Names};

use crate::error::FormatError;
use crate::letsmt::trim_java_str;
use crate::sample::SentenceSample;

/// Minimal DOM element tree built from `turbo_maxent::xml` with qualified names.
#[derive(Debug, Clone)]
pub(crate) struct XmlElement {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<XmlNode>,
}

#[derive(Debug, Clone)]
pub(crate) enum XmlNode {
    Element(XmlElement),
    Text(String),
}

impl XmlElement {
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    pub fn child_elements(&self) -> impl Iterator<Item = &XmlElement> {
        self.children.iter().filter_map(|n| match n {
            XmlNode::Element(e) => Some(e),
            XmlNode::Text(_) => None,
        })
    }

    pub fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a XmlElement> {
        self.child_elements().filter(move |e| e.name == name)
    }

    pub fn text_content(&self) -> String {
        let mut out = String::new();
        self.append_text(&mut out);
        out
    }

    fn append_text(&self, out: &mut String) {
        for child in &self.children {
            match child {
                XmlNode::Text(t) => out.push_str(t),
                XmlNode::Element(e) => e.append_text(out),
            }
        }
    }
}

#[derive(Default)]
struct TreeBuilder {
    stack: Vec<XmlElement>,
    root: Option<XmlElement>,
}

impl Handler for TreeBuilder {
    fn start_element(&mut self, name: &str, attributes: &[Attribute<'_>], _offset: usize) -> turbo_maxent::Result<()> {
        let attrs = attributes.iter().map(|a| (a.name.to_owned(), a.value.as_ref().to_owned())).collect();
        self.stack.push(XmlElement { name: name.to_owned(), attrs, children: Vec::new() });
        Ok(())
    }

    fn characters(&mut self, text: &str) {
        if let Some(top) = self.stack.last_mut() {
            if let Some(XmlNode::Text(existing)) = top.children.last_mut() {
                existing.push_str(text);
            } else {
                top.children.push(XmlNode::Text(text.to_owned()));
            }
        }
    }

    fn end_element(&mut self, _name: &str, _offset: usize) -> turbo_maxent::Result<()> {
        if let Some(elem) = self.stack.pop() {
            if let Some(parent) = self.stack.last_mut() {
                parent.children.push(XmlNode::Element(elem));
            } else {
                self.root = Some(elem);
            }
        }
        Ok(())
    }
}

pub(crate) fn parse_xml_tree(bytes: &[u8], format: &'static str) -> Result<XmlElement, FormatError> {
    let text = xml::decode(bytes).map_err(|err| FormatError::Xml { format, message: err.to_string() })?;
    parse_xml_tree_str(&text, format)
}

pub(crate) fn parse_xml_tree_str(text: &str, format: &'static str) -> Result<XmlElement, FormatError> {
    let mut builder = TreeBuilder::default();
    xml::parse_with(text, Names::Qualified, &mut builder)
        .map_err(|err| FormatError::Xml { format, message: err.to_string() })?;
    builder.root.ok_or_else(|| FormatError::Xml { format, message: "empty XML document".to_owned() })
}

/// Segment pointer into a paragraph of an [`NkjpTextDocument`].
///
/// Mirrors `opennlp.tools.formats.nkjp.NKJPSegmentationDocument.Pointer`.
/// Note that `offset` and `length` are measured in UTF-16 code units, matching
/// Java's `String.substring(offset, offset + length)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NkjpPointer {
    /// Target document file name (before `#` in `corresp`).
    pub doc: String,
    /// Target paragraph/element `xml:id`.
    pub id: String,
    /// Start offset in UTF-16 code units within the target element's text.
    pub offset: usize,
    /// Length in UTF-16 code units within the target element's text.
    pub length: usize,
    /// Whether a space follows this segment.
    ///
    /// Reproduces OpenNLP's `ptr.equals("yes")` check on `nkjp:nps`.
    pub space_after: bool,
}

impl NkjpPointer {
    /// Returns the UTF-16 code-unit span `[offset, offset + length)`.
    #[inline]
    pub fn to_span(&self) -> Span {
        Span::new(self.offset as u32, (self.offset + self.length) as u32)
    }
}

/// Parsed NKJP `ann_segmentation.xml` document.
///
/// Mirrors `opennlp.tools.formats.nkjp.NKJPSegmentationDocument`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NkjpSegmentationDocument {
    /// Ordered list of `(sentence_xml_id, Vec<(segment_xml_id, NkjpPointer)>)`.
    sentences: Vec<(String, Vec<(String, NkjpPointer)>)>,
}

impl NkjpSegmentationDocument {
    /// Parses an NKJP `ann_segmentation.xml` document from raw bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        let root = parse_xml_tree(bytes, "NKJP segmentation")?;
        Self::from_root(&root)
    }

    /// Parses an NKJP `ann_segmentation.xml` document from decoded UTF-8 text.
    pub fn parse_str(text: &str) -> Result<Self, FormatError> {
        let root = parse_xml_tree_str(text, "NKJP segmentation")?;
        Self::from_root(&root)
    }

    fn from_root(root: &XmlElement) -> Result<Self, FormatError> {
        let mut s_elements = Vec::new();
        collect_elements_by_name(root, "s", &mut s_elements);

        let mut sentences = Vec::with_capacity(s_elements.len());
        for s_elem in s_elements {
            let sent_id = xml_id(s_elem, "NKJP segmentation")?;
            let mut segs = Vec::new();
            for child in s_elem.child_elements() {
                if child.name == "seg" {
                    let seg_id = xml_id(child, "NKJP segmentation")?;
                    let ptr = pointer_from_seg(child)?;
                    segs.push((seg_id, ptr));
                } else if child.name == "choice" {
                    for choice_child in child.child_elements() {
                        if choice_child.name == "nkjp:paren" {
                            if !is_rejected(choice_child) {
                                for pchild in choice_child.child_elements() {
                                    if pchild.name == "seg" {
                                        let seg_id = xml_id(pchild, "NKJP segmentation")?;
                                        let ptr = pointer_from_seg(pchild)?;
                                        segs.push((seg_id, ptr));
                                    }
                                }
                            }
                        } else if choice_child.name == "seg" && !is_rejected(choice_child) {
                            let seg_id = xml_id(choice_child, "NKJP segmentation")?;
                            let ptr = pointer_from_seg(choice_child)?;
                            segs.push((seg_id, ptr));
                        }
                    }
                }
            }
            sentences.push((sent_id, segs));
        }

        Ok(Self { sentences })
    }

    /// Returns the ordered sentence segmentation entries.
    #[inline]
    pub fn sentences(&self) -> &[(String, Vec<(String, NkjpPointer)>)] {
        &self.sentences
    }

    /// Looks up a segment pointer by sentence `xml:id` and segment `xml:id`.
    pub fn get_segment(&self, sentence_id: &str, seg_id: &str) -> Option<&NkjpPointer> {
        self.sentences
            .iter()
            .find(|(sid, _)| sid == sentence_id)
            .and_then(|(_, segs)| segs.iter().find(|(id, _)| id == seg_id).map(|(_, p)| p))
    }
}

fn collect_elements_by_name<'a>(elem: &'a XmlElement, name: &str, out: &mut Vec<&'a XmlElement>) {
    if elem.name == name {
        out.push(elem);
    }
    for child in elem.child_elements() {
        collect_elements_by_name(child, name, out);
    }
}

fn xml_id(elem: &XmlElement, format: &'static str) -> Result<String, FormatError> {
    elem.attr("xml:id").filter(|v| !v.is_empty()).map(str::to_string).ok_or_else(|| FormatError::Xml {
        format,
        message: format!("missing required xml:id attribute on <{}>", elem.name),
    })
}

fn is_rejected(elem: &XmlElement) -> bool {
    elem.attr("nkjp:rejected") == Some("true")
}

fn pointer_from_seg(elem: &XmlElement) -> Result<NkjpPointer, FormatError> {
    let corresp = elem.attr("corresp").ok_or_else(|| FormatError::Xml {
        format: "NKJP segmentation",
        message: "missing required corresp attribute on <seg>".to_owned(),
    })?;
    let space_after = elem.attr("nkjp:nps") == Some("yes");

    let p1 = corresp.find('#').ok_or_else(|| FormatError::Xml {
        format: "NKJP segmentation",
        message: format!("invalid corresp attribute (missing '#'): {corresp}"),
    })?;
    let doc = corresp[..p1].to_string();

    let p2 = corresp[p1..].find('(').map(|i| p1 + i + 1).ok_or_else(|| FormatError::Xml {
        format: "NKJP segmentation",
        message: format!("invalid corresp attribute (missing '('): {corresp}"),
    })?;
    let p3 = corresp[p2..].find(')').map(|i| p2 + i).ok_or_else(|| FormatError::Xml {
        format: "NKJP segmentation",
        message: format!("invalid corresp attribute (missing ')'): {corresp}"),
    })?;

    let range_args = &corresp[p2..p3];
    let parts: Vec<&str> = range_args.split(',').collect();
    match parts.len() {
        3 => {
            let id = trim_java_str(parts[0]).to_owned();
            let offset = trim_java_str(parts[1]).parse::<usize>().map_err(|_| FormatError::Xml {
                format: "NKJP segmentation",
                message: format!("invalid offset in corresp: {corresp}"),
            })?;
            let length = trim_java_str(parts[2]).parse::<usize>().map_err(|_| FormatError::Xml {
                format: "NKJP segmentation",
                message: format!("invalid length in corresp: {corresp}"),
            })?;
            Ok(NkjpPointer { doc, id, offset, length, space_after })
        }
        4 => {
            let id = trim_java_str(parts[0]).to_owned();
            let os1 = trim_java_str(parts[1]).parse::<usize>().map_err(|_| FormatError::Xml {
                format: "NKJP segmentation",
                message: format!("invalid offset part 1 in corresp: {corresp}"),
            })?;
            let os2 = trim_java_str(parts[2]).parse::<usize>().map_err(|_| FormatError::Xml {
                format: "NKJP segmentation",
                message: format!("invalid offset part 2 in corresp: {corresp}"),
            })?;
            let offset = os1 * 1000 + os2;
            let length = trim_java_str(parts[3]).parse::<usize>().map_err(|_| FormatError::Xml {
                format: "NKJP segmentation",
                message: format!("invalid length in corresp: {corresp}"),
            })?;
            Ok(NkjpPointer { doc, id, offset, length, space_after })
        }
        _ => Err(FormatError::Xml {
            format: "NKJP segmentation",
            message: format!("expected 3 or 4 string-range arguments in corresp: {corresp}"),
        }),
    }
}

/// Parsed NKJP `text_structure.xml` document.
///
/// Mirrors `opennlp.tools.formats.nkjp.NKJPTextDocument`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NkjpTextDocument {
    divtypes: HashMap<String, String>,
    texts: HashMap<String, HashMap<String, HashMap<String, String>>>,
}

impl NkjpTextDocument {
    /// Parses an NKJP `text_structure.xml` document from raw bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self, FormatError> {
        let root = parse_xml_tree(bytes, "NKJP text")?;
        Self::from_root(&root)
    }

    /// Parses an NKJP `text_structure.xml` document from decoded UTF-8 text.
    pub fn parse_str(text: &str) -> Result<Self, FormatError> {
        let root = parse_xml_tree_str(text, "NKJP text")?;
        Self::from_root(&root)
    }

    fn from_root(root: &XmlElement) -> Result<Self, FormatError> {
        if !root.name.eq_ignore_ascii_case("teiCorpus") {
            return Err(FormatError::Xml {
                format: "NKJP text",
                message: format!("root node is <{}>, expected <teiCorpus>", root.name),
            });
        }

        // Collect `/teiCorpus/TEI/text/group/text`, falling back to `/teiCorpus/TEI/text`.
        let mut text_nodes = Vec::new();
        for tei in root.children_named("TEI") {
            for t in tei.children_named("text") {
                for group in t.children_named("group") {
                    for gt in group.children_named("text") {
                        text_nodes.push(gt);
                    }
                }
            }
        }
        if text_nodes.is_empty() {
            for tei in root.children_named("TEI") {
                for t in tei.children_named("text") {
                    text_nodes.push(t);
                }
            }
        }

        let mut divtypes = HashMap::new();
        let mut texts = HashMap::new();

        for text_elem in text_nodes {
            let text_id = xml_id(text_elem, "NKJP text")?;
            let mut divs_map = HashMap::new();

            for body in text_elem.children_named("body") {
                for div in body.children_named("div") {
                    let div_id = xml_id(div, "NKJP text")?;
                    if let Some(dtype) = div.attr("type").filter(|s| !s.is_empty()) {
                        divtypes.insert(div_id.clone(), dtype.to_string());
                    }
                    let mut paras_map = HashMap::new();
                    for p in div.child_elements() {
                        if p.name == "p" || p.name == "ab" {
                            let p_id = xml_id(p, "NKJP text")?;
                            if p.children.len() != 1 || !matches!(p.children[0], XmlNode::Text(_)) {
                                return Err(FormatError::Xml {
                                    format: "NKJP text",
                                    message: "Paragraph needs to have a single text element only".to_owned(),
                                });
                            }
                            paras_map.insert(p_id, p.text_content());
                        }
                    }
                    divs_map.insert(div_id, paras_map);
                }
            }
            texts.insert(text_id, divs_map);
        }

        Ok(Self { divtypes, texts })
    }

    /// Returns the map of `div` `xml:id` to `type` attribute.
    #[inline]
    pub fn divtypes(&self) -> &HashMap<String, String> {
        &self.divtypes
    }

    /// Returns the nested hierarchy `text_id -> div_id -> paragraph_id -> paragraph_text`.
    #[inline]
    pub fn texts(&self) -> &HashMap<String, HashMap<String, HashMap<String, String>>> {
        &self.texts
    }

    /// Returns a flattened map of `paragraph_id -> paragraph_text`.
    ///
    /// Mirrors `NKJPTextDocument.getParagraphs()`.
    pub fn paragraphs(&self) -> HashMap<String, String> {
        let mut paragraphs = HashMap::new();
        for dmap in self.texts.values() {
            for pmap in dmap.values() {
                for (pid, ptext) in pmap {
                    paragraphs.insert(pid.clone(), ptext.clone());
                }
            }
        }
        paragraphs
    }
}

/// Slices a UTF-8 `&str` using UTF-16 code unit `[offset, offset + length)`.
fn slice_utf16(text: &str, offset: usize, length: usize) -> Option<&str> {
    let target_end = offset.checked_add(length)?;
    let mut utf16_pos = 0usize;
    let mut byte_start = if offset == 0 { Some(0) } else { None };
    let mut byte_end = if target_end == 0 { Some(0) } else { None };

    for (byte_idx, ch) in text.char_indices() {
        if byte_start.is_none() && utf16_pos == offset {
            byte_start = Some(byte_idx);
        }
        if byte_end.is_none() && utf16_pos == target_end {
            byte_end = Some(byte_idx);
            break;
        }
        utf16_pos += ch.len_utf16();
    }
    if byte_start.is_none() && utf16_pos == offset {
        byte_start = Some(text.len());
    }
    if byte_end.is_none() && utf16_pos == target_end {
        byte_end = Some(text.len());
    }

    let start = byte_start?;
    let end = byte_end?;
    text.get(start..end)
}

/// Sentence sample stream combining an [`NkjpSegmentationDocument`] and an [`NkjpTextDocument`].
///
/// Mirrors `opennlp.tools.formats.nkjp.NKJPSentenceSampleStream`.
#[derive(Debug, Clone)]
pub struct NkjpSentenceReader {
    segmentation: NkjpSegmentationDocument,
    paragraphs: HashMap<String, String>,
    emitted: bool,
}

impl NkjpSentenceReader {
    /// Creates an NKJP sentence reader from parsed segmentation and text documents.
    #[inline]
    pub fn new(segmentation: NkjpSegmentationDocument, text: &NkjpTextDocument) -> Self {
        Self { segmentation, paragraphs: text.paragraphs(), emitted: false }
    }

    /// Parses both XML documents from bytes and returns an [`NkjpSentenceReader`].
    pub fn from_xml(segmentation_xml: &[u8], text_xml: &[u8]) -> Result<Self, FormatError> {
        let seg = NkjpSegmentationDocument::parse(segmentation_xml)?;
        let txt = NkjpTextDocument::parse(text_xml)?;
        Ok(Self::new(seg, &txt))
    }

    /// Parses both XML documents from `&str` and returns an [`NkjpSentenceReader`].
    pub fn from_xml_str(segmentation_xml: &str, text_xml: &str) -> Result<Self, FormatError> {
        let seg = NkjpSegmentationDocument::parse_str(segmentation_xml)?;
        let txt = NkjpTextDocument::parse_str(text_xml)?;
        Ok(Self::new(seg, &txt))
    }

    /// Reads the document's [`SentenceSample`], or `None` once already read.
    pub fn next_sample(&mut self) -> Option<Result<SentenceSample<'static>, FormatError>> {
        if self.emitted {
            return None;
        }
        self.emitted = true;

        let mut sentences_string = String::new();
        let mut sentence_spans = Vec::new();

        for (_, segments) in &self.segmentation.sentences {
            let mut last_id = String::new();
            let mut current_paragraph = "";
            let mut have_string = false;
            let mut begin = sentences_string.len() as u32;

            for (_, pointer) in segments {
                if last_id != pointer.id {
                    if have_string {
                        let end = sentences_string.len() as u32;
                        sentence_spans.push(Span::new(begin, end));
                        sentences_string.push(' ');
                        have_string = false;
                        begin = sentences_string.len() as u32;
                    }
                    last_id.clone_from(&pointer.id);
                    current_paragraph = self.paragraphs.get(&pointer.id).map(String::as_str).unwrap_or("");
                }

                if let Some(segment_text) = slice_utf16(current_paragraph, pointer.offset, pointer.length) {
                    sentences_string.push_str(segment_text);
                    if pointer.space_after {
                        sentences_string.push(' ');
                    }
                    have_string = true;
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

impl Iterator for NkjpSentenceReader {
    type Item = Result<SentenceSample<'static>, FormatError>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ANN_SEGMENTATION_XML: &[u8] = br##"<?xml version="1.0" encoding="UTF-8"?>
<teiCorpus xmlns:xi="http://www.w3.org/2001/XInclude" xmlns="http://www.tei-c.org/ns/1.0" xmlns:nkjp="http://www.nkjp.pl/ns/1.0">
    <TEI>
        <text xml:id="segm_text" xml:lang="pl">
            <body xml:id="segm_body">
                <p corresp="text.xml#txt_1-div" xml:id="segm_1-p">
                    <s xml:id="segm_1.1-s">
                        <seg corresp="text_structure.xml#string-range(ab-1,0,2)" xml:id="segm_1.1-seg"/>
                        <seg corresp="text_structure.xml#string-range(ab-1,3,7)" xml:id="segm_1.2-seg"/>
                        <seg corresp="text_structure.xml#string-range(ab-1,11,6)" xml:id="segm_1.3-seg"/>
                        <seg corresp="text_structure.xml#string-range(ab-1,18,1)" xml:id="segm_1.4-seg"/>
                        <seg corresp="text_structure.xml#string-range(ab-1,20,6)" xml:id="segm_1.5-seg"/>
                        <seg corresp="text_structure.xml#string-range(ab-1,27,8)" xml:id="segm_1.6-seg"/>
                        <seg corresp="text_structure.xml#string-range(ab-1,35,1)" nkjp:nps="true" xml:id="segm_1.7-seg"/>
                    </s>
                </p>
            </body>
        </text>
    </TEI>
</teiCorpus>"##;

    const TEXT_STRUCTURE_XML: &[u8] = br##"<?xml version="1.0" encoding="UTF-8"?>
<?oxygen RNGSchema="NKJP_structure.rng" type="xml"?>
<teiCorpus xmlns:xi="http://www.w3.org/2001/XInclude" xmlns="http://www.tei-c.org/ns/1.0">
    <TEI>
        <text>
            <group>
                <text decls="#bibl-1" xml:id="text-1">
                    <front>
                        <docTitle>
                            <titlePart type="main" xml:id="titlePart-1">Kr&#243;tki tekst</titlePart>
                        </docTitle>
                    </front>
                    <body>
                        <div type="article" xml:id="div-1">
                            <p xml:id="p-1">To kr&#243;tki tekst w formacie NKJP. Zawiera dwa zdania.</p>
                            <ab xml:id="ab-1">To kr&#243;tkie zdanie w drugim akapicie.</ab>
                        </div>
                    </body>
                </text>
            </group>
        </text>
    </TEI>
</teiCorpus>"##;

    #[test]
    fn parse_nkjp_segmentation_matches_opennlp() {
        let doc = NkjpSegmentationDocument::parse(ANN_SEGMENTATION_XML).unwrap();
        assert_eq!(doc.sentences().len(), 1);
        assert_eq!(doc.sentences()[0].0, "segm_1.1-s");
        assert_eq!(doc.sentences()[0].1.len(), 7);

        let ptr = doc.get_segment("segm_1.1-s", "segm_1.1-seg").unwrap();
        assert_eq!(ptr.offset, 0);
        assert_eq!(ptr.length, 2);
        let src = "To krótkie zdanie w drugim akapicie.";
        assert_eq!(slice_utf16(src, ptr.offset, ptr.length), Some("To"));
    }

    #[test]
    fn parse_nkjp_text_structure_matches_opennlp() {
        let doc = NkjpTextDocument::parse(TEXT_STRUCTURE_XML).unwrap();
        assert_eq!(doc.divtypes().len(), 1);
        assert_eq!(doc.divtypes().get("div-1").map(String::as_str), Some("article"));
        assert_eq!(doc.texts().len(), 1);
        assert_eq!(doc.texts()["text-1"].len(), 1);
        assert_eq!(doc.texts()["text-1"]["div-1"].len(), 2);
        assert_eq!(doc.texts()["text-1"]["div-1"]["p-1"], "To krótki tekst w formacie NKJP. Zawiera dwa zdania.");

        let paras = doc.paragraphs();
        assert_eq!(paras["ab-1"], "To krótkie zdanie w drugim akapicie.");
    }

    #[test]
    fn nkjp_sentence_reader_combines_segmentation_and_text() {
        let mut reader = NkjpSentenceReader::from_xml(ANN_SEGMENTATION_XML, TEXT_STRUCTURE_XML).unwrap();
        let sample = reader.next().expect("one sample").unwrap();
        assert!(reader.next().is_none());
        assert_eq!(sample.sentences().len(), 1);
        // Note: OpenNLP checks `nkjp:nps == "yes"` so `nkjp:nps="true"` results in `space_after = false`
        // for all 7 segments, concatenating "Tokrótkiezdaniewdrugimakapicie.".
        assert_eq!(sample.sentence(0), "Tokrótkiezdaniewdrugimakapicie.");
    }
}
