//! MASC's logical markup: the headings and paragraphs of a document.
//!
//! The header names one more stand-off file, `f.logical` ("Logical
//! markup"): a GrAF graph whose `region` elements anchor stretches of
//! the text in UTF-16 units, whose `node` elements each `link` to one
//! region, and whose `a` elements label a node (`cesDoc` for the whole
//! document, `head` for a heading, `p` for a paragraph, and other labels
//! the genre needs, such as `u` and `turn` in spoken text). The file
//! lists its regions, nodes and annotations in no dependable order, so
//! every reference is resolved by id after the whole file is read.
//!
//! [`MascDocument::add_logical`] keeps each labelled node as a
//! [`MascLogical`] in UTF-16 units, like every other MASC type, and
//! [`MascDocument::structure`] converts the document, heading and
//! paragraph blocks to a [`Structure`] layer over the source text,
//! through [`MascDocument::span`]. Features inside an `a` element (a
//! paragraph's `id`, for one) are not read.

use std::collections::HashMap;

use turbo_analysis::{Ix, Layer, Spans, ViewId};
use turbo_analysis_layers::{Block, Structure};
use turbo_maxent::xml::{Attribute, Handler};

use super::document::{MascDocument, attr, bad, parse_id, read_xml, valid_span};
use super::error::{AnnotationFile, MascError};
use super::java::{parse_int, split_on_spaces};

/// One labelled stretch of the logical markup: a node's region, in
/// UTF-16 units of the text, with the node's id and its label as the
/// file gives them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MascLogical {
    /// The node's `xml:id`, `logical-n` and a number, or
    /// `logical-doc-n` for the document node.
    pub id: String,
    /// The `a` element's label: `cesDoc`, `head`, `p`, or another label
    /// of the document's genre.
    pub label: String,
    pub start: i32,
    pub end: i32,
}

/// Whether `id` is a logical file's region id: `logical-r` and digits,
/// or the document region's `logical-doc-r`.
fn region_id(id: &str) -> bool {
    id == "logical-doc-r" || parse_id(Some(id), "logical-r").is_some()
}

/// Whether `id` is a logical file's node id: `logical-n` and digits, or
/// the document node's `logical-doc-n`.
fn node_id(id: &str) -> bool {
    id == "logical-doc-n" || parse_id(Some(id), "logical-n").is_some()
}

/// What a logical file holds before its references are resolved.
#[derive(Default)]
struct LogicalFile {
    /// Region id to its anchors, in UTF-16 units.
    regions: HashMap<String, (i32, i32)>,
    /// Each node once, in the order its element came, with the region
    /// its link names.
    nodes: Vec<(String, Option<String>)>,
    places: HashMap<String, usize>,
    /// Node id to its annotation's label.
    labels: HashMap<String, String>,
    /// The places of the open `node` elements.
    open: Vec<usize>,
}

impl Handler for LogicalFile {
    fn start_element(&mut self, name: &str, a: &[Attribute<'_>], at: usize) -> turbo_maxent::Result<()> {
        match name {
            "region" => {
                let id = attr(a, "xml:id").ok_or(bad(at, "a region has no id"))?;
                if !region_id(id) {
                    return Err(bad(at, "a region id is not logical-r and digits"));
                }
                let anchors = split_on_spaces(attr(a, "anchors").ok_or(bad(at, "a region has no anchors"))?);
                let [left, right] = anchors[..] else {
                    return Err(bad(at, "a region's anchors are not two numbers"));
                };
                let (Some(start), Some(end)) = (parse_int(left), parse_int(right)) else {
                    return Err(bad(at, "a region anchor is not an int"));
                };
                if !valid_span(start, end) {
                    return Err(bad(at, "a region's anchors are not a span"));
                }
                if self.regions.insert(id.to_owned(), (start, end)).is_some() {
                    return Err(bad(at, "two regions with one id"));
                }
            }
            "node" => {
                let id = attr(a, "xml:id").ok_or(bad(at, "a node has no id"))?;
                if !node_id(id) {
                    return Err(bad(at, "a node id is not logical-n and digits"));
                }
                if self.places.contains_key(id) {
                    return Err(bad(at, "two nodes with one id"));
                }
                self.places.insert(id.to_owned(), self.nodes.len());
                self.nodes.push((id.to_owned(), None));
                self.open.push(self.nodes.len() - 1);
            }
            "link" => {
                let Some(&node) = self.open.last() else {
                    return Err(bad(at, "a link outside a node"));
                };
                let targets = split_on_spaces(attr(a, "targets").ok_or(bad(at, "a link has no targets"))?);
                let [target] = targets[..] else {
                    return Err(bad(at, "a logical node links to several regions"));
                };
                if !region_id(target) {
                    return Err(bad(at, "a link's target is not a logical region"));
                }
                if self.nodes[node].1.replace(target.to_owned()).is_some() {
                    return Err(bad(at, "a logical node has two links"));
                }
            }
            "a" => {
                let node = attr(a, "ref").ok_or(bad(at, "an annotation has no ref"))?;
                if !node_id(node) {
                    return Err(bad(at, "an annotation ref is not a logical node"));
                }
                let label = attr(a, "label").ok_or(bad(at, "an annotation has no label"))?;
                if self.labels.insert(node.to_owned(), label.to_owned()).is_some() {
                    return Err(bad(at, "two annotations for one logical node"));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn end_element(&mut self, name: &str, _: usize) -> turbo_maxent::Result<()> {
        if name == "node" {
            self.open.pop();
        }
        Ok(())
    }

    fn characters(&mut self, _: &str) {}
}

fn parse_logical(bytes: &[u8]) -> Result<Vec<MascLogical>, MascError> {
    let mut file = LogicalFile::default();
    read_xml(bytes, AnnotationFile::Logical, &mut file)?;
    let fail = |message: &'static str| MascError::Annotation { file: AnnotationFile::Logical, cause: bad(0, message) };
    if file.labels.keys().any(|id| !file.places.contains_key(id)) {
        return Err(fail("an annotation names a node the file does not have"));
    }
    let mut out = Vec::with_capacity(file.nodes.len());
    for (id, target) in &file.nodes {
        let Some(target) = target else {
            return Err(fail("a logical node has no link"));
        };
        let Some(&(start, end)) = file.regions.get(target) else {
            return Err(fail("a link names a region the file does not have"));
        };
        let Some(label) = file.labels.get(id) else {
            return Err(fail("a logical node has no annotation"));
        };
        out.push(MascLogical { id: id.clone(), label: label.clone(), start, end });
    }
    out.sort_by(|a, b| (a.start, std::cmp::Reverse(a.end), &a.id).cmp(&(b.start, std::cmp::Reverse(b.end), &b.id)));
    Ok(out)
}

impl MascDocument {
    /// Reads the logical markup file and keeps each labelled node, in
    /// text order ([`MascDocument::logical`]). A missing
    /// file (`None`) leaves the document without logical markup, as a
    /// document whose header names no `f.logical` is read.
    ///
    /// Adding the markup again replaces it.
    ///
    /// # Errors
    ///
    /// [`MascError::Annotation`] for a file that is not XML, or whose
    /// references do not resolve: a region, node or annotation id of
    /// another form, a repeated id, a node with no link or two, a link
    /// to several regions or to one the file does not have, an
    /// annotation with no label or naming a node the file does not
    /// have, or a node no annotation labels.
    pub fn add_logical(&mut self, file: Option<&[u8]>) -> Result<(), MascError> {
        self.logical.clear();
        self.has_logical = false;
        let Some(bytes) = file else { return Ok(()) };
        self.logical = parse_logical(bytes)?;
        self.has_logical = true;
        Ok(())
    }

    /// The logical markup, in text order (by start, then by the later
    /// end first, then by node id); empty unless it was read
    /// ([`MascDocument::add_logical`]).
    pub fn logical(&self) -> &[MascLogical] {
        &self.logical
    }

    /// Whether the logical markup was read (`add_logical` with a file).
    pub fn has_logical(&self) -> bool {
        self.has_logical
    }

    /// The document's logical blocks as a [`Structure`] layer over the
    /// source text: a `cesDoc` block becomes a section, a `head` a
    /// heading of level 1 and a `p` a paragraph, nested by containment
    /// so the section holds the headings and paragraphs inside it.
    /// Blocks of any other label, and empty blocks, stay in
    /// [`MascDocument::logical`] and are not converted. The anchors are
    /// converted from UTF-16 units
    /// with [`MascDocument::span`], as every other offset of the
    /// document is, so the layer's spans are UTF-8 byte spans of view
    /// 0 and `StructureProjector` can carry them to a normalized view.
    /// A document without logical markup converts to an empty layer.
    ///
    /// Put the layer on an analysis document of the same text with the
    /// provenance of its source, for example
    /// `Provenance::by("masc:logical")`.
    ///
    /// # Errors
    ///
    /// [`MascError::Substring`] for an anchor outside the text and
    /// [`MascError::SplitsSurrogatePair`] for one inside a surrogate
    /// pair, as [`MascDocument::span`] fails, and
    /// [`MascError::LogicalStructure`] when two converted blocks cross
    /// (one starts inside another and ends outside it), so the blocks
    /// form no tree.
    pub fn structure(&self) -> Result<Structure, MascError> {
        type BlockRow<'a> = (turbo_analysis::Span, Block, u8, &'a str, &'a str);
        let mut blocks: Vec<BlockRow<'_>> = Vec::new();
        for block in &self.logical {
            let (kind, depth) = match block.label.as_str() {
                "cesDoc" => (Block::Section, 1),
                "head" => (Block::Heading, 1),
                "p" => (Block::Paragraph, 0),
                _ => continue,
            };
            // An empty region holds no text and is no block; the corpus
            // has such heads and paragraphs. It stays in `logical()`.
            if block.start == block.end {
                continue;
            }
            let span = self.span(i64::from(block.start), i64::from(block.end))?;
            blocks.push((span, kind, depth, &block.id, &block.label));
        }
        blocks.sort_by(|a, b| {
            (a.0.start, std::cmp::Reverse(a.0.end), a.3).cmp(&(b.0.start, std::cmp::Reverse(b.0.end), b.3))
        });
        let (mut spans, mut kind, mut parent, mut depth) =
            (Spans::new(ViewId::SOURCE), Vec::new(), Vec::new(), Vec::new());
        // The open blocks, outermost first: their places in the layer.
        let mut open: Vec<usize> = Vec::new();
        for (span, k, d, id, label) in blocks {
            while let Some(&top) = open.last() {
                let outer = spans.get(Ix::new(top as u32));
                if span.end <= outer.end {
                    break;
                }
                if span.start < outer.end {
                    return Err(MascError::LogicalStructure {
                        detail: format!("logical block {id} ({label}) crosses the block that starts it"),
                    });
                }
                open.pop();
            }
            parent.push(open.last().map(|&i| Ix::new(i as u32)));
            spans.push(span);
            kind.push(k);
            depth.push(d);
            open.push(kind.len() - 1);
        }
        let structure = Structure { spans, kind, parent, depth };
        structure.check().map_err(|detail| MascError::LogicalStructure { detail })?;
        Ok(structure)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::masc::MascFiles;

    /// The length of `s` in UTF-16 units, the units of every anchor.
    fn units(s: &str) -> usize {
        s.encode_utf16().count()
    }

    /// A document of `text` (one quark and one sentence over the whole
    /// text) with this logical file, if any.
    fn document(text: &str, logical: Option<&str>) -> Result<MascDocument, MascError> {
        let len = units(text);
        let seg = format!("<graph><region xml:id=\"seg-r0\" anchors=\"0 {len}\"/></graph>");
        let s = format!("<graph><region anchors=\"0 {len}\"/></graph>");
        let mut doc = MascDocument::parse(
            "t.hdr",
            MascFiles {
                text: text.as_bytes(),
                segments: Some(seg.as_bytes()),
                sentences: Some(s.as_bytes()),
                penn: None,
                named_entities: None,
            },
        )?;
        doc.add_logical(logical.map(str::as_bytes))?;
        Ok(doc)
    }

    /// A logical file of one block per (label, start, end), its regions,
    /// nodes and annotations written last to first.
    fn markup(blocks: &[(&str, i32, i32)]) -> String {
        let mut xml = String::from("<graph>");
        for (n, (label, _, _)) in blocks.iter().enumerate().rev() {
            xml.push_str(&format!("<a label=\"{label}\" ref=\"logical-n{n}\"/>"));
        }
        for (n, (_, start, end)) in blocks.iter().enumerate().rev() {
            xml.push_str(&format!(
                "<node xml:id=\"logical-n{n}\"><link targets=\"logical-r{n}\"/></node>\
                 <region xml:id=\"logical-r{n}\" anchors=\"{start} {end}\"/>"
            ));
        }
        xml.push_str("</graph>");
        xml
    }

    #[test]
    fn no_file_means_no_markup_and_an_empty_structure() {
        let doc = document("Some text.", None).unwrap();
        assert!(!doc.has_logical());
        assert!(doc.logical().is_empty());
        assert_eq!(doc.structure().unwrap().len(), 0);
    }

    #[test]
    fn unicode_anchors_convert_by_utf16_units_not_bytes() {
        // U+1D11E is 2 UTF-16 units and 4 UTF-8 bytes, an e-acute is 1
        // unit and 2 bytes, so unit and byte offsets part ways at once.
        let text = "𝄞 Café\n\nA paragraph.";
        let head_end = units("𝄞 Café") as i32;
        let para = (head_end + 2, units(text) as i32);
        let xml = markup(&[("head", 0, head_end), ("p", para.0, para.1)]);
        let doc = document(text, Some(&xml)).unwrap();
        assert_eq!(
            doc.logical()[0],
            MascLogical { id: "logical-n0".into(), label: "head".into(), start: 0, end: head_end }
        );
        let structure = doc.structure().unwrap();
        let texts: Vec<&str> = structure.spans.iter().map(|(_, s)| &text[s.range()]).collect();
        assert_eq!(texts, ["𝄞 Café", "A paragraph."]);
    }

    #[test]
    fn anchors_inside_a_surrogate_pair_or_past_the_text_are_refused_at_conversion() {
        let text = "𝄞 music";
        // Unit 1 is the second half of the pair for U+1D11E.
        let xml = markup(&[("p", 1, 4)]);
        let doc = document(text, Some(&xml)).unwrap();
        assert!(matches!(doc.structure(), Err(MascError::SplitsSurrogatePair { unit: 1 })));
        let xml = markup(&[("p", 0, 99)]);
        let doc = document(text, Some(&xml)).unwrap();
        assert!(matches!(doc.structure(), Err(MascError::Substring { .. })));
    }

    #[test]
    fn other_labels_are_kept_but_not_converted() {
        let text = "Hello there.\n\nGeneral.";
        let xml = markup(&[("u", 0, 12), ("p", 14, 22)]);
        let doc = document(text, Some(&xml)).unwrap();
        assert_eq!(doc.logical().len(), 2);
        let structure = doc.structure().unwrap();
        assert_eq!(structure.kind, [Block::Paragraph]);
        assert_eq!(&text[structure.spans.get(Ix::new(0)).range()], "General.");
    }

    #[test]
    fn crossing_blocks_form_no_tree_and_empty_blocks_are_skipped() {
        let text = "0123456789";
        let xml = markup(&[("cesDoc", 0, 6), ("p", 4, 9)]);
        let doc = document(text, Some(&xml)).unwrap();
        assert!(matches!(doc.structure(), Err(MascError::LogicalStructure { .. })));
        let xml = markup(&[("p", 3, 3), ("head", 0, 2)]);
        let doc = document(text, Some(&xml)).unwrap();
        assert_eq!(doc.logical().len(), 2);
        let structure = doc.structure().unwrap();
        assert_eq!(structure.kind, [Block::Heading]);
    }

    #[test]
    fn malformed_references_are_refused() {
        let region = "<region xml:id=\"logical-r0\" anchors=\"0 4\"/>";
        let node = "<node xml:id=\"logical-n0\"><link targets=\"logical-r0\"/></node>";
        let annotation = "<a label=\"p\" ref=\"logical-n0\"/>";
        let cases: Vec<String> = vec![
            // a link to a region the file does not have
            format!("<graph>{node}{annotation}</graph>"),
            // an annotation naming a node the file does not have
            format!("<graph>{region}{node}<a label=\"p\" ref=\"logical-n9\"/></graph>"),
            // a node with no link
            format!("<graph>{region}<node xml:id=\"logical-n0\"/>{annotation}</graph>"),
            // a node no annotation labels
            format!("<graph>{region}{node}</graph>"),
            // two regions with one id
            format!("<graph>{region}{region}{node}{annotation}</graph>"),
            // a node with two links
            format!(
                "<graph>{region}<node xml:id=\"logical-n0\">\
                 <link targets=\"logical-r0\"/><link targets=\"logical-r0\"/></node>{annotation}</graph>"
            ),
            // a link to several regions
            format!(
                "<graph>{region}<region xml:id=\"logical-r1\" anchors=\"5 9\"/>\
                 <node xml:id=\"logical-n0\"><link targets=\"logical-r0 logical-r1\"/></node>{annotation}</graph>"
            ),
            // an annotation with no label
            format!("<graph>{region}{node}<a ref=\"logical-n0\"/></graph>"),
            // ids of another form
            "<graph><region xml:id=\"seg-r0\" anchors=\"0 4\"/></graph>".to_owned(),
        ];
        for xml in &cases {
            let err = document("Some text here.", Some(xml)).unwrap_err();
            assert!(matches!(err, MascError::Annotation { file: AnnotationFile::Logical, .. }), "{xml}: {err}");
            assert!(err.skips_document(), "{xml}: {err}");
        }
    }

    #[test]
    fn adding_the_markup_again_replaces_it() {
        let text = "One block.";
        let xml = markup(&[("p", 0, 10)]);
        let mut doc = document(text, Some(&xml)).unwrap();
        assert_eq!(doc.logical().len(), 1);
        doc.add_logical(None).unwrap();
        assert!(!doc.has_logical());
        assert!(doc.logical().is_empty());
    }
}
