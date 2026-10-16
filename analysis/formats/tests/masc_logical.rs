//! The MASC logical annotation reader (#498): headings and paragraphs
//! from a `*-logical.xml` stand-off file become `Structure` blocks on the
//! source view, and the structure projector carries them to a normalized
//! view. Fixtures are written for the tests in the MASC layout; the one
//! corpus test runs only when `TURBO_TEST_MASC` names a MASC 3.0.0
//! `data` directory.

use std::path::PathBuf;

use turbo_analysis::{Analyzer, Bound, Document, Ix, Key, Layer, Provenance};
use turbo_analysis_layers::{Block, Structure};
use turbo_formats::masc::{MascDocument, MascFiles, read_document_with_logical};
use turbo_normalize::{NormalizeStep, Whitespace};
use turbo_structure::StructureProjector;

const TEXT: &str = "DUBLIN\n\nA first paragraph here.\n\nSIGHTS\n\nA second one follows.\n";

/// A logical file for `TEXT` with its regions, nodes and annotations in
/// shuffled order, as the real corpus writes them (n0, n1, n10 style).
fn logical() -> String {
    let doc = (0, TEXT.len());
    let head1 = (0, 6);
    let p1 = (8, 31);
    let head2 = (33, 39);
    let p2 = (41, 62);
    format!(
        "<graph>\
         <a xml:id=\"logical-a3\" label=\"head\" ref=\"logical-n3\"/>\
         <node xml:id=\"logical-n2\"><link targets=\"logical-r2\"/></node>\
         <region xml:id=\"logical-r1\" anchors=\"{} {}\"/>\
         <a xml:id=\"logical-a1\" label=\"head\" ref=\"logical-n1\"/>\
         <node xml:id=\"logical-doc-n\"><link targets=\"logical-doc-r\"/></node>\
         <region xml:id=\"logical-doc-r\" anchors=\"{} {}\"/>\
         <node xml:id=\"logical-n3\"><link targets=\"logical-r3\"/></node>\
         <region xml:id=\"logical-r3\" anchors=\"{} {}\"/>\
         <a xml:id=\"logical-doc-a\" label=\"cesDoc\" ref=\"logical-doc-n\"/>\
         <node xml:id=\"logical-n1\"><link targets=\"logical-r1\"/></node>\
         <region xml:id=\"logical-r2\" anchors=\"{} {}\"/>\
         <a xml:id=\"logical-a2\" label=\"p\" ref=\"logical-n2\"/>\
         <node xml:id=\"logical-n4\"><link targets=\"logical-r4\"/></node>\
         <region xml:id=\"logical-r4\" anchors=\"{} {}\"/>\
         <a xml:id=\"logical-a4\" label=\"p\" ref=\"logical-n4\"/>\
         </graph>",
        head1.0, head1.1, doc.0, doc.1, head2.0, head2.1, p1.0, p1.1, p2.0, p2.1,
    )
}

fn document() -> MascDocument {
    let xml = logical();
    let seg = format!("<graph><region xml:id=\"seg-r0\" anchors=\"0 {}\"/></graph>", TEXT.len());
    let s = format!("<graph><region anchors=\"0 {}\"/></graph>", TEXT.len());
    let mut doc = MascDocument::parse(
        "t.hdr",
        MascFiles {
            text: TEXT.as_bytes(),
            segments: Some(seg.as_bytes()),
            sentences: Some(s.as_bytes()),
            penn: None,
            named_entities: None,
        },
    )
    .unwrap();
    doc.add_logical(Some(xml.as_bytes())).unwrap();
    doc
}

#[test]
fn logical_blocks_become_a_structure_tree() {
    let doc = document();
    assert!(doc.has_logical());
    let labels: Vec<&str> = doc.logical().iter().map(|l| l.label.as_str()).collect();
    assert_eq!(labels, ["cesDoc", "head", "p", "head", "p"]);
    let structure = doc.structure().unwrap();
    structure.check().unwrap();
    assert_eq!(structure.kind, [Block::Section, Block::Heading, Block::Paragraph, Block::Heading, Block::Paragraph]);
    assert_eq!(structure.parent, [None, Some(Ix::new(0)), Some(Ix::new(0)), Some(Ix::new(0)), Some(Ix::new(0))]);
    let texts: Vec<&str> = structure.spans.iter().map(|(_, s)| &TEXT[s.range()]).collect();
    assert_eq!(texts, [TEXT, "DUBLIN", "A first paragraph here.", "SIGHTS", "A second one follows."]);
}

#[test]
fn imported_structure_projects_to_a_normalized_view_with_provenance() {
    let masc = document();
    let mut doc = Document::new(masc.text().to_owned());
    doc.add(&Key::<Structure>::named("source"), masc.structure().unwrap(), Provenance::by("masc:logical")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<Structure>::named("source"))
        .step(Bound::new(NormalizeStep::new(Whitespace::collapse())).write(0, "clean"))
        .step(Bound::new(StructureProjector).read(0, "clean").read(1, "source"))
        .build()
        .unwrap();
    let mut docs = [doc];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let doc = &docs[0];
    let projected = doc.get::<Structure>().unwrap();
    let clean = doc.text(projected.spans.view());
    let texts: Vec<&str> = projected.spans.iter().map(|(_, s)| &clean[s.range()]).collect();
    assert_eq!(texts[1..], ["DUBLIN", "A first paragraph here.", "SIGHTS", "A second one follows."]);
    let source = doc.provenance(Key::<Structure>::named("source").id()).unwrap();
    assert_eq!(source.annotator, "masc:logical");
}

/// The real corpus check of #498: with `TURBO_TEST_MASC` set to a MASC
/// 3.0.0 `data` directory, IntroDublin imports as one document section
/// holding 5 headings and 15 paragraphs, each span inside the text.
#[test]
fn intro_dublin_logical_import() {
    let Some(root) = std::env::var_os("TURBO_TEST_MASC").map(PathBuf::from) else { return };
    let hdr = root.join("written/travel-guides/IntroDublin.hdr");
    let doc = read_document_with_logical(&hdr).unwrap();
    assert!(doc.has_logical());
    let structure = doc.structure().unwrap();
    structure.check().unwrap();
    let count = |k: Block| structure.kind.iter().filter(|&&x| x == k).count();
    assert_eq!((count(Block::Section), count(Block::Heading), count(Block::Paragraph)), (1, 5, 15));
    for (_, span) in structure.spans.iter() {
        assert!(doc.text().get(span.range()).is_some(), "span {span:?} outside the text");
    }
}
