use turbo_analysis::*;
use turbo_analysis_layers::{Block, Structure};
use turbo_normalize::{NormalizeStep, Whitespace};
use turbo_structure::{StructureAnnotator, StructureProjector, Syntax, detect};

// Markdown document text, including literal Markdown inside an HTML block.
const NOTES: &str = "# Field notes\r\n\r\n<div class=\"notice\">\r\n**Keep** the café open.\r\n</div>\r\n\r\n- Coffee\r\n\r\n- Tea\r\n\r\n[^source]: Café register.\r\n\r\n    Checked on Friday.\r\n\r\nEnd of notes.\r\n";

#[test]
fn html_footnotes_and_loose_lists_keep_exact_spans() {
    let s = detect(NOTES, Syntax::Markdown, ViewId::SOURCE);
    s.check().unwrap();
    let texts: Vec<_> = s.spans.iter().map(|(i, span)| (s.kind[i.get()], &NOTES[span.range()])).collect();
    assert!(texts.contains(&(Block::Html, "<div class=\"notice\">\r\n**Keep** the café open.\r\n</div>")));
    assert!(texts.contains(&(Block::LooseList, "- Coffee\r\n\r\n- Tea")));
    assert!(texts.contains(&(Block::Footnote, "[^source]: Café register.\r\n\r\n    Checked on Friday.")));
    assert!(texts.contains(&(Block::Paragraph, "End of notes.")));
    assert_eq!(Block::List.list_tightness(), Some(true));
    assert_eq!(Block::LooseList.list_tightness(), Some(false));
    assert_eq!(Block::Html.list_tightness(), None);
}

#[test]
fn new_blocks_project_both_ways_after_whitespace_normalization() {
    for (text, syntax) in [(NOTES, Syntax::Markdown), ("City    Count\nParis   12\nTokyo   17\n", Syntax::Plain)] {
        let pipeline = Analyzer::builder()
            .step(Bound::new(StructureAnnotator::new(syntax)).write(0, "source"))
            .step(Bound::new(NormalizeStep::new(Whitespace::collapse())).write(0, "clean"))
            .step(Bound::new(StructureProjector).read(0, "clean").read(1, "source"))
            .build()
            .unwrap();
        let mut docs = [Document::new(text)];
        pipeline.run(&mut docs).remove(0).unwrap();
        let doc = &docs[0];
        let original = doc.get_key(&Key::<Structure>::named("source")).unwrap();
        let projected = doc.get::<Structure>().unwrap();
        assert_eq!(original.kind, projected.kind);
        assert_eq!(original.parent, projected.parent);
        assert_eq!(original.depth, projected.depth);
        for ((_, a), (_, b)) in original.spans.iter().zip(projected.spans.iter()) {
            assert_eq!(doc.to_source(projected.spans.view(), b), a);
        }
        let graph = turbo_analysis::export::export(doc, &turbo_analysis_layers::export::codecs()).unwrap();
        let restored = turbo_analysis::export::import(&graph, &turbo_analysis_layers::export::codecs()).unwrap();
        assert_eq!(restored.get::<Structure>().unwrap().kind, projected.kind);
    }
}

#[test]
fn aligned_columns_handle_unicode_crlf_and_stop_at_unaligned_text() {
    let text = "City    Count\r\nParis   12\r\nCafé    17\r\nThis is a paragraph.\r\n";
    let s = detect(text, Syntax::Plain, ViewId::SOURCE);
    s.check().unwrap();
    let cells: Vec<_> =
        s.spans.iter().filter(|(i, _)| s.kind[i.get()] == Block::TableCell).map(|(_, s)| &text[s.range()]).collect();
    assert_eq!(cells, ["City", "Count", "Paris", "12", "Café", "17"]);
    assert_eq!(s.kind.iter().filter(|&&k| k == Block::TableRow).count(), 3);
    assert_eq!(s.kind.last(), Some(&Block::Paragraph));
    assert_eq!(detect("A  line.\nAnother line.\n", Syntax::Plain, ViewId::SOURCE).kind, [Block::Paragraph]);
    let tabs = detect("City\tCount\nRome\t12\n", Syntax::Plain, ViewId::SOURCE);
    assert_eq!(tabs.kind[0], Block::Table);
}

#[test]
fn nested_and_opaque_blocks_do_not_leak_their_syntax() {
    for line in [
        "<script>\n# code\n</SCRIPT>",
        "<!--\n- text\n-->",
        "<?php\n# code\n?>",
        "<!DOCTYPE\nhtml>",
        "<![CDATA[\n# text\n]]>",
        "<custom a='x'>\n# text",
    ] {
        let s = detect(line, Syntax::Markdown, ViewId::SOURCE);
        assert_eq!(s.kind, [Block::Html], "{line}");
        assert_eq!(&line[s.spans.get(Ix::new(0)).range()], line);
    }
    let s = detect("- Coffee\n  > quoted\n  >\n- Tea\n", Syntax::Markdown, ViewId::SOURCE);
    assert_eq!(s.kind[0], Block::List);
    let s = detect("- ```\n  a\n\n  b\n  ```\n- tea", Syntax::Markdown, ViewId::SOURCE);
    assert_eq!(s.kind[0], Block::List);
    let s = detect("> [^x]: Notes.\n>\n>     More notes.\n>\n> End.", Syntax::Markdown, ViewId::SOURCE);
    assert_eq!(s.kind, [Block::Quote, Block::Footnote, Block::Paragraph]);
}
