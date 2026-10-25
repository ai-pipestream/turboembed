use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Analyzer, Bound, Document, Ix, Key, View};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{ColumnAlign, KeyValuePairs, Tables};
use turbo_normalize::{NormalizeStep, UnicodeForm};
use turbo_structure::{KeyValueAnnotator, TableGridAnnotator};

fn document(text: &str) -> Document {
    let pipeline = Analyzer::builder()
        .step(Bound::new(TableGridAnnotator::default()))
        .step(Bound::new(KeyValueAnnotator::default()))
        .build()
        .unwrap();
    let mut docs = [Document::new(text)];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    docs.into_iter().next().unwrap()
}

#[test]
fn gfm_alignment_escaped_pipes_empty_cells_and_headers() {
    let text = "| Metric | Q1 2025 | Q2 2025 |\n| :--- | :---: | ---: |\n| Revenue \\| net | 12 | |\n";
    let doc = document(text);
    let t = doc.get::<Tables>().unwrap();
    assert_eq!(t.table_bounds.len(), 1);
    assert_eq!(t.cells.len(), 6);
    assert_eq!(
        t.cells.iter().map(|c| c.align).collect::<Vec<_>>(),
        [ColumnAlign::Left, ColumnAlign::Center, ColumnAlign::Right].repeat(2)
    );
    assert_eq!(
        t.cells_in_table(0).map(|c| &text[c.span.range()]).collect::<Vec<_>>(),
        ["Metric", "Q1 2025", "Q2 2025", "Revenue \\| net", "12", ""]
    );
    let (column, row) = t.governing_headers(&t.cells[4]);
    assert_eq!(&text[column.unwrap().span.range()], "Q1 2025");
    assert_eq!(&text[row.unwrap().span.range()], "Revenue \\| net");
    assert_eq!(t.headers_for_table(0).count(), 3);
    assert_eq!(t.rows.len(), 2);
}

#[test]
fn ascii_grid_headers_multiline_utf8_and_empty_cells() {
    let text = "+--------+--------+\r\n| City   | Sales  |\r\n+========+========+\r\n| Zürich |        |\r\n| centre |        |\r\n+--------+--------+\r\n";
    let doc = document(text);
    let t = doc.get::<Tables>().unwrap();
    assert_eq!(t.cells.len(), 4);
    assert_eq!(t.cells.iter().filter(|c| c.is_header).count(), 2);
    let fragments: Vec<_> = t.spans.fragments(Ix::new(2)).map(|s| &text[s.range()]).collect();
    assert_eq!(fragments, ["Zürich", "centre"]);
    assert_eq!(t.spans.get(Ix::new(3)).start, t.spans.get(Ix::new(3)).end);
}

#[test]
fn multiline_grid_cell_source_order_can_differ_from_column_order() {
    let text = "+--------+--------+\n| City   | Sales  |\n+========+========+\n|        | 12     |\n| Zürich |        |\n+--------+--------+\n";
    let doc = document(text);
    let tables = doc.get::<Tables>().unwrap();
    let logical: Vec<_> = tables.cells_in_table(0).collect();
    assert_eq!(logical.iter().map(|c| c.cell.col_ix).collect::<Vec<_>>(), [0, 1, 0, 1]);
    assert_eq!(&text[logical[2].span.range()], "Zürich");
    assert_eq!(&text[logical[3].span.range()], "12");
    assert!(logical[2].index > logical[3].index, "storage follows source, the query follows coordinates");
    let graph = export(&doc, &codecs()).unwrap();
    let restored = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(export(&restored, &codecs()).unwrap(), graph);
}

#[test]
fn key_values_refuse_urls_timestamps_long_prose_and_code() {
    let text = "Invoice Date: 2026-03-15\nTicker = ACME\nURL: https://example.org/a\nhttps://example.org/a\n12:30\nThis long prose sentence has far too many words: no\n```text\nHidden: value\n| fake |\n| ---- |\n```\n";
    let doc = document(text);
    let kv = doc.get::<KeyValuePairs>().unwrap();
    assert_eq!(
        kv.pairs
            .iter()
            .map(|p| (&text[p.key.range()], &text[p.value.range()], &text[p.separator.range()]))
            .collect::<Vec<_>>(),
        [("Invoice Date", "2026-03-15", ":"), ("Ticker", "ACME", "="), ("URL", "https://example.org/a", ":")]
    );
    assert!(doc.get::<Tables>().unwrap().cells.is_empty());
}

#[test]
fn production_layers_round_trip_with_source_and_provenance() {
    let text = "Ticker: ACME\n\n| Metric | Q1 |\n| --- | ---: |\n| Revenue | 12 |\n";
    let doc = document(text);
    let graph = export(&doc, &codecs()).unwrap();
    let bytes = graph.to_bytes().unwrap();
    let graph = DocumentGraph::from_bytes(&bytes).unwrap();
    let restored = import(&graph, &codecs()).unwrap();
    assert_eq!(export(&restored, &codecs()).unwrap(), graph);
    assert_eq!(restored.get::<Tables>().unwrap().cells, doc.get::<Tables>().unwrap().cells);
    assert_eq!(restored.get::<KeyValuePairs>().unwrap().pairs, doc.get::<KeyValuePairs>().unwrap().pairs);
    for key in [Key::<Tables>::DEFAULT.id(), Key::<KeyValuePairs>::DEFAULT.id()] {
        assert_eq!(doc.provenance(key), restored.provenance(key));
    }
}

#[test]
fn repository_analysis_document_uses_the_production_grid_and_codec_path() {
    let text = include_str!("../../../docs/analysis.md");
    let doc = document(text);
    let tables = doc.get::<Tables>().unwrap();
    assert!(tables.table_bounds.len() > 10);
    assert!(tables.cells.len() > 100);
    for (_, span) in tables.spans.iter() {
        assert!(text.get(span.range()).is_some());
    }
    let graph = export(&doc, &codecs()).unwrap();
    let restored = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(export(&restored, &codecs()).unwrap(), graph);
}

#[test]
fn secondary_coordinates_follow_the_named_normalized_view_through_binary() {
    let text = "Cafe\u{301}: value\n\n| City | Sales |\n| --- | ---: |\n| Montre\u{301}al | 12 |\n";
    let pipeline = Analyzer::builder()
        .step(Bound::new(NormalizeStep::new(UnicodeForm::nfc())).write(0, "nfc"))
        .step(Bound::new(TableGridAnnotator::default()).read(0, "nfc"))
        .step(Bound::new(KeyValueAnnotator::default()).read(0, "nfc"))
        .build()
        .unwrap();
    let mut docs = [Document::new(text)];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let doc = &docs[0];
    let view = doc.get_key(&Key::<View>::named("nfc")).unwrap();
    let tables = doc.get::<Tables>().unwrap();
    let pairs = doc.get::<KeyValuePairs>().unwrap();
    assert_eq!(tables.spans.view(), view.id());
    assert_eq!(tables.rows.view(), view.id());
    assert_eq!(tables.table_bounds.view(), view.id());
    assert_eq!(pairs.spans.view(), view.id());
    assert_eq!(&view.text()[pairs.pairs[0].key.range()], "Café");
    assert_eq!(&view.text()[tables.spans.get(Ix::new(2)).range()], "Montréal");
    let graph = export(doc, &codecs()).unwrap();
    let restored = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(export(&restored, &codecs()).unwrap(), graph);
}
