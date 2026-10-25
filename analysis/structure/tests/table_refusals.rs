use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Annotator, Document, Ix, Key, Layer, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{KeyValuePair, KeyValuePairs, Tables};
use turbo_structure::{KeyValueAnnotator, TableGridAnnotator};

fn parsed(text: &str) -> (Document, Tables) {
    let doc = Document::new(text);
    let (tables,) = TableGridAnnotator::default().annotate((doc.get::<View>().unwrap(),)).unwrap();
    (doc, tables)
}

#[test]
fn secondary_bounds_and_rows_are_checked_against_the_view() {
    for limit in [1, u32::MAX] {
        let (mut doc, mut tables) = parsed("é\n| A |\n| --- |\n| B |\n");
        let end = tables.table_bounds.get(Ix::new(0)).end;
        let mut bounds = Spans::new(ViewId::SOURCE);
        let mut rows = Spans::new(ViewId::SOURCE);
        let row0 = tables.rows.get(Ix::new(0));
        let row1 = tables.rows.get(Ix::new(1));
        let (start, finish) = if limit == 1 { (1, end) } else { (row0.start, limit) };
        bounds.push(Span::new(start, finish));
        rows.push(Span::new(start, row0.end));
        rows.push(Span::new(row1.start, finish));
        tables.table_bounds = bounds;
        tables.rows = rows;
        assert!(tables.check().is_ok(), "the primary cells and logical coordinates remain valid");
        let error = doc.add(&Key::DEFAULT, tables, Provenance::by("test")).unwrap_err();
        assert!(error.to_string().contains("UTF-8"), "{error}");
    }
}

#[test]
fn embedded_key_value_spans_are_validated_on_add_and_wire_import() {
    let text = "é: value";
    let mut doc = Document::new(text);
    let (valid,) = KeyValueAnnotator::default().annotate((doc.get::<View>().unwrap(),)).unwrap();
    let mut invalid = valid.clone();
    invalid.pairs[0].key = Span::new(1, 2);
    assert!(invalid.check().is_ok());
    assert!(doc.add(&Key::DEFAULT, invalid, Provenance::by("test")).is_err());
    doc.add(&Key::DEFAULT, valid, Provenance::by("test")).unwrap();
    let mut graph = export(&doc, &codecs()).unwrap();
    let body = graph.layers.iter_mut().find(|l| l.layer == KeyValuePairs::ID).unwrap().body.as_mut().unwrap();
    let start = body.len() - 24;
    body[start..start + 4].copy_from_slice(&1u32.to_le_bytes());
    let graph = DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap();
    assert!(import(&graph, &codecs()).is_err());
}

#[test]
fn malformed_components_and_table_ids_are_refused() {
    let mut pairs = KeyValuePairs::new(ViewId::SOURCE);
    pairs.spans.push(Span::new(0, 8));
    pairs.pairs.push(KeyValuePair {
        key: Span::new(0, 2),
        separator: Span { start: 4, end: 3 },
        value: Span::new(5, 8),
    });
    assert!(pairs.check().is_err());
    let (mut doc, tables) = parsed("| A | B |\n| --- | --- |\n| 1 | 2 |\n");
    doc.add(&Key::DEFAULT, tables, Provenance::by("test")).unwrap();
    let graph = export(&doc, &codecs()).unwrap();
    for (offset, value) in [(0, 99), (8, 0), (10, 2), (11, 9)] {
        let mut bad = graph.clone();
        let body = bad.layers.iter_mut().find(|l| l.layer == Tables::ID).unwrap().body.as_mut().unwrap();
        let at = body.len() - 4 * 12 + offset;
        body[at] = value;
        let bad = DocumentGraph::from_bytes(&bad.to_bytes().unwrap()).unwrap();
        assert!(import(&bad, &codecs()).is_err(), "invalid metadata at {offset}");
    }
}

#[test]
fn duplicate_coordinates_cross_view_columns_and_outside_cells_fail() {
    let (_, original) = parsed("| A | B |\n| --- | --- |\n| 1 | 2 |\n");
    let mut duplicate = original.clone();
    duplicate.cells[1].col_ix = 0;
    assert!(duplicate.check().is_err());
    let mut cross_view = original.clone();
    cross_view.rows = Spans::new(ViewId(99));
    for (_, row) in original.rows.iter() {
        cross_view.rows.push(row);
    }
    assert!(cross_view.check().is_err());
    let mut outside = original;
    outside.cells[0].row_span = 3;
    assert!(outside.check().is_err());
}

#[test]
fn secondary_table_utf8_and_outside_coordinates_are_rejected_from_binary() {
    let text = "é\n| A |\n| --- |\n| B |\n";
    let (mut doc, tables) = parsed(text);
    let bounds = tables.table_bounds.get(Ix::new(0));
    let first = tables.rows.get(Ix::new(0));
    let last = tables.rows.get(Ix::new(1));
    doc.add(&Key::DEFAULT, tables, Provenance::by("test")).unwrap();
    let graph = export(&doc, &codecs()).unwrap();
    for outside in [false, true] {
        let mut bad = graph.clone();
        let body = bad.layers.iter_mut().find(|l| l.layer == Tables::ID).unwrap().body.as_mut().unwrap();
        let targets = if outside { [bounds, last] } else { [bounds, first] };
        let positions: Vec<_> = targets
            .iter()
            .map(|s| {
                let bytes: Vec<_> = s.start.to_le_bytes().into_iter().chain(s.end.to_le_bytes()).collect();
                let matches: Vec<_> =
                    body.windows(8).enumerate().filter(|(_, b)| *b == bytes).map(|(i, _)| i).collect();
                assert_eq!(matches.len(), 1);
                matches[0]
            })
            .collect();
        for position in positions {
            let (at, value) = if outside { (position + 4, u32::MAX) } else { (position, 1) };
            body[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        let bad = DocumentGraph::from_bytes(&bad.to_bytes().unwrap()).unwrap();
        let error = import(&bad, &codecs()).err().unwrap();
        assert!(error.to_string().contains("UTF-8"), "{error}");
    }
}
