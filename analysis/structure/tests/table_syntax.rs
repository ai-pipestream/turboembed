use turbo_analysis::{Annotator, Document, Layer, View, ViewId};
use turbo_analysis_layers::{ColumnAlign, Tables};
use turbo_structure::{KeyValueAnnotator, TableGridAnnotator, detect_tables};

#[test]
fn gfm_short_long_rows_block_termination_and_container_offsets() {
    for prefix in ["", "> ", "  "] {
        let text = format!(
            "{prefix}| Metric | Value |\n{prefix}| :--- | ---: |\n{prefix}| Revenue |\n{prefix}| Cost | 3 | ignored |\n\n# End\n"
        );
        let t = detect_tables(&text, ViewId::SOURCE).unwrap();
        assert_eq!(t.cells.len(), 6);
        assert_eq!(t.cells[1].align, ColumnAlign::Right);
        assert_eq!(
            t.cells_in_table(0).map(|c| &text[c.span.range()]).collect::<Vec<_>>(),
            ["Metric", "Value", "Revenue", "", "Cost", "3"]
        );
        assert!(t.check().is_ok());
    }
    let text = "| A | B |\n| --- | --- |\nvalue\n> another block\n";
    let t = detect_tables(text, ViewId::SOURCE).unwrap();
    assert_eq!(t.cells.len(), 4);
    assert_eq!(t.rows.len(), 2);
}

#[test]
fn gfm_unescaped_code_pipes_delimit_and_escaped_pipes_do_not() {
    let text = "| `a | b` | c |\n| --- | --- | --- |\n| x \\| y | z | |\n";
    let t = detect_tables(text, ViewId::SOURCE).unwrap();
    assert_eq!(
        t.cells_in_table(0).map(|c| &text[c.span.range()]).collect::<Vec<_>>(),
        ["`a", "b`", "c", "x \\| y", "z", ""]
    );
}

#[test]
fn multiple_tables_have_independent_ids_and_grid_headers_are_explicit() {
    let text = "+---+---+\n| a | b |\n+---+---+\n\n| C | D |\n|---|---|\n| 1 | 2 |\n";
    let t = detect_tables(text, ViewId::SOURCE).unwrap();
    assert_eq!(t.table_bounds.len(), 2);
    assert_eq!(t.headers_for_table(0).count(), 0);
    assert_eq!(t.headers_for_table(1).count(), 2);
    assert_eq!(t.cells_in_table(1).count(), 4);
    assert!(t.governing_headers(&t.cells[0]).0.is_none());
}

#[test]
fn whitespace_and_malformed_candidates_never_panic_or_make_invalid_layers() {
    for text in [
        "",
        " ",
        "\t",
        "\n",
        "\r\n",
        "> ",
        "+",
        "||",
        "|A|\n|---|---|\n",
        "+---+\n| unfinished |\n",
        "```\n| A |\n|---|\n```\n",
    ] {
        let mut doc = Document::new(text);
        let (tables,) = TableGridAnnotator::default().annotate((doc.get::<View>().unwrap(),)).unwrap();
        doc.add(&turbo_analysis::Key::<Tables>::DEFAULT, tables, turbo_analysis::Provenance::by("test")).unwrap();
    }
    let a = KeyValueAnnotator::default();
    let b = KeyValueAnnotator { max_key_tokens: 2 };
    assert_eq!(a.config_hash(), KeyValueAnnotator::default().config_hash());
    assert_ne!(a.config_hash(), b.config_hash());
    assert_ne!(a.config_hash(), TableGridAnnotator::default().config_hash());
}

#[test]
fn implied_cells_obey_the_explicit_resource_and_coordinate_limits() {
    let text = "| A | B |\n|---|---|\nonly-one-cell\n";
    let doc = Document::new(text);
    let view = doc.get::<View>().unwrap();
    assert!(TableGridAnnotator { max_cells: 4 }.annotate((view,)).is_ok());
    let error = TableGridAnnotator { max_cells: 3 }.annotate((view,)).err().unwrap();
    assert!(error.to_string().contains("max_cells"));
    assert!(TableGridAnnotator { max_cells: 0 }.annotate((view,)).is_err());
    assert_ne!(TableGridAnnotator { max_cells: 4 }.config_hash(), TableGridAnnotator { max_cells: 3 }.config_hash());
    let huge = format!("{}\n{}\n", "|a".repeat(65537), "|-".repeat(65537));
    assert!(detect_tables(&huge, ViewId::SOURCE).unwrap_err().contains("u16"));
    assert!(detect_tables("Metric \\| value\n---\n", ViewId::SOURCE).unwrap().cells.is_empty());
}
