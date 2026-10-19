//! The ConTEB reader on rows made for the test.

mod common;

use common::parquet::{Col, write};
use common::{copy_dir, data, query_ids, quoted, spans, text};
use turbo_eval::read::conteb;

const DOCS: &str = "documents/train-00000-of-00001.parquet";
const QUERIES: &str = "queries/train-00000-of-00001.parquet";

#[test]
fn reads_the_sample() {
    let s = conteb::read("conteb-sample", &data("conteb"), DOCS, QUERIES).unwrap();
    // Chunks in position order, whatever the row order.
    assert_eq!(
        text(&s, "paperA"),
        "Remdesivir was tested against a placebo.\nThe trial enrolled 120 patients in Zürich."
    );
    assert_eq!(text(&s, "paperB"), "Masks reduced transmission in households.");
    assert_eq!(query_ids(&s), ["0", "1"]);
    assert_eq!(s.queries[0].text, "How many patients were enrolled?");
    assert_eq!(s.qrels.grade("0", "paperA"), 1);
    assert_eq!(s.qrels.grade("0", "paperB"), 0);
    // The second chunk starts after the first and the line break; ü is
    // two bytes.
    assert_eq!(spans(&s, "0", "paperA"), [(41, 84)]);
    assert_eq!(quoted(&s, "0", "paperA"), ["The trial enrolled 120 patients in Zürich."]);
    assert_eq!(quoted(&s, "1", "paperB"), ["Masks reduced transmission in households."]);
}

fn broken(name: &str, file: &str, cols: &[Col]) -> String {
    let dir = common::scratch(name);
    copy_dir(&data("conteb"), &dir);
    write(&dir.join(file), cols);
    conteb::read("c", &dir, DOCS, QUERIES).unwrap_err()
}

#[test]
fn errors_name_the_file_and_row() {
    let e = broken("conteb-id", DOCS, &[Col::Str("chunk_id", vec!["a_0", "a_b_1"]), Col::Str("chunk", vec!["x", "y"])]);
    assert_eq!(e, format!("{DOCS}: row 1: chunk id \"a_b_1\" is not <doc id>_<position>"));
    let e =
        broken("conteb-twice", DOCS, &[Col::Str("chunk_id", vec!["a_0", "a_0"]), Col::Str("chunk", vec!["x", "y"])]);
    assert_eq!(e, format!("{DOCS}: row 1: chunk a_0 appears twice"));
    let e = broken("conteb-missing", QUERIES, &[Col::Str("chunk_id", vec!["paperA_7"]), Col::Str("query", vec!["q"])]);
    assert_eq!(e, format!("{QUERIES}: row 0: chunk paperA_7 is not in {DOCS}"));
}
