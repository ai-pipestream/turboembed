//! The Qasper reader on a sample archive made for the test.

mod common;

use common::{data, query_ids, quoted, spans, text};
use turbo_eval::read::qasper;

const P1: &str = "2101.00001";

#[test]
fn reads_the_sample_archive() {
    let s = qasper::read("qasper-sample", &data("qasper")).unwrap();
    let ids: Vec<&str> = s.docs.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(ids, [P1, "2101.00002"]);
    assert_eq!(s.docs[0].title.as_deref(), Some("A Made Up Study of Résumé Parsing"));
    // Abstract, the first section's name and paragraphs, then the second
    // section's paragraphs: it has no name, and its empty one is left out.
    assert_eq!(
        text(&s, P1),
        "We parse résumés.\n\nIntroduction\n\nParsing is hard.\n\nData comes from 3 sources.\n\nParsing is hard."
    );
    assert_eq!(text(&s, "2101.00002"), "Short.");
    assert_eq!(query_ids(&s), ["qa1", "qa2", "qa3"]);
    assert_eq!(s.qrels.grade("qa1", P1), 1);
    assert_eq!(s.qrels.grade("qa3", "2101.00002"), 1);

    // Both annotators' evidence, once each; the figure caption is left out.
    assert_eq!(quoted(&s, "qa1", P1), ["Introduction", "Data comes from 3 sources."]);
    assert_eq!(spans(&s, "qa1", P1), [(21, 33), (53, 79)]);
    // A paragraph that appears twice is relevant at both places; part of
    // the abstract is placed where it occurs, in bytes past the é.
    assert_eq!(spans(&s, "qa2", P1), [(3, 18), (35, 51), (81, 97)]);
    assert_eq!(quoted(&s, "qa2", P1), ["parse résumés", "Parsing is hard.", "Parsing is hard."]);
    assert_eq!(s.unplaced, [format!("query qa2 paper {P1}: \"Text that is nowhere in the paper.\"")]);
    // An unanswerable question has no spans.
    assert!(spans(&s, "qa3", "2101.00002").is_empty());
}

#[test]
fn json_errors_name_the_member_and_line() {
    let e = qasper::read_json("q", "a.tgz:q.json", "{\n \"x\": {\"title\": \"t\"}\n}", String::new()).unwrap_err();
    assert!(e.starts_with("a.tgz:q.json:2:"), "{e}");
    assert!(e.contains("missing field `abstract`"), "{e}");
}
