//! The HiCBench reader on a sample made for the test.

mod common;

use std::fs;

use common::{copy_dir, data, query_ids, quoted, spans};
use turbo_eval::read::hicbench;

#[test]
fn reads_the_sample() {
    let s = hicbench::read("hicbench-sample", &data("hicbench")).unwrap();
    let ids: Vec<&str> = s.docs.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(ids, ["policy", "report"]);
    assert_eq!(query_ids(&s), ["q1", "q2"]);
    assert_eq!(s.qrels.grade("q1", "policy"), 1);
    assert_eq!(s.qrels.grade("q1", "report"), 0);
    // In document order; "naïve" makes the second sentence a byte longer
    // than its characters.
    assert_eq!(quoted(&s, "q1", "policy"), ["This policy covers naïve users.", "Records are kept for 7 years."]);
    assert_eq!(spans(&s, "q1", "policy"), [(17, 49), (71, 100)]);
    assert_eq!(quoted(&s, "q2", "report"), ["Revenue grew by 12% in 2024."]);
    assert_eq!(s.unplaced, ["query q2 document report: \"A sentence that is not in the report.\""]);
}

#[test]
fn a_question_on_a_missing_document_names_its_line() {
    let dir = common::scratch("hicbench-doc");
    copy_dir(&data("hicbench"), &dir);
    fs::write(dir.join(hicbench::QAS), "{\"input\": \"q\", \"evidences\": [], \"_id\": \"policy\"}\n\n{\"input\": \"q\", \"evidences\": [], \"_id\": \"gone\"}\n").unwrap();
    assert_eq!(hicbench::read("h", &dir).unwrap_err(), "qas/HiCBench.jsonl:3: document gone is not in doc/HiCBench");
}
