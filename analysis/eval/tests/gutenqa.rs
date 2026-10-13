//! The GutenQA reader on rows made for the test.

mod common;

use std::fs;

use common::{copy_dir, data, query_ids, quoted, spans, text};
use turbo_eval::read::gutenqa;

#[test]
fn reads_the_sample() {
    let s = gutenqa::read("gutenqa-sample", &data("gutenqa")).unwrap();
    assert_eq!(s.docs[0].title.as_deref(), Some("A_Made_Up_Tale_-_Nobody"));
    assert_eq!(text(&s, "0"), "It was a cold night in the village.\nSeñor Ruiz rowed across the lake at dawn.");
    assert_eq!(query_ids(&s), ["0", "1"]);
    assert_eq!(s.queries[1].text, "What did the keeper trim?");
    assert_eq!(s.qrels.grade("0", "0"), 1);
    assert_eq!(s.qrels.grade("1", "1"), 1);
    // The passage inside chunk 1, which starts at byte 36; ñ is two bytes.
    assert_eq!(spans(&s, "0", "0"), [(36, 69)]);
    assert_eq!(quoted(&s, "0", "0"), ["Señor Ruiz rowed across the lake"]);
    // A passage its chunk does not contain is listed, not placed.
    assert!(spans(&s, "1", "1").is_empty());
    assert_eq!(s.unplaced, ["query 1 book 1 chunk 0: \"a passage the chunk does not have\""]);
}

#[test]
fn errors_name_the_file_and_line() {
    let dir = common::scratch("gutenqa-chunk");
    copy_dir(&data("gutenqa"), &dir);
    fs::write(
        dir.join(gutenqa::QUESTIONS),
        "{\"Book ID\": 0, \"Chunk ID\": 5, \"Question\": \"q\", \"Chunk Must Contain\": \"x\"}\n",
    )
    .unwrap();
    assert_eq!(
        gutenqa::read("g", &dir).unwrap_err(),
        "questions.jsonl:1: book 0 chunk 5 is not in gutenqa_chunks.jsonl"
    );

    fs::write(dir.join(gutenqa::QUESTIONS), "{\"Book ID\": \"zero\"}\n").unwrap();
    let e = gutenqa::read("g", &dir).unwrap_err();
    assert!(e.starts_with("questions.jsonl:1: invalid type: string \"zero\", expected u32"), "{e}");
}
