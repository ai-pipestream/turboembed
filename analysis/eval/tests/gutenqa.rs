//! The GutenQA reader on rows made for the test.

mod common;

use common::parquet::{Col, write};
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
fn the_digest_is_of_the_files_read() {
    // sha256sum over the two sample files, as the manifest lists them.
    let s = gutenqa::read("g", &data("gutenqa")).unwrap();
    let a = turbo_goldens::hex::to_hex(&turbo_goldens::sha::sha256(
        &std::fs::read(data("gutenqa").join(gutenqa::CHUNKS)).unwrap(),
    ));
    let b = turbo_goldens::hex::to_hex(&turbo_goldens::sha::sha256(
        &std::fs::read(data("gutenqa").join(gutenqa::QUESTIONS)).unwrap(),
    ));
    let manifest = format!("{a}  {}\n{b}  {}\n", gutenqa::CHUNKS, gutenqa::QUESTIONS);
    assert_eq!(s.sha256, turbo_goldens::hex::to_hex(&turbo_goldens::sha::sha256(manifest.as_bytes())));
}

#[test]
fn errors_name_the_file_and_row() {
    let dir = common::scratch("gutenqa-chunk");
    copy_dir(&data("gutenqa"), &dir);
    let questions = |book: Vec<i64>, chunk: Vec<i64>| {
        let n = book.len();
        write(
            &dir.join(gutenqa::QUESTIONS),
            &[
                Col::Int("Book ID", book),
                Col::Int("Chunk ID", chunk),
                Col::Str("Question", vec!["q"; n]),
                Col::Str("Chunk Must Contain", vec!["x"; n]),
            ],
        )
    };
    questions(vec![0, 0], vec![0, 5]);
    assert_eq!(
        gutenqa::read("g", &dir).unwrap_err(),
        "questions.parquet: row 1: book 0 chunk 5 is not in gutenqa_chunks.parquet"
    );
    questions(vec![-1], vec![0]);
    assert_eq!(gutenqa::read("g", &dir).unwrap_err(), "questions.parquet: column Book ID row 0 is -1, not a u32");
    write(&dir.join(gutenqa::QUESTIONS), &[Col::Int("Book ID", vec![0])]);
    assert_eq!(gutenqa::read("g", &dir).unwrap_err(), "questions.parquet: no column Chunk ID");

    write(
        &dir.join(gutenqa::CHUNKS),
        &[
            Col::Str("Book Name", vec!["A", "B"]),
            Col::Int("Book ID", vec![0, 0]),
            Col::Int("Chunk ID", vec![0, 1]),
            Col::Str("Chunk", vec!["x", "y"]),
        ],
    );
    assert_eq!(
        gutenqa::read("g", &dir).unwrap_err(),
        "gutenqa_chunks.parquet: row 1: book 0 is named \"B\" here and \"A\" before"
    );
}
