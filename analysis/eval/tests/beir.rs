//! The BEIR reader on a sample made for the test.

mod common;

use std::fs;

use common::{copy_dir, data, query_ids, scratch};
use turbo_eval::read::beir;
use turbo_eval::set::Doc;

#[test]
fn reads_the_sample() {
    let s = beir::read("beir-sample", &data("beir")).unwrap();
    assert_eq!(s.name, "beir-sample");
    assert_eq!(
        s.docs,
        vec![
            Doc {
                id: "d1".into(),
                title: Some("Café culture".into()),
                text: "Coffee houses in Vienna date from the 1680s.".into()
            },
            Doc { id: "d2".into(), title: None, text: "Tea is grown on hillsides in Darjeeling.".into() },
            Doc {
                id: "d3".into(),
                title: Some("Naïve Bayes".into()),
                text: "A classifier that assumes features are independent.".into()
            },
        ]
    );
    // q3 is in queries.jsonl but not judged in the test split.
    assert_eq!(query_ids(&s), ["q1", "q2"]);
    assert_eq!(s.queries[0].text, "When did Viennese cafés open?");
    assert_eq!(s.qrels.grade("q1", "d1"), 1);
    assert_eq!(s.qrels.grade("q1", "d3"), 0);
    assert_eq!(s.qrels.grade("q2", "d2"), 2);
    assert_eq!(s.qrels.scored_queries().collect::<Vec<_>>(), ["q1", "q2"]);
    assert!(s.spans.is_none());
    assert!(s.unplaced.is_empty());
    // `sha256sum corpus.jsonl queries.jsonl qrels/test.tsv | LC_ALL=C sort -k2 | sha256sum`
    assert_eq!(s.sha256, "c62f2c11c492de125113311a2d7905e35d97004c79d6498ab1ba6787927ea0e0");
}

/// A copy of the sample with `file` replaced by `body`.
fn broken(name: &str, file: &str, body: &str) -> String {
    let dir = scratch(name);
    copy_dir(&data("beir"), &dir);
    fs::write(dir.join(file), body).unwrap();
    beir::read("b", &dir).unwrap_err()
}

#[test]
fn errors_name_the_file_and_line() {
    let e = broken(
        "beir-corpus",
        beir::CORPUS,
        "{\"_id\": \"d1\", \"title\": \"\", \"text\": \"a\"}\n{\"_id\": \"d2\", \"title\": \"\"}\n",
    );
    assert!(e.starts_with("corpus.jsonl:2: missing field `text`"), "{e}");

    let e = broken("beir-unknown", beir::QUERIES, "{\"_id\": \"q1\", \"text\": \"a\", \"lang\": \"en\"}\n");
    assert!(e.starts_with("queries.jsonl:1: unknown field `lang`"), "{e}");

    let e =
        broken("beir-twice", beir::QUERIES, "{\"_id\": \"q1\", \"text\": \"a\"}\n{\"_id\": \"q1\", \"text\": \"b\"}\n");
    assert_eq!(e, "queries.jsonl:2: query q1 appears twice");

    let e = broken("beir-header", beir::QRELS, "q1\td1\t1\n");
    assert_eq!(e, "qrels/test.tsv:1: expected the header query-id<TAB>corpus-id<TAB>score");

    let e = broken("beir-grade", beir::QRELS, "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\thigh\n");
    assert_eq!(e, "qrels/test.tsv:3: score \"high\" is not an integer");

    let e = broken("beir-fields", beir::QRELS, "query-id\tcorpus-id\tscore\nq1 d1 1\n");
    assert_eq!(e, "qrels/test.tsv:2: expected 3 tab separated fields, found 1");

    let e = broken("beir-doc", beir::QRELS, "query-id\tcorpus-id\tscore\nq1\td9\t1\n");
    assert_eq!(e, "qrels/test.tsv:2: document d9 is not in corpus.jsonl");

    let e = broken("beir-query", beir::QRELS, "query-id\tcorpus-id\tscore\nq9\td1\t1\n");
    assert_eq!(e, "qrels/test.tsv:2: query q9 is not in queries.jsonl");
}

#[test]
fn bad_utf8_names_the_line() {
    let dir = scratch("beir-bytes");
    copy_dir(&data("beir"), &dir);
    fs::write(dir.join(beir::QUERIES), b"{\"_id\": \"q1\", \"text\": \"a\"}\n{\"_id\": \"q2\", \"text\": \"\xff\"}\n")
        .unwrap();
    assert_eq!(beir::read("b", &dir).unwrap_err(), "queries.jsonl:2: not UTF-8");
}
