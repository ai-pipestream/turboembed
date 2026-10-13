//! The LegalBench-RAG reader on a sample made for the test: code point
//! spans over text with CRLF line endings, a two byte and a four byte
//! character, and the mini subset's selection.

mod common;

use std::fs;

use common::{copy_dir, data, query_ids, quoted, scratch, spans, text};
use turbo_eval::read::legalbench_rag;

const CAFE: &str = "cuad/Café contract.txt";

#[test]
fn reads_the_sample() {
    let s = legalbench_rag::read("lb", &data("legalbench-rag")).unwrap();
    assert_eq!(query_ids(&s), ["privacy_qa/0", "cuad/0", "cuad/1"]);
    assert_eq!(s.queries[1].text, "What royalties does the Licensee pay?");
    let ids: Vec<&str> = s.docs.iter().map(|d| d.id.as_str()).collect();
    assert_eq!(ids, [CAFE, "privacy_qa/app.txt"]);

    // The file has CRLF line endings; the text is as Python reads it.
    let t = text(&s, CAFE);
    assert!(t.starts_with("MASTER AGREEMENT\nThe Licensee (\"Café Ltd\")"));
    assert!(!t.contains('\r'));

    // [43, 71) in code points: one byte more before it for the é, and
    // three more inside it for the emoji.
    assert_eq!(spans(&s, "cuad/0", CAFE), [(44, 75)]);
    assert_eq!(quoted(&s, "cuad/0", CAFE), ["shall pay 😀 royalties of 5%."]);
    // [85, 139) comes after both: four bytes more at each end.
    assert_eq!(spans(&s, "cuad/1", CAFE), [(89, 143)]);
    assert_eq!(quoted(&s, "cuad/1", CAFE), ["either party may end this Agreement on 30 days notice."]);

    // A query with snippets in two documents judges both.
    assert_eq!(quoted(&s, "privacy_qa/0", "privacy_qa/app.txt"), ["We never sell personal data to third parties."]);
    assert_eq!(quoted(&s, "privacy_qa/0", CAFE), ["MASTER AGREEMENT"]);
    assert_eq!(s.qrels.grade("privacy_qa/0", CAFE), 1);
    assert_eq!(s.qrels.grade("privacy_qa/0", "privacy_qa/app.txt"), 1);
    assert_eq!(s.qrels.grade("cuad/0", "privacy_qa/app.txt"), 0);
    assert!(s.unplaced.is_empty());
}

fn with_cuad(name: &str, cuad: &str) -> String {
    let dir = scratch(name);
    copy_dir(&data("legalbench-rag"), &dir);
    fs::write(dir.join("benchmarks/cuad.json"), cuad).unwrap();
    legalbench_rag::read("lb", &dir).unwrap_err()
}

#[test]
fn checks_spans_against_the_text() {
    let e = with_cuad(
        "lb-answer",
        r#"{"tests": [{"query": "q", "snippets": [{"file_path": "cuad/Café contract.txt", "span": [43, 71], "answer": "shall pay royalties"}]}]}"#,
    );
    assert_eq!(
        e,
        "query cuad/0: snippet [43, 71) of cuad/Café contract.txt: the text there is \
         \"shall pay 😀 royalties of 5%.\", the answer is \"shall pay royalties\""
    );
    let e = with_cuad(
        "lb-end",
        r#"{"tests": [{"query": "q", "snippets": [{"file_path": "cuad/Café contract.txt", "span": [100, 1000]}]}]}"#,
    );
    assert_eq!(e, "query cuad/0: snippet [100, 1000) of cuad/Café contract.txt: the document has 140 code points");
    let e = with_cuad("lb-json", "{\"tests\": [\n  {\"query\": \"q\", \"snippets\": [{\"span\": [0, 1]}]}\n]}");
    assert!(e.starts_with("benchmarks/cuad.json:2:"), "{e}");
    assert!(e.contains("missing field `file_path`"), "{e}");
}

/// 200 tests over 20 files: the mini subset keeps 194, chosen as
/// `benchmark.py` chooses them. The expected ids are what CPython 3.13
/// printed for the same sort.
#[test]
fn mini_keeps_what_benchmark_py_keeps() {
    let dir = scratch("lb-mini");
    copy_dir(&data("legalbench-rag"), &dir);
    for f in 0..20 {
        fs::write(dir.join(format!("corpus/cuad/f{f:02}.txt")), "text").unwrap();
    }
    let tests: Vec<String> = (0..200)
        .map(|i| {
            format!(r#"{{"query": "q{i}", "snippets": [{{"file_path": "cuad/f{:02}.txt", "span": [0, 4]}}]}}"#, i % 20)
        })
        .collect();
    fs::write(dir.join("benchmarks/cuad.json"), format!("{{\"tests\": [{}]}}", tests.join(","))).unwrap();

    let s = legalbench_rag::read("lb", &dir).unwrap();
    let cuad: Vec<&str> = query_ids(&s).into_iter().filter(|q| q.starts_with("cuad/")).collect();
    assert_eq!(cuad.len(), legalbench_rag::MINI_TESTS_PER_BENCHMARK);
    assert_eq!(cuad[..8], ["cuad/19", "cuad/39", "cuad/59", "cuad/79", "cuad/99", "cuad/119", "cuad/139", "cuad/159"]);
    for dropped in [91, 111, 131, 151, 171, 191] {
        assert!(!cuad.contains(&format!("cuad/{dropped}").as_str()), "{dropped}");
    }
}
