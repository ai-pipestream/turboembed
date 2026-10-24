use super::*;

#[test]
fn rerank_scores_come_back_in_passage_order() {
    let body = r#"[{"index":2,"score":-1.5},{"index":0,"score":3.25},{"index":1,"score":0.5}]"#;
    assert_eq!(scores(body, 3).unwrap(), [3.25, 0.5, -1.5]);
    assert!(scores(body, 4).is_err());
    assert!(scores(r#"[{"index":5,"score":1}]"#, 1).is_err());
    assert!(scores(r#"[{"index":0}]"#, 1).is_err());
}

#[test]
fn requests_take_the_passages_that_fit() {
    let text = r#"{"data":[{"paragraphs":[
        {"context":"short one","qas":[{"question":"q1"}]},
        {"context":"a very long context here","qas":[{"question":"q2"}]},
        {"context":"short two","qas":[{"question":"q3"}]},
        {"context":"short three","qas":[{"question":"q4"}]}]}]}"#;
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let tokenizer = Tokenizer::load(&Bundle::open(&dir).unwrap()).unwrap();
    let fits = |_: &Tokenizer, _: &str, c: &str| c.starts_with("short");
    let r = requests(text, &tokenizer, &fits, 2, 2).unwrap();
    assert_eq!(r[0], ("q1".to_owned(), vec!["short one".to_owned(), "short two".to_owned()]));
    assert_eq!(r[1], ("q2".to_owned(), vec!["short two".to_owned(), "short three".to_owned()]));
    assert!(requests(text, &tokenizer, &fits, 4, 2).is_err());
}

/// The real predicate: a pair fits when one window holds the whole query
/// and the whole context, and not when either would be cut.
#[test]
fn a_pair_fits_only_when_nothing_is_cut() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let t = Tokenizer::load(&Bundle::open(&dir).unwrap()).unwrap();
    assert!(fits_one_window(&t, "what", "a short context", 4, 16));
    assert!(!fits_one_window(&t, "what is this long question about", "a short context", 4, 16));
    assert!(!fits_one_window(&t, "what", "a context far too long to fit in a window of sixteen tokens at all", 4, 16));
}

/// Each committed record parses and passes `check`, and is filed under
/// the name `file_name` gives it.
#[test]
fn committed_records_pass_their_check() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(RERANK_DIR);
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    for e in entries {
        let path = e.unwrap().path();
        let r: RerankRecord = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        check(&r).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(path.file_name().unwrap().to_str().unwrap(), file_name(&r));
    }
}
