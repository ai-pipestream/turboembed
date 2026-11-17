use super::*;

use crate::test_heads::tiny_rerank_bundle;

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
    // A query budget of half the window: a longer query is cut to the same
    // pieces whichever of the two budgets is asked for, so it is counted.
    let long = "one two three four five six seven eight nine ten eleven twelve";
    assert!(fits_one_window(&t, "one two three", "world", 8, 16));
    assert!(!fits_one_window(&t, long, "world", 8, 16));
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

#[test]
fn rerank_cpu_records_no_gpu_clock() {
    use turbo::TURBO_PRECISION_MODEL;

    let mut passes = 0u32;
    let mut w = crate::gpu_clock::Window::none();
    time(1, 2, &mut w, &mut || {
        passes += 1;
        Ok(vec![1.0])
    })
    .unwrap();
    assert_eq!(passes, 3);
    assert_eq!(w.finish(), None);

    let scratch =
        turbo_scratch::Scratch::at(std::env::temp_dir().join(format!("turbo-bench-rerank-{}", std::process::id())));
    let bundle = tiny_rerank_bundle(&scratch);
    let questions_path = scratch.join("squad.json");
    let squad = json!({
        "data": [{
            "paragraphs": [
                {"context": "Paris is in France.", "qas": [{"question": "Where is Paris?"}]},
                {"context": "Berlin is in Germany.", "qas": [{"question": "Where is Berlin?"}]}
            ]
        }]
    });
    std::fs::write(&questions_path, serde_json::to_vec(&squad).unwrap()).unwrap();

    let plan = RerankPlan {
        bundle,
        device: "cpu".into(),
        precision: TURBO_PRECISION_MODEL,
        questions: questions_path,
        requests: 1,
        passages: 2,
        warmup: 1,
        iterations: 2,
        tei: None,
    };
    let p = Provenance { top: String::new(), commit: "c".repeat(40), pushed_to: vec!["origin/main".into()] };

    let cpu_record = measure(&plan, &p, "2026-10-22T00:00:00Z".into()).unwrap();
    assert_eq!(cpu_record.device.clock, None);
    check(&cpu_record).unwrap();
    std::fs::remove_dir_all(&scratch).unwrap();
}

/// The populated case, over the driver's files in a temporary tree: the
/// clock is read once after the warmup runs and once after the last timed
/// run, and not between.
#[test]
fn rerank_time_reads_the_clock_after_warmup_and_after_the_last_run() {
    use crate::gpu_clock::tests::Tree;

    let tree = Tree::new("rerank");
    let mut calls = 0u32;
    let mut w = tree.window(1);
    // Each run leaves its number in act_freq, so a reading says which run
    // it followed.
    let (ms, scores) = time(2, 3, &mut w, &mut || {
        calls += 1;
        tree.set_act(calls * 100);
        Ok(vec![calls as f32])
    })
    .unwrap();
    assert_eq!((ms.len(), scores), (3, vec![5.0]));
    assert_eq!(w.finish(), Tree::clock(200, 500));
}
