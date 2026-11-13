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

fn tiny_rerank_bundle(dir: &Path) -> PathBuf {
    let bundle = dir.join("bundle");
    std::fs::create_dir_all(bundle.join("weights")).unwrap();
    std::fs::create_dir_all(bundle.join("reference")).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let tiny = root.join("testdata/tiny-bert-bundle");
    let reference = root.join("testdata/tiny-rerank-reference");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(tiny.join("manifest.json")).unwrap()).unwrap();
    manifest.as_object_mut().unwrap().remove("embed");
    manifest["task"] = "TASK_RERANK".into();
    manifest["rerank"] = json!({"max_seq": 32, "max_batch": 4, "query_tokens": 8, "pair_template": "PAIR_BERT"});
    manifest["classifier"] = json!({"kind": "CLASSIFIER_SEQUENCE", "labels": ["LABEL_0"]});
    manifest["artifacts"][0]["files"] = json!(["weights/model.safetensors", "weights/head.safetensors"]);
    let names = manifest["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    for (k, v) in [
        ("pooler_weight", "bert.pooler.dense.weight"),
        ("pooler_bias", "bert.pooler.dense.bias"),
        ("classifier_weight", "classifier.weight"),
        ("classifier_bias", "classifier.bias"),
    ] {
        names.insert(k.into(), v.into());
    }
    manifest["reference"]["cases"] =
        serde_json::from_slice(&std::fs::read(reference.join("cases.json")).unwrap()).unwrap();
    let mut files = Vec::new();
    for (src, dest) in [
        (tiny.join("tokenizer.json"), "tokenizer.json"),
        (tiny.join("weights/model.safetensors"), "weights/model.safetensors"),
        (reference.join("head.safetensors"), "weights/head.safetensors"),
        (reference.join("reference.safetensors"), "reference/reference.safetensors"),
    ] {
        let b = std::fs::read(src).unwrap();
        std::fs::write(bundle.join(dest), &b).unwrap();
        files.push(json!({"path": dest, "size": b.len(), "sha256": turbo::bundle::sha256_hex(&b)}));
    }
    manifest["files"] = files.into();
    std::fs::write(bundle.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    bundle
}

#[test]
fn rerank_reads_gpu_clock_around_library_timed_runs() {
    use turbo::TURBO_PRECISION_MODEL;
    use turbo::record::{ClockReading, GpuClock};

    let before = ClockReading { min_mhz: 2200, max_mhz: 2200, act_mhz: 0 };
    let after = ClockReading { min_mhz: 2200, max_mhz: 2200, act_mhz: 2200 };
    let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let events_r = std::rc::Rc::clone(&events);
    let calls = std::cell::Cell::new(0u32);
    let clock = crate::gpu_clock::with_test_reader(
        move |backend| {
            let n = calls.get();
            calls.set(n + 1);
            events_r.borrow_mut().push(format!("clock:{backend}:{n}"));
            Some(if n == 0 { before } else { after })
        },
        || {
            let mut w = crate::gpu_clock::Window::new("levelzero", 1);
            time(1, 2, &mut w, &mut || {
                events.borrow_mut().push("pass".into());
                Ok(vec![1.0])
            })
            .unwrap();
            w.finish()
        },
    );
    assert_eq!(clock, Some(GpuClock { before, after }));
    assert_eq!(*events.borrow(), ["pass", "clock:levelzero:0", "pass", "pass", "clock:levelzero:1"]);

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

    let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let seen_r = std::rc::Rc::clone(&seen);
    let clocked = crate::gpu_clock::with_test_reader(
        move |backend| {
            let mut s = seen_r.borrow_mut();
            s.push(backend.to_owned());
            Some(if s.len() == 1 { before } else { after })
        },
        || measure(&plan, &p, "2026-10-22T00:00:00Z".into()).unwrap(),
    );
    assert_eq!(*seen.borrow(), ["cpu", "cpu"]);
    assert_eq!(clocked.device.clock, Some(GpuClock { before, after }));
    check(&clocked).unwrap();

    let calls = std::cell::Cell::new(0u32);
    let incomplete = crate::gpu_clock::with_test_reader(
        move |_| {
            let n = calls.get();
            calls.set(n + 1);
            (n == 0).then_some(before)
        },
        || measure(&plan, &p, "2026-10-22T00:00:00Z".into()).unwrap(),
    );
    assert_eq!(incomplete.device.clock, None);
    std::fs::remove_dir_all(&scratch).unwrap();
}
