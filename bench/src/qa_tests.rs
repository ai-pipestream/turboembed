use serde_json::json;
use turbo::TURBO_PRECISION_MODEL;

use super::*;

use crate::test_heads::tiny_qa_bundle;

#[test]
fn questions_read_squad_layout_in_order() {
    let text = r#"{"version":"v2.0","data":[{"title":"A","paragraphs":[
        {"context":"Paris is in France.","qas":[{"id":"1","question":"Where is Paris?","answers":[],"is_impossible":false},
                                            {"id":"2","question":"Who?","answers":[],"is_impossible":true}]},
        {"context":"Second.","qas":[{"id":"3","question":"What?","answers":[]}]}]}]}"#;
    let q = questions(text).unwrap();
    assert_eq!(q.len(), 3);
    assert_eq!(q[0], ("Where is Paris?".into(), "Paris is in France.".into()));
    assert_eq!(q[2], ("What?".into(), "Second.".into()));
    assert!(questions(r#"{"data":[{"paragraphs":[{"qas":[]}]}]}"#).is_err());
    assert!(questions("[]").is_err());
}

#[test]
fn bins_start_at_their_bounds() {
    assert_eq!([0, 127, 128, 255, 256, 383, 384, 5000].map(bin_of), [0, 0, 1, 1, 2, 2, 3, 3]);
}

/// Each committed record parses and passes `check`, and is filed
/// under the name `file_name` gives it.
#[test]
fn committed_records_pass_their_check() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(QA_DIR);
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    for e in entries {
        let path = e.unwrap().path();
        let r: QaRecord = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        check(&r).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(path.file_name().unwrap().to_str().unwrap(), file_name(&r));
    }
}

#[test]
fn qa_measures_bins_and_carries_gpu_clock_across_active_bins() {
    let scratch =
        turbo_scratch::Scratch::at(std::env::temp_dir().join(format!("turbo-bench-qa-{}", std::process::id())));
    let bundle = tiny_qa_bundle(&scratch);
    let questions_path = scratch.join("squad.json");
    let long_context = vec!["Paris"; 140].join(" ");
    let squad = json!({
        "data": [{
            "paragraphs": [
                {"context": "Paris is in France.", "qas": [{"question": "Where is Paris?"}]},
                {"context": long_context, "qas": [{"question": "Where?"}]}
            ]
        }]
    });
    std::fs::write(&questions_path, serde_json::to_vec(&squad).unwrap()).unwrap();

    let plan = QaPlan {
        bundle,
        device: "cpu".into(),
        precision: TURBO_PRECISION_MODEL,
        batch: 1,
        max_batch: 0,
        questions: questions_path,
        per_bin: 1,
        warmup: 1,
        iterations: 2,
    };
    let p = Provenance { top: String::new(), commit: "b".repeat(40), pushed_to: vec!["origin/main".into()] };

    let cpu_record = measure(&plan, &p, "2026-10-22T00:00:00Z".into()).unwrap();
    assert_eq!(cpu_record.bins.len(), 2);
    assert_eq!(cpu_record.device.clock, None);
    check(&cpu_record).unwrap();
    std::fs::remove_dir_all(&scratch).unwrap();
}

/// The populated case, over the driver's files in a temporary tree: with
/// two bins the clock is read once after the first bin's warmup and once
/// after the last bin's timed runs, and not between.
#[test]
fn qa_time_reads_the_clock_after_the_first_warmup_and_after_the_last_bin() {
    use crate::gpu_clock::tests::Tree;

    let tree = Tree::new("qa");
    let mut calls = 0u32;
    let mut w = tree.window(2);
    // Each run leaves its number in act_freq, so a reading says which run
    // it followed.
    let mut pass = || {
        calls += 1;
        tree.set_act(calls * 100);
        Ok(calls)
    };
    let (ms, windows) = time(1, 2, &mut w, &mut pass).unwrap();
    assert_eq!((ms.len(), windows), (2, 3));
    let (ms, windows) = time(1, 2, &mut w, &mut pass).unwrap();
    assert_eq!((ms.len(), windows), (2, 6));
    assert_eq!(w.finish(), Tree::clock(100, 600));
}
