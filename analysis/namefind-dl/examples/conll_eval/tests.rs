use std::collections::BTreeSet;

use super::support::{self, Counts, Entity, Kind};
use turbo_namefind_dl::decode::Candidate;

const GOLD: &str = "-DOCSTART- -X- -X- O\n\nAnn NNP B-NP I-PER\nLi NNP I-NP I-PER\nvisits VBZ B-VP O\nOslo NNP B-NP I-LOC\n\nÉva NNP B-NP I-PER\n";

#[test]
fn real_reader_handles_iob1_unicode_and_crlf() {
    let a = support::read(GOLD).unwrap();
    let crlf = GOLD.replace('\n', "\r\n");
    let b = support::read(&crlf).unwrap();
    support::aligned(&a, &b).unwrap();
    assert_eq!(a.len(), 2);
    assert_eq!(
        a[0].entities,
        BTreeSet::from([
            Entity { start: 0, end: 2, kind: Kind::Person },
            Entity { start: 3, end: 4, kind: Kind::Location },
        ])
    );
    assert_eq!(a[1].words[0], "Éva");
    assert_eq!(a[1].entities, b[1].entities);
}

#[test]
fn reader_rejects_empty_bad_columns_and_bad_types_without_text_in_errors() {
    for text in ["", "\n", "secret NNP O\n", "secret NNP B-NP B-NOTAKIND\n"] {
        let error = support::read(text).unwrap_err();
        assert!(!error.contains("secret"));
    }
}

#[test]
fn empty_or_whitespace_tokens_cannot_change_the_evaluation_units() {
    for text in [" X X O\n", "a\tb X X O\n"] {
        assert!(support::read(text).is_err(), "invalid token accepted");
    }
}

#[test]
fn upstream_must_match_every_token_and_sample() {
    let gold = support::read(GOLD).unwrap();
    for changed in [GOLD.replace("Ann", "Bob"), GOLD.replace("\n\nÉva", "\nÉva"), format!("{GOLD}\nExtra X X O\n")] {
        let peer = support::read(&changed).unwrap();
        assert!(support::aligned(&gold, &peer).is_err());
    }
}

fn candidate(start: usize, end: usize, kind: usize) -> Candidate {
    Candidate { start, end, kind, probability: 0.75 }
}

#[test]
fn unicode_projection_deduplicates_only_same_token_range_and_kind() {
    let (text, spans) = support::joined(&["Éva", "東京"]);
    assert_eq!(spans, [(0, 4), (5, 11)]);
    let found = support::project(
        &text,
        &spans,
        &[candidate(0, 2, 0), candidate(2, 4, 0), candidate(5, 11, 1)],
        &[Kind::Person, Kind::Location],
    )
    .unwrap();
    assert_eq!(
        found,
        BTreeSet::from([
            Entity { start: 0, end: 1, kind: Kind::Person },
            Entity { start: 1, end: 2, kind: Kind::Location },
        ])
    );
}

#[test]
fn malformed_native_entities_fail_instead_of_panicking_or_scoring() {
    let (text, spans) = support::joined(&["Éva", "東京"]);
    for c in [candidate(1, 2, 0), candidate(0, 12, 0), candidate(4, 5, 0), candidate(4, 4, 0), candidate(0, 4, 1)] {
        assert!(support::project(&text, &spans, &[c], &[Kind::Person]).is_err());
    }
    for probability in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        let c = Candidate { probability, ..candidate(0, 4, 0) };
        assert!(support::project(&text, &spans, &[c], &[Kind::Person]).is_err());
    }
}

#[test]
fn accuracy_is_exact_type_and_token_span_not_overlapping_bytes() {
    let gold = support::read(GOLD).unwrap();
    let predictions = BTreeSet::from([
        Entity { start: 0, end: 1, kind: Kind::Person },
        Entity { start: 3, end: 4, kind: Kind::Organization },
    ]);
    let mut counts = Counts::default();
    counts.update(&gold[0].entities, &predictions, None);
    assert_eq!(counts, Counts { gold: 2, predicted: 2, correct: 0 });
    assert_eq!((counts.precision(), counts.recall(), counts.f1()), (0.0, 0.0, 0.0));
    let mut by_kind = Counts::default();
    by_kind.update(&gold[0].entities, &gold[0].entities, Some(Kind::Person));
    assert_eq!(by_kind, Counts { gold: 1, predicted: 1, correct: 1 });
    assert_eq!(by_kind.f1(), 1.0);
    assert_eq!(Counts::default().f1(), 0.0);
}

#[test]
fn equal_accuracy_does_not_establish_prediction_agreement() {
    let gold = BTreeSet::from([
        Entity { start: 0, end: 1, kind: Kind::Person },
        Entity { start: 1, end: 2, kind: Kind::Person },
    ]);
    let native = BTreeSet::from([*gold.first().unwrap()]);
    let peer = BTreeSet::from([*gold.last().unwrap()]);
    let (mut a, mut b, mut agreement) = (Counts::default(), Counts::default(), Counts::default());
    a.update(&gold, &native, None);
    b.update(&gold, &peer, None);
    agreement.update(&peer, &native, None);
    assert_eq!(a.f1(), b.f1());
    assert_ne!(native, peer);
    assert_eq!(agreement.correct, 0);
}

mod tiny {
    use std::path::{Path, PathBuf};
    const BIAS: [f32; 3] = [0.0, 2.0, 0.0];
    include!("../../tests/offline/support.rs");

    pub fn bundle(name: &str) -> PathBuf {
        let dir = write_bundle(name, &["O", "B-PER", "I-PER"]);
        let path = dir.join("manifest.json");
        let mut m: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        m["model"]["id"] = json!("generated-conll-eval-test");
        m["model"]["source"] = json!({"repository": "https://example.org/generated-test", "commit": "0000000000000000000000000000000000000001"});
        fs::write(path, serde_json::to_vec(&m).unwrap()).unwrap();
        dir
    }
}

#[test]
fn native_cpu_command_reports_comparison_and_no_input_text() {
    let dir = tiny::bundle("conll-eval-native");
    let gold = "Ann X X B-PER\nmet X X O\n";
    let peer = "Ann X X B-PER\nmet X X B-PER\n";
    std::fs::write(dir.join("gold.conll"), gold).unwrap();
    std::fs::write(dir.join("upstream.conll"), peer).unwrap();
    let mut args = vec![
        dir.to_string_lossy().into_owned(),
        dir.join("gold.conll").to_string_lossy().into_owned(),
        dir.join("upstream.conll").to_string_lossy().into_owned(),
        "0000000000000000000000000000000000000001".into(),
        "word".into(),
    ];
    let result = super::run(&args).unwrap();
    assert!(result.contains("native.all: gold=1 predicted=2 correct=1"), "{result}");
    assert!(result.contains("upstream.all: gold=1 predicted=2 correct=1"));
    assert!(result.contains("parity=exact_entity_agreement"));
    assert!(result.contains(&format!("gold_sha256={}", turbo::bundle::sha256_hex(gold.as_bytes()))));
    assert!(!result.contains("Ann") && !result.contains(dir.to_str().unwrap()));
    std::fs::write(dir.join("upstream.conll"), gold).unwrap();
    assert!(super::run(&args).unwrap().contains("parity=different_entities"));
    args[3] = "0000000000000000000000000000000000000002".into();
    assert!(super::run(&args).unwrap_err().contains("same source commit"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn command_validates_arguments_before_reading_inputs() {
    assert!(super::run(&[]).unwrap_err().contains("usage:"));
    let mut args = ["missing", "missing", "missing", "short", "word"].map(str::to_owned);
    assert!(super::run(&args).unwrap_err().contains("full 40-character"));
    args[4] = "invalid".into();
    assert!(super::run(&args).unwrap_err().contains("word or piece"));
}

fn error_case_args(dir: &std::path::Path) -> Vec<String> {
    let corpus = dir.join("input.conll");
    std::fs::write(&corpus, "Ann X X B-PER\n").unwrap();
    vec![
        dir.to_string_lossy().into_owned(),
        corpus.to_string_lossy().into_owned(),
        corpus.to_string_lossy().into_owned(),
        "0000000000000000000000000000000000000001".into(),
        "word".into(),
    ]
}

#[test]
fn bundle_open_failure_keeps_the_original_cause() {
    let dir = tiny::bundle("conll-eval-missing-manifest");
    let args = error_case_args(&dir);
    std::fs::remove_file(dir.join("manifest.json")).unwrap();
    let cause = super::Bundle::open(&dir).err().unwrap().to_string();
    let error = super::run(&args).unwrap_err();
    std::fs::remove_dir_all(dir).unwrap();
    assert!(error.contains(&cause), "missing cause {cause:?} in {error:?}");
}

#[test]
fn finder_open_failure_keeps_the_original_cause() {
    let dir = tiny::bundle("conll-eval-sequence-head");
    let args = error_case_args(&dir);
    let path = dir.join("manifest.json");
    let mut manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    manifest["classifier"]["kind"] = serde_json::json!("CLASSIFIER_SEQUENCE");
    std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    super::Bundle::open(&dir).unwrap();
    let cause = super::NameFinderDl::open(&dir, &super::NameFinderDlOptions::default()).err().unwrap().to_string();
    assert!(cause.contains("not a token classifier"), "{cause}");
    let error = super::run(&args).unwrap_err();
    std::fs::remove_dir_all(dir).unwrap();
    assert!(error.contains(&cause), "missing cause {cause:?} in {error:?}");
}
