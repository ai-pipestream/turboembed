use std::collections::BTreeMap;
use turbo_trainer_recipes::corpus::{CorpusRecord, Sentence};
use turbo_trainer_recipes::ner_compare::{PredictionFile, Request, compare};

fn corpus() -> CorpusRecord {
    serde_json::from_value(serde_json::json!({
        "id":"wnut17", "name":"test", "task":"bio", "licence":"test", "spdx":"MIT",
        "terms":"test", "attribution":"test", "recorded":false,"split":"test",
        "train":{"sentences":30,"words":60},"dev":{"sentences":0,"words":0},
        "test":{"sentences":30,"words":60},"skipped":0
    }))
    .unwrap()
}
fn gold() -> Vec<Sentence> {
    (0..30)
        .map(|_| Sentence { words: vec!["A".into(), "B".into()], labels: vec!["B-person".into(), "B-location".into()] })
        .collect()
}
fn file(a: &str, b: &str) -> String {
    format!("A\t{a}\nB\t{b}\n\n").repeat(30)
}
fn run(
    candidate: &str,
    peer: &str,
    mapping: &BTreeMap<String, String>,
) -> turbo_trainer_recipes::Result<turbo_trainer_recipes::ner_compare::Report> {
    let corpus = corpus();
    let gold = gold();
    let selected = vec!["person".into()];
    let train = vec![Sentence { words: vec!["train".into()], labels: vec!["O".into()] }];
    compare(Request {
        corpus: &corpus,
        gold: &gold,
        candidate: PredictionFile { name: "candidate.bio", bytes: candidate.as_bytes() },
        peer: PredictionFile { name: "peer.bio", bytes: peer.as_bytes() },
        train: &train,
        selected_types: &selected,
        peer_mapping: mapping,
        resamples: 100,
        seed: 7,
    })
}
#[test]
fn signed_deltas_digests_exclusions_and_deterministic_bootstrap() {
    let good = file("B-person", "B-location");
    let bad = file("O", "O");
    let mapping = BTreeMap::new();
    let positive = run(&good, &bad, &mapping).unwrap();
    assert_eq!(positive.candidate_vs_peer.delta, 1.0);
    assert_eq!(positive.candidate_vs_peer.low, 1.0);
    assert_eq!(positive.candidate_vs_peer.high, 1.0);
    assert_eq!(positive, run(&good, &bad, &mapping).unwrap());
    assert_eq!(positive.candidate.excluded["location"], 30);
    assert_eq!(positive.gold_excluded["location"], 30);
    assert_eq!(positive.candidate_file.sha256, turbo::bundle::sha256_hex(good.as_bytes()));
    assert_eq!(positive.candidate.full_metrics.score(), 1.0);
    assert_eq!(run(&bad, &good, &mapping).unwrap().candidate_vs_peer.delta, -1.0);
    let equal = run(&good, &good, &mapping).unwrap();
    assert_eq!((equal.candidate_vs_peer.low, equal.candidate_vs_peer.high), (0.0, 0.0));
    let json = serde_json::to_string(&positive).unwrap();
    assert_eq!(serde_json::from_str::<turbo_trainer_recipes::ner_compare::Report>(&json).unwrap(), positive);
}
#[test]
fn explicit_mapping_and_original_bio_validation() {
    let gold = file("B-person", "B-location");
    let foreign = file("B-PER", "O");
    assert!(run(&gold, &foreign, &BTreeMap::new()).is_err());
    let dropped = BTreeMap::from([("PER".into(), "-".into())]);
    let report = run(&gold, &foreign, &dropped).unwrap();
    assert_eq!(report.peer.excluded["PER"], 30);
    assert!(run(&gold, &file("I-PER", "O"), &dropped).is_err());
    let mapping = BTreeMap::from([("PER".into(), "person".into()), ("OTHER".into(), "location".into())]);
    assert_eq!(run(&gold, &foreign, &mapping).unwrap().peer.metrics.score(), 1.0);
    assert!(run(&gold, &file("I-PER", "O"), &mapping).is_err());
    // Mapping both original types to one type must not repair an illegal transition.
    let collapse = BTreeMap::from([("PER".into(), "person".into()), ("OTHER".into(), "person".into())]);
    assert!(run(&gold, &file("B-PER", "I-OTHER"), &collapse).is_err());
    assert!(run(&file("B-person", "I-location"), &gold, &mapping).is_err());
    assert!(run(&gold, &foreign.replace("A\t", "X\t"), &mapping).is_err());
}
#[test]
fn refuses_small_empty_unsupported_and_misaligned_inputs() {
    let corpus = corpus();
    let mut gold = gold();
    let bytes = file("O", "O");
    let selected = vec!["person".into()];
    let mapping = BTreeMap::new();
    let train = vec![Sentence { words: vec!["train".into()], labels: vec!["O".into()] }];
    let attempt = |gold: &[Sentence], selected: &[String], train: &[Sentence]| {
        compare(Request {
            corpus: &corpus,
            gold,
            candidate: PredictionFile { name: "c", bytes: bytes.as_bytes() },
            peer: PredictionFile { name: "p", bytes: bytes.as_bytes() },
            train,
            selected_types: selected,
            peer_mapping: &mapping,
            resamples: 10,
            seed: 1,
        })
    };
    assert!(attempt(&gold, &[], &train).is_err());
    assert!(attempt(&gold, &["unknown".into()], &train).is_err());
    assert!(attempt(&gold, &["corporation".into()], &train).is_err());
    assert!(attempt(&gold, &selected, &[]).is_err());
    let malformed = vec![Sentence { words: vec!["x".into()], labels: vec!["I-person".into()] }];
    assert!(attempt(&gold, &selected, &malformed).is_err());
    let winning_i = vec![Sentence {
        words: vec!["a".into(), "b".into(), "c".into()],
        labels: vec!["B-person".into(), "I-person".into(), "I-person".into()],
    }];
    assert!(attempt(&gold, &selected, &winning_i).is_err());
    let tie = vec![Sentence { words: vec!["a".into(), "b".into()], labels: vec!["O".into(), "B-person".into()] }];
    assert!(attempt(&gold, &selected, &tie).unwrap().baseline.metrics.score() > 0.0);
    gold.pop();
    assert!(attempt(&gold, &selected, &train).is_err());
}

#[test]
fn sentence_pairing_keeps_identical_mixed_predictions_at_zero() {
    let mixed = format!("{}{}", "A\tB-person\nB\tO\n\n".repeat(15), "A\tO\nB\tO\n\n".repeat(15));
    let report = run(&mixed, &mixed, &BTreeMap::new()).unwrap();
    assert_eq!(report.candidate_vs_peer.delta, 0.0);
    assert_eq!(report.candidate_vs_peer.low, 0.0);
    assert_eq!(report.candidate_vs_peer.high, 0.0);
}
