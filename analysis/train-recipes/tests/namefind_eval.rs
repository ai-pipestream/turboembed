use turbo_train_recipes::namefind_eval::{Mention, Prediction, Record, Sample, evaluate};
const CORPUS: &str = include_str!("data/namefind-eval.conll");
fn fixture() -> Vec<u8> {
    std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data/fixtures/ner-bio-fixture.bin"))
        .unwrap()
}
fn record() -> Record {
    Record {
        format_version: 1,
        engine: "test".into(),
        model_sha256: "a".repeat(64),
        corpus_sha256: "b".repeat(64),
        types: "all".into(),
        samples: vec![Sample {
            tokens: 4,
            clear_adaptive_data: true,
            gold: vec![
                Mention { start: 0, end: 1, label: Some("person".into()) },
                Mention { start: 2, end: 3, label: Some("location".into()) },
            ],
            predicted: vec![Prediction {
                mention: Mention { start: 0, end: 1, label: Some("person".into()) },
                probability: 0.75,
            }],
        }],
    }
}
#[test]
fn scores_micro_counts_and_probability_parity() {
    let a = record();
    let s = a.summary().unwrap();
    assert_eq!((s.gold, s.predicted, s.correct), (2, 1, 1));
    assert_eq!((s.precision, s.recall, s.f1), (1.0, 0.5, 2.0 / 3.0));
    let mut b = a.clone();
    b.samples[0].predicted[0].probability += 5e-10;
    assert!(a.compare(&b, 1e-9).is_ok());
    b.samples[0].predicted[0].probability += 2e-9;
    assert!(a.compare(&b, 1e-9).unwrap_err().contains("probability"));
    b = a.clone();
    b.samples[0].predicted[0].mention.start = 2;
    b.samples[0].predicted[0].mention.end = 3;
    b.samples[0].predicted[0].mention.label = Some("location".into());
    assert_eq!(a.summary().unwrap(), b.summary().unwrap());
    assert!(a.compare(&b, 1e-9).unwrap_err().contains("span or type"));
}
#[test]
fn rejects_mismatched_identity_and_stream() {
    let a = record();
    for which in 0..6 {
        let mut b = a.clone();
        match which {
            0 => b.model_sha256 = "c".repeat(64),
            1 => b.corpus_sha256 = "c".repeat(64),
            2 => b.types = "per".into(),
            3 => b.samples[0].tokens = 5,
            4 => b.samples[0].clear_adaptive_data = false,
            _ => b.samples[0].gold.clear(),
        }
        assert!(a.compare(&b, 1e-9).is_err());
    }
    for tolerance in [f64::NAN, f64::INFINITY, -1.0, 2.0] {
        assert!(a.compare(&a, tolerance).is_err());
    }
}
#[test]
fn rejects_invalid_records_before_scoring() {
    for which in 0..11 {
        let mut a = record();
        match which {
            0 => a.samples.clear(),
            1 => a.format_version = 2,
            2 => a.model_sha256 = "x".repeat(64),
            3 => a.samples[0].tokens = 0,
            4 => a.samples[0].predicted[0].probability = f64::NAN,
            5 => a.samples[0].predicted[0].probability = f64::INFINITY,
            6 => a.samples[0].predicted[0].probability = 1.1,
            7 => a.samples[0].predicted[0].mention.end = 0,
            8 => a.samples[0].predicted[0].mention.end = 5,
            9 => {
                let duplicate = a.samples[0].predicted[0].clone();
                a.samples[0].predicted.push(duplicate);
            }
            _ => a.samples[0].gold.reverse(),
        }
        assert!(a.summary().is_err(), "case {which}");
    }
    let mut a = record();
    a.samples[0].gold.clear();
    a.samples[0].predicted.clear();
    assert_eq!(a.summary().unwrap().f1, -1.0);
}
#[test]
fn local_corpus_is_deterministic_and_document_reset_is_preserved() {
    let model = fixture();
    let a = evaluate(&model, CORPUS, "all").unwrap();
    assert_eq!(a.samples.iter().map(|s| s.clear_adaptive_data).collect::<Vec<_>>(), [true, false, true, false]);
    assert_eq!(a.samples.len(), 4);
    assert!(a.summary().unwrap().predicted > 0);
    assert_eq!(serde_json::to_vec(&a).unwrap(), serde_json::to_vec(&evaluate(&model, CORPUS, "all").unwrap()).unwrap());
    let b: Record = serde_json::from_slice(&serde_json::to_vec(&a).unwrap()).unwrap();
    assert_eq!(a.compare(&b, 0.0).unwrap(), 0.0);
    assert_eq!(evaluate(&model, CORPUS, "per").unwrap().summary().unwrap().gold, 4);
    assert!(evaluate(&model, "", "all").is_err());
    assert!(evaluate(&model, "word invalid columns\n", "all").is_err());
    assert!(evaluate(&model, CORPUS, "xperx").is_err());
    assert!(evaluate(b"bad-model", CORPUS, "all").is_err());
}

#[test]
fn matches_the_pinned_jvm_fixture_spans_probabilities_and_f1() {
    let reference: Record = serde_json::from_str(include_str!("data/namefind-eval-jvm.json")).unwrap();
    let actual = evaluate(&fixture(), CORPUS, "all").unwrap();
    assert!(reference.engine.starts_with("opennlp:cdf33c653d5d"));
    assert!(actual.compare(&reference, 1e-9).unwrap() <= 1e-9);
    assert_eq!(actual.summary().unwrap(), reference.summary().unwrap());
    let s = actual.summary().unwrap();
    assert_eq!((s.correct, s.predicted, s.gold), (3, 4, 7));
}
