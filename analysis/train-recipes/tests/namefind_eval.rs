use turbo_train_recipes::namefind_eval::{Mention, Prediction, Record, Sample};

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
