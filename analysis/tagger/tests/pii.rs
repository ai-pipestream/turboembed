//! End-to-end tests for `PiiTagger` (`turboembed:pii` -> `turbo:pii`, `#1888`).

mod common;

use common::bundle;
use serde_json::json;
use turbo_analysis::{Analyzer, Annotator, Document, Layer};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Pii, PiiCandidates, PiiKind, PiiScoreKind};
use turbo_pii::{Pack, PiiDetector};
use turbo_tagger::{PII_TAGGER_DETECTOR, PiiTagger};
use turbo_tokenize::SimpleTokenizer;

fn pii_bundle(name: &str) -> std::path::PathBuf {
    bundle(name, |m| {
        m["heads"] = json!([{
            "name": "pii",
            "layer": "turboembed:pii",
            "kind": "HEAD_TOKEN_TAGGER",
            "labels": [
                "O",
                "B-first_name",
                "I-first_name",
                "B-medical_record_number",
                "I-medical_record_number",
                "B-ssn",
                "I-ssn"
            ],
            "scheme": "SCHEME_BIO",
            "word_pooling": "WORD_POOLING_FIRST",
            "decode": "DECODE_ARGMAX",
            "artifact": "weights-f32",
            "tensor_prefix": "heads.pii"
        }]);
    })
}

#[test]
fn pii_tagger_runs_in_analyzer_with_rules_and_temperature_calibration() {
    let dir = pii_bundle("pii-analyzer");
    let text = "Patient Jane Doe has SSN 123-45-6789 and email desk@example.com in chart 884210.";

    // Uncalibrated run without rules: produces neural candidates with calibrated == false.
    let raw_tagger = PiiTagger::open(&dir, None, &EncoderOptions::default())
        .unwrap()
        .with_model_identity("turbo-en-nemotron-pii-modernbert-base-1", "1")
        .unwrap();
    let raw_analyzer = Analyzer::builder().add(SimpleTokenizer::default()).add(raw_tagger).build().unwrap();
    let mut raw_docs = vec![Document::new(text)];
    for r in raw_analyzer.run(&mut raw_docs) {
        r.unwrap();
    }
    let raw_pii = raw_docs[0].get::<Pii>().unwrap();
    let raw_cand = raw_docs[0].get::<PiiCandidates>().unwrap();
    raw_pii.check().unwrap();
    raw_cand.check().unwrap();
    raw_cand.check_against(raw_pii).unwrap();
    assert!(!raw_pii.is_empty(), "seeded BIO head produces neural PII mentions");
    for m in raw_pii.mentions() {
        let s = m.score.expect("neural mentions carry PiiScore");
        assert_eq!(s.kind, PiiScoreKind::Probability);
        assert!(!s.calibrated);
        assert!((0.0..=1.0).contains(&s.value));
        let (model_id, model_version) = raw_pii.detectors.get(s.model).unwrap();
        assert_eq!(model_id, "turbo-en-nemotron-pii-modernbert-base-1");
        assert_eq!(model_version, "1");
        let (det_id, _) = raw_pii.detectors.get(m.detector).unwrap();
        assert_eq!(det_id, PII_TAGGER_DETECTOR);
    }

    // Calibrated run with rules (AllStructured) and T = 2.5:
    // structured SSN and email rules win on overlap (score == None), while
    // non-overlapping neural mentions carry calibrated == true and softer probabilities.
    let cal_tagger = PiiTagger::open(&dir, None, &EncoderOptions::default())
        .unwrap()
        .with_model_identity("turbo-en-nemotron-pii-modernbert-base-1", "1")
        .unwrap()
        .with_temperature(2.5)
        .unwrap()
        .with_rules(PiiDetector::pack(Pack::AllStructured));
    let cal_analyzer = Analyzer::builder().add(SimpleTokenizer::default()).add(cal_tagger).build().unwrap();
    let mut cal_docs = vec![Document::new(text)];
    for r in cal_analyzer.run(&mut cal_docs) {
        r.unwrap();
    }
    let cal_pii = cal_docs[0].get::<Pii>().unwrap();
    let cal_cand = cal_docs[0].get::<PiiCandidates>().unwrap();
    cal_pii.check().unwrap();
    cal_cand.check().unwrap();
    cal_cand.check_against(cal_pii).unwrap();

    // Structured SSN and email rules must be present with score == None, and no overlapping neural candidate
    // may appear on those spans!
    let ssn_span = text.find("123-45-6789").map(|s| (s as u32, (s + 11) as u32)).unwrap();
    let ssn = cal_pii.mentions().find(|m| (m.span.start, m.span.end) == ssn_span).expect("SSN rule mention kept");
    assert_eq!(ssn.kind, PiiKind::UsSsn);
    assert!(ssn.score.is_none(), "rule mention keeps score == None");
    assert_eq!(
        cal_pii.mentions().filter(|m| m.span.start < ssn_span.1 && ssn_span.0 < m.span.end).count(),
        1,
        "only the rule mention survives on the SSN span"
    );

    let email = cal_pii.mentions().find(|m| m.kind == PiiKind::Email).expect("email rule mention kept");
    assert_eq!(&text[email.span.range()], "desk@example.com");
    assert!(email.score.is_none(), "rule mention keeps score == None");

    // Any kept neural mention has calibrated == true and softened probability under T = 2.5 > 1.0.
    for m in cal_pii.mentions().filter(|m| m.score.is_some()) {
        let s = m.score.unwrap();
        assert!(s.calibrated);
        if let Some(raw_m) = raw_pii.mentions().find(|r| r.span == m.span && r.kind == m.kind) {
            assert!(
                s.value <= raw_m.score.unwrap().value + 1e-5,
                "T > 1 softens top-class softmax confidence: {} vs {}",
                s.value,
                raw_m.score.unwrap().value
            );
        }
    }
}

#[test]
fn pii_tagger_validates_temperature_and_score_threshold_bounds() {
    let dir = pii_bundle("pii-bounds");
    assert!(PiiTagger::open(&dir, None, &EncoderOptions::default()).unwrap().with_temperature(0.0).is_err());
    assert!(PiiTagger::open(&dir, None, &EncoderOptions::default()).unwrap().with_temperature(-1.0).is_err());
    assert!(PiiTagger::open(&dir, None, &EncoderOptions::default()).unwrap().with_temperature(f32::NAN).is_err());
    assert!(PiiTagger::open(&dir, None, &EncoderOptions::default()).unwrap().with_score_threshold(-0.1).is_err());
    assert!(PiiTagger::open(&dir, None, &EncoderOptions::default()).unwrap().with_score_threshold(1.1).is_err());

    let t1 = PiiTagger::open(&dir, None, &EncoderOptions::default()).unwrap();
    let t2 = PiiTagger::open(&dir, None, &EncoderOptions::default()).unwrap().with_temperature(1.5).unwrap();
    assert_ne!(t1.config_hash(), t2.config_hash());
}
