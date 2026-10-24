//! Runs real mixed text through the published language model and the
//! fitted calibration, including pipeline alignment and stale profiles.
#[path = "../examples/calibration_support/mod.rs"]
mod calibration_support;
use calibration_support::{collect, transformed, verify};
use turbo_analysis::{Analyzer, Annotator, Document};
use turbo_analysis_layers::LanguageSpans;
use turbo_calibration::{
    Calibration,
    metrics::{reliability, reliability_selected},
};
use turbo_langdetect::{LanguageDetector, LanguageModel};
use turbo_langspan::{CalibratedSpanDetector, Settings, SpanDetector, calibration_identity, gold::parse};
use turbo_maxent::ModelFile;
use turbo_train::calibration::{Method, fit};

fn detector() -> LanguageDetector {
    let path = turbo_goldens::published_model("langdetect-183.bin").expect("published language model required");
    LanguageDetector::new(LanguageModel::open(ModelFile::read(path).unwrap()).unwrap())
}
#[test]
fn calibrated_real_text_keeps_spans_and_measures_the_selected_language() {
    let detector = SpanDetector::new(detector(), Settings::default());
    let dev = parse(include_str!("data/mixed-dev.txt")).unwrap();
    let test = parse(include_str!("data/mixed-test.txt")).unwrap();
    let a = collect(&detector, &dev[..20]);
    let b = collect(&detector, &dev[20..]);
    let test_rows = collect(&detector, &test);
    let identity = calibration_identity(&detector, "mixed-authored-v1");
    let profile = fit(identity.clone(), Method::Temperature, a.samples("fit20"), b.samples("conformal10")).unwrap();
    let profile = Calibration::from_bytes(&profile.to_bytes(), &identity).unwrap();
    let raw = reliability_selected(&test_rows.rows, &test_rows.gold, &test_rows.selected, 10).unwrap();
    let rows = transformed(&test_rows, &profile).unwrap();
    let measured = reliability_selected(&rows, &test_rows.gold, &test_rows.selected, 10).unwrap();
    assert_eq!(measured.samples, 82);
    assert!((raw.ece - 0.772541765).abs() < 1e-8);
    assert!((measured.ece - 0.045177907).abs() < 1e-8);
    let accepted: Vec<_> = rows.iter().enumerate().filter(|(i, r)| r[test_rows.selected[*i]] >= 0.9).collect();
    assert_eq!(accepted.len(), 81);
    assert_eq!(accepted.iter().filter(|(i, _)| test_rows.selected[*i] == test_rows.gold[*i]).count(), 78);
    // A selected second-best label must not silently use top-label ECE.
    let wrong = vec![test_rows.gold[0]; test_rows.rows.len()];
    let selected = reliability_selected(&test_rows.rows, &test_rows.gold, &wrong, 10).unwrap();
    assert_ne!(selected.accuracy, reliability(&test_rows.rows, &test_rows.gold, 10).unwrap().accuracy);
    assert!(reliability_selected(&rows, &test_rows.gold, &[0], 10).is_err());
    assert!(reliability_selected(&rows, &test_rows.gold, &vec![usize::MAX; rows.len()], 10).is_err());
    let calibrated = CalibratedSpanDetector::new(detector, profile, "mixed-authored-v1").unwrap();
    verify(&calibrated, &test).unwrap();
    calibration_support::report("test", &test_rows, calibrated.profile(), 0.9).unwrap();
    let expected_hash = calibrated.config_hash();
    assert_ne!(expected_hash, calibrated.detector().config_hash());
    let analyzer = Analyzer::builder().add(calibrated).build().unwrap();
    let mut docs: Vec<_> = test.iter().map(|d| Document::new(d.text.clone())).collect();
    for result in analyzer.run(&mut docs) {
        result.unwrap();
    }
    for (i, doc) in docs.iter().enumerate() {
        let spans = doc.get::<LanguageSpans>().unwrap();
        for run in spans.iter() {
            assert!(test[i].text.get(run.span.range()).is_some());
            assert!(run.confidence.is_finite() && (0.0..=1.0).contains(&run.confidence));
        }
    }
}
#[test]
fn actual_profile_rejects_changed_candidates_settings_and_domain() {
    let source = SpanDetector::new(detector(), Settings::default());
    let docs = parse(include_str!("data/mixed-dev.txt")).unwrap();
    let a = collect(&source, &docs[..20]);
    let b = collect(&source, &docs[20..]);
    let profile =
        fit(calibration_identity(&source, "mixed-v1"), Method::Temperature, a.samples("fit20"), b.samples("reserve10"))
            .unwrap();
    assert!(CalibratedSpanDetector::new(source, profile.clone(), "another-domain").is_err());
    let restricted = SpanDetector::restricted(detector(), Settings::default(), |code| code != "war");
    assert!(CalibratedSpanDetector::new(restricted, profile.clone(), "mixed-v1").is_err());
    let mut settings = Settings::default();
    settings.clause_letters += 1;
    let changed = SpanDetector::new(detector(), settings);
    assert!(CalibratedSpanDetector::new(changed, profile, "mixed-v1").is_err());
}
