//! End-to-end test of the n-gram language model.

use turbo_analysis::ModelRef;
use turbo_langdetect::NgramModel;

#[test]
fn trained_model_predicts_language() {
    let model = NgramModel::new(
        vec!["en".into(), "de".into()],
        vec![0.0; 2 * 16],
        vec![0.0; 2],
        (1, 3),
        16,
        ModelRef { id: "test".into(), revision: "1".into(), sha256: "0".into() },
    )
    .unwrap();
    let scores = model.scores("hello world");
    assert_eq!(scores.len(), 2);
    assert!(model.predict("hello world").is_some());
}
