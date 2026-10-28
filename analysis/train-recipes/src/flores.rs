//! FLORES+ evaluation for the language identification gate (#1525).

use turbo_langdetect::NgramModel;

/// One FLORES+ evaluation example.
#[derive(Clone, Debug)]
pub struct Example<'a> {
    pub text: &'a str,
    pub language: &'a str,
}

/// Evaluates a model on FLORES+ examples, returning accuracy.
pub fn evaluate(model: &NgramModel, examples: &[Example]) -> f32 {
    if examples.is_empty() {
        return 0.0;
    }
    let correct = examples.iter().filter(|e| model.predict(e.text).is_some_and(|p| p == e.language)).count();
    correct as f32 / examples.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::ModelRef;

    #[test]
    fn evaluates_accuracy() {
        let model = NgramModel::new(
            vec!["en".into()],
            vec![0.0; 16],
            vec![0.0; 1],
            (1, 3),
            16,
            ModelRef { id: "test".into(), revision: "1".into(), sha256: "0".into() },
        )
        .unwrap();
        let examples = [Example { text: "hello", language: "en" }, Example { text: "world", language: "de" }];
        let acc = evaluate(&model, &examples);
        assert!((acc - 0.5).abs() < 1e-6);
    }
}
