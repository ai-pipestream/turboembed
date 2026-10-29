//! FLORES+ evaluation for the language identification gate (#1525).

use turbo_langdetect::NgramModel;

/// The pinned FLORES+ revision used for evaluation only.
pub const FLORES_PLUS_REVISION: &str = "1147e2dc59f37fbb5c22fd8478770de02cd399f9";

/// One FLORES+ evaluation example.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Example<'a> {
    pub text: &'a str,
    pub language: &'a str,
}

/// Reads the converter's complete `language<TAB>text` gold output.
pub fn read_gold(source: &str) -> Result<Vec<Example<'_>>, String> {
    let mut examples = Vec::new();
    for (index, line) in source.lines().enumerate() {
        let (language, text) = line.split_once('\t').ok_or_else(|| format!("line {} has no tab", index + 1))?;
        if language.len() != 3 || !language.bytes().all(|b| b.is_ascii_lowercase()) {
            return Err(format!("line {} has an invalid language code", index + 1));
        }
        if text.trim().is_empty() || text.contains('\t') {
            return Err(format!("line {} has invalid text", index + 1));
        }
        examples.push(Example { text, language });
    }
    if examples.is_empty() {
        return Err("the gold set is empty".into());
    }
    Ok(examples)
}

/// Counts correct predictions over a nonempty gold set.
pub fn correct(model: &NgramModel, examples: &[Example]) -> usize {
    examples.iter().filter(|e| model.predict(e.text).is_some_and(|p| p == e.language)).count()
}

/// Evaluates a model on FLORES+ examples, returning accuracy.
pub fn evaluate(model: &NgramModel, examples: &[Example]) -> f32 {
    if examples.is_empty() {
        return 0.0;
    }
    correct(model, examples) as f32 / examples.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::ModelRef;

    fn model() -> NgramModel {
        NgramModel::new(
            vec!["eng".into()],
            vec![0.0; 16],
            vec![0.0; 1],
            (1, 3),
            16,
            ModelRef { id: "test".into(), revision: "1".into(), sha256: "0".into() },
        )
        .unwrap()
    }

    #[test]
    fn evaluates_accuracy() {
        let examples = [Example { text: "hello", language: "eng" }, Example { text: "world", language: "deu" }];
        assert_eq!(evaluate(&model(), &examples), 0.5);
        assert_eq!(correct(&model(), &examples), 1);
    }

    #[test]
    fn gold_reader_is_strict_and_nonempty() {
        assert_eq!(read_gold("eng\thello\ndeu\twelt\n").unwrap().len(), 2);
        assert!(read_gold("").is_err());
        assert!(read_gold("eng\n").is_err());
        assert!(read_gold("eng\t\n").is_err());
        assert!(read_gold("english\thello\n").is_err());
    }
}
