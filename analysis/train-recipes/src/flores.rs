//! FLORES+ evaluation for the language identification gate (#1525).

use std::collections::BTreeMap;

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

/// The model codes that differ from the converter's gold codes. The model
/// answers with the recipe's Tatoeba codes; the gold set uses ISO 639-3 at
/// macrolanguage grain for these six (see `convert_gold.py`).
pub const GOLD_CODES: [(&str, &str); 6] =
    [("ara", "arb"), ("cmn", "zho"), ("lvs", "lav"), ("pes", "fas"), ("swh", "swa"), ("zsm", "msa")];

/// The gold code a model language is scored as.
pub fn gold_code(model_code: &str) -> &str {
    GOLD_CODES.iter().find(|(model, _)| *model == model_code).map_or(model_code, |(_, gold)| gold)
}

/// Counts correct predictions over a nonempty gold set.
pub fn correct(model: &NgramModel, examples: &[Example]) -> usize {
    examples.iter().filter(|e| predicts(model, e)).count()
}

fn predicts(model: &NgramModel, example: &Example) -> bool {
    model.predict(example.text).is_some_and(|p| gold_code(p) == example.language)
}

/// Per gold language: texts and correct predictions, and whether the model
/// can answer that language at all.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LanguageScore {
    pub texts: usize,
    pub correct: usize,
    pub supported: bool,
}

/// Scores every gold language separately, in code order.
pub fn per_language(model: &NgramModel, examples: &[Example]) -> BTreeMap<String, LanguageScore> {
    let mut scores = BTreeMap::<String, LanguageScore>::new();
    for example in examples {
        let score = scores.entry(example.language.to_owned()).or_insert_with(|| LanguageScore {
            supported: model.supported_languages().any(|code| gold_code(code) == example.language),
            ..LanguageScore::default()
        });
        score.texts += 1;
        score.correct += usize::from(predicts(model, example));
    }
    scores
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
    fn model_codes_are_scored_as_their_gold_codes() {
        let model = NgramModel::new(
            vec!["cmn".into(), "eng".into()],
            vec![0.0; 32],
            vec![1.0, 0.0],
            (1, 3),
            16,
            ModelRef { id: "test".into(), revision: "1".into(), sha256: "0".into() },
        )
        .unwrap();
        let examples = [Example { text: "中文", language: "zho" }, Example { text: "abc", language: "yor" }];
        assert_eq!(correct(&model, &examples), 1);
        let scores = per_language(&model, &examples);
        assert_eq!(scores["zho"], LanguageScore { texts: 1, correct: 1, supported: true });
        assert_eq!(scores["yor"], LanguageScore { texts: 1, correct: 0, supported: false });
        assert_eq!(gold_code("eng"), "eng");
        assert_eq!(gold_code("zsm"), "msa");
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
