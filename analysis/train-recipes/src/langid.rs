//! Language identification trained on Tatoeba sentences (#1525).

use turbo_trainer::langid::{Config, Example, train};

use crate::error::Result;

/// Trains a language identification model on `(text, language)` pairs.
pub fn train_langid(samples: &[(String, String)], languages: &[String]) -> Result<(Vec<f32>, Vec<f32>)> {
    let config = Config::default();
    let examples: Vec<Example> = samples
        .iter()
        .filter_map(|(text, lang)| languages.iter().position(|l| l == lang).map(|language| Example { text, language }))
        .collect();
    let (weights, bias) = train::<f32>(&examples, languages.len(), &config);
    Ok((weights, bias))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trains_on_toy_sentences() {
        let samples = vec![
            ("hello world".to_owned(), "en".to_owned()),
            ("hallo welt".to_owned(), "de".to_owned()),
            ("hello there".to_owned(), "en".to_owned()),
            ("hallo dort".to_owned(), "de".to_owned()),
        ];
        let languages = vec!["en".to_owned(), "de".to_owned()];
        let (weights, bias) = train_langid(&samples, &languages).unwrap();
        assert!(!weights.is_empty());
        assert_eq!(bias.len(), 2);
    }
}
