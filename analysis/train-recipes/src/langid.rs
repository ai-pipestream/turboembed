//! Language identification trained on Tatoeba sentences (#1525).

use std::path::{Path, PathBuf};

use turbo_analysis::ModelRef;
use turbo_langdetect::{NgramModel, ngram::MODEL_ENTRY};
use turbo_trainer::langid::{Config, Example, train};

use crate::error::{Error, Result};

/// Trains a loadable language identification model on `(text, language)` pairs.
pub fn train_langid(samples: &[(String, String)], languages: &[String], model_ref: ModelRef) -> Result<NgramModel> {
    let config = Config::default();
    let examples: Vec<Example> = samples
        .iter()
        .filter_map(|(text, lang)| languages.iter().position(|l| l == lang).map(|language| Example { text, language }))
        .collect();
    let (weights, bias) = train::<f32>(&examples, languages.len(), &config);
    NgramModel::from_trained(languages.to_vec(), weights, bias, config.ngram_range, config.buckets, model_ref)
        .map_err(|e| Error::Model(e.to_string()))
}

/// Writes the native model entry into `dir` and returns its path.
pub fn write_langid(model: &NgramModel, dir: &Path) -> Result<PathBuf> {
    let path = dir.join(MODEL_ENTRY);
    std::fs::write(&path, model.to_bytes()).map_err(|e| Error::Io { path: path.clone(), message: e.to_string() })?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trained_model_loads_and_predicts() {
        let samples = vec![
            ("hello world".to_owned(), "eng".to_owned()),
            ("hallo welt".to_owned(), "deu".to_owned()),
            ("hello there".to_owned(), "eng".to_owned()),
            ("hallo dort".to_owned(), "deu".to_owned()),
        ];
        let languages = vec!["eng".to_owned(), "deu".to_owned()];
        let model_ref = ModelRef { id: "test-langid".into(), revision: "1".into(), sha256: "0".into() };
        let model = train_langid(&samples, &languages, model_ref.clone()).unwrap();
        assert_eq!(model.predict("hello"), Some("eng"));
        assert_eq!(model.predict("hallo"), Some("deu"));
        let dir = std::env::temp_dir().join(format!("langid-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_langid(&model, &dir).unwrap();
        assert_eq!(path.file_name().unwrap(), MODEL_ENTRY);
        let loaded = NgramModel::from_bytes(&std::fs::read(&path).unwrap(), model_ref).unwrap();
        assert_eq!(loaded.predict("hello"), Some("eng"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
