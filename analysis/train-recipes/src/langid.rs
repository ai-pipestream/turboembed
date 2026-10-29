//! Native language identification on the pinned Tatoeba and FineWeb plans (#1525).

use std::path::{Path, PathBuf};

use turbo_analysis::ModelRef;
use turbo_langdetect::{NgramModel, ngram::MODEL_ENTRY};
use turbo_trainer::langid::{Config, Example, train};

use crate::error::{Error, Result};
use crate::fineweb::WebPlan;
use crate::langdetect::{LANGDETECT, Plan};

/// A native model and its exact held-out coverage.
#[derive(Debug)]
pub struct NativeOutcome {
    pub model: NgramModel,
    pub training: usize,
    pub held_out: usize,
    pub correct: usize,
}

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

/// Trains and scores on the checked Tatoeba plan and optional checked web plan.
pub fn train_planned(plan: &Plan, web: Option<&WebPlan>, model_ref: ModelRef) -> Result<NativeOutcome> {
    let languages: Vec<String> = LANGDETECT.languages.iter().map(|p| p.tatoeba.to_owned()).collect();
    let mut samples = Vec::new();
    for sample in &plan.training {
        samples.push((sample.text.to_string(), languages[sample.language].clone()));
    }
    if let Some(web) = web {
        for sample in &web.training {
            samples.push((sample.window.clone(), languages[sample.language].clone()));
        }
    }
    let model = train_langid(&samples, &languages, model_ref)?;
    let mut held_out = plan.held_out.len();
    let mut correct = plan
        .held_out
        .iter()
        .filter(|sample| model.predict(&sample.text).is_some_and(|p| p == languages[sample.language]))
        .count();
    if let Some(web) = web {
        held_out += web.held_out.len();
        correct += web
            .held_out
            .iter()
            .filter(|sample| model.predict(&sample.window).is_some_and(|p| p == languages[sample.language]))
            .count();
    }
    Ok(NativeOutcome { model, training: samples.len(), held_out, correct })
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
