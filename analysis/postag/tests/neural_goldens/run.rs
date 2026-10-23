//! The runs: each one's goldens, model file, settings and corpus.

use std::path::Path;

use turbo_goldens::{Golden, Shape};
use turbo_postag::neural::{BilstmSettings, FeedforwardSettings, PosSample};

use crate::golden::*;

/// The value of a golden's `config` record.
pub(crate) fn config<'g>(g: &'g Golden, key: &str) -> Option<&'g str> {
    g.header.config.iter().find(|c| c.key == key).map(|c| c.value.as_str())
}

/// The value of `name=` in a Java record's `toString`.
pub(crate) fn setting<T: std::str::FromStr>(settings: &str, name: &str) -> T {
    let at = settings.find(&format!("[{name}=")).or_else(|| settings.find(&format!(" {name}=")));
    let start = at.unwrap_or_else(|| panic!("no {name} in {settings}")) + name.len() + 2;
    let end = settings[start..].find([',', ']']).map_or(settings.len(), |e| start + e);
    settings[start..end].parse().unwrap_or_else(|_| panic!("{name} in {settings}"))
}

pub(crate) fn bilstm_settings(s: &str) -> BilstmSettings {
    BilstmSettings::new(
        setting(s, "wordEmbeddingSize"),
        setting(s, "charEmbeddingSize"),
        setting(s, "charHiddenSize"),
        setting(s, "hiddenSize"),
        setting(s, "epochs"),
        setting(s, "batchSize"),
        setting(s, "learningRate"),
        setting(s, "clipNorm"),
        setting(s, "dropout"),
        setting(s, "wordCutoff"),
        setting(s, "maxWordLength"),
        setting(s, "seed"),
        setting(s, "threads"),
        setting(s, "wordDropout"),
        setting(s, "learningRateHalfLife"),
        setting(s, "crf"),
        setting(s, "encoderLayers"),
        setting(s, "pretrainedDropout"),
        setting(s, "encoderDropout"),
        setting(s, "auxLossWeight"),
        setting(s, "pretrainedTuning"),
        setting(s, "pretrainedAdapter"),
    )
    .unwrap()
}

pub(crate) fn feedforward_settings(s: &str) -> FeedforwardSettings {
    FeedforwardSettings::new(
        setting(s, "embeddingSize"),
        setting(s, "hiddenSize"),
        setting(s, "epochs"),
        setting(s, "batchSize"),
        setting(s, "learningRate"),
        setting(s, "l2"),
        setting(s, "dropout"),
        setting(s, "wordCutoff"),
        setting(s, "suffixCutoff"),
        setting(s, "seed"),
    )
    .unwrap()
}

/// One run's two goldens, its model file and its training corpus.
pub(crate) struct Run {
    pub(crate) name: String,
    pub(crate) training: Golden,
    pub(crate) tokens: Golden,
    pub(crate) bytes: Vec<u8>,
    pub(crate) samples: Vec<PosSample>,
    /// The non-empty samples' sentences, which the training golden tags.
    pub(crate) train_sentences: Vec<Vec<String>>,
}

impl Run {
    pub(crate) fn read(dir: &Path, shapes: &[Shape]) -> Run {
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let read = |file: &str| {
            let path = dir.join(file);
            let g = Golden::read_with(&path, shapes).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(config(&g, "run"), Some(name.as_str()), "{}", path.display());
            g
        };
        let training = read("tagged/wordtag.golden");
        let tokens = read("sequence/sentences.golden");
        // The fixed token list is the checked in input.
        tokens.input_text(turbo_goldens::data_dir()).unwrap_or_else(|e| panic!("{e}"));
        if let Some(v) = config(&training, "vectors") {
            assert_eq!(v, "length*0.1f,0.5f", "{name}: the vectors this test gives the trainers");
        }
        let bytes = model_bytes(&training);
        let samples = corpus(&training);
        let train_sentences: Vec<Vec<String>> =
            samples.iter().filter(|s| !s.sentence.is_empty()).map(|s| s.sentence.clone()).collect();
        assert_eq!(train_sentences, sentences(&training), "{name}: training sentences");
        Run { name, training, tokens, bytes, samples, train_sentences }
    }

    pub(crate) fn is_bilstm(&self) -> bool {
        let annotator = self.training.header.annotator.class.as_str();
        let bilstm = annotator.ends_with(".BilstmPOSTagger");
        assert!(bilstm || annotator.ends_with(".FeedforwardPOSTagger"), "{}: annotator {annotator}", self.name);
        bilstm
    }

    pub(crate) fn settings(&self) -> &str {
        config(&self.training, "settings").expect("the settings record")
    }

    pub(crate) fn with_vectors(&self) -> bool {
        config(&self.training, "vectors").is_some()
    }

    pub(crate) fn goldens(&self) -> [(String, &Golden); 2] {
        [
            (format!("{}/tagged/wordtag", self.name), &self.training),
            (format!("{}/sequence/sentences", self.name), &self.tokens),
        ]
    }
}

/// The runs under `postag-neural`, read and checked against their inputs.
pub(crate) fn runs() -> Vec<Run> {
    let root = turbo_goldens::data_dir().join("postag-neural");
    assert!(root.is_dir(), "{} is absent; write the goldens with `{GENERATE}`", root.display());
    let shapes = shapes();
    let runs: Vec<Run> = sorted(&root).iter().map(|dir| Run::read(dir, &shapes)).collect();
    assert_eq!(runs.len(), 8, "{}: runs", root.display());
    runs
}
