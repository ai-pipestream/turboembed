//! Reading the goldens and dumps `turbo-goldens generate` wrote for the
//! chunker (analysis/goldens/data/chunk, format in docs/analysis.md),
//! through `turbo_goldens`.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use turbo_goldens::{Dump, Golden, data_dir, sha512};

/// The generator's fixtures, `chunker-fixture.bin` among them.
pub fn fixtures_dir() -> PathBuf {
    data_dir().join("fixtures")
}

/// One golden of a run and its input text, checked against the header.
pub struct Case {
    pub path: PathBuf,
    pub golden: Golden,
    pub text: String,
}

impl Case {
    pub fn set(&self) -> &str {
        &self.golden.header.input.set
    }

    pub fn id(&self) -> &str {
        &self.golden.header.input.id
    }

    /// The beam size of the run's `config`, when it names one: a run on a
    /// model rebuilt with another beam size. The runs have no other option.
    pub fn beam_size(&self) -> Option<usize> {
        let config = &self.golden.header.config;
        let beam = config
            .iter()
            .find(|c| c.key == "beamSize")
            .map(|c| c.value.parse().unwrap_or_else(|e| panic!("{}: {e}", self.path.display())));
        assert_eq!(config.len(), usize::from(beam.is_some()), "{}: {config:?}", self.path.display());
        beam
    }

    /// Asserts the annotator that ran.
    #[track_caller]
    pub fn assert_annotator(&self) {
        let a = &self.golden.header.annotator;
        assert_eq!(
            (a.class.as_str(), a.method.as_str()),
            ("opennlp.tools.chunker.ChunkerME", "chunk"),
            "{}",
            self.path.display()
        );
    }

    /// Asserts the model records: the chunker model at `path`, by name
    /// and SHA-512, then the sentence, token and part-of-speech models
    /// that cut and tagged the input.
    #[track_caller]
    pub fn assert_models(&self, path: &Path) {
        let models = &self.golden.header.models;
        assert_eq!(models.len(), 4, "{}", self.path.display());
        assert_eq!(Some(models[0].file.as_str()), path.file_name().and_then(|n| n.to_str()));
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(sha512(&bytes), models[0].sha512, "{}: {}", self.path.display(), path.display());
        assert!(models[1].file.contains("-sentence-") && models[2].file.contains("-tokens-"), "{models:?}");
        assert!(models[3].file.contains("-pos-"), "{models:?}");
    }

    /// The decision point dump beside the golden.
    pub fn dump(&self) -> Dump {
        let d = Dump::read(self.path.with_extension("dump")).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.header, self.golden.header, "{}", self.path.display());
        d
    }
}

/// Every golden of `chunk/<run>`, over all its input sets, sorted by set
/// and id.
pub fn run(name: &str) -> Vec<Case> {
    let dir = data_dir().join("chunk").join(name);
    let mut paths = Vec::new();
    for set in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        for f in std::fs::read_dir(set.unwrap().path()).unwrap() {
            let p = f.unwrap().path();
            if p.extension().is_some_and(|x| x == "golden") {
                paths.push(p);
            }
        }
    }
    paths.sort();
    assert!(!paths.is_empty(), "no goldens in {}", dir.display());
    paths
        .into_iter()
        .map(|path| {
            let golden = Golden::read(&path).unwrap_or_else(|e| panic!("{e}"));
            let text = golden.input_text(data_dir()).unwrap_or_else(|e| panic!("{e}"));
            Case { path, golden, text }
        })
        .collect()
}
