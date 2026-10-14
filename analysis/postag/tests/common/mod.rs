//! Reading the goldens and dumps `turbo-goldens generate` wrote for the
//! part-of-speech tagger (analysis/goldens/data/postag, format in
//! docs/analysis.md), through `turbo_goldens`.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use turbo_goldens::{Dump, Golden, data_dir, sha512};
use turbo_postag::PosTagFormat;

/// The published part-of-speech models, under tests/models.
pub fn models_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models")
}

/// The generator's fixture models, `pos-fixture.bin` among them.
pub fn fixtures_dir() -> PathBuf {
    data_dir().join("fixtures")
}

/// How a run's tagger differed from the model as it is: the golden's
/// `config` record.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Setting {
    /// The model as it is, asked for UD tags.
    Default,
    /// Asked for another tag format (`config format <NAME>`).
    Format(PosTagFormat),
    /// Rebuilt with another beam size (`config beamSize <n>`).
    BeamSize(usize),
}

impl Setting {
    /// What the run's directory appends to the model's base name.
    pub fn suffix(self) -> String {
        match self {
            Setting::Default => String::new(),
            Setting::Format(f) => format!("+{}", f.as_str().to_ascii_lowercase()),
            Setting::BeamSize(n) => format!("+beam{n}"),
        }
    }
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

    /// The setting the run used; no other option is set.
    pub fn setting(&self) -> Setting {
        let config = &self.golden.header.config;
        match config.iter().map(|c| (c.key.as_str(), c.value.as_str())).collect::<Vec<_>>()[..] {
            [] => Setting::Default,
            [("format", "PENN")] => Setting::Format(PosTagFormat::Penn),
            [("format", "CUSTOM")] => Setting::Format(PosTagFormat::Custom),
            [("format", "UNKNOWN")] => Setting::Format(PosTagFormat::Unknown),
            [("beamSize", n)] => Setting::BeamSize(n.parse().unwrap()),
            _ => panic!("{}: options {config:?}", self.path.display()),
        }
    }

    /// Asserts the annotator that ran.
    #[track_caller]
    pub fn assert_annotator(&self, class: &str) {
        let a = &self.golden.header.annotator;
        assert_eq!((a.class.as_str(), a.method.as_str()), (class, "tag"), "{}", self.path.display());
    }

    /// Asserts the model records: the part-of-speech model at `path`, by
    /// name and SHA-512, then the sentence and token models that cut the
    /// input.
    #[track_caller]
    pub fn assert_models(&self, path: &Path) {
        let models = &self.golden.header.models;
        assert_eq!(models.len(), 3, "{}", self.path.display());
        assert_eq!(Some(models[0].file.as_str()), path.file_name().and_then(|n| n.to_str()));
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(sha512(&bytes), models[0].sha512, "{}: {}", self.path.display(), path.display());
        assert!(models[1].file.contains("-sentence-") && models[2].file.contains("-tokens-"), "{models:?}");
    }

    /// The decision point dump beside the golden.
    pub fn dump(&self) -> Dump {
        let d = Dump::read(self.path.with_extension("dump")).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.header, self.golden.header, "{}", self.path.display());
        d
    }
}

/// Every golden of `postag/<run>`, over all its input sets, sorted by set
/// and id.
pub fn run(name: &str) -> Vec<Case> {
    let dir = data_dir().join("postag").join(name);
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
