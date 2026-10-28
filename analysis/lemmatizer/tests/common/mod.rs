//! Reading the goldens and dumps `turbo-goldens generate` wrote for the
//! lemmatizers (analysis/goldens/data/lemmatize, format in
//! docs/analysis.md), through `turbo_goldens`.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

pub use turbo_goldens::published_model;
use turbo_goldens::{Dump, Golden, data_dir, sha512};

/// The generator's fixtures, `lemmatizer-dictionary.txt` among them.
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

    /// The runs have no options.
    pub fn assert_no_config(&self) {
        assert!(self.golden.header.config.is_empty(), "{}: {:?}", self.path.display(), self.golden.header.config);
    }

    /// Asserts the annotator that ran.
    #[track_caller]
    pub fn assert_annotator(&self, class: &str) {
        let a = &self.golden.header.annotator;
        assert_eq!((a.class.as_str(), a.method.as_str()), (class, "lemmatize"), "{}", self.path.display());
    }

    /// Asserts the model records: the lemmatizer model or dictionary at
    /// `path`, by name and SHA-512, then the sentence, token and
    /// part-of-speech models that cut and tagged the input.
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

/// Every golden of `<annotator>/<run>`, over all its input sets, sorted
/// by set and id, and how many goldens the reader refused because a lemma
/// holds half of a surrogate pair: the JVM lemmatizes a token that is
/// half of an emoji (`en/emoji-split`) into that half, which is no
/// string here.
pub fn run(annotator: &str, name: &str) -> (Vec<Case>, usize) {
    let dir = data_dir().join(annotator).join(name);
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
    let mut cases = Vec::new();
    let mut surrogate_lemmas = 0;
    for path in paths {
        let golden = match Golden::read(&path) {
            Ok(g) => g,
            Err(e) if e.to_string().contains("lemma: ") && e.to_string().contains("surrogate") => {
                assert!(path.ends_with("en/emoji-split.golden"), "{e}");
                surrogate_lemmas += 1;
                continue;
            }
            Err(e) => panic!("{e}"),
        };
        let text = golden.input_text(data_dir()).unwrap_or_else(|e| panic!("{e}"));
        cases.push(Case { path, golden, text });
    }
    (cases, surrogate_lemmas)
}
