//! Reading the goldens and dumps `turbo-goldens generate` wrote for the
//! sentence detectors (analysis/goldens/data/sentdetect, format in
//! docs/analysis.md), through `turbo_goldens`.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use turbo_goldens::{Dump, Golden, data_dir, sha512};
use turbo_sentdetect::WhitespaceMode;
use turbo_unicode::utf16::byte_to_utf16;

/// A published sentence model, where `turbo-goldens fetch-models` puts
/// it: the directory in `TURBO_GOLDENS_MODELS`, or
/// `target/turbo-goldens/models` under the repository root. The models
/// are trained on CC BY-SA treebanks and are not checked in, so a test
/// that needs one says so and passes without it when it is missing,
/// unless `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, which makes a missing
/// model fail it, as CI sets it.
pub fn published_model(file: &str) -> Option<PathBuf> {
    let dir = match std::env::var_os("TURBO_GOLDENS_MODELS") {
        Some(d) => PathBuf::from(d),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/turbo-goldens/models"),
    };
    let path = dir.join(file);
    if path.is_file() {
        Some(path)
    } else if std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1") {
        panic!("{} not found and TURBO_TEST_REQUIRE_OPENNLP_MODELS=1", path.display());
    } else {
        eprintln!("{} not found: skipping the test that needs it", path.display());
        None
    }
}

/// The generator's fixture models, `en-abbrev.bin` and `en-plain.bin`
/// among them.
pub fn fixtures_dir() -> PathBuf {
    data_dir().join("fixtures")
}

/// One span with both kinds of offsets and, for the maxent detector, its
/// probability.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GoldenSpan {
    pub utf8: (u32, u32),
    pub utf16: (usize, usize),
    pub prob: Option<f64>,
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

    /// The whitespace mode the run used: `config whitespace legacy`, or
    /// the default; no other option is set.
    pub fn whitespace(&self) -> WhitespaceMode {
        let config = &self.golden.header.config;
        match config.iter().map(|c| (c.key.as_str(), c.value.as_str())).collect::<Vec<_>>()[..] {
            [] => WhitespaceMode::Unicode,
            [("whitespace", "legacy")] => WhitespaceMode::Legacy,
            _ => panic!("{}: options {config:?}", self.path.display()),
        }
    }

    /// Asserts the annotator that ran.
    #[track_caller]
    pub fn assert_annotator(&self, class: &str) {
        let a = &self.golden.header.annotator;
        assert_eq!((a.class.as_str(), a.method.as_str()), (class, "sentPosDetect"), "{}", self.path.display());
    }

    /// Asserts the one model record names `path` and its SHA-512, or that
    /// there is none.
    #[track_caller]
    pub fn assert_model(&self, path: Option<&Path>) {
        let models = &self.golden.header.models;
        match path {
            None => assert!(models.is_empty(), "{}", self.path.display()),
            Some(path) => {
                assert_eq!(models.len(), 1, "{}", self.path.display());
                assert_eq!(Some(models[0].file.as_str()), path.file_name().and_then(|n| n.to_str()));
                let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                assert_eq!(sha512(&bytes), models[0].sha512, "{}: {}", self.path.display(), path.display());
            }
        }
    }

    /// The sentences the JVM returned. No sentence starts or ends inside a
    /// surrogate pair in these goldens, so the UTF-16 offsets follow from
    /// the UTF-8 ones.
    pub fn spans(&self) -> Vec<GoldenSpan> {
        let g = &self.golden;
        assert!(g.layer("turbo:utf16<opennlp:sentences>").is_none(), "{}", self.path.display());
        let layer = g.layer("opennlp:sentences").unwrap_or_else(|| panic!("{}: no sentences", self.path.display()));
        let (starts, ends) = (layer.u32s("start").unwrap(), layer.u32s("end").unwrap());
        let prob = g.layer("turbo:confidence<opennlp:sentences>").map(|l| l.f64s("value").unwrap());
        let unit = |b: u32| byte_to_utf16(&self.text, b as usize).expect("a character boundary");
        (0..layer.len())
            .map(|i| GoldenSpan {
                utf8: (starts[i], ends[i]),
                utf16: (unit(starts[i]), unit(ends[i])),
                prob: prob.map(|p| p[i]),
            })
            .collect()
    }

    /// The decision point dump beside the golden.
    pub fn dump(&self) -> Dump {
        let d = Dump::read(self.path.with_extension("dump")).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(d.header, self.golden.header, "{}", self.path.display());
        d
    }
}

/// Every golden of `sentdetect/<run>`, over all its input sets, sorted by
/// set and id.
pub fn run(name: &str) -> Vec<Case> {
    let dir = data_dir().join("sentdetect").join(name);
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
