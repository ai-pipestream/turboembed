//! Where the runs find their models, dictionaries and fixtures: the
//! published models `turbo-goldens fetch-models` downloaded, the checked
//! in fixtures under `analysis/goldens/data/fixtures`, and the WordPiece
//! bundle under `testdata`.

use std::path::{Path, PathBuf};

/// The directories the runs load their files from.
#[derive(Clone, Debug)]
pub struct Resources {
    /// The published models and the MeCab dictionaries, as
    /// `turbo-goldens fetch-models` lays them out.
    pub models: PathBuf,
    /// `analysis/goldens/data`: the fixtures, the rate snapshot, the
    /// checked in inputs.
    pub data: PathBuf,
    /// The repository root, for `testdata/tiny-bert-bundle`.
    pub workspace: PathBuf,
}

/// Why a run cannot be opened here: a file it needs is not present, or
/// the differential does not cover the run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unavailable {
    /// A model or dictionary `fetch-models` would download is missing.
    Missing(PathBuf),
    /// The run is not one the Rust side reproduces (training, the format
    /// readers, the evaluators, the detokenizer, the BPE trainer).
    NotCovered(String),
    /// A file is present and does not load.
    Broken(String),
}

impl std::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unavailable::Missing(p) => write!(f, "{} is not present; run turbo-goldens fetch-models", p.display()),
            Unavailable::NotCovered(what) => write!(f, "{what} is not a run the differential covers"),
            Unavailable::Broken(what) => write!(f, "{what}"),
        }
    }
}

impl Resources {
    /// Resources with the models at `models` and the data directory of the
    /// `turbo-goldens` crate, which is inside the repository.
    pub fn new(models: impl Into<PathBuf>) -> Resources {
        let data = turbo_goldens::data_dir();
        let workspace = data.join("../../..");
        Resources { models: models.into(), data, workspace }
    }

    /// The models directory the tests use: `TURBO_GOLDENS_MODELS`, or
    /// `target/turbo-goldens/models` under the repository root.
    pub fn from_env() -> Resources {
        let data = turbo_goldens::data_dir();
        let models = match std::env::var_os("TURBO_GOLDENS_MODELS") {
            Some(d) => PathBuf::from(d),
            None => data.join("../../../target/turbo-goldens/models"),
        };
        Resources::new(models)
    }

    pub fn fixtures(&self) -> PathBuf {
        self.data.join("fixtures")
    }

    /// A checked in fixture by file name; fixtures are always present.
    pub fn fixture(&self, name: &str) -> Result<PathBuf, Unavailable> {
        present(self.fixtures().join(name))
    }

    /// A published model or dictionary by file or directory name.
    pub fn model(&self, name: &str) -> Result<PathBuf, Unavailable> {
        present(self.models.join(name))
    }

    /// The WordPiece bundle the subword goldens were made against.
    pub fn bert_bundle(&self) -> Result<PathBuf, Unavailable> {
        present(self.workspace.join("testdata").join("tiny-bert-bundle"))
    }

    /// The ECB rate snapshot the money goldens convert with.
    pub fn rates(&self) -> PathBuf {
        self.data.join("rates").join("ecb-snapshot.csv")
    }
}

fn present(path: PathBuf) -> Result<PathBuf, Unavailable> {
    if Path::exists(&path) { Ok(path) } else { Err(Unavailable::Missing(path)) }
}
