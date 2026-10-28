//! Recipes for the Rust trainer (#1518): the corpora heads train on,
//! pinned by digest; the majority baseline and the scoring every run
//! reports; the receipt a run writes; and the `turbo-train-head` command
//! that ties them to `turbo-trainer` and the encoder.
//!
//! | Module | What it holds |
//! |---|---|
//! | `corpus` | Sentences, splits, file pins and the corpus record |
//! | `masc_pos` | MASC 3.0.0 Penn tags on the recipes' recorded release and split |
//! | `multiconer` | MultiCoNER v2 English, pinned file by file, and its coarse groups |
//! | `eval` | Accuracy, span F1, the majority baseline, the paired bootstrap |
//! | `run` | Head spec, encoder features, prediction, outputs |
//! | `receipt` | `receipt.json` |
//! | `safetensors` | Writing the head's F32 tensors |

pub mod cache;
pub mod corpus;
pub mod eval;
pub mod masc_pos;
pub mod multiconer;
pub mod receipt;
pub mod run;
pub mod safetensors;

/// An error, as the message the command prints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// The corpora the trainer reads, by id.
pub const CORPORA: [&str; 2] = [masc_pos::ID, multiconer::ID];

/// Reads the corpus `id` from `dir`.
///
/// # Errors
///
/// An unknown id, or what the corpus's reader refuses.
pub fn read_corpus(id: &str, dir: &std::path::Path, unrecorded: bool) -> Result<corpus::Corpus> {
    match id {
        masc_pos::ID => masc_pos::read(dir, unrecorded),
        multiconer::ID => multiconer::read(dir, unrecorded),
        _ => Err(Error(format!("unknown corpus {id:?}: one of {}", CORPORA.join(", ")))),
    }
}

/// The coarse groups of a corpus's types, where it has them.
pub fn coarse(id: &str) -> Option<fn(&str) -> &str> {
    (id == multiconer::ID).then_some(multiconer::coarse as fn(&str) -> &str)
}
