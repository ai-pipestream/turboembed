//! A corpus as the trainer reads it: sentences of words, each with its
//! gold label, in training, dev and test splits, and the record of the
//! files they were read from, each pinned by size and SHA-256.

use std::path::Path;

use serde::{Deserialize, Serialize};
use turbo::bundle::sha256_hex;

use crate::{Error, Result};

/// One sentence: its words and the gold label of each.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sentence {
    pub words: Vec<String>,
    pub labels: Vec<String>,
}

/// The three splits of a corpus.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Splits {
    pub train: Vec<Sentence>,
    pub dev: Vec<Sentence>,
    pub test: Vec<Sentence>,
}

/// How a corpus's labels read: one tag a word, or BIO entity spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Tags,
    Bio,
}

/// A file a corpus was read from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileRecord {
    /// Relative to the corpus directory, `/`-separated.
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

/// A pinned file: where it sits in the corpus directory, and its size and
/// digest as recorded when the release was downloaded and checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pin {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// Counts of one split.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    pub sentences: usize,
    pub words: usize,
}

impl Counts {
    pub fn of(sentences: &[Sentence]) -> Counts {
        Counts { sentences: sentences.len(), words: sentences.iter().map(|s| s.words.len()).sum() }
    }
}

/// What a receipt records of the corpus a run read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusRecord {
    /// The trainer's id of the corpus (`masc-pos`, `multiconer2-en`).
    pub id: String,
    pub name: String,
    pub task: Task,
    pub licence: String,
    /// The licence's SPDX id, as the model zoo knows it, from which the
    /// head's `license_class` is taken (`CorpusRecord::class`). Empty in
    /// a receipt written before it was recorded.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub spdx: String,
    /// Where the publisher states the licence.
    pub terms: String,
    /// What a model trained on the corpus credits.
    pub attribution: String,
    /// Whether every file matched its recorded pin. A run on files that
    /// did not is refused unless asked for, and is never a release.
    pub recorded: bool,
    /// The digest of the whole corpus directory, for a corpus pinned that
    /// way (MASC), or empty.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub corpus_sha256: String,
    /// The files read, for a corpus pinned file by file.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<FileRecord>,
    /// How each split was made, in a sentence.
    pub split: String,
    pub train: Counts,
    pub dev: Counts,
    pub test: Counts,
    /// Sentences the reader left out (no words, a word without a label).
    pub skipped: usize,
    /// Whether a head trained on the corpus stays local whatever its
    /// licence says: the corpus's text rights are not settled (UD
    /// Portuguese Porttinari), so it is a recipe a user runs, never a
    /// shipped model.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub recipe_only: bool,
}

impl CorpusRecord {
    /// The class of the corpus's licence by the model zoo's rules
    /// (docs/zoo.md): what a head trained on it carries as
    /// `license_class`, so a bundle holding the head is offered only as
    /// its terms allow (a CC BY-SA corpus makes the bundle accept-first).
    /// A recipe-only corpus's head is local-only, whatever its licence.
    ///
    /// # Errors
    ///
    /// A licence the zoo does not know or refuses: no head is made from
    /// such a corpus.
    pub fn class(&self) -> Result<turbo::manifest::LicenseClass> {
        use turbo::manifest::LicenseClass;
        use turbo_zoo::{Licence, LicenceClass};
        let licence: Licence = self.spdx.parse().map_err(|e| Error(format!("corpus {}: licence {e}", self.id)))?;
        let class = licence.class();
        if self.recipe_only && class != LicenceClass::Refused {
            return Ok(LicenseClass::LocalOnly);
        }
        Ok(match class {
            LicenceClass::Permissive => LicenseClass::Permissive,
            LicenceClass::Lgpl => LicenseClass::Lgpl,
            LicenceClass::ShareAlike => LicenseClass::ShareAlike,
            LicenceClass::Gpl => LicenseClass::Gpl,
            LicenceClass::LocalOnly => LicenseClass::LocalOnly,
            LicenceClass::Refused => {
                return Err(Error(format!("corpus {}: the zoo refuses its licence {licence}", self.id)));
            }
        })
    }
}

/// A corpus read and checked.
#[derive(Debug, Clone)]
pub struct Corpus {
    pub record: CorpusRecord,
    pub splits: Splits,
}

impl Corpus {
    /// The head's label set, sorted with "O" first. Fixed-type NER recipes
    /// use their declared schema, including labels absent from training;
    /// other tasks use observed training labels. Held-out labels never
    /// determine the vocabulary.
    pub fn labels(&self) -> Vec<String> {
        let mut all: Vec<String> = match crate::fixed_entity_types(&self.record.id) {
            Some(types) => std::iter::once("O".to_owned())
                .chain(types.iter().flat_map(|kind| [format!("B-{kind}"), format!("I-{kind}")]))
                .collect(),
            None => self.splits.train.iter().flat_map(|s| s.labels.iter().cloned()).collect(),
        };
        all.sort();
        all.dedup();
        if let Some(o) = all.iter().position(|l| l == "O") {
            let o = all.remove(o);
            all.insert(0, o);
        }
        all
    }
}

/// The record of the file at `rel` under `dir`, and its bytes.
///
/// # Errors
///
/// A file that cannot be read.
pub fn read_file(dir: &Path, rel: &str) -> Result<(FileRecord, Vec<u8>)> {
    let path = dir.join(rel);
    let bytes = std::fs::read(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    let record = FileRecord { path: rel.to_owned(), size: bytes.len() as u64, sha256: sha256_hex(&bytes) };
    Ok((record, bytes))
}

/// Whether `file` is the file `pin` records, and if not, why.
pub fn check_pin(file: &FileRecord, pin: &Pin) -> std::result::Result<(), String> {
    if file.size == pin.size && file.sha256 == pin.sha256 {
        Ok(())
    } else {
        Err(format!(
            "{}: {} bytes, sha256 {}; the release records {} bytes, sha256 {}",
            file.path, file.size, file.sha256, pin.size, pin.sha256
        ))
    }
}
