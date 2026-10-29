//! MultiCoNER v2 (SemEval-2023 Task 2), English: fine-grained named
//! entities in BIO over lowercased, short sentences, 33 types under six
//! coarse groups, published by Amazon on the AWS Open Data Registry;
//! fetched by the command in its readme, not checked in. The registry
//! states CC BY 4.0, the dataset paper (arXiv 2310.13213, appendix B.5)
//! CC BY-SA 4.0, and the sentences are Wikipedia text, so it is taken as
//! CC BY-SA 4.0: a head trained on it is share-alike, offered for
//! download once a user accepts, never in a default bundle.
//!
//! The files are pinned by size and SHA-256 as downloaded from
//! `s3://multiconer/multiconer2023/` on 2026-10-28. The release gives its
//! own train, dev and test files, and the trainer uses them as given.

use std::path::Path;

use crate::corpus::{Corpus, CorpusRecord, Counts, FileRecord, Pin, Sentence, Splits, Task, check_pin, read_file};
use crate::{Error, Result};

pub const ID: &str = "multiconer2-en";
pub const NAME: &str = "MultiCoNER v2, English";
pub const LICENCE: &str = "CC BY-SA 4.0";
/// LICENCE as the zoo names it.
pub const SPDX: &str = "CC-BY-SA-4.0";
pub const TERMS: &str = "https://arxiv.org/abs/2310.13213";
pub const ATTRIBUTION: &str = "Trained on MultiCoNER v2 (Fetahu et al., SemEval-2023 Task 2: Fine-grained \
     Multilingual Named Entity Recognition), by Amazon, published at https://registry.opendata.aws/multiconer/; \
     its sentences are from Wikipedia. Used under the Creative Commons Attribution-ShareAlike 4.0 International \
     licence (https://creativecommons.org/licenses/by-sa/4.0/), which the dataset paper states.";

/// The English files, relative to the directory the release's readme
/// has `aws s3 cp` write (`multiconer2023/`).
pub const TRAIN: Pin = Pin {
    path: "EN-English/en_train.conll",
    size: 4_348_311,
    sha256: "1e1af77f95e92aa287c40feb7c10b384192c8ff026c916114a54713f331138ea",
};
pub const DEV: Pin = Pin {
    path: "EN-English/en_dev.conll",
    size: 227_151,
    sha256: "cef0633da09ab2ee3386576818acad0aef47e124ca7c7ec557e18de6974326df",
};
pub const TEST: Pin = Pin {
    path: "EN-English/en_test.conll",
    size: 70_582_472,
    sha256: "67ff1545244c757c2d56edb50eb5afd0d9bd8bdd877ea719a3a1e5932f5d7d77",
};

/// The six coarse groups of the 33 fine types, as the task's taxonomy
/// gives them.
pub const COARSE: [(&str, &[&str]); 6] = [
    ("Location", &["Facility", "OtherLOC", "HumanSettlement", "Station"]),
    ("CreativeWork", &["VisualWork", "MusicalWork", "WrittenWork", "ArtWork", "Software"]),
    (
        "Group",
        &["MusicalGRP", "PublicCorp", "PrivateCorp", "AerospaceManufacturer", "SportsGRP", "CarManufacturer", "ORG"],
    ),
    ("Person", &["Scientist", "Artist", "Athlete", "Politician", "Cleric", "SportsManager", "OtherPER"]),
    ("Product", &["Clothing", "Vehicle", "Food", "Drink", "OtherPROD"]),
    ("Medical", &["Medication/Vaccine", "MedicalProcedure", "AnatomicalStructure", "Symptom", "Disease"]),
];

/// The coarse group of a fine type, or the type itself when it has none.
pub fn coarse(fine: &str) -> &str {
    COARSE.iter().find(|(_, f)| f.contains(&fine)).map_or(fine, |(c, _)| c)
}

/// Reads the sentences of one file: a `# id` line, then one
/// `word _ _ LABEL` line a word, then blank lines.
///
/// # Errors
///
/// Any other line, a label that is not `O`, `B-<type>` or `I-<type>`, or
/// text that is not UTF-8, naming the file and line.
pub fn parse(file: &str, bytes: &[u8]) -> Result<Vec<Sentence>> {
    let text = std::str::from_utf8(bytes).map_err(|e| Error(format!("{file}: not UTF-8: {e}")))?;
    let mut out = Vec::new();
    let mut current: Option<Sentence> = None;
    for (n, line) in text.lines().enumerate() {
        let at = || format!("{file}:{}", n + 1);
        if line.starts_with("# id ") {
            out.extend(current.take().filter(|s| !s.words.is_empty()));
            current = Some(Sentence { words: Vec::new(), labels: Vec::new() });
        } else if line.trim().is_empty() {
            out.extend(current.take().filter(|s| !s.words.is_empty()));
        } else {
            let fields: Vec<&str> = line.split(' ').collect();
            let [word, "_", "_", label] = fields[..] else {
                return Err(Error(format!("{}: {line:?} is not `word _ _ LABEL`", at())));
            };
            let ok = label == "O"
                || label.strip_prefix("B-").or_else(|| label.strip_prefix("I-")).is_some_and(|t| !t.is_empty());
            if word.is_empty() || !ok {
                return Err(Error(format!(
                    "{}: {line:?}: an empty word, or a label not O, B-<type> or I-<type>",
                    at()
                )));
            }
            let Some(s) = current.as_mut() else {
                return Err(Error(format!("{}: a word before the first `# id` line", at())));
            };
            s.words.push(word.to_owned());
            s.labels.push(label.to_owned());
        }
    }
    out.extend(current.filter(|s| !s.words.is_empty()));
    Ok(out)
}

/// Reads the English train, dev and test files under `dir`. Files that
/// do not match their pins are refused unless `unrecorded`, and the
/// record then says `recorded: false`.
///
/// # Errors
///
/// A file that cannot be read or parsed, or a pin mismatch.
pub fn read(dir: &Path, unrecorded: bool) -> Result<Corpus> {
    let mut files: Vec<FileRecord> = Vec::new();
    let mut parts = Vec::new();
    let mut mismatches = Vec::new();
    for pin in [TRAIN, DEV, TEST] {
        let (record, bytes) = read_file(dir, pin.path)?;
        if let Err(m) = check_pin(&record, &pin) {
            mismatches.push(m);
        }
        parts.push(parse(pin.path, &bytes)?);
        files.push(record);
    }
    if !mismatches.is_empty() && !unrecorded {
        return Err(Error(format!(
            "{NAME} is not the recorded release (pass --unrecorded-corpus to train on it anyway): {}",
            mismatches.join("; ")
        )));
    }
    let test = parts.pop().unwrap_or_default();
    let dev = parts.pop().unwrap_or_default();
    let train = parts.pop().unwrap_or_default();
    let record = CorpusRecord {
        id: ID.into(),
        name: NAME.into(),
        task: Task::Bio,
        licence: LICENCE.into(),
        spdx: SPDX.into(),
        terms: TERMS.into(),
        attribution: ATTRIBUTION.into(),
        recorded: mismatches.is_empty(),
        corpus_sha256: String::new(),
        files,
        split: "the release's own train, dev and test files".into(),
        train: Counts::of(&train),
        dev: Counts::of(&dev),
        test: Counts::of(&test),
        skipped: 0,
        recipe_only: false,
    };
    Ok(Corpus { record, splits: Splits { train, dev, test } })
}
