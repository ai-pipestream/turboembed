//! Re-DocRED (Tan et al., EMNLP 2022, "Revisiting DocRED"): the DocRED
//! documents with their missing relations added back, each a Wikipedia
//! lead section in tokenized sentences, its entities as sets of mentions
//! of one of six types, and its relations as Wikidata property ids
//! between two entities, a pair holding several at once. The repository
//! is MIT, but the documents are Wikipedia text, so the corpus is taken
//! as CC BY-SA 4.0: a head trained on it is share-alike, offered for
//! download once a user accepts, never in a default bundle.
//!
//! The files are pinned by size and SHA-256 as fetched from the
//! repository at commit [`COMMIT`] on 2026-10-31. The release gives its
//! own train, dev and test files, and the trainer uses them as given.

use std::path::Path;

use serde::Deserialize;
use turbo::head::relation::Mention;

use crate::corpus::{CorpusRecord, Counts, FileRecord, Pin, Task, check_pin, read_file};
use crate::{Error, Result};

pub const ID: &str = "re-docred";
pub const NAME: &str = "Re-DocRED";
pub const LICENCE: &str = "CC BY-SA 4.0";
/// LICENCE as the zoo names it.
pub const SPDX: &str = "CC-BY-SA-4.0";
pub const TERMS: &str = "https://github.com/tonytan48/Re-DocRED";
pub const ATTRIBUTION: &str = "Trained on Re-DocRED (Tan, Xu, Bing, Ng and Aljunied, Revisiting DocRED: \
     Addressing the False Negative Problem in Relation Extraction, EMNLP 2022), published at \
     https://github.com/tonytan48/Re-DocRED under the MIT licence; its documents are from Wikipedia and are \
     used under the Creative Commons Attribution-ShareAlike 4.0 International licence \
     (https://creativecommons.org/licenses/by-sa/4.0/).";
/// The repository commit the pins are from.
pub const COMMIT: &str = "ccfb54f5ddf5836027c87badda10f6dfc56efaac";

/// The head a relation corpus trains, and the layer it writes.
pub const HEAD: &str = "relation";
pub const LAYER: &str = "turboembed:relations";

/// The entity types the release uses, in order: a head's type index.
pub const ENTITY_TYPES: [&str; 6] = ["LOC", "MISC", "NUM", "ORG", "PER", "TIME"];

/// The files, relative to a checkout of the repository at [`COMMIT`].
pub const TRAIN: Pin = Pin {
    path: "data/train_revised.json",
    size: 18_602_933,
    sha256: "c40a137a1c57f3e2daaf5ebf511eaa1518f9305ffaa3052d46271921a91826f2",
};
pub const DEV: Pin = Pin {
    path: "data/dev_revised.json",
    size: 3_245_588,
    sha256: "051ee1d057204a5d08ef5502beacdadf191245b5eaf0e29ec2c607cf002c016f",
};
pub const TEST: Pin = Pin {
    path: "data/test_revised.json",
    size: 3_207_228,
    sha256: "ea673b7ef91e16510d7eb88863d3903e47318dc5653ef361266f27ae3b84723c",
};
/// The repository's MIT licence, read and pinned with the data.
pub const LICENSE: Pin =
    Pin { path: "LICENSE", size: 1066, sha256: "0a06d894a38e1948af6fb9e76e0415ecc7ec36dfe6c9fd5f21d470b1bb5714a5" };

/// A gold relation: head entity, tail entity, and its property id.
pub type Triple = (u32, u32, String);

/// One document: its words (the sentences joined), its entities' mentions
/// by word, each entity's type, and its gold relations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub title: String,
    pub words: Vec<String>,
    pub mentions: Vec<Mention>,
    /// Each entity's type, its first mention's: the release gives a few
    /// entities mentions of two types.
    pub types: Vec<String>,
    pub relations: Vec<Triple>,
}

/// The three splits of a relation corpus.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Documents {
    pub train: Vec<Document>,
    pub dev: Vec<Document>,
    pub test: Vec<Document>,
}

/// A relation corpus read and checked.
#[derive(Debug, Clone)]
pub struct RelationCorpus {
    pub record: CorpusRecord,
    pub splits: Documents,
}

impl RelationCorpus {
    /// The head's labels: `none`, then every property id training gives,
    /// sorted. Held-out relations never add one.
    pub fn labels(&self) -> Vec<String> {
        let mut all: Vec<String> =
            self.splits.train.iter().flat_map(|d| d.relations.iter().map(|r| r.2.clone())).collect();
        all.sort();
        all.dedup();
        all.insert(0, turbo::manifest::NO_RELATION.to_owned());
        all
    }
}

#[derive(Deserialize)]
struct RawDoc {
    title: String,
    sents: Vec<Vec<String>>,
    #[serde(rename = "vertexSet")]
    vertex_set: Vec<Vec<RawMention>>,
    #[serde(default)]
    labels: Vec<RawLabel>,
}

#[derive(Deserialize)]
struct RawMention {
    pos: [usize; 2],
    #[serde(rename = "type")]
    kind: String,
    sent_id: usize,
}

#[derive(Deserialize)]
struct RawLabel {
    r: String,
    h: usize,
    t: usize,
}

/// Reads the documents of one split file: a JSON array of documents with
/// `title`, `sents` (sentences of words), `vertexSet` (each entity's
/// mentions, by sentence and word range `pos`, with a `type`) and
/// `labels` (relations `r` from entity `h` to entity `t`).
///
/// # Errors
///
/// Text that is not that, an empty word, a mention outside its sentence,
/// an entity with no mention or an empty type, or a relation between an
/// entity and itself, to an entity the document lacks, repeated, or with
/// no label, naming the file and document.
pub fn parse(file: &str, bytes: &[u8]) -> Result<Vec<Document>> {
    let raw: Vec<RawDoc> = serde_json::from_slice(bytes).map_err(|e| Error(format!("{file}: {e}")))?;
    raw.into_iter()
        .enumerate()
        .map(|(i, d)| document(d).map_err(|e| Error(format!("{file}: document {i}: {e}"))))
        .collect()
}

fn document(d: RawDoc) -> std::result::Result<Document, String> {
    let mut starts = Vec::with_capacity(d.sents.len());
    let mut words = Vec::new();
    for s in d.sents {
        starts.push(words.len());
        if s.iter().any(String::is_empty) {
            return Err(format!("sentence {} has an empty word", starts.len() - 1));
        }
        words.extend(s);
    }
    let len = |s: usize| starts.get(s + 1).copied().unwrap_or(words.len()) - starts[s];
    let (mut mentions, mut types) = (Vec::new(), Vec::new());
    for (e, entity) in d.vertex_set.iter().enumerate() {
        let first = entity.first().ok_or_else(|| format!("entity {e} has no mention"))?;
        if first.kind.is_empty() {
            return Err(format!("entity {e} has no type"));
        }
        types.push(first.kind.clone());
        for m in entity {
            let [a, b] = m.pos;
            if m.sent_id >= starts.len() || a >= b || b > len(m.sent_id) {
                return Err(format!("entity {e}: a mention {:?} outside sentence {}", m.pos, m.sent_id));
            }
            let at = starts[m.sent_id];
            mentions.push(Mention { first: (at + a) as u32, last: (at + b - 1) as u32, entity: e as u32 });
        }
    }
    let mut relations: Vec<Triple> = Vec::with_capacity(d.labels.len());
    for l in d.labels {
        let r = (l.h as u32, l.t as u32, l.r);
        if l.h == l.t || l.h >= types.len() || l.t >= types.len() || r.2.is_empty() || relations.contains(&r) {
            return Err(format!("relation {r:?}: to itself, to no entity, unlabelled or repeated"));
        }
        relations.push(r);
    }
    Ok(Document { title: d.title, words, mentions, types, relations })
}

fn counts(docs: &[Document]) -> Counts {
    Counts { sentences: docs.len(), words: docs.iter().map(|d| d.words.len()).sum() }
}

/// Reads the train, dev and test files under `dir`, a checkout of the
/// repository. Files that do not match their pins are refused unless
/// `unrecorded`, and the record then says `recorded: false`.
///
/// # Errors
///
/// A file that cannot be read or parsed, or a pin mismatch.
pub fn read(dir: &Path, unrecorded: bool) -> Result<RelationCorpus> {
    let mut files: Vec<FileRecord> = Vec::new();
    let mut parts = Vec::new();
    let mut mismatches = Vec::new();
    for pin in [TRAIN, DEV, TEST, LICENSE] {
        let (record, bytes) = read_file(dir, pin.path)?;
        if let Err(m) = check_pin(&record, &pin) {
            mismatches.push(m);
        }
        if pin != LICENSE {
            parts.push(parse(pin.path, &bytes)?);
        }
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
        task: Task::Relations,
        licence: LICENCE.into(),
        spdx: SPDX.into(),
        terms: TERMS.into(),
        attribution: ATTRIBUTION.into(),
        recorded: mismatches.is_empty(),
        corpus_sha256: String::new(),
        files,
        split: "the release's own train, dev and test files; counts are of documents".into(),
        train: counts(&train),
        dev: counts(&dev),
        test: counts(&test),
        skipped: 0,
        recipe_only: false,
    };
    Ok(RelationCorpus { record, splits: Documents { train, dev, test } })
}
