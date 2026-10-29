//! Universal Dependencies treebanks, read as syntactic words (a
//! multiword token's words, never its surface line) with every column a
//! head trains on: UPOS, XPOS, features, lemma, head and relation.
//!
//! Each treebank is pinned by release tag, commit and the size and
//! SHA-256 of its train, dev and test files, and fetched by the command
//! in the recipes' readme, never checked in. Every English UD treebank
//! is CC BY-SA or NC, so a head trained on one is share-alike: offered
//! for download once a user accepts, never in a default bundle.
//!
//! A tag task is a corpus of its own, `<treebank>-<column>`
//! (`ud-en-ewt-upos`), whose labels are that column, so `train` takes it
//! as it takes MASC. Lemma and dependency heads read [`UdSentence`]s.

use std::path::Path;

use turbo_formats::conllu::ConlluWordReader;

use crate::corpus::{Corpus, CorpusRecord, Counts, FileRecord, Pin, Sentence, Splits, Task, check_pin, read_file};
use crate::{Error, Result};

/// A pinned treebank release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Treebank {
    /// The trainer's id, the prefix of each column's corpus id.
    pub id: &'static str,
    pub name: &'static str,
    /// The release tag and the commit it names.
    pub tag: &'static str,
    pub commit: &'static str,
    pub licence: &'static str,
    pub spdx: &'static str,
    pub terms: &'static str,
    pub attribution: &'static str,
    pub train: Pin,
    pub dev: Pin,
    pub test: Pin,
}

/// UD English EWT r2.18. Its test file is the one the peer records under
/// `analysis/eval/records/peer-analysis-1521` score (spaCy, Stanza,
/// Trankit), so the numbers compare.
pub const EN_EWT: Treebank = Treebank {
    id: "ud-en-ewt",
    name: "Universal Dependencies English Web Treebank r2.18",
    tag: "r2.18",
    commit: "b7711cce01cdd4f5fcc0a8199b8a50d951b16c0c",
    licence: "CC BY-SA 4.0",
    spdx: "CC-BY-SA-4.0",
    terms: "https://github.com/UniversalDependencies/UD_English-EWT/blob/r2.18/README.md",
    attribution: "Trained on the Universal Dependencies English Web Treebank (Silveira et al., 2014), \
         annotations by The Board of Trustees of The Leland Stanford Junior University, published at \
         https://github.com/UniversalDependencies/UD_English-EWT; used under the Creative Commons \
         Attribution-ShareAlike 4.0 International licence (https://creativecommons.org/licenses/by-sa/4.0/).",
    train: Pin {
        path: "en_ewt-ud-train.conllu",
        size: 15_034_493,
        sha256: "d68e06122a702464c613076523d56740f047e5bbe89dd90ec32737e04d952143",
    },
    dev: Pin {
        path: "en_ewt-ud-dev.conllu",
        size: 1_898_574,
        sha256: "39239e0a60db3ae68f4b7036189f11b6692741d10ff8240dd91f74f2760d90f8",
    },
    test: Pin {
        path: "en_ewt-ud-test.conllu",
        size: 1_900_101,
        sha256: "fa024f43dc5da3c5ac02563bc9bd0e974f46cbb1560823976a8f342a37dc494a",
    },
};

/// The treebanks the trainer reads.
pub const TREEBANKS: [Treebank; 1] = [EN_EWT];

/// The columns a tag head trains on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Column {
    Upos,
    Xpos,
    Feats,
}

impl Column {
    pub const ALL: [Column; 3] = [Column::Upos, Column::Xpos, Column::Feats];

    /// The suffix of the column's corpus id.
    pub fn suffix(self) -> &'static str {
        match self {
            Column::Upos => "upos",
            Column::Xpos => "xpos",
            Column::Feats => "feats",
        }
    }

    /// The head a corpus of this column trains: its name and layer.
    pub fn head(self) -> (&'static str, &'static str) {
        match self {
            Column::Upos => ("upos", "turboembed:pos"),
            Column::Xpos => ("xpos", "turboembed:xpos"),
            Column::Feats => ("feats", "turboembed:feats"),
        }
    }

    fn of(self, s: &UdSentence) -> Vec<String> {
        match self {
            Column::Upos => s.upos.clone(),
            Column::Xpos => s.xpos.clone(),
            Column::Feats => s.feats.clone(),
        }
    }
}

/// One sentence's syntactic words and their columns. `heads` count from
/// 1, with 0 the root, as CoNLL-U writes them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UdSentence {
    pub words: Vec<String>,
    pub lemmas: Vec<String>,
    pub upos: Vec<String>,
    pub xpos: Vec<String>,
    pub feats: Vec<String>,
    pub heads: Vec<u32>,
    pub rels: Vec<String>,
}

/// The sentences of one CoNLL-U file. Empty nodes (`8.1`) are left out,
/// as they carry only enhanced dependencies.
///
/// # Errors
///
/// Text that is not UTF-8, a sentence the CoNLL-U reader refuses, or a
/// word whose head is not a number from 0 to the sentence's length,
/// naming the file and line.
pub fn parse(file: &str, bytes: &[u8]) -> Result<Vec<UdSentence>> {
    let text = std::str::from_utf8(bytes).map_err(|e| Error(format!("{file}: not UTF-8: {e}")))?;
    let reader = ConlluWordReader::new(text).map_err(|e| Error(format!("{file}: {e}")))?;
    let mut out = Vec::new();
    for sentence in reader {
        let sentence = sentence.map_err(|e| Error(format!("{file}: {e}")))?;
        let n = sentence.words.len();
        let mut s = UdSentence::default();
        for w in &sentence.words {
            let head = w.head.text.parse::<u32>().ok().filter(|&h| h as usize <= n).ok_or_else(|| {
                Error(format!("{file}:{}: head {:?} is not a word of the sentence or 0", w.line, w.head.text))
            })?;
            s.words.push(w.form.text.to_string());
            s.lemmas.push(w.lemma.text.to_string());
            s.upos.push(w.upos.text.to_string());
            s.xpos.push(w.xpos.text.to_string());
            s.feats.push(w.feats.text.to_string());
            s.heads.push(head);
            s.rels.push(w.deprel.text.to_string());
        }
        if !s.words.is_empty() {
            out.push(s);
        }
    }
    Ok(out)
}

/// A treebank's three splits.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UdSplits {
    pub train: Vec<UdSentence>,
    pub dev: Vec<UdSentence>,
    pub test: Vec<UdSentence>,
}

/// Reads `bank`'s train, dev and test files under `dir`. Files that do
/// not match their pins are refused unless `unrecorded`; the returned
/// flag says whether every file matched.
///
/// # Errors
///
/// A file that cannot be read or parsed, or a pin mismatch.
pub fn read_treebank(bank: &Treebank, dir: &Path, unrecorded: bool) -> Result<(UdSplits, Vec<FileRecord>, bool)> {
    let (mut files, mut parts, mut mismatches) = (Vec::new(), Vec::new(), Vec::new());
    for pin in [bank.train, bank.dev, bank.test] {
        let (record, bytes) = read_file(dir, pin.path)?;
        if let Err(m) = check_pin(&record, &pin) {
            mismatches.push(m);
        }
        parts.push(parse(pin.path, &bytes)?);
        files.push(record);
    }
    if !mismatches.is_empty() && !unrecorded {
        return Err(Error(format!(
            "{} is not the recorded release (pass --unrecorded-corpus to train on it anyway): {}",
            bank.name,
            mismatches.join("; ")
        )));
    }
    let test = parts.pop().unwrap_or_default();
    let dev = parts.pop().unwrap_or_default();
    let train = parts.pop().unwrap_or_default();
    Ok((UdSplits { train, dev, test }, files, mismatches.is_empty()))
}

/// The treebank and column a corpus id names (`ud-en-ewt-upos`), if any.
pub fn find(id: &str) -> Option<(Treebank, Column)> {
    TREEBANKS.iter().find_map(|b| {
        let suffix = id.strip_prefix(b.id)?.strip_prefix('-')?;
        Column::ALL.iter().find(|c| c.suffix() == suffix).map(|&c| (*b, c))
    })
}

/// Every tag corpus id the treebanks give.
pub fn corpus_ids() -> Vec<String> {
    TREEBANKS.iter().flat_map(|b| Column::ALL.iter().map(move |c| format!("{}-{}", b.id, c.suffix()))).collect()
}

/// The tag corpus of `bank`'s `column`, read from `dir`.
///
/// # Errors
///
/// What [`read_treebank`] refuses.
pub fn read(bank: &Treebank, column: Column, dir: &Path, unrecorded: bool) -> Result<Corpus> {
    let (splits, files, recorded) = read_treebank(bank, dir, unrecorded)?;
    let tags = |part: &[UdSentence]| -> Vec<Sentence> {
        part.iter().map(|s| Sentence { words: s.words.clone(), labels: column.of(s) }).collect()
    };
    let (train, dev, test) = (tags(&splits.train), tags(&splits.dev), tags(&splits.test));
    let record = CorpusRecord {
        id: format!("{}-{}", bank.id, column.suffix()),
        name: format!("{}, {}", bank.name, column.suffix().to_uppercase()),
        task: Task::Tags,
        licence: bank.licence.into(),
        spdx: bank.spdx.into(),
        terms: bank.terms.into(),
        attribution: bank.attribution.into(),
        recorded,
        corpus_sha256: String::new(),
        files,
        split: format!("the release's own train, dev and test files at {} ({})", bank.tag, bank.commit),
        train: Counts::of(&train),
        dev: Counts::of(&dev),
        test: Counts::of(&test),
        skipped: 0,
    };
    Ok(Corpus { record, splits: Splits { train, dev, test } })
}
