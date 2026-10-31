//! Native GermEval 2014 recipe, from the publisher's four-column format.
//! Licence: CC BY 4.0, University of Hamburg's German NER data page.
//! This does not depend on the retired compatibility corpus reader.
use crate::corpus::{Corpus, CorpusRecord, Counts, Pin, Sentence, Splits, Task, check_pin, read_file};
use crate::{Error, Result};
use std::path::Path;

const ATTRIBUTION: &str = "GermEval 2014 / NoSta-D, Darina Benikova, Chris Biemann, Marc Reznicek, Max Kisselew and Sebastian Pado (2014), CC BY 4.0. https://sites.google.com/site/germeval2014ner/";

pub const OUTER: &str = "germeval2014-outer";
pub const INNER: &str = "germeval2014-inner";
pub const TERMS: &str = "https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germaner.html";
pub const TYPES: [&str; 12] = [
    "PER", "ORG", "LOC", "OTH", "PERderiv", "ORGderiv", "LOCderiv", "OTHderiv", "PERpart", "ORGpart", "LOCpart",
    "OTHpart",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Outer,
    Inner,
}

impl Level {
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            OUTER => Some(Self::Outer),
            INNER => Some(Self::Inner),
            _ => None,
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Self::Outer => OUTER,
            Self::Inner => INNER,
        }
    }
}

/// Corrected 2022-04-28 publisher files. Every byte, including metadata
/// comments and the one trailing tab in test, is covered by these pins.
pub const FILES: [(Pin, &str); 3] = [
    (
        Pin {
            path: "NER-de-train.tsv",
            size: 7_882_321,
            sha256: "98682c21700e5bd7959ac361150679929e4af5d91e4dd7c379cc89133a810994",
        },
        "1Jjhbal535VVz2ap4v4r_rN1UEHTdLK5P",
    ),
    (
        Pin {
            path: "NER-de-dev.tsv",
            size: 723_854,
            sha256: "2ffe8405acc6bb6111b62a88623aa1e1f2e6db16b77fc005c54163541807aa05",
        },
        "1ZfRcQThdtAR5PPRjIDtrVP7BtXSCUBbm",
    ),
    (
        Pin {
            path: "NER-de-test.tsv",
            size: 1_682_725,
            sha256: "ec6e6adce5ed71cee119ae758e569ea4be54c6be5319b3b49dfcdcb05e51301b",
        },
        "1u9mb7kNJHWQCWyweMDRMuTFoOHOfeBTH",
    ),
];

pub fn read(level: Level, dir: &Path, unrecorded: bool) -> Result<Corpus> {
    let mut records = Vec::new();
    let mut splits = Vec::new();
    let mut errors = Vec::new();
    for (pin, _) in FILES {
        let (record, bytes) = read_file(dir, pin.path)?;
        if let Err(e) = check_pin(&record, &pin) {
            errors.push(e);
        }
        splits.push(parse(pin.path, &bytes, level)?);
        records.push(record);
    }
    if !unrecorded && !errors.is_empty() {
        return Err(Error(format!("GermEval is not the recorded release: {}", errors.join("; "))));
    }
    let test = splits.pop().unwrap_or_default();
    let dev = splits.pop().unwrap_or_default();
    let train = splits.pop().unwrap_or_default();
    if [&train, &dev, &test].iter().any(|s| s.is_empty()) {
        return Err(Error("GermEval requires nonempty train, dev and test splits".into()));
    }
    let record = CorpusRecord {
        id: level.id().into(),
        name: "GermEval 2014, corrected 2022-04-28".into(),
        task: Task::Bio,
        licence: "CC BY 4.0".into(),
        spdx: "CC-BY-4.0".into(),
        terms: TERMS.into(),
        attribution: ATTRIBUTION.into(),
        recorded: errors.is_empty(),
        corpus_sha256: String::new(),
        files: records,
        split: format!("publisher train/dev/test; {level:?} BIO column; no repartitioning or subtype folding"),
        train: Counts::of(&train),
        dev: Counts::of(&dev),
        test: Counts::of(&test),
        skipped: 0,
        recipe_only: false,
    };
    Ok(Corpus { record, splits: Splits { train, dev, test } })
}

/// Validate both columns even when only one is trained. Keep token spelling,
/// consecutive one-based IDs, BIO entity boundaries and all twelve subtypes.
pub fn parse(file: &str, bytes: &[u8], level: Level) -> Result<Vec<Sentence>> {
    let text = std::str::from_utf8(bytes).map_err(|e| Error(format!("{file}: UTF-8: {e}")))?;
    let mut out = Vec::new();
    let mut words = Vec::new();
    let mut labels = Vec::new();
    let mut previous = [None, None];
    for (i, line) in text.lines().chain(std::iter::once("")).enumerate() {
        let fail = |message| Error(format!("{file}:{}: {message}", i + 1));
        if line.is_empty() {
            if !words.is_empty() {
                out.push(Sentence { words: std::mem::take(&mut words), labels: std::mem::take(&mut labels) });
            }
            previous = [None, None];
            continue;
        }
        if line.starts_with('#') {
            if !words.is_empty() {
                return Err(fail("source comment inside a sentence"));
            }
            continue;
        }
        // The pinned test file has one optional trailing delimiter. It is
        // framing only: never trim tokens, label text or nonempty fields.
        let mut fields = line.strip_suffix('\t').unwrap_or(line).split('\t');
        let row = [fields.next(), fields.next(), fields.next(), fields.next()];
        let [Some(id), Some(word), Some(outer), Some(inner)] = row else {
            return Err(fail("expected four TSV fields"));
        };
        if fields.next().is_some() || word.is_empty() || id.parse::<usize>().ok() != Some(words.len() + 1) {
            return Err(fail("expected four fields, nonempty token and consecutive one-based token ID"));
        }
        for (column, label) in [outer, inner].into_iter().enumerate() {
            if label == "O" {
                previous[column] = None;
                continue;
            }
            let Some((prefix, kind)) = label.split_once('-') else {
                return Err(fail("invalid BIO label"));
            };
            if !matches!(prefix, "B" | "I")
                || !TYPES.contains(&kind)
                || (prefix == "I" && previous[column] != Some(kind))
            {
                return Err(fail("invalid BIO label or continuation"));
            }
            previous[column] = Some(kind);
        }
        words.push(word.to_owned());
        labels.push(
            match level {
                Level::Outer => outer,
                Level::Inner => inner,
            }
            .to_owned(),
        );
    }
    Ok(out)
}
