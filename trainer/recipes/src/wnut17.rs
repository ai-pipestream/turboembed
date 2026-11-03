//! WNUT-17 Emerging Entities: six types, kept separate from MASC's schema.
//! Pins and publisher attribution come from the existing corpus recipe.
//! The #1524 ruling permits this CC BY 4.0 corpus for the default head.
use crate::corpus::{Corpus, CorpusRecord, Counts, Pin, Sentence, Splits, Task, check_pin, read_file};
use crate::{Error, Result};
use std::path::Path;
use turbo_formats::ner::{Wnut17Reader, Wnut17Types};
use turbo_train_recipes::wnut17::{WNUT_17, files};

const ATTRIBUTION: &str = "Trained on WNUT 2017 Emerging and Rare Entity Recognition, Derczynski et al., CC BY 4.0. Corpus: https://github.com/leondz/emerging_entities_17";

pub const ID: &str = "wnut17";
/// The single-label test file used by the publisher's Hugging Face loader:
/// https://huggingface.co/datasets/leondz/wnut_17/blob/main/wnut_17.py
/// `emerging.test.conll` instead has alternative comma-separated labels;
/// it must not be substituted or reduced to its first label.
pub const TEST: Pin = Pin {
    path: "emerging.test.annotated",
    size: 192_425,
    sha256: "2aa79b764e56ec9264a1b30fdd9b70195bd00ff400b62edd8f399d5f13c178f0",
};
pub const TYPES: [&str; 6] = ["person", "location", "group", "creative-work", "corporation", "product"];

/// Read pinned train/dev/test files. Unrecorded data is for local experiments,
/// never a release. Missing types in another corpus are not negative labels here.
pub fn read(dir: &Path, unrecorded: bool) -> Result<Corpus> {
    let mut records = Vec::new();
    let mut parts = Vec::new();
    let mut mismatches = Vec::new();
    for name in ["wnut17train.conll", "emerging.dev.conll", TEST.path] {
        let pin = if name == TEST.path {
            TEST
        } else {
            let pin = files().find(|f| f.path == name).ok_or_else(|| Error(format!("WNUT pin missing: {name}")))?;
            Pin { path: pin.path, size: pin.size, sha256: pin.sha256 }
        };
        let (record, bytes) = read_file(dir, name)?;
        if let Err(e) = check_pin(&record, &pin) {
            mismatches.push(e);
        }
        parts.push(parse(name, &bytes)?);
        records.push(record);
    }
    if !unrecorded && !mismatches.is_empty() {
        return Err(Error(format!("WNUT-17 is not the recorded release: {}", mismatches.join("; "))));
    }
    let test = parts.pop().unwrap_or_default();
    let dev = parts.pop().unwrap_or_default();
    let train = parts.pop().unwrap_or_default();
    if [&train, &dev, &test].iter().any(|s| s.is_empty()) {
        return Err(Error("WNUT-17 requires nonempty train, dev and test files".into()));
    }
    let record = CorpusRecord {
        id: ID.into(),
        name: WNUT_17.name.into(),
        task: Task::Bio,
        licence: "CC BY 4.0".into(),
        spdx: "CC-BY-4.0".into(),
        terms: WNUT_17.licence_terms.into(),
        attribution: ATTRIBUTION.into(),
        recorded: mismatches.is_empty(),
        corpus_sha256: String::new(),
        files: records,
        split: format!("publisher train/dev/test files at {}; no repartitioning", WNUT_17.revision),
        train: Counts::of(&train),
        dev: Counts::of(&dev),
        test: Counts::of(&test),
        skipped: 0,
        recipe_only: false,
    };
    Ok(Corpus { record, splits: Splits { train, dev, test } })
}

/// The corpus reader retains token boundaries. Validate labels before its
/// optional type filter, so an unknown annotation cannot silently become O.
pub fn parse(file: &str, bytes: &[u8]) -> Result<Vec<Sentence>> {
    let text = std::str::from_utf8(bytes).map_err(|e| Error(format!("{file}: UTF-8: {e}")))?;
    let mut previous = None;
    for (i, line) in text.lines().enumerate() {
        let fail = |m| Error(format!("{file}:{}: {m}", i + 1));
        if line.trim().is_empty() {
            previous = None;
            continue;
        }
        let mut fields = line.split('\t');
        let word = fields.next().unwrap_or_default();
        let label = fields.next().ok_or_else(|| fail("expected token and BIO label"))?;
        if word.is_empty() || fields.next().is_some() {
            return Err(fail("expected nonempty token and BIO label"));
        }
        if label == "O" {
            previous = None;
            continue;
        }
        let (prefix, kind) = label.split_once('-').ok_or_else(|| fail("invalid BIO label"))?;
        if !matches!(prefix, "B" | "I") || !TYPES.contains(&kind) {
            return Err(fail("unknown BIO label"));
        }
        if prefix == "I" && previous != Some(kind) {
            return Err(fail("I label does not continue the preceding entity"));
        }
        previous = Some(kind);
    }
    // The pinned train file has 2,394 tab-only sentence separators. Pass
    // borrowed sentence blocks to the reader, without rewriting tokens or
    // building normalized copies of the corpus.
    let mut out = Vec::new();
    let (mut start, mut offset) = (0, 0);
    for line in text.split_inclusive('\n') {
        if line.trim().is_empty() {
            out.extend(block(file, &text[start..offset])?);
            start = offset + line.len();
        }
        offset += line.len();
    }
    out.extend(block(file, &text[start..])?);
    Ok(out)
}

fn block(file: &str, text: &str) -> Result<Vec<Sentence>> {
    Wnut17Reader::new(text, Wnut17Types::ALL)
        .map_err(|e| Error(format!("{file}: {e}")))?
        .map(|sample| {
            let sample = sample.map_err(|e| Error(format!("{file}: {e}")))?;
            let words = sample.tokens().iter().map(|t| t.text.to_string()).collect::<Vec<_>>();
            let mut labels = vec!["O".to_owned(); words.len()];
            for name in sample.names() {
                let kind = name.kind.ok_or_else(|| Error(format!("{file}: untyped entity")))?;
                for (i, label) in labels[name.start as usize..name.end as usize].iter_mut().enumerate() {
                    *label = format!("{}-{kind}", if i == 0 { "B" } else { "I" });
                }
            }
            Ok(Sentence { words, labels })
        })
        .collect()
}
