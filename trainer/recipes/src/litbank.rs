//! LitBank coreference reader (100 English literary works, CC BY 4.0, #1527).
//!
//! LitBank (`dbamman/litbank`) provides 100 public-domain English fiction
//! excerpts (~2,000 words each, 210,532 tokens total) annotated with
//! OntoNotes-style CoNLL coreference brackets in `coref/conll/*.conll`.
//! Both the public-domain texts (Project Gutenberg) and the CC BY 4.0
//! annotations are permissive, so `CorpusRecord::class` gives
//! `LicenseClass::Permissive` on both the pinned release and
//! `--unrecorded-corpus` fixtures.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use turbo::manifest::COREF_LABELS;
pub use turbo_train_recipes::litbank::{FILES, FILES_NAME, File, LITBANK, SPLIT, Split, WORKS, coref_files};

use crate::corpus::{Corpus, CorpusRecord, Counts, FileRecord, Pin, Sentence, Splits, Task, check_pin, read_file};
use crate::{Error, Result};

pub const ID: &str = "litbank";
pub const NAME: &str = "LitBank";
pub const LICENCE: &str = "CC BY 4.0";
pub const SPDX: &str = "CC-BY-4.0";
pub const ATTRIBUTION: &str = "Trained on LitBank (Bamman, Popat and Shen, NAACL 2019; \
    Bamman, Lewke and Mansoor, LREC 2020), published at https://github.com/dbamman/litbank \
    under the Creative Commons Attribution 4.0 International licence \
    (https://creativecommons.org/licenses/by/4.0/) over public-domain texts from Project Gutenberg.";

/// One gold coreference mention span in document word coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct MentionSpan {
    /// Inclusive 0-based document word index of the mention start.
    pub first: u32,
    /// Inclusive 0-based document word index of the mention end.
    pub last: u32,
    /// Gold coreference cluster identifier within the document.
    pub cluster: u32,
}

/// One LitBank document parsed from a CoNLL coreference file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorefDoc {
    /// Document identifier from the file stem or `#begin document` header.
    pub id: String,
    /// Sentences as sequences of word forms.
    pub sentences: Vec<Vec<String>>,
    /// Flattened document words in reading order.
    pub words: Vec<String>,
    /// Gold mention spans `(first_word, last_word)` sorted by `(first, last)`.
    pub mentions: Vec<(u32, u32)>,
    /// Gold cluster ID for each entry in [`Self::mentions`].
    pub chains: Vec<u32>,
    /// Gold coreference clusters, each a non-empty sorted list of `(first, last)` spans.
    pub clusters: Vec<Vec<(u32, u32)>>,
}

/// Alias for [`CorefDoc`].
pub type Document = CorefDoc;

/// The LitBank train/dev/test coreference documents and provenance record.
#[derive(Debug, Clone)]
pub struct CorefSplits {
    pub record: CorpusRecord,
    pub train: Vec<CorefDoc>,
    pub dev: Vec<CorefDoc>,
    pub test: Vec<CorefDoc>,
}

/// The 100 pinned `coref/conll/*.conll` files from `litbank-files.txt`.
pub fn pins() -> Vec<Pin> {
    coref_files().map(|f| Pin { path: f.path, size: f.size, sha256: f.sha256 }).collect()
}

/// Parses one LitBank CoNLL coreference file from raw bytes.
///
/// # Errors
///
/// Refuses invalid UTF-8, lines outside `#begin document` / `#end document`,
/// unclosed `#begin document`, fewer than 12 tab-separated columns,
/// non-consecutive 0-based sentence token indices in column 3, empty word
/// forms, malformed or unmatched coreference brackets, brackets crossing
/// sentence boundaries, conflicting cluster IDs on identical spans, or
/// empty documents.
pub fn parse(file: &str, bytes: &[u8]) -> Result<CorefDoc> {
    let text = std::str::from_utf8(bytes).map_err(|e| Error(format!("{file}: not UTF-8: {e}")))?;

    let mut id = doc_stem(file);
    let mut in_doc = false;
    let mut saw_doc = false;
    let mut sentences: Vec<Vec<String>> = Vec::new();
    let mut words: Vec<String> = Vec::new();
    let mut cur_sentence: Vec<String> = Vec::new();
    let mut open: BTreeMap<u32, Vec<(u32, usize)>> = BTreeMap::new();
    let mut raw_mentions: Vec<MentionSpan> = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let ln = idx + 1;
        let line = raw.trim_end_matches('\r');
        if line.trim().is_empty() {
            if in_doc && !cur_sentence.is_empty() {
                check_no_open(file, ln, &open)?;
                sentences.push(std::mem::take(&mut cur_sentence));
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("#begin document") {
            if in_doc {
                return Err(Error(format!("{file}:{ln}: nested #begin document before #end document")));
            }
            if let Some(parsed) = parse_doc_header(rest) {
                id = parsed;
            }
            in_doc = true;
            saw_doc = true;
            continue;
        }
        if line.starts_with("#end document") {
            if !in_doc {
                return Err(Error(format!("{file}:{ln}: #end document without matching #begin document")));
            }
            if !cur_sentence.is_empty() {
                sentences.push(std::mem::take(&mut cur_sentence));
            }
            check_no_open(file, ln, &open)?;
            in_doc = false;
            continue;
        }
        if !in_doc {
            return Err(Error(format!("{file}:{ln}: line outside #begin/#end document framing")));
        }

        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 12 {
            return Err(Error(format!("{file}:{ln}: expected at least 12 tab-separated columns, got {}", cols.len())));
        }
        let tok_idx: usize =
            cols[2].trim().parse().map_err(|_| Error(format!("{file}:{ln}: invalid token index {:?}", cols[2])))?;
        if tok_idx != cur_sentence.len() {
            return Err(Error(format!(
                "{file}:{ln}: expected sentence token index {}, got {tok_idx}",
                cur_sentence.len()
            )));
        }
        let word = cols[3].trim();
        if word.is_empty() {
            return Err(Error(format!("{file}:{ln}: empty word form in column 4")));
        }
        let doc_word =
            u32::try_from(words.len()).map_err(|_| Error(format!("{file}:{ln}: document exceeds u32 word limit")))?;
        cur_sentence.push(word.to_owned());
        words.push(word.to_owned());

        let coref = cols[cols.len() - 1].trim();
        if !coref.is_empty() && coref != "_" && coref != "-" {
            parse_coref_column(file, ln, doc_word, coref, &mut open, &mut raw_mentions)?;
        }
    }

    if in_doc {
        return Err(Error(format!("{file}: unclosed #begin document at end of file")));
    }
    if !saw_doc || sentences.is_empty() || words.is_empty() {
        return Err(Error(format!("{file}: empty document")));
    }

    raw_mentions.sort();
    raw_mentions.dedup();
    for pair in raw_mentions.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if a.first == b.first && a.last == b.last && a.cluster != b.cluster {
            return Err(Error(format!(
                "{file}: conflicting cluster IDs {} and {} on mention span ({}, {})",
                a.cluster, b.cluster, a.first, a.last
            )));
        }
    }

    let mut mentions = Vec::with_capacity(raw_mentions.len());
    let mut chains = Vec::with_capacity(raw_mentions.len());
    let mut by_cluster: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
    for m in raw_mentions {
        mentions.push((m.first, m.last));
        chains.push(m.cluster);
        by_cluster.entry(m.cluster).or_default().push((m.first, m.last));
    }
    let clusters: Vec<Vec<(u32, u32)>> = by_cluster.into_values().collect();

    Ok(CorefDoc { id, sentences, words, mentions, chains, clusters })
}

fn check_no_open(file: &str, ln: usize, open: &BTreeMap<u32, Vec<(u32, usize)>>) -> Result<()> {
    if let Some((&cluster, stack)) = open.iter().find(|(_, s)| !s.is_empty()) {
        let (_, open_ln) = stack[0];
        return Err(Error(format!(
            "{file}:{ln}: unclosed coreference bracket for cluster {cluster} opened at line {open_ln}"
        )));
    }
    Ok(())
}

fn parse_coref_column(
    file: &str,
    ln: usize,
    word_idx: u32,
    coref: &str,
    open: &mut BTreeMap<u32, Vec<(u32, usize)>>,
    mentions: &mut Vec<MentionSpan>,
) -> Result<()> {
    for item in coref.split('|') {
        if item.is_empty() {
            return Err(Error(format!("{file}:{ln}: empty item in coreference column {coref:?}")));
        }
        if let Some(inner) = item.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
            let cluster = parse_cluster_id(file, ln, inner, item)?;
            mentions.push(MentionSpan { first: word_idx, last: word_idx, cluster });
        } else if let Some(inner) = item.strip_prefix('(') {
            let cluster = parse_cluster_id(file, ln, inner, item)?;
            open.entry(cluster).or_default().push((word_idx, ln));
        } else if let Some(inner) = item.strip_suffix(')') {
            let cluster = parse_cluster_id(file, ln, inner, item)?;
            let Some(stack) = open.get_mut(&cluster) else {
                return Err(Error(format!("{file}:{ln}: unmatched closing bracket {item:?} for cluster {cluster}")));
            };
            let Some((first, _)) = stack.pop() else {
                return Err(Error(format!("{file}:{ln}: unmatched closing bracket {item:?} for cluster {cluster}")));
            };
            if stack.is_empty() {
                open.remove(&cluster);
            }
            mentions.push(MentionSpan { first, last: word_idx, cluster });
        } else {
            return Err(Error(format!("{file}:{ln}: malformed coreference bracket {item:?}")));
        }
    }
    Ok(())
}

fn parse_cluster_id(file: &str, ln: usize, digits: &str, raw: &str) -> Result<u32> {
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Error(format!("{file}:{ln}: invalid cluster ID in coreference bracket {raw:?}")));
    }
    digits.parse::<u32>().map_err(|_| Error(format!("{file}:{ln}: invalid cluster ID in coreference bracket {raw:?}")))
}

fn parse_doc_header(rest: &str) -> Option<String> {
    let trimmed = rest.trim();
    if let Some(after_open) = trimmed.strip_prefix('(')
        && let Some((inside, _)) = after_open.split_once(')')
        && !inside.trim().is_empty()
    {
        return Some(inside.trim().to_owned());
    }
    None
}

fn doc_stem(file: &str) -> String {
    Path::new(file).file_stem().and_then(|s| s.to_str()).unwrap_or(file).to_owned()
}

/// Reads the LitBank coreference splits (`train`, `dev`, `test`) from `dir`.
///
/// When `unrecorded` is `false`, all 100 pinned `coref/conll/*.conll` files
/// must be present at their recorded sizes and SHA-256 digests, and are
/// partitioned into 80 train, 10 dev, and 10 test works via [`SPLIT`].
/// When `unrecorded` is `true`, custom `.conll` fixtures in `dir/coref/conll`
/// (or directly in `dir`) are accepted and partitioned in sorted filename order.
pub fn read_coref(dir: &Path, unrecorded: bool) -> Result<CorefSplits> {
    let expected = pins();
    let all_pinned_present = expected.iter().all(|p| dir.join(p.path).is_file());

    let (files, docs, recorded): (Vec<FileRecord>, Vec<CorefDoc>, bool) = if !unrecorded || all_pinned_present {
        let mut files = Vec::with_capacity(expected.len());
        let mut docs = Vec::with_capacity(expected.len());
        let mut mismatches = Vec::new();
        for pin in expected {
            let (record, bytes) = read_file(dir, pin.path)?;
            if let Err(m) = check_pin(&record, &pin) {
                mismatches.push(m);
            }
            docs.push(parse(pin.path, &bytes)?);
            files.push(record);
        }
        if !mismatches.is_empty() && !unrecorded {
            return Err(Error(format!(
                "{NAME} is not the recorded release (pass --unrecorded-corpus to train on it anyway): {}",
                mismatches.join("; ")
            )));
        }
        (files, docs, mismatches.is_empty())
    } else {
        let rel_paths = discover_conll_files(dir)?;
        if rel_paths.len() < 3 {
            return Err(Error(format!(
                "{}: {NAME} requires at least 3 .conll documents for train/dev/test, found {}",
                dir.display(),
                rel_paths.len()
            )));
        }
        let mut files = Vec::with_capacity(rel_paths.len());
        let mut docs = Vec::with_capacity(rel_paths.len());
        for (rel, _) in rel_paths {
            let (record, bytes) = read_file(dir, &rel)?;
            docs.push(parse(&rel, &bytes)?);
            files.push(record);
        }
        (files, docs, false)
    };

    let (train, dev, test) = partition_docs(docs)?;
    let (train_s, dev_s, test_s) = (docs_to_sentences(&train), docs_to_sentences(&dev), docs_to_sentences(&test));

    let record = CorpusRecord {
        id: ID.into(),
        name: NAME.into(),
        task: Task::Coref,
        licence: LICENCE.into(),
        spdx: SPDX.into(),
        terms: LITBANK.licence_terms.into(),
        attribution: ATTRIBUTION.into(),
        recorded,
        corpus_sha256: String::new(),
        files,
        split: format!(
            "sorted coref/conll works partitioned {}/{}/{} train/dev/test at {}",
            SPLIT.train, SPLIT.dev, SPLIT.test, LITBANK.revision
        ),
        train: Counts::of(&train_s),
        dev: Counts::of(&dev_s),
        test: Counts::of(&test_s),
        skipped: 0,
        recipe_only: false,
    };
    Ok(CorefSplits { record, train, dev, test })
}

/// Reads LitBank as [`Corpus`] for `check --corpus litbank`.
pub fn read(dir: &Path, unrecorded: bool) -> Result<Corpus> {
    let coref = read_coref(dir, unrecorded)?;
    Ok(Corpus {
        record: coref.record,
        splits: Splits {
            train: docs_to_sentences(&coref.train),
            dev: docs_to_sentences(&coref.dev),
            test: docs_to_sentences(&coref.test),
        },
    })
}

fn partition_docs(docs: Vec<CorefDoc>) -> Result<(Vec<CorefDoc>, Vec<CorefDoc>, Vec<CorefDoc>)> {
    if docs.len() == WORKS {
        let (tr, dv, te) = SPLIT.partition();
        let by_idx = |indices: Vec<usize>| indices.into_iter().map(|i| docs[i].clone()).collect();
        return Ok((by_idx(tr), by_idx(dv), by_idx(te)));
    }
    let n = docs.len();
    let dev_n = (n * SPLIT.dev / WORKS).max(1);
    let test_n = (n * SPLIT.test / WORKS).max(1);
    if n <= dev_n + test_n {
        return Err(Error(format!("litbank: cannot partition {n} documents into non-empty train/dev/test")));
    }
    let train_n = n - dev_n - test_n;
    let mut it = docs.into_iter();
    let train: Vec<CorefDoc> = it.by_ref().take(train_n).collect();
    let dev: Vec<CorefDoc> = it.by_ref().take(dev_n).collect();
    let test: Vec<CorefDoc> = it.collect();
    Ok((train, dev, test))
}

fn docs_to_sentences(docs: &[CorefDoc]) -> Vec<Sentence> {
    let label = COREF_LABELS[0];
    let mut out = Vec::new();
    for doc in docs {
        for s in &doc.sentences {
            out.push(Sentence { words: s.clone(), labels: vec![label.to_owned(); s.len()] });
        }
    }
    out
}

fn discover_conll_files(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let sub = dir.join("coref/conll");
    let (scan_dir, prefix) = if sub.is_dir() { (sub, "coref/conll/") } else { (dir.to_path_buf(), "") };
    let entries = fs::read_dir(&scan_dir).map_err(|e| Error(format!("{}: {e}", scan_dir.display())))?;
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| Error(format!("{}: {e}", scan_dir.display())))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("conll") && path.is_file() {
            let name = entry.file_name().to_string_lossy().into_owned();
            out.push((format!("{prefix}{name}"), path));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}
