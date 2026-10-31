//! Auditable, sentence-paired NER comparison on explicitly selected types.
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::corpus::{CorpusRecord, FileRecord, Sentence, Task};
use crate::eval::{self, Comparison, Metrics, Seen};
use crate::{Error, Result, ner_peer};

#[derive(Clone, Copy)]
pub struct PredictionFile<'a> {
    pub name: &'a str,
    pub bytes: &'a [u8],
}

pub struct Request<'a> {
    pub corpus: &'a CorpusRecord,
    pub gold: &'a [Sentence],
    pub candidate: PredictionFile<'a>,
    pub peer: PredictionFile<'a>,
    pub train: &'a [Sentence],
    pub selected_types: &'a [String],
    /// Explicit peer type to corpus type mapping, or "-" to exclude.
    /// Unmapped known types retain their names; unknown types require a mapping.
    pub peer_mapping: &'a BTreeMap<String, String>,
    pub resamples: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scores {
    pub metrics: Metrics,
    /// Peer full scores omit types explicitly mapped to "-".
    pub full_metrics: Metrics,
    /// Entity spans excluded by the selected-type filter, before mapping for peers.
    pub excluded: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub corpus: CorpusRecord,
    pub candidate_file: FileRecord,
    pub peer_file: FileRecord,
    pub selected_types: Vec<String>,
    pub peer_mapping: BTreeMap<String, String>,
    pub gold_excluded: BTreeMap<String, usize>,
    pub candidate: Scores,
    pub peer: Scores,
    pub baseline: Scores,
    pub baseline_description: String,
    pub candidate_vs_peer: Comparison,
    pub candidate_vs_baseline: Comparison,
}

/// Compare strict, token-aligned BIO files. No mapping repairs malformed BIO.
pub fn compare(request: Request<'_>) -> Result<Report> {
    let Request { corpus, gold, candidate, peer, train, selected_types, peer_mapping, resamples, seed } = request;
    if corpus.task != Task::Bio || gold.len() < 30 || resamples == 0 {
        return Err(Error("NER comparison requires BIO, at least 30 sentences and positive resamples".into()));
    }
    let selected: BTreeSet<_> = selected_types.iter().cloned().collect();
    if selected.is_empty() || selected.len() != selected_types.len() {
        return Err(Error("selected types must be nonempty and unique".into()));
    }
    let gold_labels: Vec<_> = gold.iter().map(|s| s.labels.clone()).collect();
    for sentence in gold {
        if sentence.words.is_empty() || sentence.words.len() != sentence.labels.len() {
            return Err(Error("gold requires nonempty sentences and one label per word".into()));
        }
    }
    ner_peer::validate(&corpus.id, &gold_labels)?;
    for kind in &selected {
        let one = vec![vec![format!("B-{kind}")]];
        ner_peer::validate(&corpus.id, &one)?;
        if !gold_labels.iter().any(|s| eval::spans(s).iter().any(|span| &span.2 == kind)) {
            return Err(Error(format!("selected type {kind:?} has no gold support")));
        }
    }
    for (from, to) in peer_mapping {
        if from.is_empty() || from == "O" || from.contains(['\t', '\n', '\r']) {
            return Err(Error("invalid peer mapping source type".into()));
        }
        if to != "-" {
            ner_peer::validate(&corpus.id, &[vec![format!("B-{to}")]])?;
        }
    }
    let candidate_labels = ner_peer::read(candidate.name, candidate.bytes, gold)?;
    // Check the original sequence before filtering, including excluded types.
    strict(&candidate_labels)?;
    ner_peer::validate(&corpus.id, &candidate_labels)?;
    let peer_labels = ner_peer::read(peer.name, peer.bytes, gold)?;
    strict(&peer_labels)?;
    let mapped = map_peer(&corpus.id, &peer_labels, peer_mapping)?;
    let (gold_filtered, gold_excluded) = filter(&gold_labels, &selected);
    let filtered_gold: Vec<_> =
        gold.iter().zip(gold_filtered).map(|(s, labels)| Sentence { words: s.words.clone(), labels }).collect();
    let (candidate_filtered, candidate_excluded) = filter(&candidate_labels, &selected);
    let (peer_filtered, _) = filter(&mapped, &selected);
    let mut peer_excluded = BTreeMap::new();
    for sentence in &peer_labels {
        for (_, _, kind) in eval::spans(sentence) {
            let mapped_kind = peer_mapping.get(&kind).unwrap_or(&kind);
            if !selected.contains(mapped_kind) {
                *peer_excluded.entry(kind).or_default() += 1;
            }
        }
    }
    let mut counts = BTreeMap::<String, usize>::new();
    let train_labels: Vec<_> = train.iter().map(|s| s.labels.clone()).collect();
    if train.iter().any(|s| s.words.len() != s.labels.len()) {
        return Err(Error("training requires one label per word".into()));
    }
    ner_peer::validate(&corpus.id, &train_labels)?;
    for label in train_labels.iter().flatten() {
        *counts.entry(label.clone()).or_default() += 1;
    }
    let winner = counts
        .iter()
        .rev()
        .max_by_key(|(_, n)| **n)
        .map(|(label, _)| label.clone())
        .ok_or_else(|| Error("empty baseline training labels".into()))?;
    let baseline: Vec<_> = gold.iter().map(|s| vec![winner.clone(); s.words.len()]).collect();
    strict(&baseline)?;
    ner_peer::validate(&corpus.id, &baseline)?;
    let (baseline_filtered, baseline_excluded) = filter(&baseline, &selected);
    let score = |labels: &[Vec<String>], full: &[Vec<String>], excluded| Scores {
        full_metrics: eval::evaluate(Task::Bio, gold, full, &Seen::default(), None),
        metrics: eval::evaluate(Task::Bio, &filtered_gold, labels, &Seen::default(), None),
        excluded,
    };
    let paired =
        |other: &[Vec<String>]| eval::bootstrap(Task::Bio, &filtered_gold, &candidate_filtered, other, resamples, seed);
    Ok(Report {
        corpus: corpus.clone(),
        candidate_file: digest(candidate),
        peer_file: digest(peer),
        selected_types: selected.into_iter().collect(),
        peer_mapping: peer_mapping.clone(),
        gold_excluded,
        candidate: score(&candidate_filtered, &candidate_labels, candidate_excluded),
        peer: score(&peer_filtered, &mapped, peer_excluded),
        baseline: score(&baseline_filtered, &baseline, baseline_excluded),
        baseline_description: "global training-token-majority label".into(),
        candidate_vs_peer: paired(&peer_filtered),
        candidate_vs_baseline: paired(&baseline_filtered),
    })
}

fn digest(file: PredictionFile<'_>) -> FileRecord {
    FileRecord { path: file.name.into(), size: file.bytes.len() as u64, sha256: turbo::bundle::sha256_hex(file.bytes) }
}

fn strict(sentences: &[Vec<String>]) -> Result<()> {
    for tags in sentences {
        let mut previous = None;
        for tag in tags {
            if tag == "O" {
                previous = None;
                continue;
            }
            let Some((prefix, kind)) = tag.split_once('-') else {
                return Err(Error(format!("not a BIO label: {tag:?}")));
            };
            if kind.is_empty() || !matches!(prefix, "B" | "I") || (prefix == "I" && previous != Some(kind)) {
                return Err(Error(format!("invalid original BIO sequence at {tag:?}")));
            }
            previous = Some(kind);
        }
    }
    Ok(())
}

fn map_peer(corpus: &str, labels: &[Vec<String>], mapping: &BTreeMap<String, String>) -> Result<Vec<Vec<String>>> {
    let mapped: Vec<_> = labels
        .iter()
        .map(|s| {
            s.iter()
                .map(|tag| {
                    if tag == "O" {
                        return tag.clone();
                    }
                    let (prefix, kind) = tag.split_once('-').expect("strict BIO checked");
                    let mapped = mapping.get(kind).map(String::as_str).unwrap_or(kind);
                    if mapped == "-" { "O".into() } else { format!("{prefix}-{mapped}") }
                })
                .collect()
        })
        .collect();
    ner_peer::validate(corpus, &mapped)?;
    Ok(mapped)
}

fn filter(labels: &[Vec<String>], selected: &BTreeSet<String>) -> (Vec<Vec<String>>, BTreeMap<String, usize>) {
    let mut excluded = BTreeMap::new();
    let out = labels
        .iter()
        .map(|sentence| {
            for (_, _, kind) in eval::spans(sentence) {
                if !selected.contains(&kind) {
                    *excluded.entry(kind).or_default() += 1;
                }
            }
            sentence
                .iter()
                .map(|label| {
                    if label.split_once('-').is_some_and(|(_, kind)| selected.contains(kind)) {
                        label.clone()
                    } else {
                        "O".to_owned()
                    }
                })
                .collect()
        })
        .collect();
    (out, excluded)
}
