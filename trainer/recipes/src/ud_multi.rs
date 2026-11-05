//! Multi-treebank UD tag corpora. Train and dev concatenate in the given
//! treebank order, test remains in that same order for later per-language
//! scoring, and XPOS labels carry their treebank id so equal strings from
//! different tag sets cannot collide.

use std::path::Path;

use crate::corpus::{Corpus, CorpusRecord, Counts, Splits, Task};
use crate::ud::{Column, TREEBANKS};
use crate::{Error, Result};

/// A combined corpus and the language-table row of every sentence.
#[derive(Debug)]
pub struct MultiCorpus {
    pub corpus: Corpus,
    pub languages: Vec<String>,
    pub train_languages: Vec<Option<u32>>,
    pub dev_languages: Vec<Option<u32>>,
    pub test_languages: Vec<Option<u32>>,
}

/// A UD tag column's command-line name.
pub fn column(name: &str) -> Result<Column> {
    match name {
        "upos" => Ok(Column::Upos),
        "xpos" => Ok(Column::Xpos),
        "feats" => Ok(Column::Feats),
        _ => Err(Error(format!("--column {name:?}: upos, xpos or feats"))),
    }
}

/// The common column of a comma-separated tag corpus id, if it names one.
pub fn find_column(id: &str) -> Option<Column> {
    let mut columns = id.split(',').map(|part| crate::ud::find(part).map(|(_, column)| column));
    let first = columns.next()??;
    columns.all(|column| column == Some(first)).then_some(first)
}

fn language_names(ids: &[&str]) -> Result<Vec<String>> {
    let mut languages = Vec::with_capacity(ids.len());
    for id in ids {
        let language = id
            .strip_prefix("ud-")
            .and_then(|rest| rest.split('-').next())
            .filter(|language| !language.is_empty())
            .ok_or_else(|| Error(format!("treebank {id:?} has no language code")))?;
        if languages.iter().any(|known| known == language) {
            return Err(Error(format!("language {language:?} is listed twice; use one treebank per language")));
        }
        languages.push(language.to_owned());
    }
    Ok(languages)
}

/// Reads a comma-separated list of treebank ids as one tag corpus.
/// `sample` is the temperature exponent: 1 keeps each split unchanged,
/// 0 gives every treebank the same training sentence count, and values
/// between interpolate by `count.powf(sample)`. Resampling repeats each
/// bank's own training sentences cyclically and never touches dev or test.
pub fn read(ids: &str, column: Column, dir: &Path, unrecorded: bool, sample: f64) -> Result<MultiCorpus> {
    let ids: Vec<&str> = ids.split(',').collect();
    if ids.len() < 2 || ids.iter().any(|id| id.is_empty()) {
        return Err(Error("a multi-treebank corpus needs at least two comma-separated treebank ids".into()));
    }
    let mut corpora = Vec::with_capacity(ids.len());
    for id in ids.iter().copied() {
        if corpora.iter().any(|c: &Corpus| c.record.id == format!("{id}-{}", column.suffix())) {
            return Err(Error(format!("treebank {id:?} is listed twice")));
        }
        let bank = TREEBANKS
            .iter()
            .find(|bank| bank.id == id)
            .ok_or_else(|| Error(format!("unknown treebank {id:?}: one of {}", TREEBANKS.map(|b| b.id).join(", "))))?;
        let mut corpus = crate::ud::read(bank, column, dir, unrecorded)?;
        if column == Column::Xpos {
            for sentence in corpora_sentences_mut(&mut corpus.splits) {
                for label in &mut sentence.labels {
                    *label = format!("{}:{label}", bank.id);
                }
            }
        }
        corpora.push(corpus);
    }
    if corpora.windows(2).any(|pair| pair[0].record.spdx != pair[1].record.spdx) {
        return Err(Error("a multi-treebank run needs one SPDX licence id across its corpora".into()));
    }

    let counts: Vec<usize> = corpora.iter().map(|c| c.splits.train.len()).collect();
    let targets = sample_counts(&counts, sample)?;
    for (corpus, target) in corpora.iter_mut().zip(&targets) {
        if corpus.splits.train.len() != *target {
            let original = std::mem::take(&mut corpus.splits.train);
            corpus.splits.train = (0..*target).map(|i| original[i % original.len()].clone()).collect();
        }
    }

    let id = ids.iter().map(|id| format!("{id}-{}", column.suffix())).collect::<Vec<_>>().join(",");
    let name = corpora.iter().map(|c| c.record.name.as_str()).collect::<Vec<_>>().join("; ");
    let licence = corpora[0].record.licence.clone();
    let spdx = corpora[0].record.spdx.clone();
    let terms = corpora.iter().map(|c| c.record.terms.as_str()).collect::<Vec<_>>().join("\n");
    let attribution = corpora.iter().map(|c| c.record.attribution.as_str()).collect::<Vec<_>>().join("\n");
    let recorded = corpora.iter().all(|c| c.record.recorded);
    let files = corpora.iter().flat_map(|c| c.record.files.clone()).collect();
    let skipped = corpora.iter().map(|c| c.record.skipped).sum();
    let recipe_only = corpora.iter().any(|c| c.record.recipe_only);
    let split = format!(
        "each release's own train, dev and test files; --sample {sample} training sentence targets {:?}",
        targets
    );
    let languages = language_names(&ids)?;
    let mut splits = Splits::default();
    let (mut train_languages, mut dev_languages, mut test_languages) = (Vec::new(), Vec::new(), Vec::new());
    for (row, mut corpus) in corpora.into_iter().enumerate() {
        train_languages.extend(std::iter::repeat_n(Some(row as u32), corpus.splits.train.len()));
        dev_languages.extend(std::iter::repeat_n(Some(row as u32), corpus.splits.dev.len()));
        test_languages.extend(std::iter::repeat_n(Some(row as u32), corpus.splits.test.len()));
        splits.train.append(&mut corpus.splits.train);
        splits.dev.append(&mut corpus.splits.dev);
        splits.test.append(&mut corpus.splits.test);
    }
    let record = CorpusRecord {
        id,
        name,
        task: Task::Tags,
        licence,
        spdx,
        terms,
        attribution,
        recorded,
        corpus_sha256: String::new(),
        files,
        split,
        train: Counts::of(&splits.train),
        dev: Counts::of(&splits.dev),
        test: Counts::of(&splits.test),
        skipped,
        recipe_only,
    };
    Ok(MultiCorpus { corpus: Corpus { record, splits }, languages, train_languages, dev_languages, test_languages })
}

fn corpora_sentences_mut(splits: &mut Splits) -> impl Iterator<Item = &mut crate::corpus::Sentence> {
    splits.train.iter_mut().chain(&mut splits.dev).chain(&mut splits.test)
}

/// Training sentence counts after temperature resampling, preserving the
/// total. Largest remainders receive the sentences left after flooring;
/// ties go to the earlier treebank, so the result is deterministic.
pub fn sample_counts(counts: &[usize], sample: f64) -> Result<Vec<usize>> {
    if counts.is_empty() || counts.contains(&0) {
        return Err(Error("every treebank needs a nonempty training split".into()));
    }
    if !sample.is_finite() || !(0.0..=1.0).contains(&sample) {
        return Err(Error(format!("--sample {sample}: a finite value from 0 to 1")));
    }
    if sample == 1.0 {
        return Ok(counts.to_vec());
    }
    let total = counts.iter().sum::<usize>();
    let weights: Vec<f64> = counts.iter().map(|&count| (count as f64).powf(sample)).collect();
    let scale = total as f64 / weights.iter().sum::<f64>();
    let raw: Vec<f64> = weights.into_iter().map(|weight| weight * scale).collect();
    let mut targets: Vec<usize> = raw.iter().map(|value| value.floor() as usize).collect();
    let left = total - targets.iter().sum::<usize>();
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by(|a, b| {
        raw[*b].fract().partial_cmp(&raw[*a].fract()).expect("finite sample weights").then_with(|| a.cmp(b))
    });
    for index in order.into_iter().take(left) {
        targets[index] += 1;
    }
    Ok(targets)
}
