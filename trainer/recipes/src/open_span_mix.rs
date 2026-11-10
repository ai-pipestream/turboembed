//! Multi-corpus open-span training records and independent zero-shot scoring.

use std::path::Path;

use turbo::head::OpenSpanMatcher;
use turbo::manifest::{LicenseClass, WordPooling};
use turbo_tagger::WordEncoder;

use crate::corpus::{Corpus, CorpusRecord, Counts, Task};
use crate::eval::{Seen, bootstrap, evaluate};
use crate::open_span::{
    dictionary_baseline, embed_label_names, entity_types, filter_sentences, predict_from_features,
    read_open_span_corpus,
};
use crate::receipt::OpenSpanDatasetEval;
use crate::{Error, Result, coarse};

pub fn read_corpora(ids: &str, dir: &Path, unrecorded: bool) -> Result<Vec<Corpus>> {
    let ids: Vec<&str> = ids.split(',').map(str::trim).collect();
    let dirs_text = dir.to_string_lossy();
    let dirs: Vec<&str> = dirs_text.split(',').map(str::trim).collect();
    if ids.is_empty()
        || ids.len() != dirs.len()
        || ids.iter().any(|id| id.is_empty())
        || dirs.iter().any(|d| d.is_empty())
    {
        return Err(Error("open-span --corpus and --dir need equal, non-empty comma-separated lists".into()));
    }
    let mut out = Vec::with_capacity(ids.len());
    for (id, dir) in ids.into_iter().zip(dirs) {
        if out.iter().any(|c: &Corpus| c.record.id == id) {
            return Err(Error(format!("duplicate open-span corpus {id:?}")));
        }
        out.push(read_open_span_corpus(id, Path::new(dir), unrecorded)?);
    }
    Ok(out)
}

/// Captures each corpus's full schema before any training truncation, plus
/// the union in deterministic first-seen order.
pub fn type_schema(corpora: &[Corpus]) -> Result<(Vec<String>, Vec<Vec<String>>)> {
    let mut all = Vec::new();
    let mut declared = Vec::with_capacity(corpora.len());
    for corpus in corpora {
        let types = entity_types(corpus)?;
        for kind in &types {
            if !all.contains(kind) {
                all.push(kind.clone());
            }
        }
        declared.push(types);
    }
    Ok((all, declared))
}

/// Returns training feature rows in the same order as the transformed
/// sentences. `period` is the sentence count after optional truncation and
/// before optional balancing, so repeated sentences reuse their own rows.
pub fn align_train_features(features: &[Vec<f32>], sentences: usize, period: usize) -> Result<Vec<Vec<f32>>> {
    if sentences == 0 {
        return Ok(Vec::new());
    }
    if period == 0 || period > features.len() {
        return Err(Error(format!(
            "open-span training features do not match sentences: {sentences} rows requested with period {period} over {} features",
            features.len()
        )));
    }
    Ok((0..sentences).map(|i| features[i % period].clone()).collect())
}

/// Keeps the global label order while restricting supervision to the types
/// declared by one corpus. Types absent from that corpus are not outside.
pub fn corpus_active_types(declared: &[String], active: &[String]) -> Vec<String> {
    active.iter().filter(|kind| declared.contains(*kind)).cloned().collect()
}

pub fn combine(corpora: &[Corpus]) -> Result<Corpus> {
    let Some(first) = corpora.first() else { return Err(Error("open-span needs at least one corpus".into())) };
    if corpora.len() == 1 {
        return Ok(first.clone());
    }
    let rank = |c: LicenseClass| match c {
        LicenseClass::Permissive => 0,
        LicenseClass::Lgpl => 1,
        LicenseClass::ShareAlike => 2,
        LicenseClass::Gpl => 3,
        LicenseClass::LocalOnly => 4,
    };
    let mut strict = &first.record;
    for c in &corpora[1..] {
        if rank(c.record.class()?) > rank(strict.class()?) {
            strict = &c.record;
        }
    }
    let join = |f: fn(&CorpusRecord) -> &str| corpora.iter().map(|c| f(&c.record)).collect::<Vec<_>>().join(" + ");
    let counts = |part: fn(&Corpus) -> usize, words: fn(&Corpus) -> usize| Counts {
        sentences: corpora.iter().map(part).sum(),
        words: corpora.iter().map(words).sum(),
    };
    let train_n = |c: &Corpus| c.splits.train.len();
    let dev_n = |c: &Corpus| c.splits.dev.len();
    let test_n = |c: &Corpus| c.splits.test.len();
    let train_w = |c: &Corpus| c.splits.train.iter().map(|s| s.words.len()).sum();
    let dev_w = |c: &Corpus| c.splits.dev.iter().map(|s| s.words.len()).sum();
    let test_w = |c: &Corpus| c.splits.test.iter().map(|s| s.words.len()).sum();
    let mut record = strict.clone();
    record.id = corpora.iter().map(|c| c.record.id.as_str()).collect::<Vec<_>>().join(",");
    record.name = join(|c| &c.name);
    record.task = Task::Bio;
    record.licence = join(|c| &c.licence);
    record.terms = join(|c| &c.terms);
    record.attribution = join(|c| &c.attribution);
    record.recorded = corpora.iter().all(|c| c.record.recorded);
    record.corpus_sha256 = String::new();
    record.files = corpora
        .iter()
        .flat_map(|c| {
            c.record.files.iter().map(|f| crate::corpus::FileRecord {
                path: format!("{}/{}", c.record.id, f.path),
                size: f.size,
                sha256: f.sha256.clone(),
            })
        })
        .collect();
    record.split = join(|c| &c.split);
    record.train = counts(train_n, train_w);
    record.dev = counts(dev_n, dev_w);
    record.test = counts(test_n, test_w);
    record.skipped = corpora.iter().map(|c| c.record.skipped).sum();
    record.recipe_only = corpora.iter().any(|c| c.record.recipe_only);
    let mut splits = crate::corpus::Splits::default();
    for c in corpora {
        splits.train.extend(c.splits.train.clone());
        splits.dev.extend(c.splits.dev.clone());
        splits.test.extend(c.splits.test.clone());
    }
    Ok(Corpus { record, splits })
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_dataset(
    matcher: &OpenSpanMatcher,
    enc: &mut WordEncoder,
    corpus: &Corpus,
    test_wf: &[Vec<f32>],
    all_lf: &[f32],
    all_types: &[String],
    seen_types: &[String],
    hold_out: &[String],
    pooling: WordPooling,
    resamples: usize,
    seed: u64,
) -> Result<OpenSpanDatasetEval> {
    let predicted = predict_from_features(matcher, &corpus.splits.test, test_wf, all_lf, all_types)?;
    let held = if hold_out.is_empty() {
        None
    } else {
        let held_lf = embed_label_names(enc, hold_out, pooling)?;
        let gold = filter_sentences(&corpus.splits.test, hold_out);
        Some(predict_from_features(matcher, &gold, test_wf, &held_lf, hold_out)?)
    };
    evaluate_predictions(corpus, all_types, seen_types, hold_out, &predicted, held.as_deref(), "test", resamples, seed)
}

/// Scores predictions from any open-span implementation against the same
/// majority and dictionary baselines and held-out slices as a trained head.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_predictions(
    corpus: &Corpus,
    all_types: &[String],
    seen_types: &[String],
    hold_out: &[String],
    predicted: &[Vec<String>],
    held_predicted: Option<&[Vec<String>]>,
    split: &str,
    resamples: usize,
    seed: u64,
) -> Result<OpenSpanDatasetEval> {
    let test = match split {
        "dev" => &corpus.splits.dev,
        "test" => &corpus.splits.test,
        _ => return Err(Error("open-span evaluation split must be dev or test".into())),
    };
    if predicted.len() != test.len() || hold_out.is_empty() != held_predicted.is_none() {
        return Err(Error("open-span predictions do not match the evaluation split or held-out schema".into()));
    }
    if held_predicted.is_some_and(|p| p.len() != test.len()) {
        return Err(Error("open-span held-out predictions do not match the evaluation split".into()));
    }
    let train = filter_sentences(&corpus.splits.train, seen_types);
    let majority = crate::eval::Majority::train(&train);
    let base: Vec<Vec<String>> = test.iter().map(|s| majority.predict(&s.words)).collect();
    let dictionary = dictionary_baseline(test, all_types);
    let seen = Seen::of(&train);
    let groups = coarse(&corpus.record.id);
    let model = evaluate(Task::Bio, test, predicted, &seen, groups);
    let baseline = evaluate(Task::Bio, test, &base, &seen, groups);
    let dict = evaluate(Task::Bio, test, &dictionary, &seen, groups);
    let comparison = bootstrap(Task::Bio, test, predicted, &base, resamples, seed);
    let dictionary_comparison = bootstrap(Task::Bio, test, predicted, &dictionary, resamples, seed);
    let mut held = (None, None, None, None, None);
    if let Some(pred) = held_predicted {
        let gold = filter_sentences(test, hold_out);
        let base_held = filter_sentences(
            &base
                .iter()
                .zip(test)
                .map(|(b, s)| crate::corpus::Sentence { words: s.words.clone(), labels: b.clone() })
                .collect::<Vec<_>>(),
            hold_out,
        );
        let base_held: Vec<Vec<String>> = base_held.into_iter().map(|s| s.labels).collect();
        let dict_held = dictionary_baseline(&gold, hold_out);
        held = (
            Some(evaluate(Task::Bio, &gold, pred, &seen, None)),
            Some(evaluate(Task::Bio, &gold, &base_held, &seen, None)),
            Some(evaluate(Task::Bio, &gold, &dict_held, &seen, None)),
            Some(bootstrap(Task::Bio, &gold, pred, &base_held, resamples, seed)),
            Some(bootstrap(Task::Bio, &gold, pred, &dict_held, resamples, seed)),
        );
    }
    Ok(OpenSpanDatasetEval {
        corpus_id: corpus.record.id.clone(),
        model,
        baseline,
        dictionary: dict,
        comparison,
        dictionary_comparison,
        held_out_model: held.0,
        held_out_baseline: held.1,
        held_out_dictionary: held.2,
        held_out_comparison: held.3,
        held_out_dictionary_comparison: held.4,
    })
}
