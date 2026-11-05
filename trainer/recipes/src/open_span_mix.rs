//! Multi-corpus open-span training records and independent zero-shot scoring.

use std::path::Path;

use turbo::head::OpenSpanMatcher;
use turbo::manifest::{LicenseClass, WordPooling};
use turbo_tagger::WordEncoder;

use crate::corpus::{Corpus, CorpusRecord, Counts, Task};
use crate::eval::{Seen, bootstrap, evaluate};
use crate::open_span::{
    dictionary_baseline, embed_label_names, filter_sentences, predict_from_features, read_open_span_corpus,
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
    let test = &corpus.splits.test;
    let train = filter_sentences(&corpus.splits.train, seen_types);
    let predicted = predict_from_features(matcher, test, test_wf, all_lf, all_types)?;
    let majority = crate::eval::Majority::train(&train);
    let base: Vec<Vec<String>> = test.iter().map(|s| majority.predict(&s.words)).collect();
    let dictionary = dictionary_baseline(test, all_types);
    let seen = Seen::of(&train);
    let groups = coarse(&corpus.record.id);
    let model = evaluate(Task::Bio, test, &predicted, &seen, groups);
    let baseline = evaluate(Task::Bio, test, &base, &seen, groups);
    let dict = evaluate(Task::Bio, test, &dictionary, &seen, groups);
    let comparison = bootstrap(Task::Bio, test, &predicted, &base, resamples, seed);
    let dictionary_comparison = bootstrap(Task::Bio, test, &predicted, &dictionary, resamples, seed);
    let mut held = (None, None, None, None, None);
    if !hold_out.is_empty() {
        let held_lf = embed_label_names(enc, hold_out, pooling)?;
        let gold = filter_sentences(test, hold_out);
        let pred = predict_from_features(matcher, &gold, test_wf, &held_lf, hold_out)?;
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
            Some(evaluate(Task::Bio, &gold, &pred, &seen, None)),
            Some(evaluate(Task::Bio, &gold, &base_held, &seen, None)),
            Some(evaluate(Task::Bio, &gold, &dict_held, &seen, None)),
            Some(bootstrap(Task::Bio, &gold, &pred, &base_held, resamples, seed)),
            Some(bootstrap(Task::Bio, &gold, &pred, &dict_held, resamples, seed)),
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
