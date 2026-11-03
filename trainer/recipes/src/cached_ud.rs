//! Cached layer and scalar-mix training for UD lemma, dependency parse,
//! and multi-word token split heads (`#1822`), plus shared `Receipt`
//! generation over cached test splits.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use turbo::head::TokenTagger;
use turbo::head::tagger::Work as TagWork;
use turbo::manifest::{Biaffine, HeadSpec};
use turbo_trainer::mix::{MixExample, fit_mix};
use turbo_trainer::{
    Config, Example, LemmaExample, MixLemmaExample, MixParseExample, ParseExample, Report, fit_lemmatizer_mix,
    fit_lemmatizer_with, fit_parser_mix, fit_parser_with, mine_trees,
};

use crate::cache::{CacheRecord, layer_mix_for, read_layer, same_corpus_release, stack_layers};
use crate::corpus::{Sentence, Task};
use crate::eval::{Majority, Seen, bootstrap, evaluate};
use crate::fit_cache::{CachedOutcome, Mix, TestEval, examples_at, finish_cached};
use crate::receipt::{EvalRecord, Uas};
use crate::ud::{TREEBANKS, Treebank, UdSentence, read_treebank};
use crate::ud_heads::{
    UdHead, UdOptions, deps_baseline, lemma_baseline, predict_lemmas, predict_parses, record as ud_record, relations,
    spec as ud_spec, uas,
};
use crate::ud_split as split;
use crate::{Error, Result, coarse};

const RESAMPLES: usize = 1000;

/// Options for a cached training run across `tag`, `lemma`, `parse`, or `split`.
#[derive(Debug, Clone)]
pub struct CachedParams {
    pub argv: Vec<String>,
    pub commit: String,
    pub corpus_id: String,
    pub corpus_dir: std::path::PathBuf,
    pub cache_dir: std::path::PathBuf,
    pub out_dir: std::path::PathBuf,
    pub unrecorded: bool,
    pub layers: Vec<u32>,
    pub mixed: bool,
    pub crf: bool,
    pub bilstm: Option<(u32, u32)>,
    pub hash: Option<turbo::manifest::HashEmbed>,
    pub min_tree_freq: usize,
    pub biaffine: Biaffine,
    pub config: Config,
    pub max_train: usize,
}

/// Resolves a UD treebank from `id`, stripping any task suffix (`-lemma`,
/// `-deps`, `-parse`, `-split`, `-upos`, `-xpos`, `-feats`).
pub fn resolve_bank(id: &str) -> Result<Treebank> {
    let base = ["lemma", "deps", "parse", "split", "upos", "xpos", "feats"]
        .iter()
        .find_map(|s| id.strip_suffix(&format!("-{s}")))
        .unwrap_or(id);
    TREEBANKS
        .iter()
        .find(|b| b.id == base)
        .copied()
        .ok_or_else(|| Error(format!("unknown treebank {id:?}: one of {}", TREEBANKS.map(|b| b.id).join(", "))))
}

/// Runs a cached `lemma` or `parse` head (`head_kind`) and writes
/// `head.safetensors`, `head.json`, `fit.json`, and (when `"test"` is
/// cached) `receipt.json`.
pub fn run_ud(p: &CachedParams, head_kind: UdHead, cache: CacheRecord) -> Result<CachedOutcome> {
    if p.crf || p.hash.is_some() {
        return Err(Error(format!("{} heads take neither --crf nor --hash-embed", head_kind.suffix())));
    }
    let bank = resolve_bank(&p.corpus_id)?;
    let t_read = Instant::now();
    let (full_splits, files, recorded) = read_treebank(&bank, &p.corpus_dir, p.unrecorded)?;
    let full_record = ud_record(&bank, head_kind, &full_splits, files.clone(), recorded);
    if !same_corpus_release(&full_record, &cache.corpus) {
        return Err(Error(format!("{}: a cache of another corpus or release", p.cache_dir.display())));
    }
    let full_train_sents = head_kind.sentences(&full_splits.train);
    let dev_sents = head_kind.sentences(&full_splits.dev);
    let test_sents = head_kind.sentences(&full_splits.test);
    let mut splits = full_splits;
    if p.max_train > 0 {
        splits.train.truncate(p.max_train);
    }
    let record = ud_record(&bank, head_kind, &splits, files, recorded);
    let labels = match head_kind {
        UdHead::Lemma => mine_trees(
            splits.train.iter().flat_map(|s| s.words.iter().zip(&s.lemmas).map(|(w, l)| (w.as_str(), l.as_str()))),
            p.min_tree_freq,
        ),
        UdHead::Deps => relations(&splits.train),
    };
    let mut head = ud_spec(
        head_kind,
        labels,
        &record,
        &UdOptions { pooling: cache.pooling, bilstm: p.bilstm, biaffine: p.biaffine },
    )?;
    head.layer_mix = layer_mix_for(&p.layers, p.mixed, cache.encoder.layers);
    let hidden = cache.encoder.hidden;

    let train_layers = read_ud_layers(&p.cache_dir, &cache, "train", &p.layers, &full_train_sents, splits.train.len())?;
    let dev_layers = read_ud_layers(&p.cache_dir, &cache, "dev", &p.layers, &dev_sents, splits.dev.len())?;
    let read_s = t_read.elapsed().as_secs_f64();
    let feature_words: u64 = splits.train.iter().chain(&splits.dev).map(|s| s.words.len() as u64).sum();

    let t_train = Instant::now();
    let (tensors, report, mix, dev_left_out, test_eval) = match head_kind {
        UdHead::Lemma => fit_cached_lemma(
            p,
            &head,
            hidden,
            &splits.train,
            &splits.dev,
            &splits.test,
            &test_sents,
            &cache,
            train_layers,
            dev_layers,
        )?,
        UdHead::Deps => fit_cached_parse(
            p,
            &head,
            hidden,
            &splits.train,
            &splits.dev,
            &splits.test,
            &test_sents,
            &cache,
            train_layers,
            dev_layers,
        )?,
    };
    let train_s = t_train.elapsed().as_secs_f64();
    finish_cached(
        p,
        cache,
        record,
        head,
        tensors,
        report,
        mix,
        0,
        dev_left_out,
        read_s,
        train_s,
        feature_words,
        test_eval,
    )
}

fn read_ud_layers(
    cache_dir: &Path,
    cache: &CacheRecord,
    split: &str,
    layers: &[u32],
    full_sents: &[Sentence],
    keep: usize,
) -> Result<Vec<Vec<Vec<f32>>>> {
    layers
        .iter()
        .map(|&l| {
            let mut v = read_layer(cache_dir, cache, split, l, full_sents)?;
            v.truncate(keep);
            Ok(v)
        })
        .collect()
}

type UdFitResult = (Vec<(String, Vec<f32>)>, Report, Option<Mix>, usize, TestEval);

#[allow(clippy::too_many_arguments)]
fn fit_cached_lemma(
    p: &CachedParams,
    head: &HeadSpec,
    hidden: usize,
    train_ud: &[UdSentence],
    dev_ud: &[UdSentence],
    test_ud: &[UdSentence],
    test_sents: &[Sentence],
    cache: &CacheRecord,
    train_layers: Vec<Vec<Vec<f32>>>,
    dev_layers: Vec<Vec<Vec<f32>>>,
) -> Result<UdFitResult> {
    let build = |part: &[UdSentence], per_layer: Vec<Vec<Vec<f32>>>| -> Vec<MixLemmaExample> {
        (0..part.len())
            .map(|i| MixLemmaExample {
                layers: per_layer.iter().map(|l| l[i].clone()).collect(),
                tags: Vec::new(),
                forms: part[i].words.clone(),
                lemmas: part[i].lemmas.clone(),
            })
            .collect()
    };
    let train_mix = build(train_ud, train_layers);
    let dev_mix = build(dev_ud, dev_layers);
    let (tensors, core, report, mix) = if p.layers.len() == 1 && !p.mixed {
        let one = |v: Vec<MixLemmaExample>| -> Vec<LemmaExample> {
            v.into_iter()
                .map(|mut e| LemmaExample {
                    features: e.layers.remove(0),
                    tags: e.tags,
                    forms: e.forms,
                    lemmas: e.lemmas,
                })
                .collect()
        };
        let (m, r) = fit_lemmatizer_with(head, hidden, one(train_mix), one(dev_mix), &p.config, &mut |_, _, _| Ok(()))
            .map_err(|e| Error(e.message))?;
        (m.tensors(), m.to_core().map_err(|e| Error(e.message))?, r, None)
    } else {
        let (m, r) =
            fit_lemmatizer_mix(head, hidden, p.layers.len(), train_mix, dev_mix, &p.config, &mut |_, _, _| Ok(()))
                .map_err(|e| Error(e.message))?;
        let (w, g) = m.weights();
        let mix = Mix { weights: w.iter().map(|&v| f64::from(v)).collect(), gain: f64::from(g) };
        (m.tensors(), m.to_core().map_err(|e| Error(e.message))?, r, Some(mix))
    };
    let test_eval = if cache.splits.iter().any(|s| s.name == "test") && !test_ud.is_empty() {
        let t_eval = Instant::now();
        let test_layers = read_ud_layers(&p.cache_dir, cache, "test", &p.layers, test_sents, test_ud.len())?;
        let test_f = stack_layers(&test_layers);
        let predicted = predict_lemmas(&core, test_ud, &test_f)?;
        let base = lemma_baseline(train_ud, test_ud);
        let train_s = UdHead::Lemma.sentences(train_ud);
        let test_gold = UdHead::Lemma.sentences(test_ud);
        let seen = Seen::of(&train_s);
        let eval = EvalRecord {
            split: "test".into(),
            model: Some(evaluate(Task::Tags, &test_gold, &predicted, &seen, None)),
            baseline: evaluate(Task::Tags, &test_gold, &base, &seen, None),
            rules: None,
            dictionary: None,
            open_span: None,
            comparison: Some(bootstrap(Task::Tags, &test_gold, &predicted, &base, RESAMPLES, p.config.seed)),
            uas: None,
        };
        Some((eval, t_eval.elapsed().as_secs_f64(), test_gold, predicted))
    } else {
        None
    };
    Ok((tensors, report, mix, 0, test_eval))
}

#[allow(clippy::too_many_arguments)]
fn fit_cached_parse(
    p: &CachedParams,
    head: &HeadSpec,
    hidden: usize,
    train_ud: &[UdSentence],
    dev_ud: &[UdSentence],
    test_ud: &[UdSentence],
    test_sents: &[Sentence],
    cache: &CacheRecord,
    train_layers: Vec<Vec<Vec<f32>>>,
    dev_layers: Vec<Vec<Vec<f32>>>,
) -> Result<UdFitResult> {
    let index: BTreeMap<&str, u32> = head.labels.iter().enumerate().map(|(i, l)| (l.as_str(), i as u32)).collect();
    let build = |part: &[UdSentence], per_layer: Vec<Vec<Vec<f32>>>| -> (Vec<MixParseExample>, usize) {
        let mut out = Vec::with_capacity(part.len());
        let mut left = 0;
        for (i, s) in part.iter().enumerate() {
            match s.rels.iter().map(|r| index.get(r.as_str()).copied()).collect::<Option<Vec<u32>>>() {
                Some(rels) => out.push(MixParseExample {
                    layers: per_layer.iter().map(|l| l[i].clone()).collect(),
                    heads: s.heads.clone(),
                    rels,
                }),
                None => left += 1,
            }
        }
        (out, left)
    };
    let (train_mix, _) = build(train_ud, train_layers);
    let (dev_mix, dev_left_out) = build(dev_ud, dev_layers);
    let (tensors, core, report, mix) = if p.layers.len() == 1 && !p.mixed {
        let one = |v: Vec<MixParseExample>| -> Vec<ParseExample> {
            v.into_iter()
                .map(|mut e| ParseExample { features: e.layers.remove(0), heads: e.heads, rels: e.rels })
                .collect()
        };
        let (tr, dv) = (one(train_mix), one(dev_mix));
        let (m, r) =
            fit_parser_with(head, hidden, &tr, &dv, &p.config, &mut |_, _, _| Ok(())).map_err(|e| Error(e.message))?;
        (m.tensors(), m.to_core().map_err(|e| Error(e.message))?, r, None)
    } else {
        let (m, r) =
            fit_parser_mix(head, hidden, p.layers.len(), &train_mix, &dev_mix, &p.config, &mut |_, _, _| Ok(()))
                .map_err(|e| Error(e.message))?;
        let (w, g) = m.weights();
        let mix = Mix { weights: w.iter().map(|&v| f64::from(v)).collect(), gain: f64::from(g) };
        (m.tensors(), m.to_core().map_err(|e| Error(e.message))?, r, Some(mix))
    };
    let test_eval = if cache.splits.iter().any(|s| s.name == "test") && !test_ud.is_empty() {
        let t_eval = Instant::now();
        let test_layers = read_ud_layers(&p.cache_dir, cache, "test", &p.layers, test_sents, test_ud.len())?;
        let test_f = stack_layers(&test_layers);
        let predicted = predict_parses(&core, test_ud, &test_f)?;
        let base = deps_baseline(train_ud, test_ud);
        let train_s = UdHead::Deps.sentences(train_ud);
        let test_gold = UdHead::Deps.sentences(test_ud);
        let seen = Seen::of(&train_s);
        let eval = EvalRecord {
            split: "test".into(),
            model: Some(evaluate(Task::Tags, &test_gold, &predicted, &seen, None)),
            baseline: evaluate(Task::Tags, &test_gold, &base, &seen, None),
            rules: None,
            dictionary: None,
            open_span: None,
            comparison: Some(bootstrap(Task::Tags, &test_gold, &predicted, &base, RESAMPLES, p.config.seed)),
            uas: Some(Uas { model: uas(&test_gold, &predicted), baseline: uas(&test_gold, &base) }),
        };
        Some((eval, t_eval.elapsed().as_secs_f64(), test_gold, predicted))
    } else {
        None
    };
    Ok((tensors, report, mix, dev_left_out, test_eval))
}

/// Runs a cached `split` head and writes `head.safetensors`, `head.json`,
/// `fit.json`, and (when `"test"` is cached) `receipt.json`.
pub fn run_split(p: &CachedParams, cache: CacheRecord) -> Result<CachedOutcome> {
    if p.crf || p.hash.is_some() {
        return Err(Error("split heads take neither --crf nor --hash-embed".into()));
    }
    let bank = resolve_bank(&p.corpus_id)?;
    let t_read = Instant::now();
    let (full_splits, files, recorded) = split::read_tokens(&bank, &p.corpus_dir, p.unrecorded)?;
    let full_record = split::record(&bank, &full_splits, files.clone(), recorded);
    if !same_corpus_release(&full_record, &cache.corpus) {
        return Err(Error(format!("{}: a cache of another corpus or release", p.cache_dir.display())));
    }
    let mut splits = full_splits.clone();
    if p.max_train > 0 {
        splits.train.truncate(p.max_train);
    }
    let record = split::record(&bank, &splits, files, recorded);
    let chosen = crate::split_labels::choose(&splits.train);
    let labels = split::mine_labels(&splits.train, &chosen, p.min_tree_freq);
    if labels.len() < 2 {
        return Err(Error(format!(
            "{}: no multiword token's split comes up {} times in training",
            bank.id, p.min_tree_freq
        )));
    }
    let mut head = split::spec(labels, &record, cache.pooling, p.bilstm)?;
    head.layer_mix = layer_mix_for(&p.layers, p.mixed, cache.encoder.layers);
    let hidden = cache.encoder.hidden;

    let full_train_s = split::labelled(&full_splits.train, &chosen, &head.labels);
    let dev_s = split::labelled(&splits.dev, &chosen, &head.labels);
    let (mut train_ex, train_left_out) =
        examples_at(&p.cache_dir, &cache, "train", &p.layers, &full_train_s, &head.labels)?;
    if p.max_train > 0 && train_ex.len() > p.max_train {
        train_ex.truncate(p.max_train);
    }
    let (dev_ex, dev_left_out) = examples_at(&p.cache_dir, &cache, "dev", &p.layers, &dev_s, &head.labels)?;
    let read_s = t_read.elapsed().as_secs_f64();
    let feature_words: u64 = train_ex.iter().chain(&dev_ex).map(|e| e.labels.len() as u64).sum();

    let t_train = Instant::now();
    let (tensors, splitter, report, mix) = if p.layers.len() == 1 && !p.mixed {
        let one = |v: Vec<MixExample>| -> Vec<Example> {
            v.into_iter()
                .map(|mut e| Example { features: e.layers.remove(0), labels: e.labels, forms: e.forms })
                .collect()
        };
        let (tr, dv) = (one(train_ex), one(dev_ex));
        let (m, r) = turbo_trainer::fit(&head, hidden, &tr, &dv, &p.config).map_err(|e| Error(e.message))?;
        (m.tensors(), m.to_splitter().map_err(|e| Error(e.message))?, r, None)
    } else {
        let (m, r) = fit_mix(&head, hidden, p.layers.len(), &train_ex, &dev_ex, &p.config, &mut |_, _, _| Ok(()))
            .map_err(|e| Error(e.message))?;
        let (w, g) = m.weights();
        let mix = Mix { weights: w.iter().map(|&v| f64::from(v)).collect(), gain: f64::from(g) };
        (m.tensors(), m.to_token_splitter().map_err(|e| Error(e.message))?, r, Some(mix))
    };
    let train_s = t_train.elapsed().as_secs_f64();

    let test_eval = if cache.splits.iter().any(|s| s.name == "test") && !splits.test.is_empty() {
        let t_eval = Instant::now();
        let test_gold = split::sentences(&splits.test);
        let test_layers = read_ud_layers(&p.cache_dir, &cache, "test", &p.layers, &test_gold, splits.test.len())?;
        let test_f = stack_layers(&test_layers);
        let predicted = split::predict(&splitter, &splits.test, &test_f)?;
        let base = split::baseline(&splits.train, &splits.test);
        let seen = Seen::of(&split::sentences(&splits.train));
        let eval = EvalRecord {
            split: "test".into(),
            model: Some(evaluate(Task::Tags, &test_gold, &predicted, &seen, None)),
            baseline: evaluate(Task::Tags, &test_gold, &base, &seen, None),
            rules: None,
            dictionary: None,
            open_span: None,
            comparison: Some(bootstrap(Task::Tags, &test_gold, &predicted, &base, RESAMPLES, p.config.seed)),
            uas: None,
        };
        Some((eval, t_eval.elapsed().as_secs_f64(), test_gold, predicted))
    } else {
        None
    };
    finish_cached(
        p,
        cache,
        record,
        head,
        tensors,
        report,
        mix,
        train_left_out,
        dev_left_out,
        read_s,
        train_s,
        feature_words,
        test_eval,
    )
}

/// Evaluates a trained tagger on cached test features and builds its `TestEval`.
pub fn eval_cached_tagger(
    p: &CachedParams,
    cache: &CacheRecord,
    tagger: &TokenTagger,
    train_sents: &[Sentence],
    test_sents: &[Sentence],
    task: Task,
) -> Result<TestEval> {
    if !cache.splits.iter().any(|s| s.name == "test") || test_sents.is_empty() {
        return Ok(None);
    }
    let t_eval = Instant::now();
    let test_layers = read_ud_layers(&p.cache_dir, cache, "test", &p.layers, test_sents, test_sents.len())?;
    let test_f = stack_layers(&test_layers);
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), TagWork::default());
    let mut predicted = Vec::with_capacity(test_sents.len());
    for (s, f) in test_sents.iter().zip(&test_f) {
        let forms: Vec<&str> = s.words.iter().map(String::as_str).collect();
        let ids: Vec<u32> = (0..forms.len() as u32).collect();
        tagger.scores_forms(f, &ids, Some(&forms), &mut scores, &mut work).map_err(|e| Error(e.message))?;
        tagger.decode(&scores, forms.len(), &mut tags).map_err(|e| Error(e.message))?;
        predicted.push(tags.iter().map(|&t| tagger.labels()[t as usize].clone()).collect());
    }
    let m = Majority::train(train_sents);
    let base: Vec<Vec<String>> = test_sents.iter().map(|s| m.predict(&s.words)).collect();
    let seen = Seen::of(train_sents);
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(evaluate(task, test_sents, &predicted, &seen, coarse(&p.corpus_id))),
        baseline: evaluate(task, test_sents, &base, &seen, coarse(&p.corpus_id)),
        rules: None,
        dictionary: None,
        open_span: None,
        comparison: Some(bootstrap(task, test_sents, &predicted, &base, RESAMPLES, p.config.seed)),
        uas: None,
    };
    Ok(Some((eval, t_eval.elapsed().as_secs_f64(), test_sents.to_vec(), predicted)))
}
