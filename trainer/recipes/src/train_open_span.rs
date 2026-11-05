//! `train-open-span`: one open-label head over one or more BIO corpora,
//! with genuinely held-out types and independent per-dataset zero-shot
//! metrics, dictionary baselines and paired bootstrap comparisons.

use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer::OpenSpans;
use turbo_trainer_recipes::cache::{CacheRecord, read_layer};
use turbo_trainer_recipes::corpus::{Corpus, Task};
use turbo_trainer_recipes::eval::{Majority, Seen, evaluate};
use turbo_trainer_recipes::open_span::{
    self, DEFAULT_MAX_WORDS, DEFAULT_WIDTH_DIM, dictionary_baseline, embed_label_names, encode_sentences, entity_types,
    examples_from_features, filter_sentences, predict_from_features,
};
use turbo_trainer_recipes::open_span_mix::{combine, evaluate_dataset, read_corpora};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH, named, replace};
use turbo_trainer_recipes::{Error, Result, safetensors};

use super::{Args, Opts, RESAMPLES, encoder_record, finish, options, saved_any};

struct Features {
    train: Vec<Vec<f32>>,
    dev: Vec<Vec<f32>>,
    test: Vec<Vec<f32>>,
}

pub(super) fn train(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let max_words: u32 = a.parsed("--max-words", DEFAULT_MAX_WORDS)?;
    let width_dim: u32 = a.parsed("--width-dim", DEFAULT_WIDTH_DIM)?;
    let nested = a.flag("--nested");
    let balance = a.flag("--balance");
    let hold_out_arg = a.value("--hold-out")?;
    let cache_dir = a.value("--cache")?.map(PathBuf::from);
    let layer_arg: Option<u32> =
        a.value("--layer")?.map(|v| v.parse()).transpose().map_err(|_| Error("--layer N".into()))?;
    let Opts { bundle, mut out, device, precision, pooling, crf, bilstm, hash, c, max_train, commit } =
        options(&mut a)?;
    a.done()?;
    if crf || hash.is_some() {
        return Err(Error("open-span heads take neither --crf nor --hash-embed".into()));
    }
    if layer_arg.is_some() && cache_dir.is_none() {
        return Err(Error("--layer requires --cache".into()));
    }

    let t = Instant::now();
    let mut corpora = read_corpora(corpus_id, dir, unrecorded)?;
    if corpora.len() > 1 && cache_dir.is_some() {
        return Err(Error("--cache is supported for one open-span corpus at a time".into()));
    }
    let full_train: Vec<Vec<turbo_trainer_recipes::corpus::Sentence>> =
        corpora.iter().map(|c| c.splits.train.clone()).collect();
    let mut all_types = Vec::new();
    for corpus in &corpora {
        for kind in entity_types(corpus)? {
            if !all_types.contains(&kind) {
                all_types.push(kind);
            }
        }
    }
    if max_train > 0 {
        let mut remaining = max_train;
        for c in &mut corpora {
            let keep = remaining.min(c.splits.train.len());
            c.splits.train.truncate(keep);
            remaining = remaining.saturating_sub(keep);
        }
    }
    if balance && corpora.len() > 1 {
        let target = corpora.iter().map(|c| c.splits.train.len()).max().unwrap_or(0);
        for c in &mut corpora {
            if c.splits.train.is_empty() {
                return Err(Error("--balance needs a nonempty training split in every corpus".into()));
            }
            let original = std::mem::take(&mut c.splits.train);
            c.splits.train = (0..target).map(|i| original[i % original.len()].clone()).collect();
        }
    }
    let hold_out: Vec<String> = match hold_out_arg {
        Some(s) if !s.trim().is_empty() => s.split(',').map(|x| x.trim().to_owned()).collect(),
        _ => Vec::new(),
    };
    for h in &hold_out {
        if !all_types.contains(h) {
            return Err(Error(format!("--hold-out {h:?} is not a type of {corpus_id} ({all_types:?})")));
        }
    }
    let seen_types: Vec<String> = all_types.iter().filter(|t| !hold_out.contains(t)).cloned().collect();
    if seen_types.is_empty() {
        return Err(Error("--hold-out leaves no training entity types".into()));
    }
    let combined = combine(&corpora)?;
    let read = t.elapsed().as_secs_f64();
    let spec = open_span::spec(seen_types.clone(), &combined.record, pooling, bilstm, max_words, width_dim, nested)?;
    let mut enc = WordEncoder::open(&bundle, &EncoderOptions { device, precision, ..EncoderOptions::default() })
        .map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;
    let hidden = enc.hidden();

    let t = Instant::now();
    let mut extracted = Vec::with_capacity(corpora.len());
    for (corpus, full) in corpora.iter().zip(&full_train) {
        extracted.push(features(&mut enc, corpus, full, pooling, cache_dir.as_deref(), layer_arg, &encoder)?);
    }
    let (mut train_wf, mut dev_wf, mut test_wf) = (Vec::new(), Vec::new(), Vec::new());
    for (corpus, f) in corpora.iter().zip(&extracted) {
        train_wf.extend(f.train.iter().take(corpus.splits.train.len()).cloned());
        dev_wf.extend(f.dev.clone());
        test_wf.extend(f.test.clone());
    }
    let seen_lf = embed_label_names(&mut enc, &seen_types, pooling)?;
    let all_lf = embed_label_names(&mut enc, &all_types, pooling)?;
    let (train_ex, _) = examples_from_features(
        &combined.splits.train,
        &train_wf,
        &seen_lf,
        &seen_types,
        &hold_out,
        max_words,
        hidden,
        Some(c.seed),
    )?;
    let (dev_ex, dev_left_out) = examples_from_features(
        &combined.splits.dev,
        &dev_wf,
        &seen_lf,
        &seen_types,
        &hold_out,
        max_words,
        hidden,
        None,
    )?;
    let features_s = t.elapsed().as_secs_f64();
    let feature_words: u64 =
        combined.splits.train.iter().chain(&combined.splits.dev).map(|s| s.words.len() as u64).sum();

    let t = Instant::now();
    let spec_json = serde_json::to_vec_pretty(&spec).expect("a head spec serializes");
    let out_dir = out.path.clone();
    let mut saved = false;
    let mut save = |m: &OpenSpans<f32>, epoch: usize, dev: f64| -> Result<()> {
        if !saved {
            replace(&out_dir, "head.json", &spec_json)?;
            saved = true;
        }
        let at = [(CHECKPOINT_EPOCH, (epoch + 1).to_string()), (CHECKPOINT_DEV, dev.to_string())];
        replace(&out_dir, "head.safetensors", &safetensors::write_with(&named(&spec, hidden, m.tensors()), &at))?;
        Ok(())
    };
    let fitted = turbo_trainer::fit_open_spans_with(&spec, hidden, &train_ex, &dev_ex, &c, |m, e, d| {
        save(m, e, d).map_err(|x| turbo::status::invalid(x.0))
    });
    out.keep = out.keep || saved_any(&out_dir);
    let (model, report) = fitted.map_err(|e| Error(e.message))?;
    drop((train_ex, dev_ex));
    let trained = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let matcher = model.to_core().map_err(|e| Error(e.message))?;
    let overall = evaluate_dataset(
        &matcher,
        &mut enc,
        &combined,
        &test_wf,
        &all_lf,
        &all_types,
        &seen_types,
        &hold_out,
        pooling,
        RESAMPLES,
        c.seed,
    )?;
    let mut datasets = Vec::with_capacity(corpora.len());
    for (corpus, features) in corpora.iter().zip(&extracted) {
        datasets.push(evaluate_dataset(
            &matcher,
            &mut enc,
            corpus,
            &features.test,
            &all_lf,
            &all_types,
            &seen_types,
            &hold_out,
            pooling,
            RESAMPLES,
            c.seed,
        )?);
    }
    let train_filtered = filter_sentences(&combined.splits.train, &seen_types);
    let majority = Majority::train(&train_filtered);
    let base: Vec<Vec<String>> = combined.splits.test.iter().map(|s| majority.predict(&s.words)).collect();
    let seen = Seen::of(&train_filtered);
    let test_seen = filter_sentences(&combined.splits.test, &seen_types);
    let pred_seen = predict_from_features(&matcher, &test_seen, &test_wf, &seen_lf, &seen_types)?;
    let dict_seen = dictionary_baseline(&test_seen, &seen_types);
    let seen_m = evaluate(Task::Bio, &test_seen, &pred_seen, &seen, None);
    let seen_b = evaluate(Task::Bio, &test_seen, &base, &seen, None);
    let seen_d = evaluate(Task::Bio, &test_seen, &dict_seen, &seen, None);
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(overall.model.clone()),
        baseline: overall.baseline.clone(),
        dictionary: Some(overall.dictionary.clone()),
        open_span: Some(OpenSpanEvalRecord {
            seen_types,
            held_out_types: hold_out,
            seen_model: seen_m,
            seen_baseline: seen_b,
            seen_dictionary: seen_d,
            held_out_model: overall.held_out_model.clone(),
            held_out_baseline: overall.held_out_baseline.clone(),
            held_out_dictionary: overall.held_out_dictionary.clone(),
            dictionary_comparison: Some(overall.dictionary_comparison.clone()),
            held_out_comparison: overall.held_out_comparison.clone(),
            held_out_dictionary_comparison: overall.held_out_dictionary_comparison.clone(),
            datasets,
        }),
        comparison: Some(overall.comparison.clone()),
        uas: None,
        rules: None,
        against: None,
        by_language: Vec::new(),
    };
    let eval_s = t.elapsed().as_secs_f64();

    let best = report.best_epoch;
    let at = [(CHECKPOINT_EPOCH, (best + 1).to_string()), (CHECKPOINT_DEV, report.dev[best].to_string())];
    let weights = safetensors::write_with(&named(&spec, hidden, model.tensors()), &at);
    let outputs = vec![replace(&out.path, "head.safetensors", &weights)?, replace(&out.path, "head.json", &spec_json)?];
    let receipt = Receipt {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        unix_time: unix_time(),
        corpus: combined.record,
        encoder: Some(encoder),
        head: Some(spec),
        peer: None,
        config: Some(ConfigRecord {
            seed: c.seed,
            epochs: c.epochs,
            batch: c.batch,
            lr: c.lr,
            weight_decay: c.weight_decay,
            clip: c.clip,
            warmup: c.warmup,
            patience: c.patience,
            tolerance: c.tolerance,
            threads: c.threads,
            max_train,
            dev_left_out,
            sample: None,
        }),
        training: Some(TrainingRecord {
            train_loss: report.train_loss,
            dev: report.dev,
            best_epoch: best + 1,
            epoch_seconds: report.epoch_seconds,
        }),
        eval,
        timing_s: Timing { read, features: features_s, train: trained, eval: eval_s, feature_words },
        outputs,
    };
    finish(out, receipt)
}

fn features(
    enc: &mut WordEncoder,
    corpus: &Corpus,
    full_train: &[turbo_trainer_recipes::corpus::Sentence],
    pooling: turbo::manifest::WordPooling,
    cache_dir: Option<&Path>,
    layer_arg: Option<u32>,
    encoder: &EncoderRecord,
) -> Result<Features> {
    let Some(cdir) = cache_dir else {
        return Ok(Features {
            train: encode_sentences(enc, &corpus.splits.train, pooling)?,
            dev: encode_sentences(enc, &corpus.splits.dev, pooling)?,
            test: encode_sentences(enc, &corpus.splits.test, pooling)?,
        });
    };
    let p = cdir.join("cache.json");
    let text = std::fs::read(&p).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let cache: CacheRecord = serde_json::from_slice(&text).map_err(|e| Error(format!("{}: {e}", p.display())))?;
    let corpus_ok = cache.corpus == corpus.record
        || (cache.corpus.files == corpus.record.files
            && cache.corpus.recorded == corpus.record.recorded
            && cache.corpus.id == format!("{}-upos", corpus.record.id));
    if !corpus_ok
        || cache.encoder.model != encoder.model
        || cache.encoder.revision != encoder.revision
        || cache.encoder.manifest_sha256 != encoder.manifest_sha256
        || cache.encoder.hidden != encoder.hidden
        || cache.encoder.layers != encoder.layers
        || cache.encoder.width != encoder.width
        || cache.pooling != pooling
    {
        return Err(Error(format!("{}: cache does not match corpus, encoder or pooling", cdir.display())));
    }
    let layer = match layer_arg {
        Some(l) if cache.layers.contains(&l) => l,
        Some(l) => return Err(Error(format!("{}: no layer {l}; it holds {:?}", cdir.display(), cache.layers))),
        None => *cache.layers.last().ok_or_else(|| Error(format!("{}: cache has no layers", cdir.display())))?,
    };
    Ok(Features {
        train: read_layer(cdir, &cache, "train", layer, full_train)?,
        dev: read_layer(cdir, &cache, "dev", layer, &corpus.splits.dev)?,
        test: read_layer(cdir, &cache, "test", layer, &corpus.splits.test)?,
    })
}
