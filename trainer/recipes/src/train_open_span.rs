//! `train-open-span`: training an open-label span typing head
//! (`HEAD_OPEN_SPAN`, #1735) on a BIO NER corpus (`wnut17`, `multiconer`),
//! with optional `--hold-out <type1,type2>` zero-shot evaluation on types
//! excluded from training, scored against both the majority baseline and a
//! dictionary matcher over the label names.

use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer::OpenSpans;
use turbo_trainer_recipes::cache::{CacheRecord, read_layer};
use turbo_trainer_recipes::corpus::Task;
use turbo_trainer_recipes::eval::{Seen, bootstrap, evaluate};
use turbo_trainer_recipes::open_span::{
    self, DEFAULT_MAX_WORDS, DEFAULT_WIDTH_DIM, dictionary_baseline, embed_label_names, encode_sentences, entity_types,
    examples_from_features, filter_sentences, predict_from_features, read_open_span_corpus,
};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH, named, replace};
use turbo_trainer_recipes::{Error, Result, coarse, safetensors};

use super::{Args, Opts, RESAMPLES, baseline, encoder_record, finish, options, saved_any};

pub(super) fn train(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let max_words: u32 = a.parsed("--max-words", DEFAULT_MAX_WORDS)?;
    let width_dim: u32 = a.parsed("--width-dim", DEFAULT_WIDTH_DIM)?;
    let nested = a.flag("--nested");
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
    let corpus = read_open_span_corpus(corpus_id, dir, unrecorded)?;
    let read = t.elapsed().as_secs_f64();
    let all_types = entity_types(&corpus)?;
    let hold_out: Vec<String> = match hold_out_arg {
        Some(s) if !s.trim().is_empty() => s.split(',').map(|x| x.trim().to_owned()).collect(),
        _ => Vec::new(),
    };
    for h in &hold_out {
        if !all_types.iter().any(|t| t == h) {
            return Err(Error(format!("--hold-out {h:?} is not a type of {corpus_id} ({all_types:?})")));
        }
    }
    let seen_types: Vec<String> = all_types.iter().filter(|t| !hold_out.contains(t)).cloned().collect();
    if seen_types.is_empty() {
        return Err(Error("--hold-out leaves no training entity types".into()));
    }

    let spec = open_span::spec(seen_types.clone(), &corpus.record, pooling, bilstm, max_words, width_dim, nested)?;
    let mut enc = WordEncoder::open(&bundle, &EncoderOptions { device, precision, ..EncoderOptions::default() })
        .map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;
    let hidden = enc.hidden();

    let t = Instant::now();
    let train_slice = match max_train {
        0 => &corpus.splits.train[..],
        n => &corpus.splits.train[..n.min(corpus.splits.train.len())],
    };
    let (train_wf, dev_wf, test_wf, encoder) = if let Some(ref cdir) = cache_dir {
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
        let full_train = read_layer(cdir, &cache, "train", layer, &corpus.splits.train)?;
        let dev = read_layer(cdir, &cache, "dev", layer, &corpus.splits.dev)?;
        let test = read_layer(cdir, &cache, "test", layer, &corpus.splits.test)?;
        (full_train.into_iter().take(train_slice.len()).collect(), dev, test, cache.encoder)
    } else {
        (
            encode_sentences(&mut enc, train_slice, pooling)?,
            encode_sentences(&mut enc, &corpus.splits.dev, pooling)?,
            encode_sentences(&mut enc, &corpus.splits.test, pooling)?,
            encoder,
        )
    };
    let seen_lf = embed_label_names(&mut enc, &seen_types, pooling)?;
    let all_lf = embed_label_names(&mut enc, &all_types, pooling)?;
    let (train_ex, _) = examples_from_features(
        train_slice,
        &train_wf,
        &seen_lf,
        &seen_types,
        &hold_out,
        max_words,
        hidden,
        Some(c.seed),
    )?;
    let (dev_ex, dev_left_out) =
        examples_from_features(&corpus.splits.dev, &dev_wf, &seen_lf, &seen_types, &hold_out, max_words, hidden, None)?;
    let features_s = t.elapsed().as_secs_f64();
    let feature_words: u64 = train_slice.iter().chain(&corpus.splits.dev).map(|s| s.words.len() as u64).sum();

    let t = Instant::now();
    let spec_json = serde_json::to_vec_pretty(&spec).expect("a head spec serializes");
    let out_dir = out.path.clone();
    let mut saved = false;
    let mut save = |m: &OpenSpans<f32>, epoch: usize, dev: f64| -> Result<()> {
        if !saved {
            replace(&out_dir, "head.json", &spec_json)?;
            saved = true;
        }
        let at = [(CHECKPOINT_EPOCH, epoch.to_string()), (CHECKPOINT_DEV, dev.to_string())];
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
    let test = &corpus.splits.test;
    let train_filtered = filter_sentences(train_slice, &seen_types);
    let predicted = predict_from_features(&matcher, test, &test_wf, &all_lf, &all_types)?;
    let base = baseline(&train_filtered, test);
    let dict_all = dictionary_baseline(test, &all_types);
    let seen = Seen::of(&train_filtered);
    let model_metrics = evaluate(Task::Bio, test, &predicted, &seen, coarse(corpus_id));
    let base_metrics = evaluate(Task::Bio, test, &base, &seen, coarse(corpus_id));
    let dict_metrics = evaluate(Task::Bio, test, &dict_all, &seen, coarse(corpus_id));
    println!(
        "{corpus_id} open-span all ({} types): model {:.4}, majority {:.4}, dictionary {:.4}",
        all_types.len(),
        model_metrics.score(),
        base_metrics.score(),
        dict_metrics.score(),
    );
    let test_seen = filter_sentences(test, &seen_types);
    let pred_seen = predict_from_features(&matcher, &test_seen, &test_wf, &seen_lf, &seen_types)?;
    let dict_seen = dictionary_baseline(&test_seen, &seen_types);
    let seen_m = evaluate(Task::Bio, &test_seen, &pred_seen, &seen, None);
    let seen_b = evaluate(Task::Bio, &test_seen, &base, &seen, None);
    let seen_d = evaluate(Task::Bio, &test_seen, &dict_seen, &seen, None);
    let (held_out_model, held_out_baseline, held_out_dictionary) = if !hold_out.is_empty() {
        let held_lf = embed_label_names(&mut enc, &hold_out, pooling)?;
        let test_held = filter_sentences(test, &hold_out);
        let pred_held = predict_from_features(&matcher, &test_held, &test_wf, &held_lf, &hold_out)?;
        let base_held = filter_sentences(
            &base
                .iter()
                .zip(test)
                .map(|(b, s)| turbo_trainer_recipes::corpus::Sentence { words: s.words.clone(), labels: b.clone() })
                .collect::<Vec<_>>(),
            &hold_out,
        )
        .into_iter()
        .map(|s| s.labels)
        .collect::<Vec<_>>();
        let dict_held = dictionary_baseline(&test_held, &hold_out);
        let held_m = evaluate(Task::Bio, &test_held, &pred_held, &seen, None);
        let held_b = evaluate(Task::Bio, &test_held, &base_held, &seen, None);
        let held_d = evaluate(Task::Bio, &test_held, &dict_held, &seen, None);
        println!(
            "{corpus_id} open-span seen ({seen_types:?}): model {:.4}, majority {:.4}, dictionary {:.4}",
            seen_m.score(),
            seen_b.score(),
            seen_d.score(),
        );
        println!(
            "{corpus_id} open-span zero-shot held-out ({hold_out:?}): model {:.4}, majority {:.4}, dictionary {:.4}",
            held_m.score(),
            held_b.score(),
            held_d.score(),
        );
        (Some(held_m), Some(held_b), Some(held_d))
    } else {
        (None, None, None)
    };
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(model_metrics),
        baseline: base_metrics,
        dictionary: Some(dict_metrics),
        open_span: Some(OpenSpanEvalRecord {
            seen_types,
            held_out_types: hold_out,
            seen_model: seen_m,
            seen_baseline: seen_b,
            seen_dictionary: seen_d,
            held_out_model,
            held_out_baseline,
            held_out_dictionary,
        }),
        comparison: Some(bootstrap(Task::Bio, test, &predicted, &base, RESAMPLES, c.seed)),
        uas: None,
        rules: None,
    };
    let eval_s = t.elapsed().as_secs_f64();

    let best = report.best_epoch;
    let at = [(CHECKPOINT_EPOCH, best.to_string()), (CHECKPOINT_DEV, report.dev[best].to_string())];
    let weights = safetensors::write_with(&named(&spec, hidden, model.tensors()), &at);
    let outputs = vec![replace(&out.path, "head.safetensors", &weights)?, replace(&out.path, "head.json", &spec_json)?];
    let receipt = Receipt {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        host: host(),
        unix_time: unix_time(),
        corpus: corpus.record,
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
        }),
        training: Some(TrainingRecord {
            train_loss: report.train_loss,
            dev: report.dev,
            best_epoch: report.best_epoch,
            epoch_seconds: report.epoch_seconds,
        }),
        eval,
        timing_s: Timing { read, features: features_s, train: trained, eval: eval_s, feature_words },
        outputs,
    };
    finish(out, receipt)
}
