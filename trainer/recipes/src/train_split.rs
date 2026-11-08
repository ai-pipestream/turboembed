//! `train-split`: a token split head on a UD treebank's surface
//! tokens, written and scored as `train` writes and scores a tagger
//! (the head's tensors are a tagger's; its own type decodes).

use std::path::Path;
use std::time::Instant;

use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer_recipes::corpus::Task;
use turbo_trainer_recipes::eval::{Seen, bootstrap, evaluate};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH, named, replace};
use turbo_trainer_recipes::ud::TREEBANKS;
use turbo_trainer_recipes::ud_split as split;
use turbo_trainer_recipes::{Error, Result, safetensors};

use super::{Args, Opts, RESAMPLES, encoder_record, finish, options, saved_any};

/// spaCy's default, as `train-lemma` keeps: a split's trees must come
/// up this often in training to be a label of the head's.
const MIN_TREE_FREQ: usize = 3;

pub(super) fn train(argv: &[String], bank: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    if a.rest.iter().any(|x| x == "--cache") {
        return turbo_trainer_recipes::fit_cache::command_with_task(
            argv,
            bank,
            dir,
            unrecorded,
            Some(turbo_trainer_recipes::fit_cache::CacheTask::Split),
            a,
        );
    }
    let bank_id = bank;
    let bank = TREEBANKS.iter().find(|b| b.id == bank_id);
    if bank.is_none() && !bank_id.contains(',') {
        return Err(Error(format!("unknown treebank {bank_id:?}: one of {}", TREEBANKS.map(|b| b.id).join(", "))));
    }
    let sample = if bank.is_none() { a.parsed("--sample", 1.0)? } else { 1.0 };
    let min_freq: usize = a.parsed("--min-tree-freq", MIN_TREE_FREQ)?;
    let seen_k: u32 = a.parsed("--seen-lookup", 0)?;
    let Opts { bundle, mut out, device, precision, pooling, crf, bilstm, hash, c, max_train, commit } =
        options(&mut a)?;
    a.done()?;
    if crf || hash.is_some() {
        return Err(Error("split heads take neither --crf nor --hash-embed".into()));
    }

    let t = Instant::now();
    let (mut splits, record, languages) = match bank {
        Some(bank) => {
            let (splits, files, recorded) = split::read_tokens(bank, dir, unrecorded)?;
            let record = split::record(bank, &splits, files, recorded);
            (splits, record, None)
        }
        None => {
            let multi = turbo_trainer_recipes::ud_multi::read_tokens(bank_id, dir, unrecorded, sample)?;
            (
                multi.splits,
                multi.record,
                Some((multi.languages, multi.train_languages, multi.dev_languages, multi.test_languages)),
            )
        }
    };
    if max_train > 0 {
        splits.train.truncate(max_train);
    }
    let read = t.elapsed().as_secs_f64();
    let chosen = turbo_trainer_recipes::split_labels::choose(&splits.train);
    let labels = split::mine_labels(&splits.train, &chosen, min_freq);
    if labels.len() < 2 {
        return Err(Error(format!("{bank_id}: no multiword token's split comes up {min_freq} times in training")));
    }
    // Mined from the training split alone; the receipt's head carries it.
    let seen = (seen_k > 0).then(|| split::seen_lookup(&splits.train, &chosen, &labels, seen_k));
    let mut spec = split::spec(labels, &record, pooling, bilstm, seen)?;
    if let Some((names, _, _, _)) = &languages {
        spec.language = Some(turbo::manifest::LanguageInput { languages: names.clone(), dim: 16 });
        spec.validate("head").map_err(|e| Error(e.message))?;
    }
    let mut enc = WordEncoder::open(&bundle, &EncoderOptions { device, precision, ..EncoderOptions::default() })
        .map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;
    let hidden = enc.hidden();

    let t = Instant::now();
    let train_s = split::labelled(&splits.train, &chosen, &spec.labels);
    let dev_s = split::labelled(&splits.dev, &chosen, &spec.labels);
    let train_languages = languages.as_ref().map(|(_, rows, _, _)| &rows[..train_s.len()]);
    let dev_languages = languages.as_ref().map(|(_, _, rows, _)| &rows[..]);
    let (train_ex, _) = turbo_trainer_recipes::run::examples_with_languages(
        &mut enc,
        &train_s,
        &spec.labels,
        pooling,
        train_languages,
    )?;
    let (dev_ex, dev_left_out) =
        turbo_trainer_recipes::run::examples_with_languages(&mut enc, &dev_s, &spec.labels, pooling, dev_languages)?;
    let features_s = t.elapsed().as_secs_f64();
    let feature_words = train_ex.iter().chain(&dev_ex).map(|e| e.labels.len() as u64).sum();

    // Each better dev score publishes a checkpoint, as `train` does.
    let t = Instant::now();
    let spec_json = serde_json::to_vec_pretty(&spec).expect("a head spec serializes");
    let dir = out.path.clone();
    let mut saved = false;
    let mut save = |m: &turbo_trainer::Tagger<f32>, epoch: usize, dev: f64| -> Result<()> {
        if !saved {
            replace(&dir, "head.json", &spec_json)?;
            saved = true;
        }
        let at = [(CHECKPOINT_EPOCH, (epoch + 1).to_string()), (CHECKPOINT_DEV, dev.to_string())];
        replace(&dir, "head.safetensors", &safetensors::write_with(&named(&spec, hidden, m.tensors()), &at))?;
        Ok(())
    };
    let fitted = turbo_trainer::fit_with(&spec, hidden, &train_ex, &dev_ex, &c, &mut |m, e, d| {
        save(m, e, d).map_err(|x| turbo::status::invalid(x.0))
    });
    out.keep = out.keep || saved_any(&dir);
    let (model, report) = fitted.map_err(|e| Error(e.message))?;
    drop((train_ex, dev_ex));
    let trained = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let splitter = model.to_splitter().map_err(|e| Error(e.message))?;
    let test_f = split::features(&mut enc, &splits.test, pooling)?;
    let test_languages = languages.as_ref().map(|(_, _, _, rows)| &rows[..]);
    let predicted = split::predict_with_languages(&splitter, &splits.test, &test_f, test_languages)?;
    let base = split::baseline(&splits.train, &splits.test);
    let test = split::sentences(&splits.test);
    let train_s = split::sentences(&splits.train);
    let seen = Seen::of(&train_s);
    let by_language = match &languages {
        Some((names, train_rows, _, test_rows)) => turbo_trainer_recipes::eval::language_evals(
            &train_s,
            &test,
            &predicted,
            &train_rows[..train_s.len()],
            test_rows,
            names,
            turbo_trainer_recipes::eval::LanguageEvalOptions {
                task: Task::Tags,
                baseline: |train_rows: &[usize], test_rows: &[usize]| {
                    let train_part: Vec<split::TokenSentence> =
                        train_rows.iter().map(|&i| splits.train[i].clone()).collect();
                    let test_part: Vec<split::TokenSentence> =
                        test_rows.iter().map(|&i| splits.test[i].clone()).collect();
                    split::baseline(&train_part, &test_part)
                },
                resamples: RESAMPLES,
                seed: c.seed,
                attachment: None,
            },
        )?,
        None => Vec::new(),
    };
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(evaluate(Task::Tags, &test, &predicted, &seen, None)),
        baseline: evaluate(Task::Tags, &test, &base, &seen, None),
        dictionary: None,
        open_span: None,
        comparison: Some(bootstrap(Task::Tags, &test, &predicted, &base, RESAMPLES, c.seed)),
        uas: None,
        rules: None,
        against: None,
        by_language,
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
        corpus: record,
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
            sample: languages.map(|_| sample),
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
