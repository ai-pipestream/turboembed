//! `train-relation`: a relation classifier head on Re-DocRED's documents,
//! written and scored as `train` writes and scores a tagger, by exact
//! relation triples against gold on the test split.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use turbo::head::relation::tensors;
use turbo::manifest::HeadSpec;
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer::{Relations, fit_relations_with};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::redocred::{self, RelationCorpus};
use turbo_trainer_recipes::redocred_eval::*;
use turbo_trainer_recipes::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH, replace};
use turbo_trainer_recipes::{Error, Result, safetensors};

use super::{Args, Opts, RESAMPLES, encoder_record, finish, options, saved_any};

/// The learning rate a relation head trains at unless `--lr` says
/// otherwise: on Re-DocRED over a frozen ModernBERT-base, the trainer's
/// default (2e-3) has the loss climb again from the second epoch, and
/// 3e-4 falls steadily to a dev micro F1 of 0.446 at epoch 7.
pub(super) const LR: f64 = 3e-4;

/// Reads the relation corpus `id` from `dir`.
pub(super) fn read(id: &str, dir: &Path, unrecorded: bool) -> Result<RelationCorpus> {
    match id {
        redocred::ID => redocred::read(dir, unrecorded),
        _ => Err(Error(format!("unknown relation corpus {id:?}: {}", redocred::ID))),
    }
}

pub(super) fn train(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let d = RelationOptions::default();
    if !a.rest.iter().any(|x| x == "--lr") {
        a.rest.extend(["--lr".to_owned(), LR.to_string()]);
    }
    let mlp = a.parsed("--mlp", d.mlp)?;
    let type_dim = a.parsed("--type-dim", d.type_dim)?;
    let Opts { bundle, mut out, device, precision, pooling, crf, bilstm, hash, c, max_train, commit } =
        options(&mut a)?;
    a.done()?;
    if crf || hash.is_some() {
        return Err(Error("relation heads take neither --crf nor --hash-embed".into()));
    }

    let t = Instant::now();
    let mut corpus = read(corpus_id, dir, unrecorded)?;
    if max_train > 0 {
        corpus.splits.train.truncate(max_train);
    }
    let read_s = t.elapsed().as_secs_f64();
    let spec = spec(&corpus, &RelationOptions { pooling, mlp, type_dim, bilstm })?;
    let mut enc = WordEncoder::open(&bundle, &EncoderOptions { device, precision, ..EncoderOptions::default() })
        .map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;
    let hidden = enc.hidden();

    let t = Instant::now();
    let s = &corpus.splits;
    let (train_ex, _) = examples(&mut enc, &s.train, &spec.labels, pooling)?;
    let (dev_ex, dev_left_out) = examples(&mut enc, &s.dev, &spec.labels, pooling)?;
    let features_s = t.elapsed().as_secs_f64();
    let feature_words = s.train.iter().chain(&s.dev).map(|d| d.words.len() as u64).sum();

    // Each better dev score publishes a checkpoint, as `train` does.
    let t = Instant::now();
    let spec_json = serde_json::to_vec_pretty(&spec).expect("a head spec serializes");
    let dir = out.path.clone();
    let mut saved = false;
    let mut save = |m: &Relations<f32>, epoch: usize, dev: f64| -> Result<()> {
        if !saved {
            replace(&dir, "head.json", &spec_json)?;
            saved = true;
        }
        let at = [(CHECKPOINT_EPOCH, (epoch + 1).to_string()), (CHECKPOINT_DEV, dev.to_string())];
        replace(&dir, "head.safetensors", &safetensors::write_with(&named(&spec, hidden, m.tensors()), &at))?;
        Ok(())
    };
    let fitted = fit_relations_with(&spec, hidden, &train_ex, &dev_ex, &c, &mut |m, e, d| {
        save(m, e, d).map_err(|x| turbo::status::invalid(x.0))
    });
    out.keep = out.keep || saved_any(&dir);
    let (model, report) = fitted.map_err(|e| Error(e.message))?;
    drop((train_ex, dev_ex));
    let trained = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let head = model.to_core().map_err(|e| Error(e.message))?;
    let test = &corpus.splits.test;
    let (test_ex, _) = examples(&mut enc, test, &spec.labels, pooling)?;
    let predicted = predict(&head, &test_ex)?;
    let base = baseline(test);
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(evaluate(test, &predicted)),
        baseline: evaluate(test, &base),
        dictionary: None,
        open_span: None,
        comparison: Some(bootstrap(test, &predicted, &base, RESAMPLES, c.seed)),
        uas: None,
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
            best_epoch: best + 1,
            epoch_seconds: report.epoch_seconds,
        }),
        eval,
        timing_s: Timing { read: read_s, features: features_s, train: trained, eval: eval_s, feature_words },
        outputs,
    };
    finish(out, receipt)
}

/// The head's tensors under their bundle names, with their shapes.
fn named(spec: &HeadSpec, hidden: usize, values: Vec<(String, Vec<f32>)>) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let shapes: BTreeMap<String, Vec<u64>> = tensors(spec, hidden).into_iter().collect();
    values
        .into_iter()
        .map(|(suffix, v)| {
            let shape = shapes.get(&suffix).cloned().unwrap_or_else(|| vec![v.len() as u64]);
            (format!("{}.{suffix}", spec.tensor_prefix), shape, v)
        })
        .collect()
}
