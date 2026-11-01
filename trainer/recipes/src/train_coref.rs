//! `train-coref`: an antecedent-ranking coreference head (`HEAD_COREF_RANKER`)
//! trained on LitBank's 100 literary works (#1527), written and scored with
//! CoNLL coreference metrics (MUC, B-cubed, CEAF-e, and CoNLL F1) beside the
//! deterministic singleton baseline.

use std::path::Path;
use std::time::Instant;

use turbo::manifest::{CorefShape, SpanShape};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer::fit_coref_with;
use turbo_trainer_recipes::corpus::Counts;
use turbo_trainer_recipes::litbank::{self, CorefDoc};
use turbo_trainer_recipes::litbank_eval::{
    CorefOptions, DEFAULT_ANTECEDENTS, DEFAULT_MAX_WORDS, DEFAULT_MENTION_DIM, DEFAULT_WIDTH_DIM, baseline,
    bootstrap_coref, evaluate_coref, examples, features, named, predict, spec,
};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH, replace};
use turbo_trainer_recipes::{Error, Result, safetensors};

use super::{Args, Opts, RESAMPLES, encoder_record, finish, options, saved_any};

pub(super) fn train(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    if corpus_id != litbank::ID {
        return Err(Error(format!("unknown coref corpus {corpus_id:?}: expected {}", litbank::ID)));
    }
    let dim = parse_alias(&mut a, &["--mlp", "--dim", "--mention-dim"], DEFAULT_MENTION_DIM)?;
    let antecedents =
        parse_alias(&mut a, &["--max-antecedents", "--antecedents", "--rough-top-k"], DEFAULT_ANTECEDENTS)?;
    let max_words: u32 = a.parsed("--max-words", DEFAULT_MAX_WORDS)?;
    let width_dim: u32 = a.parsed("--width-dim", DEFAULT_WIDTH_DIM)?;
    let Opts { bundle, mut out, device, precision, pooling, crf, bilstm, hash, c, max_train, commit } =
        options(&mut a)?;
    a.done()?;
    if crf || hash.is_some() {
        return Err(Error("coref heads take neither --crf nor --hash-embed".into()));
    }

    let t = Instant::now();
    let mut splits = litbank::read_coref(dir, unrecorded)?;
    if max_train > 0 && max_train < splits.train.len() {
        splits.train.truncate(max_train);
        splits.record.train = doc_counts(&splits.train);
    }
    let read = t.elapsed().as_secs_f64();

    let head_opts = CorefOptions {
        pooling,
        bilstm,
        span: SpanShape { max_words, width_dim },
        coref: CorefShape { dim, antecedents },
    };
    let head = spec(&splits.record, &head_opts)?;
    let mut enc = WordEncoder::open(&bundle, &EncoderOptions { device, precision, ..EncoderOptions::default() })
        .map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;
    let hidden = enc.hidden();

    let t = Instant::now();
    let train_f = features(&mut enc, &splits.train, pooling)?;
    let dev_f = features(&mut enc, &splits.dev, pooling)?;
    let features_s = t.elapsed().as_secs_f64();
    let feature_words: u64 = splits.train.iter().chain(&splits.dev).map(|d| d.words.len() as u64).sum();
    let (train_ex, _) = examples(&splits.train, train_f, max_words);
    let (dev_ex, dev_left_out) = examples(&splits.dev, dev_f, max_words);

    let t = Instant::now();
    let spec_json = serde_json::to_vec_pretty(&head).expect("a head spec serializes");
    let out_dir = out.path.clone();
    let mut saved = false;
    let mut save = |tensors: Vec<(String, Vec<f32>)>, epoch: usize, dev: f64| -> Result<()> {
        if !saved {
            replace(&out_dir, "head.json", &spec_json)?;
            saved = true;
        }
        let at = [(CHECKPOINT_EPOCH, epoch.to_string()), (CHECKPOINT_DEV, dev.to_string())];
        replace(&out_dir, "head.safetensors", &safetensors::write_with(&named(&head, hidden, tensors), &at))?;
        Ok(())
    };
    let fitted = fit_coref_with(&head, hidden, &train_ex, &dev_ex, &c, &mut |m, e, d| {
        save(m.tensors(), e, d).map_err(|x| turbo::status::invalid(x.0))
    });
    out.keep = out.keep || saved_any(&out_dir);
    let (model, report) = fitted.map_err(|e| Error(e.message))?;
    let core_head = model.to_core().map_err(|e| Error(e.message))?;
    let tensors = model.tensors();
    drop((train_ex, dev_ex));
    let trained = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let test_f = features(&mut enc, &splits.test, pooling)?;
    let predicted = predict(&core_head, &splits.test, &test_f)?;
    let base = baseline(&splits.test);
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(evaluate_coref(&splits.test, &predicted)?),
        baseline: evaluate_coref(&splits.test, &base)?,
        comparison: Some(bootstrap_coref(&splits.test, &predicted, &base, RESAMPLES, c.seed)?),
        uas: None,
    };
    let eval_s = t.elapsed().as_secs_f64();

    let best = report.best_epoch;
    let at = [(CHECKPOINT_EPOCH, best.to_string()), (CHECKPOINT_DEV, report.dev[best].to_string())];
    let weights = safetensors::write_with(&named(&head, hidden, tensors), &at);
    let outputs = vec![replace(&out.path, "head.safetensors", &weights)?, replace(&out.path, "head.json", &spec_json)?];
    let receipt = Receipt {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        host: host(),
        unix_time: unix_time(),
        corpus: splits.record,
        encoder: Some(encoder),
        head: Some(head),
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

fn parse_alias(a: &mut Args, flags: &[&str], default: u32) -> Result<u32> {
    let mut found: Option<u32> = None;
    for &flag in flags {
        if let Some(v) = a.value(flag)? {
            let parsed: u32 = v.parse().map_err(|_| Error(format!("{flag} {v:?}: expected a positive integer")))?;
            if found.is_some() {
                return Err(Error(format!("conflicting flags in {flags:?}")));
            }
            found = Some(parsed);
        }
    }
    Ok(found.unwrap_or(default))
}

fn doc_counts(docs: &[CorefDoc]) -> Counts {
    Counts { sentences: docs.iter().map(|d| d.sentences.len()).sum(), words: docs.iter().map(|d| d.words.len()).sum() }
}
