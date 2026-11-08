//! `train-lemma` and `train-parse`: an edit-tree lemmatizer or a biaffine
//! parser on a UD treebank's syntactic words, written and scored as
//! `train` writes and scores a tagger.

use std::path::Path;
use std::time::Instant;

use turbo::manifest::Biaffine;
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer::{fit_lemmatizer_with, fit_parser_with, mine_seen_lookup, mine_trees};
use turbo_trainer_recipes::corpus::Task;
use turbo_trainer_recipes::eval::{Seen, bootstrap, evaluate};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH, replace};
use turbo_trainer_recipes::ud::{TREEBANKS, read_treebank};
use turbo_trainer_recipes::ud_heads::*;
use turbo_trainer_recipes::{Error, Result, safetensors};

use super::{Args, Opts, RESAMPLES, encoder_record, finish, options, saved_any};

/// spaCy's default: a tree must come up this often in training.
const MIN_TREE_FREQ: usize = 3;
/// Arc and relation projection widths (Dozat and Manning use 500 and
/// 100 over a wider BiLSTM).
const ARC: u32 = 256;
const REL: u32 = 64;

pub(super) fn train(
    argv: &[String],
    head: UdHead,
    bank: &str,
    dir: &Path,
    unrecorded: bool,
    mut a: Args,
) -> Result<()> {
    if a.rest.iter().any(|x| x == "--cache") {
        let task = match head {
            UdHead::Lemma => turbo_trainer_recipes::fit_cache::CacheTask::Lemma,
            UdHead::Deps => turbo_trainer_recipes::fit_cache::CacheTask::Parse,
        };
        return turbo_trainer_recipes::fit_cache::command_with_task(argv, bank, dir, unrecorded, Some(task), a);
    }
    let tune = a.flag("--tune-encoder");
    let lora = a.parsed("--lora", 0)?;
    let bf16 = a.flag("--bf16");
    if tune {
        if head != UdHead::Deps {
            return Err(Error("--tune-encoder is a train-parse option".into()));
        }
        return super::train_tuned::train(argv, bank, dir, unrecorded, (lora > 0).then_some(lora), bf16, a);
    }
    if lora > 0 || bf16 {
        return Err(Error("--lora and --bf16 require --tune-encoder".into()));
    }
    let bank_id = bank;
    let bank = TREEBANKS.iter().find(|b| b.id == bank_id);
    if bank.is_none() && !bank_id.contains(',') {
        return Err(Error(format!("unknown treebank {bank:?}: one of {}", TREEBANKS.map(|b| b.id).join(", "))));
    }
    let sample = if bank.is_none() { a.parsed("--sample", 1.0)? } else { 1.0 };
    let min_freq: usize = a.parsed("--min-tree-freq", MIN_TREE_FREQ)?;
    let seen_lookup: Option<usize> = match a.value("--seen-lookup")? {
        None => None,
        Some(v) => Some(
            v.parse::<usize>()
                .ok()
                .filter(|&k| k > 0)
                .ok_or_else(|| Error(format!("--seen-lookup {v:?}: a positive count")))?,
        ),
    };
    let biaffine = Biaffine { arc: a.parsed("--arc", ARC)?, rel: a.parsed("--rel", REL)? };
    let Opts { bundle, mut out, device, precision, pooling, crf, bilstm, hash, c, max_train, commit } =
        options(&mut a)?;
    a.done()?;
    if crf || hash.is_some() {
        return Err(Error(format!("{} heads take neither --crf nor --hash-embed", head.suffix())));
    }
    if head != UdHead::Lemma && seen_lookup.is_some() {
        return Err(Error("--seen-lookup is a lemma head option".into()));
    }

    let t = Instant::now();
    let (mut splits, record, languages) = match bank {
        Some(bank) => {
            let (splits, files, recorded) = read_treebank(bank, dir, unrecorded)?;
            let record = record(bank, head, &splits, files, recorded);
            (splits, record, None)
        }
        None => {
            let multi = turbo_trainer_recipes::ud_multi::read_ud(bank_id, head, dir, unrecorded, sample)?;
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
    let pairs =
        || splits.train.iter().flat_map(|s| s.words.iter().zip(&s.lemmas).map(|(w, l)| (w.as_str(), l.as_str())));
    let labels = match head {
        UdHead::Lemma => mine_trees(pairs(), min_freq),
        UdHead::Deps => relations(&splits.train),
    };
    let lookup = seen_lookup.map(|k| mine_seen_lookup(pairs(), k));
    let mut spec = spec_with_lookup(head, labels, lookup, &record, &UdOptions { pooling, bilstm, biaffine })?;
    if let Some((names, _, _, _)) = &languages {
        spec.language = Some(turbo::manifest::LanguageInput { languages: names.clone(), dim: 16 });
        spec.validate("head").map_err(|e| Error(e.message))?;
    }
    let mut enc = WordEncoder::open(&bundle, &EncoderOptions { device, precision, ..EncoderOptions::default() })
        .map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;
    let hidden = enc.hidden();

    let t = Instant::now();
    let train_f = features(&mut enc, &splits.train, pooling)?;
    let dev_f = features(&mut enc, &splits.dev, pooling)?;
    let features_s = t.elapsed().as_secs_f64();
    let feature_words = splits.train.iter().chain(&splits.dev).map(|s| s.words.len() as u64).sum();

    // Each better dev score publishes a checkpoint, as `train` does: the
    // spec once, then the head with the epoch and dev score it is from,
    // each replaced by one rename; from the first one on, the directory
    // stays if the run stops.
    let t = Instant::now();
    let spec_json = serde_json::to_vec_pretty(&spec).expect("a head spec serializes");
    let dir = out.path.clone();
    let mut saved = false;
    let mut save = |tensors: Vec<(String, Vec<f32>)>, epoch: usize, dev: f64| -> Result<()> {
        if !saved {
            replace(&dir, "head.json", &spec_json)?;
            saved = true;
        }
        let at = [(CHECKPOINT_EPOCH, (epoch + 1).to_string()), (CHECKPOINT_DEV, dev.to_string())];
        replace(&dir, "head.safetensors", &safetensors::write_with(&named(head, &spec, hidden, tensors), &at))?;
        Ok(())
    };
    let err = |e: turbo::status::Error| Error(e.message);
    let fitted = match head {
        UdHead::Lemma => {
            let train_languages = languages.as_ref().map(|(_, rows, _, _)| &rows[..splits.train.len()]);
            let dev_languages = languages.as_ref().map(|(_, _, rows, _)| &rows[..]);
            let tr = lemma_examples_with_languages(&splits.train, train_f, train_languages);
            let dv = lemma_examples_with_languages(&splits.dev, dev_f, dev_languages);
            fit_lemmatizer_with(&spec, hidden, tr, dv, &c, &mut |m, e, d| {
                save(m.tensors(), e, d).map_err(|x| turbo::status::invalid(x.0))
            })
            .and_then(|(m, r)| Ok((m.tensors(), Model::Lemma(Box::new(m.to_core()?)), r, 0)))
        }
        UdHead::Deps => {
            let train_languages = languages.as_ref().map(|(_, rows, _, _)| &rows[..splits.train.len()]);
            let dev_languages = languages.as_ref().map(|(_, _, rows, _)| &rows[..]);
            let (tr, _) = parse_examples_with_languages(&splits.train, train_f, &spec.labels, train_languages);
            let (dv, left) = parse_examples_with_languages(&splits.dev, dev_f, &spec.labels, dev_languages);
            fit_parser_with(&spec, hidden, &tr, &dv, &c, &mut |m, e, d| {
                save(m.tensors(), e, d).map_err(|x| turbo::status::invalid(x.0))
            })
            .and_then(|(m, r)| Ok((m.tensors(), Model::Parser(Box::new(m.to_core()?)), r, left)))
        }
    };
    out.keep = out.keep || saved_any(&dir);
    let (tensors, model, report, dev_left_out) = fitted.map_err(err)?;
    let trained = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let test_f = features(&mut enc, &splits.test, pooling)?;
    let (predicted, base) = match &model {
        Model::Lemma(m) => {
            let test_languages = languages.as_ref().map(|(_, _, _, rows)| &rows[..]);
            (
                predict_lemmas_with_languages(m, &splits.test, &test_f, test_languages)?,
                lemma_baseline(&splits.train, &splits.test),
            )
        }
        Model::Parser(m) => {
            let test_languages = languages.as_ref().map(|(_, _, _, rows)| &rows[..]);
            (
                predict_parses_with_languages(m, &splits.test, &test_f, test_languages)?,
                deps_baseline(&splits.train, &splits.test),
            )
        }
    };
    let (train_s, test) = (head.sentences(&splits.train), head.sentences(&splits.test));
    let seen = Seen::of(&train_s);
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(evaluate(Task::Tags, &test, &predicted, &seen, None)),
        baseline: evaluate(Task::Tags, &test, &base, &seen, None),
        dictionary: None,
        open_span: None,
        comparison: Some(bootstrap(Task::Tags, &test, &predicted, &base, RESAMPLES, c.seed)),
        uas: (head == UdHead::Deps).then(|| Uas { model: uas(&test, &predicted), baseline: uas(&test, &base) }),
        rules: None,
        against: None,
    };
    let eval_s = t.elapsed().as_secs_f64();

    let best = report.best_epoch;
    let at = [(CHECKPOINT_EPOCH, (best + 1).to_string()), (CHECKPOINT_DEV, report.dev[best].to_string())];
    let weights = safetensors::write_with(&named(head, &spec, hidden, tensors), &at);
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

/// The trained head, as the core runs it.
enum Model {
    Lemma(Box<turbo::head::EditTreeLemmatizer>),
    Parser(Box<turbo::head::BiaffineParser>),
}
