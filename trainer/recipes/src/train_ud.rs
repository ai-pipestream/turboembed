//! `train-lemma` and `train-parse`: an edit-tree lemmatizer or a biaffine
//! parser on a UD treebank's syntactic words, written and scored as
//! `train` writes and scores a tagger.

use std::path::Path;
use std::time::Instant;

use turbo::manifest::Biaffine;
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer::{fit_lemmatizer_with, fit_parser_with, mine_trees};
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
    let bank = *TREEBANKS
        .iter()
        .find(|b| b.id == bank)
        .ok_or_else(|| Error(format!("unknown treebank {bank:?}: one of {}", TREEBANKS.map(|b| b.id).join(", "))))?;
    let min_freq: usize = a.parsed("--min-tree-freq", MIN_TREE_FREQ)?;
    let biaffine = Biaffine { arc: a.parsed("--arc", ARC)?, rel: a.parsed("--rel", REL)? };
    let Opts { bundle, mut out, device, precision, pooling, crf, bilstm, hash, c, max_train, commit } =
        options(&mut a)?;
    a.done()?;
    if crf || hash.is_some() {
        return Err(Error(format!("{} heads take neither --crf nor --hash-embed", head.suffix())));
    }

    let t = Instant::now();
    let (mut splits, files, recorded) = read_treebank(&bank, dir, unrecorded)?;
    if max_train > 0 {
        splits.train.truncate(max_train);
    }
    let record = record(&bank, head, &splits, files, recorded);
    let read = t.elapsed().as_secs_f64();
    let labels = match head {
        UdHead::Lemma => mine_trees(
            splits.train.iter().flat_map(|s| s.words.iter().zip(&s.lemmas).map(|(w, l)| (w.as_str(), l.as_str()))),
            min_freq,
        ),
        UdHead::Deps => relations(&splits.train),
    };
    let spec = spec(head, labels, &record, &UdOptions { pooling, bilstm, biaffine })?;
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
        let at = [(CHECKPOINT_EPOCH, epoch.to_string()), (CHECKPOINT_DEV, dev.to_string())];
        replace(&dir, "head.safetensors", &safetensors::write_with(&named(head, &spec, hidden, tensors), &at))?;
        Ok(())
    };
    let err = |e: turbo::status::Error| Error(e.message);
    let fitted = match head {
        UdHead::Lemma => {
            let (tr, dv) = (lemma_examples(&splits.train, train_f), lemma_examples(&splits.dev, dev_f));
            fit_lemmatizer_with(&spec, hidden, tr, dv, &c, &mut |m, e, d| {
                save(m.tensors(), e, d).map_err(|x| turbo::status::invalid(x.0))
            })
            .and_then(|(m, r)| Ok((m.tensors(), Model::Lemma(Box::new(m.to_core()?)), r, 0)))
        }
        UdHead::Deps => {
            let (tr, _) = parse_examples(&splits.train, train_f, &spec.labels);
            let (dv, left) = parse_examples(&splits.dev, dev_f, &spec.labels);
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
        Model::Lemma(m) => (predict_lemmas(m, &splits.test, &test_f)?, lemma_baseline(&splits.train, &splits.test)),
        Model::Parser(m) => (predict_parses(m, &splits.test, &test_f)?, deps_baseline(&splits.train, &splits.test)),
    };
    let (train_s, test) = (head.sentences(&splits.train), head.sentences(&splits.test));
    let seen = Seen::of(&train_s);
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(evaluate(Task::Tags, &test, &predicted, &seen, None)),
        baseline: evaluate(Task::Tags, &test, &base, &seen, None),
        comparison: Some(bootstrap(Task::Tags, &test, &predicted, &base, RESAMPLES, c.seed)),
        uas: (head == UdHead::Deps).then(|| Uas { model: uas(&test, &predicted), baseline: uas(&test, &base) }),
        rules: None,
    };
    let eval_s = t.elapsed().as_secs_f64();

    let best = report.best_epoch;
    let at = [(CHECKPOINT_EPOCH, best.to_string()), (CHECKPOINT_DEV, report.dev[best].to_string())];
    let weights = safetensors::write_with(&named(head, &spec, hidden, tensors), &at);
    let outputs = vec![replace(&out.path, "head.safetensors", &weights)?, replace(&out.path, "head.json", &spec_json)?];
    let receipt = Receipt {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        host: host(),
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

/// The trained head, as the core runs it.
enum Model {
    Lemma(Box<turbo::head::EditTreeLemmatizer>),
    Parser(Box<turbo::head::BiaffineParser>),
}
