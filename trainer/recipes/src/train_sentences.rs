//! `train-sentences`: the sentence boundary head (`sentences` on
//! `turboembed:sentences`, #1725), trained on MASC or a UD treebank's
//! multi-sentence documents and evaluated by exact sentence span F1
//! against both the newline baseline and the rule-based sentence
//! splitter in `analysis/sentdetect`.

use std::path::Path;
use std::time::Instant;

use turbo::manifest::{Adapter, Bilstm, Decode, HeadKind, HeadSpec, Scheme};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH, examples, named, replace};
use turbo_trainer_recipes::sentences::{
    SENTENCES_PER_DOCUMENT, bootstrap_spans, evaluate_spans, newline_baseline, predict, read, rule_baseline,
};
use turbo_trainer_recipes::{Error, Result, safetensors};

use super::{Args, Opts, RESAMPLES, encoder_record, finish, options, saved_any};

pub(super) fn train(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let Opts { bundle, mut out, device, precision, pooling, crf, bilstm, hash, c, max_train, commit } =
        options(&mut a)?;
    a.done()?;

    let t = Instant::now();
    let corpus = read(corpus_id, dir, unrecorded)?;
    let class = corpus.record.class()?;
    let mut train_s: Vec<_> = corpus.train.iter().flat_map(|d| d.train_passages(SENTENCES_PER_DOCUMENT)).collect();
    if max_train > 0 && max_train < train_s.len() {
        train_s.truncate(max_train);
    }
    let dev_s: Vec<_> = corpus.dev.iter().map(|d| d.to_sentence()).collect();
    let test_s: Vec<_> = corpus.test.iter().map(|d| d.to_sentence()).collect();
    let read_s = t.elapsed().as_secs_f64();

    let spec = HeadSpec {
        name: "sentences".into(),
        layer: "turboembed:sentences".into(),
        kind: HeadKind::SentenceBoundary,
        labels: vec!["inside".into(), "last".into()],
        scheme: Scheme::Tags,
        word_pooling: pooling,
        adapter: bilstm.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode: if crf { Decode::Crf } else { Decode::Argmax },
        biaffine: None,
        pos_input: None,
        hash_embed: hash,
        layer_mix: None,
        span: None,
        relation: None,
        coref: None,
        license_class: Some(class),
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.sentences".into(),
    };

    let mut enc = WordEncoder::open(&bundle, &EncoderOptions { device, precision, ..EncoderOptions::default() })
        .map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;
    let hidden = enc.hidden();

    let t = Instant::now();
    let (train_ex, _) = examples(&mut enc, &train_s, &spec.labels, pooling)?;
    let (dev_ex, dev_left_out) = examples(&mut enc, &dev_s, &spec.labels, pooling)?;
    let features_s = t.elapsed().as_secs_f64();
    let feature_words = train_ex.iter().chain(&dev_ex).map(|e| e.labels.len() as u64).sum();

    let t = Instant::now();
    let spec_json = serde_json::to_vec_pretty(&spec).expect("a head spec serializes");
    let out_dir = out.path.clone();
    let mut saved = false;
    let mut save = |m: &turbo_trainer::Tagger<f32>, epoch: usize, dev: f64| -> Result<()> {
        if !saved {
            replace(&out_dir, "head.json", &spec_json)?;
            saved = true;
        }
        let at = [(CHECKPOINT_EPOCH, epoch.to_string()), (CHECKPOINT_DEV, dev.to_string())];
        replace(&out_dir, "head.safetensors", &safetensors::write_with(&named(&spec, hidden, m.tensors()), &at))?;
        Ok(())
    };
    let fitted = turbo_trainer::fit_with(&spec, hidden, &train_ex, &dev_ex, &c, &mut |m, e, d| {
        save(m, e, d).map_err(|x| turbo::status::invalid(x.0))
    });
    out.keep = out.keep || saved_any(&out_dir);
    let (model, report) = fitted.map_err(|e| Error(e.message))?;
    drop((train_ex, dev_ex));
    let trained = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let head = model.to_boundary().map_err(|e| Error(e.message))?;
    let (test_ex, _) = examples(&mut enc, &test_s, &spec.labels, pooling)?;
    let test_f: Vec<Vec<f32>> = test_ex.into_iter().map(|e| e.features).collect();
    let predicted = predict(&head, &corpus.test, &test_f)?;
    let nl = newline_baseline(&corpus.test);
    let rules = rule_baseline(corpus.language, &corpus.test);
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(evaluate_spans(&corpus.test, &predicted)),
        baseline: evaluate_spans(&corpus.test, &nl),
        dictionary: None,
        open_span: None,
        comparison: Some(bootstrap_spans(&corpus.test, &predicted, &nl, RESAMPLES, c.seed)),
        uas: None,
        rules: Some(BaselineEval {
            name: format!("sentdetect-rules-{}", corpus.language),
            baseline: evaluate_spans(&corpus.test, &rules),
            comparison: bootstrap_spans(&corpus.test, &predicted, &rules, RESAMPLES, c.seed),
        }),
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
        timing_s: Timing { read: read_s, features: features_s, train: trained, eval: eval_s, feature_words },
        outputs,
    };
    finish(out, receipt)
}
