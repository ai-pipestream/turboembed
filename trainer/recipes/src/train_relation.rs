//! `train-relation`: a relation classifier head on Re-DocRED's documents,
//! written and scored as `train` writes and scores a tagger, by exact
//! relation triples against gold on the test split. And `baseline
//! --dependency-path`: the dependency-path rule row (#1825, the row the
//! #1734 receipt lists as missing), the documents' own sentences parsed by
//! our dependency head (`--parser`, a `train-parse` run directory, over
//! `--bundle`), the rule of `redocred_path` mined from the training split
//! and applied to the test split, the receipt's `baseline` the rule's
//! score, its `encoder` and `head` naming what produced the parses, and
//! `outputs` holding the rule's predictions, digested.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo::head::BiaffineParser;
use turbo::head::relation::tensors;
use turbo::manifest::HeadSpec;
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer::{Relations, fit_relations_with};
use turbo_trainer_recipes::cli::{OutDir, commit, encoder_args};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::redocred::{self, RelationCorpus};
use turbo_trainer_recipes::redocred_eval::*;
use turbo_trainer_recipes::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH, replace};
use turbo_trainer_recipes::{Error, Result, redocred_path, safetensors};

use super::{Args, Opts, RESAMPLES, encoder_record, finish, options, saved_any};

/// The learning rate a relation head trains at unless `--lr` says
/// otherwise: on Re-DocRED over a frozen ModernBERT-base, the trainer's
/// default (2e-3) has the loss climb again from the second epoch, and
/// 3e-4 falls steadily to a dev micro F1 of 0.446 at epoch 7.
pub(super) const LR: f64 = 3e-4;

/// The labels, most gold relations first, the per-label table prints.
const TABLE: usize = 10;

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
    let decode = a.value("--decode")?.map_or(Ok(d.decode), |v| decode_named(&v))?;
    let against = a.value("--against")?.map(PathBuf::from);
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
    let spec = spec(&corpus, &RelationOptions { pooling, mlp, type_dim, bilstm, decode })?;
    let mut enc = WordEncoder::open(&bundle, &EncoderOptions { device, precision, ..EncoderOptions::default() })
        .map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;
    let hidden = enc.hidden();
    // The head to score against, read before training so a bad one fails
    // first.
    let prior = against.as_deref().map(|dir| load_head(dir, &spec, &encoder)).transpose()?;

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
    let against = match prior {
        Some(PriorHead { head: prior, name, files }) => {
            let theirs = predict(&prior, &test_ex)?;
            let comparison = bootstrap(test, &predicted, &theirs, RESAMPLES, c.seed);
            Some(BaselineEval { name, baseline: evaluate(test, &theirs), comparison, files })
        }
        None => None,
    };
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(evaluate(test, &predicted)),
        baseline: evaluate(test, &base),
        dictionary: None,
        open_span: None,
        comparison: Some(bootstrap(test, &predicted, &base, RESAMPLES, c.seed)),
        uas: None,
        rules: None,
        against,
    };
    if let Some(m) = &eval.model {
        println!("{}", label_table(m, TABLE));
    }
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
            sample: None,
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

pub(super) fn dependency_path(
    argv: &[String],
    corpus_id: &str,
    dir: &Path,
    unrecorded: bool,
    mut a: Args,
) -> Result<()> {
    if corpus_id != redocred::ID {
        return Err(Error(format!("{corpus_id}: --dependency-path is re-docred's baseline")));
    }
    let parser_dir = PathBuf::from(a.required("--parser")?);
    let bundle = PathBuf::from(a.required("--bundle")?);
    let (device, precision, _) = encoder_args(&mut a)?;
    let out = OutDir::claim(&a.required("--out")?)?;
    let commit = commit(a.value("--commit")?);
    a.done()?;

    let t = Instant::now();
    let c = redocred::read(dir, unrecorded)?;
    let read = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let spec_bytes =
        std::fs::read(parser_dir.join("head.json")).map_err(|e| Error(format!("{}: {e}", parser_dir.display())))?;
    let spec: HeadSpec = serde_json::from_slice(&spec_bytes).map_err(|e| Error(format!("head.json: {e}")))?;
    if spec.kind != turbo::manifest::HeadKind::BiaffineParser {
        return Err(Error(format!("{}: {:?} is not a parser head", parser_dir.display(), spec.kind)));
    }
    let weights = std::fs::read(parser_dir.join("head.safetensors"))
        .map_err(|e| Error(format!("{}: {e}", parser_dir.display())))?;
    let tensors = turbo::safetensors::File::parse("head.safetensors", &weights).map_err(|e| Error(e.message))?;
    let mut enc = WordEncoder::open(&bundle, &EncoderOptions { device, precision, ..EncoderOptions::default() })
        .map_err(|e| Error(e.message))?;
    let encoder = encoder_record(&bundle, &enc)?;
    let get = |s: &str| tensors.tensor(&format!("{}.{s}", spec.tensor_prefix)).map(|t| t.f32s());
    let parser = BiaffineParser::from_tensors(&spec, enc.hidden(), &get).map_err(|e| Error(e.message))?;

    let train_parsed = parse_documents(&mut enc, &parser, &c.splits.train)?;
    let test_parsed = parse_documents(&mut enc, &parser, &c.splits.test)?;
    let features_s = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let table = redocred_path::mine(&c.splits.train, &train_parsed);
    let predicted = redocred_path::predict(&c.splits.test, &test_parsed, &table);
    let baseline = evaluate(&c.splits.test, &predicted);
    let eval_s = t.elapsed().as_secs_f64();

    let predictions = serde_json::to_vec_pretty(
        &c.splits
            .test
            .iter()
            .zip(&predicted)
            .map(|(d, rs)| {
                serde_json::json!({
                    "title": d.title,
                    "relations": rs.iter().map(|(h, t, r)| serde_json::json!([h, t, r])).collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>(),
    )
    .expect("predictions serialize");
    let outputs = vec![replace(&out.path, "predictions.json", &predictions)?];

    let receipt = Receipt {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        unix_time: unix_time(),
        corpus: c.record.clone(),
        encoder: Some(encoder),
        head: Some(spec),
        peer: None,
        config: None,
        training: None,
        eval: EvalRecord::baseline_only(baseline),
        timing_s: Timing { read, features: features_s, eval: eval_s, ..Timing::default() },
        outputs,
    };
    println!(
        "re-docred: dependency-path rule, {} paths mined, test micro F1 {:.4}; {}/receipt.json",
        table.len(),
        receipt.eval.baseline.score(),
        out.path.display(),
    );
    finish(out, receipt)
}

/// Each document's sentences parsed by `parser` over the encoder's
/// features, the head's own pooling.
fn parse_documents(
    enc: &mut WordEncoder,
    parser: &BiaffineParser,
    docs: &[redocred::Document],
) -> Result<Vec<redocred_path::Parsed>> {
    let pooling = parser.spec().word_pooling;
    let (mut arcs, mut heads, mut scores, mut rels, mut w) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Default::default());
    let mut out = Vec::with_capacity(docs.len());
    for d in docs {
        let mut at = 0;
        let mut sentences: Vec<Vec<&str>> = Vec::with_capacity(d.sentences.len());
        for &n in &d.sentences {
            sentences.push(d.words[at..at + n].iter().map(String::as_str).collect());
            at += n;
        }
        let features = enc.features(&sentences, pooling).map_err(|e| Error(e.message))?;
        let mut parsed = Vec::with_capacity(sentences.len());
        for (f, words) in features.iter().zip(&sentences) {
            let n = words.len();
            let ids: Vec<u32> = (0..n as u32).collect();
            parser.arcs(f, &ids, &mut arcs, &mut w).map_err(|e| Error(e.message))?;
            parser.decode(&arcs, n, &mut heads).map_err(|e| Error(e.message))?;
            parser.relations(&heads, &mut scores, &w).map_err(|e| Error(e.message))?;
            parser.relation_labels(&scores, n, &mut rels).map_err(|e| Error(e.message))?;
            parsed.push(heads.iter().zip(&rels).map(|(&h, &r)| (h, parser.labels()[r as usize].clone())).collect());
        }
        out.push(parsed);
    }
    Ok(out)
}
