//! Fine-tune a converted GLiNER output MLP without Python or test labels.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo_gliner::ExtractionSchema;
use turbo_gliner::package::Manifest;
use turbo_gliner::train::{OutputTrainer, output_optimizer};
use turbo_trainer::adamw::Settings;
use turbo_trainer::rng::Rng;
use turbo_trainer_recipes::corpus::{FileRecord, Sentence};
use turbo_trainer_recipes::eval::{Majority, Seen, evaluate};
use turbo_trainer_recipes::open_span::{dictionary_baseline, filter_sentences};
use turbo_trainer_recipes::open_span_mix::{combine, evaluate_predictions, read_corpora, type_schema};
use turbo_trainer_recipes::peer::{Column, PeerRecord};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::run::replace;
use turbo_trainer_recipes::{Error, Result, gliner_eval};

use super::{Args, OutDir, commit, finish};

struct Predictions {
    all: Vec<Vec<String>>,
    held: Vec<Vec<String>>,
    seen: Vec<Vec<String>>,
}

pub(super) fn train(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let package = PathBuf::from(a.required("--package")?);
    let hold_out_arg = a.required("--hold-out")?;
    let balance = a.flag("--balance");
    let guidelines = a.value("--guidelines")?.map(read_guidelines).transpose()?;
    let max_train: usize = a.parsed("--max-train", 0)?;
    let head = super::head_args(&mut a)?;
    if head.crf || head.bilstm.is_some() || head.hash.is_some() {
        return Err(Error("train-gliner takes neither token-tagging adapters nor CRF".into()));
    }
    let c = head.config;
    let out = OutDir::claim(&a.required("--out")?)?;
    let commit = commit(a.value("--commit")?);
    a.done()?;
    if c.epochs == 0 || c.batch == 0 {
        return Err(Error("train-gliner needs positive epochs and batch size".into()));
    }

    let started = Instant::now();
    let mut corpora = read_corpora(corpus_id, dir, unrecorded)?;
    let (all_types, declared) = type_schema(&corpora)?;
    if let Some(guidelines) = &guidelines {
        let unknown: Vec<&str> = guidelines.keys().filter(|x| !all_types.contains(*x)).map(String::as_str).collect();
        if !unknown.is_empty() {
            return Err(Error(format!("--guidelines names labels outside the corpus: {unknown:?}")));
        }
    }
    let hold_out: Vec<String> = hold_out_arg.split(',').map(|x| x.trim().to_owned()).collect();
    if hold_out.iter().any(|x| x.is_empty() || !all_types.contains(x)) {
        return Err(Error(format!("--hold-out must name corpus entity types, got {hold_out:?}")));
    }
    let seen_types: Vec<String> = all_types.iter().filter(|x| !hold_out.contains(x)).cloned().collect();
    if seen_types.is_empty() {
        return Err(Error("--hold-out leaves no seen entity types".into()));
    }
    for (corpus, declared) in corpora.iter_mut().zip(&declared) {
        corpus.splits.train = filter_sentences(&corpus.splits.train, &seen_types);
        if max_train > 0 {
            corpus.splits.train.truncate(max_train);
        }
        let active = turbo_trainer_recipes::open_span_mix::corpus_active_types(declared, &seen_types);
        if active.len() > turbo_gliner::MAX_LABELS && !corpus.splits.train.is_empty() {
            return Err(Error("train-gliner currently requires at most 32 active labels per corpus".into()));
        }
    }
    if balance {
        let target = corpora.iter().map(|x| x.splits.train.len()).max().unwrap_or(0);
        for corpus in &mut corpora {
            if corpus.splits.train.is_empty() {
                return Err(Error("--balance needs nonempty filtered training splits".into()));
            }
            let original = std::mem::take(&mut corpus.splits.train);
            corpus.splits.train = (0..target).map(|i| original[i % original.len()].clone()).collect();
        }
    }
    let combined = combine(&corpora)?;
    let read = started.elapsed().as_secs_f64();

    let original_heads = turbo_gliner::package::read(&package.join("heads.safetensors"), 256 << 20).map_err(Error)?;
    let mut model = Manifest::load(&package).map_err(Error)?;
    let mut trainer = OutputTrainer::load(model.config(), &original_heads).map_err(Error)?;
    let total: usize = corpora.iter().map(|x| x.splits.train.len().div_ceil(c.batch)).sum();
    let steps = (total * c.epochs as usize) as u64;
    let settings = Settings {
        lr: c.lr,
        weight_decay: c.weight_decay,
        clip: c.clip,
        warmup: (steps as f64 * c.warmup) as u64,
        total: steps,
        ..Settings::default()
    };
    let mut optimizer = output_optimizer(settings, &trainer).map_err(Error)?;
    let threshold = 0.8;
    let (initial, initial_eval) =
        score(&model, &combined, &all_types, &seen_types, &hold_out, threshold, guidelines.as_ref())?;
    let mut best_score = initial_eval.model.score();
    let mut best_heads = original_heads.clone();
    let mut best_predictions = initial;
    let mut best_epoch = 0;
    let mut dev = vec![best_score];
    let mut train_loss = Vec::new();
    let mut epoch_seconds = Vec::new();
    let mut rng = Rng::new(c.seed);
    let mut stale = 0;
    eprintln!("epoch 0 dev {best_score:.4}");

    for epoch in 0..c.epochs {
        let at = Instant::now();
        let mut epoch_loss = 0.0;
        let mut completed = 0;
        for (corpus, declared) in corpora.iter().zip(&declared) {
            let active = turbo_trainer_recipes::open_span_mix::corpus_active_types(declared, &seen_types);
            if active.is_empty() {
                continue;
            }
            let schema = gliner_eval::schema(&active, threshold, guidelines.as_ref())?;
            let mut order: Vec<usize> = (0..corpus.splits.train.len()).collect();
            rng.shuffle(&mut order);
            for batch in order.chunks(c.batch) {
                let lanes = if c.threads == 0 { batch.len() } else { c.threads.min(batch.len()) }.max(1);
                let mut results = Vec::with_capacity(batch.len());
                for lane in batch.chunks(lanes) {
                    if lane.len() == 1 {
                        results.push(sentence_gradients(
                            &model,
                            &trainer,
                            &corpus.splits.train[lane[0]],
                            &schema,
                            &active,
                        )?);
                        continue;
                    }
                    let mut computed = std::thread::scope(|scope| {
                        let mut handles = Vec::new();
                        for &index in lane {
                            let model = &model;
                            let trainer = &trainer;
                            let schema = &schema;
                            let active = &active;
                            handles.push(scope.spawn(move || {
                                sentence_gradients(model, trainer, &corpus.splits.train[index], schema, active)
                            }));
                        }
                        let mut computed = Vec::with_capacity(handles.len());
                        for handle in handles {
                            computed
                                .push(handle.join().map_err(|_| Error("a GLiNER training worker panicked".into()))??);
                        }
                        Ok::<_, Error>(computed)
                    })?;
                    results.append(&mut computed);
                }
                let mut grads = trainer.zero_grads();
                for (loss, local) in results {
                    epoch_loss += loss;
                    for (sum, part) in grads.iter_mut().zip(local) {
                        for (a, b) in sum.iter_mut().zip(part) {
                            *a += b;
                        }
                    }
                }
                trainer.step(&mut optimizer, &mut grads).map_err(Error)?;
                model.apply_output(&trainer).map_err(Error)?;
                completed += 1;
                if completed % 100 == 0 {
                    eprintln!("epoch {} steps {completed}/{total}", epoch + 1);
                }
            }
        }
        train_loss.push(epoch_loss);
        let (predictions, evaluation) =
            score(&model, &combined, &all_types, &seen_types, &hold_out, threshold, guidelines.as_ref())?;
        let score = evaluation.model.score();
        dev.push(score);
        epoch_seconds.push(at.elapsed().as_secs_f64());
        eprintln!("epoch {} dev {score:.4}", epoch + 1);
        let progressed = score > best_score + c.tolerance;
        if score > best_score {
            best_score = score;
            best_heads = trainer.write_heads(&original_heads).map_err(Error)?;
            best_predictions = predictions;
            best_epoch = (epoch + 1) as usize;
        }
        if progressed {
            stale = 0;
        } else {
            stale += 1;
            if c.patience > 0 && stale >= c.patience {
                break;
            }
        }
    }

    let tuned = out.path.join("model");
    Manifest::write_tuned(&package, &tuned, &best_heads).map_err(Error)?;
    let manifest_bytes = std::fs::read(tuned.join("gliner.json")).map_err(|e| Error(e.to_string()))?;
    let manifest_record = file_record("model/gliner.json", &manifest_bytes);
    let prediction_bytes =
        super::eval_gliner::prediction_text(&combined.splits.dev, &best_predictions.all)?.into_bytes();
    let prediction_record = replace(&out.path, "predictions.bio", &prediction_bytes)?;
    let identity = model.identity();
    let peer_name = format!("gliner-tuned:{}@{}#{}", identity.id, identity.revision, manifest_record.sha256);
    let evaluation = score_parts(&combined, &all_types, &seen_types, &hold_out, &best_predictions)?;
    let train_filtered = filter_sentences(&combined.splits.train, &seen_types);
    let majority = Majority::train(&train_filtered);
    let base: Vec<Vec<String>> = combined.splits.dev.iter().map(|s| majority.predict(&s.words)).collect();
    let dev_seen = filter_sentences(&combined.splits.dev, &seen_types);
    let seen = Seen::of(&train_filtered);
    let dictionary = dictionary_baseline(&dev_seen, &seen_types);
    let eval = EvalRecord {
        split: "dev".into(),
        model: Some(evaluation.model.clone()),
        baseline: evaluation.baseline.clone(),
        dictionary: Some(evaluation.dictionary.clone()),
        open_span: Some(OpenSpanEvalRecord {
            seen_types,
            held_out_types: hold_out,
            seen_model: evaluate(combined.record.task, &dev_seen, &best_predictions.seen, &seen, None),
            seen_baseline: evaluate(combined.record.task, &dev_seen, &base, &seen, None),
            seen_dictionary: evaluate(combined.record.task, &dev_seen, &dictionary, &seen, None),
            held_out_model: evaluation.held_out_model.clone(),
            held_out_baseline: evaluation.held_out_baseline.clone(),
            held_out_dictionary: evaluation.held_out_dictionary.clone(),
            dictionary_comparison: Some(evaluation.dictionary_comparison.clone()),
            held_out_comparison: evaluation.held_out_comparison.clone(),
            held_out_dictionary_comparison: evaluation.held_out_dictionary_comparison.clone(),
            datasets: Vec::new(),
        }),
        comparison: Some(evaluation.comparison.clone()),
        uas: None,
        rules: None,
        against: None,
        by_language: Vec::new(),
    };
    let receipt = Receipt {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        unix_time: unix_time(),
        corpus: combined.record.clone(),
        encoder: None,
        head: None,
        peer: Some(PeerRecord {
            name: peer_name,
            column: Column::Bio,
            predictions: prediction_record.clone(),
            map: BTreeMap::new(),
        }),
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
            dev_left_out: 0,
            sample: None,
        }),
        training: Some(TrainingRecord { train_loss, dev, best_epoch, epoch_seconds }),
        eval,
        timing_s: Timing { read, train: started.elapsed().as_secs_f64(), ..Timing::default() },
        outputs: vec![manifest_record, prediction_record],
    };
    finish(out, receipt)
}

fn read_guidelines(path: String) -> Result<BTreeMap<String, String>> {
    let bytes = std::fs::read(&path).map_err(|e| Error(format!("{path}: {e}")))?;
    serde_json::from_slice(&bytes).map_err(|e| Error(format!("{path}: {e}")))
}

#[allow(clippy::too_many_arguments)]
fn score(
    model: &turbo_gliner::Model,
    corpus: &turbo_trainer_recipes::corpus::Corpus,
    all: &[String],
    seen: &[String],
    held: &[String],
    threshold: f32,
    guidelines: Option<&BTreeMap<String, String>>,
) -> Result<(Predictions, turbo_trainer_recipes::receipt::OpenSpanDatasetEval)> {
    let predictions = std::thread::scope(|scope| {
        let held_task = scope.spawn(|| gliner_eval::predict(model, &corpus.splits.dev, held, threshold, guidelines));
        let seen_task = scope.spawn(|| gliner_eval::predict(model, &corpus.splits.dev, seen, threshold, guidelines));
        let all = gliner_eval::predict(model, &corpus.splits.dev, all, threshold, guidelines)?;
        let held = held_task.join().map_err(|_| Error("a GLiNER dev scoring worker panicked".into()))??;
        let seen = seen_task.join().map_err(|_| Error("a GLiNER dev scoring worker panicked".into()))??;
        Ok::<_, Error>(Predictions { all, held, seen })
    })?;
    let evaluation = score_parts(corpus, all, seen, held, &predictions)?;
    Ok((predictions, evaluation))
}

fn score_parts(
    corpus: &turbo_trainer_recipes::corpus::Corpus,
    all: &[String],
    seen: &[String],
    held: &[String],
    predictions: &Predictions,
) -> Result<turbo_trainer_recipes::receipt::OpenSpanDatasetEval> {
    evaluate_predictions(corpus, all, seen, held, &predictions.all, Some(&predictions.held), "dev", 0, 1)
}

fn sentence_gradients(
    model: &turbo_gliner::Model,
    trainer: &OutputTrainer,
    sentence: &Sentence,
    schema: &ExtractionSchema,
    labels: &[String],
) -> Result<(f64, Vec<Vec<f32>>)> {
    let words: Vec<&str> = sentence.words.iter().map(String::as_str).collect();
    let (input, candidates) = model.training_input(&words, schema).map_err(Error)?;
    let gold = gold_spans(sentence, labels)?;
    let mut grads = trainer.zero_grads();
    let loss = trainer.gradients(&input, &candidates, &gold, &mut grads).map_err(Error)?;
    Ok((loss, grads))
}

fn gold_spans(sentence: &Sentence, labels: &[String]) -> Result<Vec<(usize, usize, usize)>> {
    let mut out = Vec::new();
    let mut open: Option<(usize, usize)> = None;
    for (at, tag) in sentence.labels.iter().map(String::as_str).chain(["O"]).enumerate() {
        let kind =
            tag.strip_prefix("B-").or_else(|| tag.strip_prefix("I-")).and_then(|x| labels.iter().position(|y| y == x));
        if tag.starts_with("B-") || kind != open.map(|x| x.0) {
            if let Some((label, start)) = open.take() {
                out.push((start, at, label));
            }
            if tag.starts_with("B-") {
                open = kind.map(|x| (x, at));
            }
        }
    }
    Ok(out)
}

fn file_record(path: &str, bytes: &[u8]) -> FileRecord {
    FileRecord { path: path.into(), size: bytes.len() as u64, sha256: turbo::bundle::sha256_hex(bytes) }
}
