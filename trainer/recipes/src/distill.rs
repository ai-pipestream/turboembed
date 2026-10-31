//! `turbo-train-head distill-head` (#1520): distils a teacher head into a
//! student head (either equal-width on `--bundle` or cross-width on a smaller
//! `--student-bundle` encoder) using temperature-scaled KL + hard
//! cross-entropy, verifies the student accuracy margin gate and end-to-end
//! encoder + head throughput comparison, and writes `head.safetensors`,
//! `head.json`, and `receipt.json`.
//!
//! Two runs of one command at one commit on one device produce identical
//! student weights (`student_sha256`), training losses (`train_loss`,
//! `final_loss`), and accuracy scores (`val_scores`, `teacher_val_accuracy`,
//! `student_val_accuracy`, `teacher_test_accuracy`, `student_test_accuracy`,
//! `accuracy_drop`, `within_margin`); `host`, `unix_time`, and `throughput`
//! record the execution environment and wall-clock benchmark on `throughput.device`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use turbo::head::tagger::{Work as CoreWork, tensors};
use turbo::manifest::HeadSpec;
use turbo::safetensors::{Dtype, File};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer::{
    DistillOptions, Example, StudentGate, Tagger, ThroughputComparison, distill_tagger, fit, make_distill_examples,
    score_tagger,
};

use crate::cli::{Args, HeadArgs, OutDir, commit, encoder_args, head_args};
use crate::corpus::{CorpusRecord, FileRecord, Sentence};
use crate::receipt::{host, unix_time};
use crate::run::{CHECKPOINT_DEV, CHECKPOINT_EPOCH, HeadOptions, PREDICT_CHUNK, examples, named, replace, spec, write};
use crate::{Error, Result, read_corpus, safetensors};

pub const SCHEMA: &str = "turboembed.distill-head.v1";

/// Default number of timed passes over the test split for end-to-end throughput benchmarking.
const BENCHMARK_PASSES: usize = 8;

/// End-to-end throughput receipt block comparing teacher and student inference on `device`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThroughputRecord {
    pub device: String,
    pub backend: String,
    pub tokens: usize,
    pub sentences: usize,
    pub teacher_seconds: f64,
    pub student_seconds: f64,
    pub teacher_tokens_per_sec: f64,
    pub student_tokens_per_sec: f64,
    pub teacher_sentences_per_sec: f64,
    pub student_sentences_per_sec: f64,
    pub speedup_ratio: f64,
}

impl ThroughputRecord {
    pub fn from_comparison(t: ThroughputComparison, device: impl Into<String>, backend: impl Into<String>) -> Self {
        Self {
            device: device.into(),
            backend: backend.into(),
            tokens: t.tokens,
            sentences: t.sentences,
            teacher_seconds: t.teacher_seconds,
            student_seconds: t.student_seconds,
            teacher_tokens_per_sec: t.teacher_tokens_per_sec,
            student_tokens_per_sec: t.student_tokens_per_sec,
            teacher_sentences_per_sec: t.teacher_sentences_per_sec,
            student_sentences_per_sec: t.student_sentences_per_sec,
            speedup_ratio: t.speedup_ratio,
        }
    }
}

/// Distillation receipt written to `receipt.json`.
///
/// Training outputs (`student_sha256`, `train_loss`, `val_scores`, accuracy metrics)
/// are deterministic for a fixed seed; `host`, `unix_time`, and `throughput`
/// record runtime metadata and wall-clock timings on `throughput.device`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistillReceipt {
    pub schema: String,
    pub command: Vec<String>,
    pub commit: String,
    pub host: String,
    pub unix_time: u64,
    pub corpus: CorpusRecord,
    pub teacher_head: HeadSpec,
    pub student_head: HeadSpec,
    pub teacher_hidden: usize,
    pub student_hidden: usize,
    pub teacher_sha256: String,
    pub student_sha256: String,
    pub temperature: f32,
    pub alpha: f32,
    pub seed: u64,
    pub epochs: u32,
    pub batch_size: usize,
    pub lr: f64,
    pub weight_decay: f64,
    pub epochs_run: u32,
    pub best_epoch: usize,
    pub final_loss: f64,
    pub train_loss: Vec<f64>,
    pub val_scores: Vec<f64>,
    pub teacher_val_accuracy: f64,
    pub student_val_accuracy: f64,
    pub teacher_test_accuracy: f64,
    pub student_test_accuracy: f64,
    pub accuracy_drop: f64,
    pub max_accuracy_drop: f64,
    pub within_margin: bool,
    pub throughput: ThroughputRecord,
    pub outputs: Vec<FileRecord>,
}

/// Loads a trained [`Tagger<f32>`] and its `head.safetensors` SHA-256 digest
/// from `dir` (`head.json` + `head.safetensors`).
pub fn load_head(dir: &Path, hidden: usize) -> Result<(HeadSpec, Tagger<f32>, String)> {
    let spec_path = dir.join("head.json");
    let spec_bytes = std::fs::read(&spec_path).map_err(|e| Error(format!("{}: {e}", spec_path.display())))?;
    let head_spec: HeadSpec =
        serde_json::from_slice(&spec_bytes).map_err(|e| Error(format!("{}: {e}", spec_path.display())))?;
    let st_path = dir.join("head.safetensors");
    let st_bytes = std::fs::read(&st_path).map_err(|e| Error(format!("{}: {e}", st_path.display())))?;
    let sha256 = turbo::bundle::sha256_hex(&st_bytes);
    let file = File::parse("head.safetensors", &st_bytes).map_err(|e| Error(e.message))?;
    let wanted = tensors(&head_spec, hidden);
    let mut params = Vec::with_capacity(wanted.len());
    for (suffix, shape) in &wanted {
        let key = format!("{}.{suffix}", head_spec.tensor_prefix);
        let t = file.get(&key, Dtype::F32, shape.len()).map_err(|e| Error(e.message))?;
        if t.shape != *shape {
            return Err(Error(format!("{key}: shape {:?} does not match expected {shape:?}", t.shape)));
        }
        params.push(t.f32s());
    }
    let tagger = Tagger::<f32>::with_params(&head_spec, hidden, params).map_err(|e| Error(e.message))?;
    Ok((head_spec, tagger, sha256))
}

fn time_end_to_end_passes(
    enc: &mut WordEncoder,
    model: &Tagger<f32>,
    sentences: &[Sentence],
    passes: usize,
) -> Result<f64> {
    let core = model.to_token_tagger().map_err(|e| Error(e.message))?;
    let pooling = core.spec().word_pooling;
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), CoreWork::default());
    let mut run_once = || -> Result<()> {
        for chunk in sentences.chunks(PREDICT_CHUNK) {
            let words: Vec<Vec<&str>> = chunk.iter().map(|s| s.words.iter().map(String::as_str).collect()).collect();
            for (f, forms) in enc.features(&words, pooling).map_err(|e| Error(e.message))?.iter().zip(&words) {
                let ids: Vec<u32> = (0..forms.len() as u32).collect();
                core.scores_forms(f, &ids, Some(forms), &mut scores, &mut work).map_err(|e| Error(e.message))?;
                core.decode(&scores, forms.len(), &mut tags).map_err(|e| Error(e.message))?;
                std::hint::black_box(&tags);
            }
        }
        Ok(())
    };
    run_once()?;
    let start = Instant::now();
    for _ in 0..passes.max(1) {
        run_once()?;
    }
    Ok(start.elapsed().as_secs_f64().max(1e-9))
}

/// Benchmarks full end-to-end teacher vs. student inference (`WordEncoder::features` +
/// `TokenTagger::scores_forms` + `TokenTagger::decode`) over `sentences` on the
/// encoders' execution device after one untimed warmup pass.
pub fn measure_end_to_end_throughput(
    teacher_enc: &mut WordEncoder,
    teacher: &Tagger<f32>,
    student_enc: &mut WordEncoder,
    student: &Tagger<f32>,
    sentences: &[Sentence],
    passes: usize,
) -> Result<ThroughputComparison> {
    if sentences.is_empty() {
        return Err(Error("distill: non-empty evaluation sentences are required for throughput benchmarking".into()));
    }
    let passes = passes.max(1);
    let total_sentences = sentences.len() * passes;
    let total_tokens: usize = sentences.iter().map(|s| s.words.len()).sum::<usize>() * passes;
    let t_sec = time_end_to_end_passes(teacher_enc, teacher, sentences, passes)?;
    let s_sec = time_end_to_end_passes(student_enc, student, sentences, passes)?;
    let (t_tps, s_tps) = (total_tokens as f64 / t_sec, total_tokens as f64 / s_sec);
    Ok(ThroughputComparison {
        tokens: total_tokens,
        sentences: total_sentences,
        teacher_seconds: t_sec,
        student_seconds: s_sec,
        teacher_tokens_per_sec: t_tps,
        student_tokens_per_sec: s_tps,
        teacher_sentences_per_sec: total_sentences as f64 / t_sec,
        student_sentences_per_sec: total_sentences as f64 / s_sec,
        speedup_ratio: s_tps / t_tps.max(1e-12),
    })
}

/// Runs teacher-to-student distillation on prepared splits, checks the student
/// gate, and writes `head.safetensors`, `head.json`, and `receipt.json` to `out_dir`.
#[allow(clippy::too_many_arguments)]
pub fn distill_head_splits<F>(
    teacher: &Tagger<f32>,
    teacher_sha256: String,
    student_spec: &HeadSpec,
    student_hidden: usize,
    teacher_splits: (&[Example], &[Example], &[Example]),
    student_splits: (&[Example], &[Example], &[Example]),
    options: &DistillOptions,
    min_speedup_ratio: f64,
    allow_gate_failure: bool,
    bench: F,
    out_dir: &Path,
    command: Vec<String>,
    commit: String,
    corpus: CorpusRecord,
) -> Result<(Tagger<f32>, DistillReceipt)>
where
    F: FnOnce(&Tagger<f32>) -> Result<(ThroughputComparison, String, String)>,
{
    let (t_train, t_val, t_test) = teacher_splits;
    let (s_train, s_val, s_test) = student_splits;
    let d_train = make_distill_examples(teacher, t_train, s_train).map_err(|e| Error(e.message))?;
    let d_val = make_distill_examples(teacher, t_val, s_val).map_err(|e| Error(e.message))?;
    let d_test = make_distill_examples(teacher, t_test, s_test).map_err(|e| Error(e.message))?;

    let (student, mut report) = distill_tagger(student_spec, student_hidden, &d_train, &d_val, &d_test, options)
        .map_err(|e| Error(e.message))?;
    let eval_t_val = if t_val.is_empty() { t_train } else { t_val };
    let eval_t_test = if t_test.is_empty() { eval_t_val } else { t_test };
    report.teacher_val_accuracy = score_tagger(teacher, eval_t_val).map_err(|e| Error(e.message))?;
    report.teacher_test_accuracy = score_tagger(teacher, eval_t_test).map_err(|e| Error(e.message))?;
    report.accuracy_drop = report.teacher_test_accuracy - report.student_test_accuracy;
    report.within_margin = report.accuracy_drop <= options.max_accuracy_drop;

    let (throughput, device, backend) = bench(&student)?;
    let gate = StudentGate::evaluate(&report, throughput, options.max_accuracy_drop, min_speedup_ratio);
    if !allow_gate_failure {
        gate.verify().map_err(|e| Error(e.message))?;
    }

    let best = report.best_epoch;
    let best_val = report.val_scores.get(best).copied().unwrap_or(report.student_val_accuracy);
    let at = [(CHECKPOINT_EPOCH, best.to_string()), (CHECKPOINT_DEV, best_val.to_string())];
    let weights = safetensors::write_with(&named(student_spec, student_hidden, student.tensors()), &at);
    let spec_json = serde_json::to_vec_pretty(student_spec).expect("a head spec serializes");
    let st_rec = replace(out_dir, "head.safetensors", &weights)?;
    let student_sha256 = st_rec.sha256.clone();
    let json_rec = replace(out_dir, "head.json", &spec_json)?;
    let outputs = vec![st_rec, json_rec];

    let receipt = DistillReceipt {
        schema: SCHEMA.into(),
        command,
        commit,
        host: host(),
        unix_time: unix_time(),
        corpus,
        teacher_head: teacher.spec().clone(),
        student_head: student_spec.clone(),
        teacher_hidden: teacher.hidden(),
        student_hidden,
        teacher_sha256,
        student_sha256,
        temperature: options.temperature,
        alpha: options.alpha,
        seed: options.seed,
        epochs: options.epochs,
        batch_size: options.batch_size,
        lr: options.lr,
        weight_decay: options.weight_decay,
        epochs_run: report.epochs_run,
        best_epoch: report.best_epoch,
        final_loss: report.final_loss,
        train_loss: report.train_loss,
        val_scores: report.val_scores,
        teacher_val_accuracy: report.teacher_val_accuracy,
        student_val_accuracy: report.student_val_accuracy,
        teacher_test_accuracy: report.teacher_test_accuracy,
        student_test_accuracy: report.student_test_accuracy,
        accuracy_drop: report.accuracy_drop,
        max_accuracy_drop: options.max_accuracy_drop,
        within_margin: report.within_margin,
        throughput: ThroughputRecord::from_comparison(throughput, device, backend),
        outputs,
    };
    let text = serde_json::to_string_pretty(&receipt).expect("the receipt serializes");
    write(out_dir, "receipt.json", text.as_bytes())?;
    Ok((student, receipt))
}

/// CLI entry point for `turbo-train-head distill-head`.
pub fn command(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let bundle = PathBuf::from(a.required("--bundle")?);
    let student_bundle = a.value("--student-bundle")?.map(PathBuf::from);
    let teacher_dir = a.value("--teacher")?.map(PathBuf::from);
    let mut out = OutDir::claim(&a.required("--out")?)?;
    let (device, precision, pooling) = encoder_args(&mut a)?;
    let defaults = DistillOptions::default();
    let temperature: f32 = a.parsed("--temperature", defaults.temperature)?;
    let alpha: f32 = a.parsed("--alpha", defaults.alpha)?;
    let max_accuracy_drop: f64 = a.parsed("--max-accuracy-drop", defaults.max_accuracy_drop)?;
    let min_speedup: f64 = a.parsed("--min-speedup", 0.0)?;
    let allow_gate_failure = a.flag("--allow-gate-failure");
    let HeadArgs { crf, bilstm, hash, config: c } = head_args(&mut a)?;
    let max_train: usize = a.parsed("--max-train", 0)?;
    let commit = commit(a.value("--commit")?);
    a.done()?;

    let options = DistillOptions {
        epochs: c.epochs,
        batch_size: c.batch,
        lr: c.lr,
        weight_decay: c.weight_decay,
        clip: c.clip,
        warmup: c.warmup,
        seed: c.seed,
        temperature,
        alpha,
        max_accuracy_drop,
    };
    options.validate().map_err(|e| Error(e.message))?;

    let corpus = read_corpus(corpus_id, dir, unrecorded)?;
    let student_spec = spec(&corpus, &HeadOptions { pooling, crf: false, bilstm, hash })?;
    let enc_opts = EncoderOptions { device, precision, ..EncoderOptions::default() };
    let mut teacher_enc = WordEncoder::open(&bundle, &enc_opts).map_err(|e| Error(e.message))?;
    let teacher_hidden = teacher_enc.hidden();

    let train_set = match max_train {
        0 => &corpus.splits.train[..],
        n => &corpus.splits.train[..n.min(corpus.splits.train.len())],
    };
    let (t_train, _) = examples(&mut teacher_enc, train_set, &student_spec.labels, pooling)?;
    let (t_dev, _) = examples(&mut teacher_enc, &corpus.splits.dev, &student_spec.labels, pooling)?;
    let (t_test, _) = examples(&mut teacher_enc, &corpus.splits.test, &student_spec.labels, pooling)?;

    let (teacher, teacher_sha256) = match teacher_dir {
        Some(ref td) => {
            let (_, tagger, sha) = load_head(td, teacher_hidden)?;
            (tagger, sha)
        }
        None => {
            let teacher_spec = spec(&corpus, &HeadOptions { pooling, crf, bilstm, hash })?;
            let (tagger, _) = fit(&teacher_spec, teacher_hidden, &t_train, &t_dev, &c).map_err(|e| Error(e.message))?;
            let bytes = safetensors::write(&named(&teacher_spec, teacher_hidden, tagger.tensors()));
            let sha = turbo::bundle::sha256_hex(&bytes);
            (tagger, sha)
        }
    };

    let mut student_enc_opt = match student_bundle {
        Some(ref sb) if *sb != bundle => Some(WordEncoder::open(sb, &enc_opts).map_err(|e| Error(e.message))?),
        _ => None,
    };
    let student_hidden = student_enc_opt.as_ref().map_or(teacher_hidden, WordEncoder::hidden);
    let owned_student_splits = if let Some(ref mut s_enc) = student_enc_opt {
        let (s_tr, _) = examples(s_enc, train_set, &student_spec.labels, pooling)?;
        let (s_dv, _) = examples(s_enc, &corpus.splits.dev, &student_spec.labels, pooling)?;
        let (s_te, _) = examples(s_enc, &corpus.splits.test, &student_spec.labels, pooling)?;
        Some((s_tr, s_dv, s_te))
    } else {
        None
    };
    let (s_train, s_dev, s_test) = match owned_student_splits {
        Some((ref tr, ref dv, ref te)) => (&tr[..], &dv[..], &te[..]),
        None => (&t_train[..], &t_dev[..], &t_test[..]),
    };

    let eval_sentences = if corpus.splits.test.is_empty() { &corpus.splits.dev } else { &corpus.splits.test };
    let (exec_dev, exec_back) = {
        let x = student_enc_opt.as_ref().unwrap_or(&teacher_enc).execution();
        (x.device.clone(), x.backend.clone())
    };
    let (_, receipt) = distill_head_splits(
        &teacher,
        teacher_sha256,
        &student_spec,
        student_hidden,
        (&t_train, &t_dev, &t_test),
        (s_train, s_dev, s_test),
        &options,
        min_speedup,
        allow_gate_failure,
        |student| {
            let cmp = match student_enc_opt {
                Some(ref mut s_enc) => measure_end_to_end_throughput(
                    &mut teacher_enc,
                    &teacher,
                    s_enc,
                    student,
                    eval_sentences,
                    BENCHMARK_PASSES,
                )?,
                None => {
                    let t_sec = time_end_to_end_passes(&mut teacher_enc, &teacher, eval_sentences, BENCHMARK_PASSES)?;
                    let s_sec = time_end_to_end_passes(&mut teacher_enc, student, eval_sentences, BENCHMARK_PASSES)?;
                    let sentences = eval_sentences.len() * BENCHMARK_PASSES;
                    let tokens: usize = eval_sentences.iter().map(|s| s.words.len()).sum::<usize>() * BENCHMARK_PASSES;
                    let (t_tps, s_tps) = (tokens as f64 / t_sec, tokens as f64 / s_sec);
                    ThroughputComparison {
                        tokens,
                        sentences,
                        teacher_seconds: t_sec,
                        student_seconds: s_sec,
                        teacher_tokens_per_sec: t_tps,
                        student_tokens_per_sec: s_tps,
                        teacher_sentences_per_sec: sentences as f64 / t_sec,
                        student_sentences_per_sec: sentences as f64 / s_sec,
                        speedup_ratio: s_tps / t_tps.max(1e-12),
                    }
                }
            };
            Ok((cmp, exec_dev, exec_back))
        },
        &out.path,
        argv.to_vec(),
        commit,
        corpus.record,
    )?;
    out.keep = true;
    println!(
        "{}: teacher {:.4} -> student {:.4} (drop {:.4}, max {:.4}, speedup {:.2}x on {}); {}",
        receipt.corpus.id,
        receipt.teacher_test_accuracy,
        receipt.student_test_accuracy,
        receipt.accuracy_drop,
        receipt.max_accuracy_drop,
        receipt.throughput.speedup_ratio,
        receipt.throughput.device,
        out.path.join("receipt.json").display()
    );
    Ok(())
}
