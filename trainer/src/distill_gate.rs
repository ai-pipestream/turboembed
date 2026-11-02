//! Student distillation training loop, multi-head runner, throughput benchmark,
//! and accuracy-margin gate (#1520).

use std::time::Instant;

use turbo::head::boundary::LAST;
use turbo::head::tagger::Work as CoreWork;
use turbo::manifest::{Decode, HeadKind, HeadSpec};
use turbo::status::{Result, invalid};

use crate::adamw::{AdamW, Settings};
use crate::distill::{DistillOptions, DistillWork, tagger_distill_loss};
use crate::fit::{Example, score, score_boundary, score_split};
use crate::rng::Rng;
use crate::tagger::{Tagger, Work};
use crate::tally::Tally;

/// One sequence for student distillation.
#[derive(Debug, Clone, PartialEq)]
pub struct DistillExample {
    pub features: Vec<f32>,
    pub teacher_scores: Vec<f32>,
    pub teacher_tags: Vec<u32>,
    pub labels: Vec<u32>,
    pub forms: Vec<String>,
}

impl DistillExample {
    pub fn forms(&self) -> Option<Vec<&str>> {
        (!self.forms.is_empty()).then(|| self.forms.iter().map(String::as_str).collect())
    }
    pub fn to_example(&self) -> Example {
        Example { features: self.features.clone(), labels: self.labels.clone(), forms: self.forms.clone() }
    }
}

/// Decodes one sequence's `scores` (`[n, labels]`) using `teacher`'s runtime
/// decoder (including Viterbi decoding for CRF teachers, valid-tree filtering
/// for `TokenSplit`, and final-token boundary enforcement for `SentenceBoundary`).
pub fn decode_teacher_example(teacher: &Tagger<f32>, scores: &[f32], n: usize, forms: &[String]) -> Result<Vec<u32>> {
    match teacher.spec().kind {
        HeadKind::SentenceBoundary => {
            let b = teacher.to_boundary()?;
            let mut ends = Vec::new();
            b.decode(scores, n, &mut ends)?;
            Ok(ends.into_iter().map(u32::from).collect())
        }
        HeadKind::TokenSplit => {
            let s = teacher.to_splitter()?;
            let form_refs: Vec<&str> = forms.iter().map(String::as_str).collect();
            let mut tags = Vec::new();
            s.decode(scores, &form_refs, &mut tags)?;
            Ok(tags)
        }
        _ => {
            let t = teacher.to_token_tagger()?;
            let mut tags = Vec::new();
            t.decode(scores, n, &mut tags)?;
            Ok(tags)
        }
    }
}

/// Builds [`DistillExample`] sequences by running `teacher` over `teacher_examples`.
pub fn make_distill_examples(
    teacher: &Tagger<f32>,
    teacher_examples: &[Example],
    student_examples: &[Example],
) -> Result<Vec<DistillExample>> {
    if teacher_examples.len() != student_examples.len() {
        return Err(invalid("distill: teacher_examples and student_examples must have the same length"));
    }
    let mut work = Work::default();
    teacher_examples
        .iter()
        .zip(student_examples)
        .map(|(te, se)| {
            if te.labels != se.labels || te.forms != se.forms {
                return Err(invalid("distill: teacher and student examples must share gold labels and forms"));
            }
            teacher.forward_forms(&te.features, te.forms().as_deref(), &mut work)?;
            let teacher_scores = work.scores().to_vec();
            let teacher_tags = decode_teacher_example(teacher, &teacher_scores, se.labels.len(), &se.forms)?;
            Ok(DistillExample {
                features: se.features.clone(),
                teacher_scores,
                teacher_tags,
                labels: se.labels.clone(),
                forms: se.forms.clone(),
            })
        })
        .collect()
}

/// Summary of a distillation run and its accuracy gate check.
#[derive(Debug, Clone, PartialEq)]
pub struct DistillReport {
    pub teacher_val_accuracy: f64,
    pub student_val_accuracy: f64,
    pub teacher_test_accuracy: f64,
    pub student_test_accuracy: f64,
    pub accuracy_drop: f64,
    pub within_margin: bool,
    pub epochs_run: u32,
    pub best_epoch: usize,
    pub final_loss: f64,
    pub train_loss: Vec<f64>,
    pub val_scores: Vec<f64>,
}

/// Inference throughput comparison between teacher and student heads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThroughputComparison {
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

/// Per-student release gate combining the accuracy margin and throughput check.
#[derive(Debug, Clone, PartialEq)]
pub struct StudentGate {
    pub max_accuracy_drop: f64,
    pub min_speedup_ratio: f64,
    pub teacher_accuracy: f64,
    pub student_accuracy: f64,
    pub accuracy_drop: f64,
    pub within_margin: bool,
    pub throughput: ThroughputComparison,
    pub passed: bool,
}

impl StudentGate {
    pub fn evaluate(r: &DistillReport, throughput: ThroughputComparison, max_drop: f64, min_speedup: f64) -> Self {
        let within_margin = r.accuracy_drop <= max_drop;
        let mut gate = Self {
            max_accuracy_drop: max_drop,
            min_speedup_ratio: min_speedup,
            teacher_accuracy: r.teacher_test_accuracy,
            student_accuracy: r.student_test_accuracy,
            accuracy_drop: r.accuracy_drop,
            within_margin,
            throughput,
            passed: false,
        };
        gate.passed = gate.verify().is_ok();
        gate
    }

    /// Refuses undefined thresholds, accuracies and throughput measurements.
    /// This validation also applies when a caller allows a measured gate failure.
    pub fn validate(&self) -> Result<()> {
        let t = &self.throughput;
        let positive = [
            t.teacher_seconds,
            t.student_seconds,
            t.teacher_tokens_per_sec,
            t.student_tokens_per_sec,
            t.teacher_sentences_per_sec,
            t.student_sentences_per_sec,
            t.speedup_ratio,
        ];
        if !self.max_accuracy_drop.is_finite()
            || self.max_accuracy_drop < 0.0
            || !self.min_speedup_ratio.is_finite()
            || self.min_speedup_ratio < 0.0
            || !self.teacher_accuracy.is_finite()
            || !(0.0..=1.0).contains(&self.teacher_accuracy)
            || !self.student_accuracy.is_finite()
            || !(0.0..=1.0).contains(&self.student_accuracy)
            || !self.accuracy_drop.is_finite()
            || t.tokens == 0
            || t.sentences == 0
            || positive.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(invalid(
                "distill gate requires finite nonnegative thresholds, finite accuracies and positive throughput measurements",
            ));
        }
        Ok(())
    }

    pub fn verify(&self) -> Result<()> {
        self.validate()?;
        if !self.within_margin
            || self.accuracy_drop > self.max_accuracy_drop
            || self.throughput.speedup_ratio < self.min_speedup_ratio
        {
            return Err(invalid(format!(
                "distill gate failed: accuracy drop {:.4} (max {:.4}), speedup {:.3}x (min {:.3}x)",
                self.accuracy_drop, self.max_accuracy_drop, self.throughput.speedup_ratio, self.min_speedup_ratio
            )));
        }
        Ok(())
    }
}

fn check_distill_examples(examples: &[DistillExample], hidden: usize, labels: usize, forms: bool) -> Result<()> {
    for (i, e) in examples.iter().enumerate() {
        let n = e.labels.len();
        if hidden == 0
            || n.checked_mul(hidden) != Some(e.features.len())
            || n.checked_mul(labels) != Some(e.teacher_scores.len())
            || e.teacher_tags.len() != n
            || e.teacher_tags.iter().any(|&l| l as usize >= labels)
            || e.labels.iter().any(|&l| l as usize >= labels)
            || (forms && e.forms.len() != n)
        {
            return Err(invalid(format!(
                "distill: example {i} has mismatched features, teacher scores, teacher tags, or forms"
            )));
        }
    }
    Ok(())
}

/// Scores a [`Tagger<f32>`] on `examples` using its runtime decoder (including
/// Viterbi decoding when the head uses CRF transitions).
pub fn score_tagger(tagger: &Tagger<f32>, examples: &[Example]) -> Result<f64> {
    match tagger.spec().kind {
        HeadKind::SentenceBoundary => score_boundary(&tagger.to_boundary()?, examples),
        HeadKind::TokenSplit => score_split(&tagger.to_splitter()?, examples),
        _ => score(&tagger.to_token_tagger()?, examples),
    }
}

fn score_student(student: &Tagger<f32>, examples: &[DistillExample]) -> Result<f64> {
    let plain: Vec<Example> = examples.iter().map(DistillExample::to_example).collect();
    score_tagger(student, &plain)
}

fn score_teacher_tags(spec: &HeadSpec, examples: &[DistillExample]) -> Result<f64> {
    if spec.kind == HeadKind::SentenceBoundary {
        let (mut right, mut total) = (0usize, 0usize);
        for e in examples {
            let n = e.labels.len();
            right += e
                .teacher_tags
                .iter()
                .enumerate()
                .zip(&e.labels)
                .filter(|&((i, &pred), &gold)| (pred == LAST || i + 1 == n) == (gold == LAST))
                .count();
            total += n;
        }
        return Ok(if total == 0 { 0.0 } else { right as f64 / total as f64 });
    }
    if spec.kind == HeadKind::TokenSplit {
        let (mut right, mut total) = (0usize, 0usize);
        for e in examples {
            right += e.teacher_tags.iter().zip(&e.labels).filter(|(a, b)| a == b).count();
            total += e.labels.len();
        }
        return Ok(if total == 0 { 0.0 } else { right as f64 / total as f64 });
    }
    let mut tally = Tally::new(spec);
    for e in examples {
        tally.add(&e.teacher_tags, &e.labels)?;
    }
    Ok(tally.score())
}

/// Distils a student [`Tagger<f32>`] of hidden width `student_hidden` from teacher soft logits and gold labels.
pub fn distill_tagger(
    spec: &HeadSpec,
    student_hidden: usize,
    train: &[DistillExample],
    val: &[DistillExample],
    test: &[DistillExample],
    options: &DistillOptions,
) -> Result<(Tagger<f32>, DistillReport)> {
    options.validate()?;
    if spec.decode == Decode::Crf {
        return Err(invalid("distill: student head must not use Decode::Crf (only emission logits are distilled)"));
    }
    if train.is_empty() {
        return Err(invalid("distill: training set must not be empty"));
    }
    let (labels, forms) = (spec.labels.len(), spec.hash_embed.is_some() || spec.kind == HeadKind::TokenSplit);
    for s in [train, val, test] {
        check_distill_examples(s, student_hidden, labels, forms)?;
    }
    let mut rng = Rng::new(options.seed);
    let mut student = Tagger::<f32>::new(spec, student_hidden, &mut rng)?;
    let shapes: Vec<usize> = student.params().iter().map(Vec::len).collect();
    let total = train.len().div_ceil(options.batch_size) as u64 * options.epochs as u64;
    let s = Settings {
        lr: options.lr,
        weight_decay: options.weight_decay,
        clip: options.clip,
        warmup: (total as f64 * options.warmup) as u64,
        total,
        ..Settings::default()
    };
    let (mut opt, decay) = (AdamW::new(s, &shapes)?, student.decays());
    let mut order: Vec<usize> = (0..train.len()).collect();
    let (mut grads, mut work) = (student.zero_grads(), DistillWork::<f32>::default());
    let (temperature, alpha) = (f64::from(options.temperature), f64::from(options.alpha));
    let eval_val = if val.is_empty() { train } else { val };
    let eval_test = if test.is_empty() { eval_val } else { test };
    let (mut train_loss, mut val_scores) = (Vec::new(), Vec::new());
    let (mut best_val, mut best_epoch, mut best_params) = (f64::NEG_INFINITY, 0usize, None);

    for epoch in 0..options.epochs as usize {
        rng.shuffle(&mut order);
        let (mut epoch_loss, mut epoch_words) = (0.0, 0usize);
        for chunk in order.chunks(options.batch_size) {
            grads.iter_mut().for_each(|g| g.fill(0.0));
            let mut batch_words = 0usize;
            for &idx in chunk {
                let ex = &train[idx];
                let f = ex.forms();
                epoch_loss += tagger_distill_loss(
                    &student,
                    &ex.features,
                    f.as_deref(),
                    &ex.teacher_scores,
                    &ex.labels,
                    temperature,
                    alpha,
                    &mut grads,
                    &mut work,
                )?;
                batch_words += ex.labels.len();
            }
            epoch_words += batch_words;
            if batch_words > 0 {
                let inv = 1.0 / batch_words as f32;
                grads.iter_mut().flatten().for_each(|g| *g *= inv);
                student.step(&mut opt, &mut grads, &decay)?;
            }
        }
        train_loss.push(if epoch_words == 0 { 0.0 } else { epoch_loss / epoch_words as f64 });
        let v = score_student(&student, eval_val)?;
        val_scores.push(v);
        if v > best_val {
            (best_val, best_epoch, best_params) = (v, epoch, Some(student.params().to_vec()));
        }
    }
    if let Some(params) = best_params {
        student.set_params(params)?;
    }
    let (t_val, s_val) = (score_teacher_tags(spec, eval_val)?, score_student(&student, eval_val)?);
    let (t_test, s_test) = (score_teacher_tags(spec, eval_test)?, score_student(&student, eval_test)?);
    let accuracy_drop = t_test - s_test;
    let final_loss = train_loss.last().copied().unwrap_or(0.0);
    Ok((
        student,
        DistillReport {
            teacher_val_accuracy: t_val,
            student_val_accuracy: s_val,
            teacher_test_accuracy: t_test,
            student_test_accuracy: s_test,
            accuracy_drop,
            within_margin: accuracy_drop <= options.max_accuracy_drop,
            epochs_run: options.epochs,
            best_epoch,
            final_loss,
            train_loss,
            val_scores,
        },
    ))
}

/// One task in a multi-head distillation run.
#[derive(Debug, Clone)]
pub struct DistillTask {
    pub spec: HeadSpec,
    pub train: Vec<DistillExample>,
    pub val: Vec<DistillExample>,
    pub test: Vec<DistillExample>,
}

/// Distils multiple student heads across tasks and returns each head with its report.
pub fn distill_multi_head(
    student_hidden: usize,
    tasks: &[DistillTask],
    options: &DistillOptions,
) -> Result<Vec<(Tagger<f32>, DistillReport)>> {
    if tasks.is_empty() {
        return Err(invalid("distill: tasks must not be empty"));
    }
    tasks.iter().map(|t| distill_tagger(&t.spec, student_hidden, &t.train, &t.val, &t.test, options)).collect()
}

/// Measures head-level teacher vs. student inference throughput over `iterations`
/// passes on pre-encoded [`Example`] feature sequences (after one untimed warmup pass).
/// For full end-to-end encoder + head benchmarking on CPU/GPU bundles, see
/// `turbo-trainer-recipes::distill::measure_end_to_end_throughput`.
pub fn measure_throughput(
    teacher: &Tagger<f32>,
    teacher_examples: &[Example],
    student: &Tagger<f32>,
    student_examples: &[Example],
    iterations: usize,
) -> Result<ThroughputComparison> {
    if teacher_examples.is_empty() || teacher_examples.len() != student_examples.len() {
        return Err(invalid("distill: non-empty matching teacher and student examples are required"));
    }
    let passes = iterations.max(1);
    let sentences = teacher_examples.len() * passes;
    let tokens: usize = teacher_examples.iter().map(|e| e.labels.len()).sum::<usize>() * passes;
    let run_bench = |model: &Tagger<f32>, examples: &[Example]| -> Result<f64> {
        let (core, hidden) = (model.to_token_tagger()?, model.hidden());
        let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), CoreWork::default());
        let words_buf: Vec<Vec<u32>> =
            examples.iter().map(|e| (0..(e.features.len() / hidden) as u32).collect()).collect();
        let forms_buf: Vec<Option<Vec<&str>>> = examples.iter().map(Example::forms).collect();
        for ((e, words), forms) in examples.iter().zip(&words_buf).zip(&forms_buf) {
            core.scores_forms(&e.features, words, forms.as_deref(), &mut scores, &mut work)?;
            core.decode(&scores, words.len(), &mut tags)?;
            std::hint::black_box(&tags);
        }
        let start = Instant::now();
        for _ in 0..passes {
            for ((e, words), forms) in examples.iter().zip(&words_buf).zip(&forms_buf) {
                core.scores_forms(&e.features, words, forms.as_deref(), &mut scores, &mut work)?;
                core.decode(&scores, words.len(), &mut tags)?;
                std::hint::black_box(&tags);
            }
        }
        Ok(start.elapsed().as_secs_f64().max(1e-9))
    };
    let (t_sec, s_sec) = (run_bench(teacher, teacher_examples)?, run_bench(student, student_examples)?);
    let (t_tps, s_tps) = (tokens as f64 / t_sec, tokens as f64 / s_sec);
    Ok(ThroughputComparison {
        tokens,
        sentences,
        teacher_seconds: t_sec,
        student_seconds: s_sec,
        teacher_tokens_per_sec: t_tps,
        student_tokens_per_sec: s_tps,
        teacher_sentences_per_sec: sentences as f64 / t_sec,
        student_sentences_per_sec: sentences as f64 / s_sec,
        speedup_ratio: s_tps / t_tps.max(1e-12),
    })
}
