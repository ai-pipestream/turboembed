//! Civil Comments incivility head training (with and without parent context),
//! fusion gate and severe toxicity rule tuning, and Toxic Spans evidence span
//! evaluation (`#888`, `#963`, `#890`).
//!
//! ```text
//! cargo run --release -p turbo-analysis-embed --example incivility_eval -- \
//!     --bundle <bundle dir> --device cuda --civil <dir> --toxic-spans <dir> \
//!     --cache <dir> --head-out <head.tivh> --context-head-out <ctx_head.tivh> \
//!     --out <file.json>
//! ```

#[path = "../incivility_bias/data.rs"]
mod data;
#[path = "../incivility_bias/embed.rs"]
mod embed;
mod source;
mod spans_eval;

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use serde_json::{Value, json};
use turbo_analysis_embed::{Device, EncoderOptions};
use turbo_analysis_layers::{IncivilityLabel, IncivilityScores};
use turbo_doccat::{
    INCIVILITY_DIMENSIONS, IncivilityAnnotator, IncivilityFitOptions, IncivilityMetrics, IncivilityStage,
    IncivilityVectorHead, IncivilityVectorScorer, evaluate_incivility_corpus,
    evaluate_incivility_corpus_with_thresholds,
};
use turbo_zoo::{Sha256, Sha512};

use data::Unit;
mod prepared;
use prepared::{Prepared, prepare, score_split};

const THRESHOLDS: [f64; 10] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50];
const FUSION_GATES: [f64; 7] = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 1.0];

struct Args {
    bundle: PathBuf,
    device: Device,
    civil: PathBuf,
    toxic_spans: PathBuf,
    cache: PathBuf,
    head_out: Option<PathBuf>,
    context_head_out: Option<PathBuf>,
    out: PathBuf,
    fit: IncivilityFitOptions,
}

fn args() -> Result<Args, String> {
    let mut a = std::env::args().skip(1);
    let (mut bundle, mut device, mut civil, mut toxic_spans, mut cache, mut head_out, mut context_head_out, mut out) =
        (None, Device::Cpu, None, None, None, None, None, None);
    let mut fit = IncivilityFitOptions { epochs: 500, learning_rate: 16.0, l2: 1e-4 };
    while let Some(flag) = a.next() {
        let mut value = || a.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--bundle" => bundle = Some(PathBuf::from(value()?)),
            "--device" => {
                device = match value()?.as_str() {
                    "cpu" => Device::Cpu,
                    d => Device::Backend(d.into()),
                }
            }
            "--civil" => civil = Some(PathBuf::from(value()?)),
            "--toxic-spans" => toxic_spans = Some(PathBuf::from(value()?)),
            "--cache" => cache = Some(PathBuf::from(value()?)),
            "--head-out" => head_out = Some(PathBuf::from(value()?)),
            "--context-head-out" => context_head_out = Some(PathBuf::from(value()?)),
            "--out" => out = Some(PathBuf::from(value()?)),
            "--epochs" => fit.epochs = value()?.parse().map_err(|e| format!("--epochs: {e}"))?,
            "--learning-rate" => fit.learning_rate = value()?.parse().map_err(|e| format!("--learning-rate: {e}"))?,
            "--l2" => fit.l2 = value()?.parse().map_err(|e| format!("--l2: {e}"))?,
            f => return Err(format!("unknown flag {f}")),
        }
    }
    let need = |v: Option<PathBuf>, f: &str| v.ok_or_else(|| format!("{f} is required"));
    Ok(Args {
        bundle: need(bundle, "--bundle")?,
        device,
        civil: need(civil, "--civil")?,
        toxic_spans: need(toxic_spans, "--toxic-spans")?,
        cache: need(cache, "--cache")?,
        head_out,
        context_head_out,
        out: need(out, "--out")?,
        fit,
    })
}

fn peak_rss_kb() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

mod tuning;
use tuning::tune_thresholds;

fn corpus_json(m: &IncivilityMetrics) -> Value {
    let labels: Vec<Value> = m
        .per_label
        .iter()
        .map(|(l, x)| {
            json!({
                "label": l.name(), "tp": x.tp, "fp": x.fp, "fn": x.fn_count, "tn": x.tn,
                "precision": x.precision, "recall": x.recall, "f1": x.f1,
            })
        })
        .collect();
    json!({
        "per_label": labels,
        "macro": { "precision": m.macro_precision, "recall": m.macro_recall, "f1": m.macro_f1 },
        "micro": { "precision": m.micro_precision, "recall": m.micro_recall, "f1": m.micro_f1 },
        "positive_predictions": m.total_positive_predictions,
        "positive_predictions_with_evidence": m.positive_with_evidence,
    })
}

fn eval_stage_on_test(
    annotator: &IncivilityAnnotator,
    val_p: &Prepared<'_>,
    val_v: &[f32],
    test_p: &Prepared<'_>,
    test_v: &[f32],
    parent_lookup: Option<&HashMap<String, &[f32]>>,
    dim: usize,
) -> Result<(Value, f64, [f64; INCIVILITY_DIMENSIONS]), String> {
    let (uni_t, val_uni_micro, per_label_t, val_pl_macro, val_pl_micro) =
        tune_thresholds(annotator, val_p, val_v, parent_lookup, dim)?;
    let per_label_map: serde_json::Map<String, Value> = IncivilityLabel::ALL
        .into_iter()
        .enumerate()
        .map(|(i, l)| (l.name().to_owned(), json!(per_label_t[i])))
        .collect();
    let layer_def = score_split(annotator, test_p, test_v, parent_lookup, dim)?;
    let m_def = evaluate_incivility_corpus(&test_p.units, &layer_def, 0.5);
    let layer_uni = score_split(&annotator.clone().with_threshold(uni_t), test_p, test_v, parent_lookup, dim)?;
    let m_uni = evaluate_incivility_corpus(&test_p.units, &layer_uni, uni_t);
    let layer_pl = score_split(&annotator.clone().with_thresholds(per_label_t), test_p, test_v, parent_lookup, dim)?;
    let m_pl = evaluate_incivility_corpus_with_thresholds(&test_p.units, &layer_pl, &per_label_t);

    let summary = json!({
        "val_tuning": {
            "uniform_threshold": uni_t, "val_uniform_micro_f1": val_uni_micro,
            "per_label_thresholds": per_label_map,
            "val_per_label_macro_f1": val_pl_macro, "val_per_label_micro_f1": val_pl_micro,
        },
        "test": {
            "default_threshold_0_5": corpus_json(&m_def),
            "tuned_uniform_threshold": { "threshold": uni_t, "metrics": corpus_json(&m_uni) },
            "tuned_per_label_thresholds": { "thresholds": per_label_map, "metrics": corpus_json(&m_pl) },
        },
    });
    Ok((summary, uni_t, per_label_t))
}

fn head_meta_json(bytes: &[u8], fit_seconds: f64) -> Value {
    json!({
        "fit_seconds": fit_seconds,
        "size_bytes": bytes.len(),
        "sha256": Sha256::of(bytes).to_string(),
        "sha512": Sha512::of(bytes).to_string(),
    })
}

fn run() -> Result<(), String> {
    let a = args()?;
    let source = source::verify(&a.civil)?;
    let data::Civil { train, val, test } = data::civil(&a.civil)?;
    let toxic_ds = spans_eval::load_toxic_spans(&a.toxic_spans)?;
    eprintln!(
        "{} train, {} val, {} test; {} ACL2022 toxic_spans ({} non-ASCII), {} raw ({} non-ASCII)",
        train.len(),
        val.len(),
        test.len(),
        toxic_ds.acl_texts.len(),
        toxic_ds.acl_non_ascii,
        toxic_ds.raw_texts.len(),
        toxic_ds.raw_non_ascii
    );

    let mut embedder =
        embed::Embedder::open(&a.bundle, &EncoderOptions { device: a.device.clone(), ..Default::default() })?;
    let dim = embedder.dim;
    let mut embed_units = |units: &[Unit], name: &str| {
        eprintln!("embedding {name}");
        let texts: Vec<&str> = units.iter().map(|u| u.text.as_str()).collect();
        embedder.embed(&texts, &a.cache)
    };
    let (train_v, val_v, test_v) =
        (embed_units(&train, "train")?, embed_units(&val, "val")?, embed_units(&test, "test")?);
    let acl_refs: Vec<&str> = toxic_ds.acl_texts.iter().map(String::as_str).collect();
    let raw_refs: Vec<&str> = toxic_ds.raw_texts.iter().map(String::as_str).collect();
    let (acl_v, raw_v) = (embedder.embed(&acl_refs, &a.cache)?, embedder.embed(&raw_refs, &a.cache)?);

    let parent_map = spans_eval::build_parent_lookup(&train, &train_v, &val, &val_v, &test, &test_v, dim);
    let train_rows: Vec<&[f32]> = train_v.chunks_exact(dim).collect();
    let train_contexts: Vec<Option<&[f32]>> =
        train.iter().map(|u| u.parent_id.as_deref().and_then(|pid| parent_map.get(pid).copied())).collect();
    let train_with_parent = train_contexts.iter().filter(|c| c.is_some()).count();

    let soft_targets: Vec<IncivilityScores> = train.iter().map(|u| u.targets).collect();
    let hard_targets: Vec<IncivilityScores> = train
        .iter()
        .map(|u| {
            let mut t = u.targets;
            for l in IncivilityLabel::ALL {
                t.set(l, if t.get(l) >= 0.5 { 1.0 } else { 0.0 });
            }
            t
        })
        .collect();

    let rss_before_soft = peak_rss_kb();
    let fit_one = |ctx: Option<&[Option<&[f32]>]>,
                   targets: &[IncivilityScores],
                   label: &str|
     -> Result<(IncivilityVectorHead, f64), String> {
        eprintln!("fitting {label}");
        let t0 = Instant::now();
        let h = IncivilityVectorHead::fit(&train_rows, ctx, targets, &a.fit)
            .and_then(|h| h.with_model(embedder.model.clone()))
            .map_err(|e| e.to_string())?;
        Ok((h, t0.elapsed().as_secs_f64()))
    };
    let (head_soft, fit_soft_secs) = fit_one(None, &soft_targets, "soft targets (no context)")?;
    let rss_after_soft = peak_rss_kb();
    let (head_soft_ctx, fit_soft_ctx_secs) =
        fit_one(Some(&train_contexts), &soft_targets, "soft targets (parent context)")?;
    let rss_after_soft_ctx = peak_rss_kb();
    let (head_hard, fit_hard_secs) = fit_one(None, &hard_targets, "hard targets (no context)")?;
    let (head_hard_ctx, fit_hard_ctx_secs) =
        fit_one(Some(&train_contexts), &hard_targets, "hard targets (parent context)")?;
    let rss_peak_kb = peak_rss_kb();

    let (head_soft_bytes, head_soft_ctx_bytes, head_hard_bytes, head_hard_ctx_bytes) =
        (head_soft.to_bytes(), head_soft_ctx.to_bytes(), head_hard.to_bytes(), head_hard_ctx.to_bytes());
    for (opt_path, bytes) in [(&a.head_out, &head_soft_bytes), (&a.context_head_out, &head_soft_ctx_bytes)] {
        if let Some(p) = opt_path {
            if let Some(parent) = p.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::write(p, bytes).map_err(|e| format!("{}: {e}", p.display()))?;
        }
    }

    let (val_p, test_p) = (prepare(&val)?, prepare(&test)?);
    let val_with_parent =
        val_p.parent_ids.iter().filter(|pid| pid.as_deref().is_some_and(|id| parent_map.contains_key(id))).count();
    let test_with_parent =
        test_p.parent_ids.iter().filter(|pid| pid.as_deref().is_some_and(|id| parent_map.contains_key(id))).count();

    let scorer_soft: Arc<dyn IncivilityVectorScorer> = Arc::new(head_soft);
    let scorer_soft_ctx: Arc<dyn IncivilityVectorScorer> = Arc::new(head_soft_ctx);
    let scorer_hard: Arc<dyn IncivilityVectorScorer> = Arc::new(head_hard);
    let scorer_hard_ctx: Arc<dyn IncivilityVectorScorer> = Arc::new(head_hard_ctx);

    let lex_ann = IncivilityAnnotator::new().with_stage(IncivilityStage::Lexicon);
    let (lex_res, _, lex_pl_t) = eval_stage_on_test(&lex_ann, &val_p, &val_v, &test_p, &test_v, None, dim)?;

    let learned_soft_ann =
        IncivilityAnnotator::new().with_vector_scorer(Arc::clone(&scorer_soft)).with_stage(IncivilityStage::Learned);
    let (learned_soft_res, _, learned_soft_pl_t) =
        eval_stage_on_test(&learned_soft_ann, &val_p, &val_v, &test_p, &test_v, None, dim)?;

    let learned_soft_ctx_ann = IncivilityAnnotator::new()
        .with_vector_scorer(Arc::clone(&scorer_soft_ctx))
        .with_stage(IncivilityStage::Learned);
    let (learned_soft_ctx_res, _, _) =
        eval_stage_on_test(&learned_soft_ctx_ann, &val_p, &val_v, &test_p, &test_v, Some(&parent_map), dim)?;

    let fused_soft_ann =
        IncivilityAnnotator::new().with_vector_scorer(Arc::clone(&scorer_soft)).with_stage(IncivilityStage::Fused);
    let (fused_soft_res, _, fused_soft_pl_t) =
        eval_stage_on_test(&fused_soft_ann, &val_p, &val_v, &test_p, &test_v, None, dim)?;

    let fused_soft_ctx_ann =
        IncivilityAnnotator::new().with_vector_scorer(Arc::clone(&scorer_soft_ctx)).with_stage(IncivilityStage::Fused);
    let (fused_soft_ctx_res, _, _) =
        eval_stage_on_test(&fused_soft_ctx_ann, &val_p, &val_v, &test_p, &test_v, Some(&parent_map), dim)?;

    let learned_hard_ann =
        IncivilityAnnotator::new().with_vector_scorer(Arc::clone(&scorer_hard)).with_stage(IncivilityStage::Learned);
    let (learned_hard_res, _, learned_hard_pl_t) =
        eval_stage_on_test(&learned_hard_ann, &val_p, &val_v, &test_p, &test_v, None, dim)?;

    let learned_hard_ctx_ann = IncivilityAnnotator::new()
        .with_vector_scorer(Arc::clone(&scorer_hard_ctx))
        .with_stage(IncivilityStage::Learned);
    let (learned_hard_ctx_res, _, _) =
        eval_stage_on_test(&learned_hard_ctx_ann, &val_p, &val_v, &test_p, &test_v, Some(&parent_map), dim)?;

    let fused_hard_ann =
        IncivilityAnnotator::new().with_vector_scorer(Arc::clone(&scorer_hard)).with_stage(IncivilityStage::Fused);
    let (fused_hard_res, _, fused_hard_pl_t) =
        eval_stage_on_test(&fused_hard_ann, &val_p, &val_v, &test_p, &test_v, None, dim)?;

    let mut fusion_sweep = Vec::new();
    for &gate in &FUSION_GATES {
        for sev_rule in [true, false] {
            let ann = IncivilityAnnotator::new()
                .with_vector_scorer(Arc::clone(&scorer_soft))
                .with_stage(IncivilityStage::Fused)
                .with_fusion_gate(gate)
                .with_severe_toxicity_rule(sev_rule);
            let (uni_t, val_uni_micro, per_label_t, val_pl_macro, val_pl_micro) =
                tune_thresholds(&ann, &val_p, &val_v, None, dim)?;
            let test_def =
                evaluate_incivility_corpus(&test_p.units, &score_split(&ann, &test_p, &test_v, None, dim)?, 0.5);
            let test_pl = evaluate_incivility_corpus_with_thresholds(
                &test_p.units,
                &score_split(&ann.with_thresholds(per_label_t), &test_p, &test_v, None, dim)?,
                &per_label_t,
            );
            fusion_sweep.push(json!({
                "fusion_gate": gate, "severe_toxicity_rule": sev_rule,
                "val_uniform_threshold": uni_t, "val_uniform_micro_f1": val_uni_micro,
                "val_per_label_macro_f1": val_pl_macro, "val_per_label_micro_f1": val_pl_micro,
                "test_default_0_5": corpus_json(&test_def), "test_per_label_tuned": corpus_json(&test_pl),
            }));
        }
    }

    let default_ts = [0.5f64; INCIVILITY_DIMENSIONS];
    let eval_acl = |ann: &IncivilityAnnotator, ts: &[f64; INCIVILITY_DIMENSIONS]| {
        spans_eval::eval_subset(
            ann,
            &toxic_ds.acl_views,
            &toxic_ds.acl_tokens,
            &toxic_ds.acl_token_lower,
            &toxic_ds.acl_gold,
            &acl_v,
            dim,
            ts,
        )
    };
    let eval_raw = |ann: &IncivilityAnnotator, ts: &[f64; INCIVILITY_DIMENSIONS]| {
        spans_eval::eval_subset(
            ann,
            &toxic_ds.raw_views,
            &toxic_ds.raw_tokens,
            &toxic_ds.raw_token_lower,
            &toxic_ds.raw_gold,
            &raw_v,
            dim,
            ts,
        )
    };

    let record = json!({
        "source": source,
        "evaluation_protocol": "unicode-production-keys; rescore-candidate-thresholds; severe-last-v2",
        "encoder": { "id": embedder.model.id, "revision": embedder.model.revision, "manifest_sha256": embedder.model.sha256, "dim": dim },
        "civil_comments": {
            "train_comments": train.len(), "val_comments": val.len(), "test_comments": test.len(),
            "train_with_parent_in_dataset": train_with_parent, "val_with_parent_in_dataset": val_with_parent, "test_with_parent_in_dataset": test_with_parent,
        },
        "heads": {
            "fit_options": { "epochs": a.fit.epochs, "learning_rate": a.fit.learning_rate, "l2": a.fit.l2 },
            "rss_kb": { "before_fit_hwm_kb": rss_before_soft, "after_soft_no_ctx_hwm_kb": rss_after_soft, "after_soft_with_ctx_hwm_kb": rss_after_soft_ctx, "peak_hwm_kb": rss_peak_kb },
            "soft_targets_no_context": head_meta_json(&head_soft_bytes, fit_soft_secs),
            "soft_targets_parent_context": head_meta_json(&head_soft_ctx_bytes, fit_soft_ctx_secs),
            "hard_targets_no_context": head_meta_json(&head_hard_bytes, fit_hard_secs),
            "hard_targets_parent_context": head_meta_json(&head_hard_ctx_bytes, fit_hard_ctx_secs),
        },
        "stages_888": {
            "lexicon": lex_res,
            "learned_soft_no_context": learned_soft_res, "learned_soft_parent_context": learned_soft_ctx_res,
            "fused_soft_no_context": fused_soft_res, "fused_soft_parent_context": fused_soft_ctx_res,
            "learned_hard_no_context": learned_hard_res, "learned_hard_parent_context": learned_hard_ctx_res,
            "fused_hard_no_context": fused_hard_res,
        },
        "fusion_tuning_963": fusion_sweep,
        "toxic_spans_890": {
            "acl2022_11006": {
                "non_ascii_comments": toxic_ds.acl_non_ascii,
                "lexicon_default_0_5": eval_acl(&lex_ann, &default_ts)?,
                "learned_soft_default_0_5": eval_acl(&learned_soft_ann, &default_ts)?,
                "learned_soft_per_label_tuned": eval_acl(&learned_soft_ann.clone().with_thresholds(learned_soft_pl_t), &learned_soft_pl_t)?,
                "learned_hard_per_label_tuned": eval_acl(&learned_hard_ann.clone().with_thresholds(learned_hard_pl_t), &learned_hard_pl_t)?,
                "fused_soft_default_0_5": eval_acl(&fused_soft_ann, &default_ts)?,
                "fused_soft_per_label_tuned": eval_acl(&fused_soft_ann.clone().with_thresholds(fused_soft_pl_t), &fused_soft_pl_t)?,
                "fused_hard_per_label_tuned": eval_acl(&fused_hard_ann.clone().with_thresholds(fused_hard_pl_t), &fused_hard_pl_t)?,
            },
            "raw_16100": {
                "non_ascii_comments": toxic_ds.raw_non_ascii,
                "lexicon_tuned": eval_raw(&lex_ann, &lex_pl_t)?,
                "learned_soft_per_label_tuned": eval_raw(&learned_soft_ann.with_thresholds(learned_soft_pl_t), &learned_soft_pl_t)?,
                "fused_soft_per_label_tuned": eval_raw(&fused_soft_ann.with_thresholds(fused_soft_pl_t), &fused_soft_pl_t)?,
            },
        },
    });
    if let Some(parent) = a.out.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?;
    fs::write(&a.out, text + "\n").map_err(|e| format!("{}: {e}", a.out.display()))
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("incivility_eval: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests;
