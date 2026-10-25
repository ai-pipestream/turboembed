//! Reproducible WILDS Civil Comments 5-label sigmoid threshold calibration test (`#980`).
//!
//! Skips when neither `TURBO_CIVIL_COMMENTS_DIR` nor `TURBO_CIVIL_BUNDLE_DIR` is
//! set; panics naming both when only one is set. When both are set, reads the
//! `comment_text`, `split`, and five toxicity columns (`insult`, `obscene`,
//! `threat`, `identity_attack`, `severe_toxicity`) from
//! `all_data_with_identities.csv`, embeds each split's `comment_text` column
//! with the bundle at `TURBO_CIVIL_BUNDLE_DIR` (`add_special_tokens = true`,
//! `Truncation::Right`, `max_tokens = info.max_seq`, `TURBO_POOLING_MODEL`, and
//! `info.l2`, optionally caching `{manifest_sha256[..16]}-{fnv(texts):016x}.f32`
//! in `TURBO_CIVIL_CACHE_DIR` as `examples/incivility_bias` does), binds the
//! resulting `Vectors<Sentences>` to the bundle's `ModelRef`, fits the linear
//! `OutputKind::Sigmoid` head on `train`, calibrates per-label thresholds on
//! `val` via `ThresholdCalibrator::default()`, evaluates on `val` and `test`,
//! and verifies the recorded metrics in `docs/dense-classifier.md`.

use std::collections::BTreeSet;
use std::path::Path;

use turbo::TURBO_POOLING_MODEL;
use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo_analysis::{ModelRef, TagSet};
use turbo_analysis_embed::{Device, Encoder, EncoderOptions, Rows, dense::*};
use turbo_analysis_layers::{Sentences, Vectors};

fn fnv(texts: &[String]) -> u64 {
    let mut h = turbo_analysis::Fnv1a::new();
    for t in texts {
        h.str(t);
    }
    h.finish()
}

fn embed_split(
    encoder: &mut Encoder,
    tokenizer: &Tokenizer,
    model: &ModelRef,
    texts: &[String],
    cache_dir: Option<&Path>,
) -> Vectors<Sentences> {
    let info = encoder.info().clone();
    let dim = info.dim as usize;
    let cache_file =
        cache_dir.map(|dir| dir.join(format!("{}-{:016x}.f32", model.sha256.get(..16).unwrap_or("model"), fnv(texts))));
    if let Some(file) = &cache_file
        && let Ok(bytes) = std::fs::read(file)
        && bytes.len() == texts.len() * dim * 4
    {
        let data: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect();
        return Vectors::new(info.dim, data, vec![], model.clone());
    }
    let opts = Encode {
        add_special_tokens: true,
        truncation: Truncation::Right,
        max_tokens: info.max_seq,
        prompt: PromptRole::None,
    };
    let mut rows = Rows::new(info.max_seq, tokenizer.fill_id());
    let (mut ids, mut batch, mut out) = (Vec::new(), Vec::new(), Vec::with_capacity(texts.len() * dim));
    for chunk in texts.chunks(info.max_batch as usize) {
        rows.clear();
        for t in chunk {
            ids.clear();
            tokenizer.encode_into(t, opts, &mut ids).unwrap();
            rows.push(&[&ids]);
        }
        encoder.run_rows_pooled(&rows, TURBO_POOLING_MODEL, info.l2, &mut batch).unwrap();
        out.extend_from_slice(&batch);
    }
    if let Some(file) = &cache_file {
        if let Some(parent) = file.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(file, bytes).unwrap();
    }
    Vectors::new(info.dim, out, vec![], model.clone())
}

fn assert_close(actual: f64, expected: f64, label: &str) {
    assert!(
        (actual - expected).abs() < 1.5e-4,
        "{label}: expected {expected:.4}, got {actual:.4} (diff {:.6})",
        (actual - expected).abs()
    );
}

#[allow(clippy::too_many_arguments)]
fn check_report(
    r: &MultiLabelReport,
    mi_p: f64,
    mi_r: f64,
    mi_f1: f64,
    ma_p: f64,
    ma_r: f64,
    ma_f1: f64,
    exact: f64,
    tag: &str,
) {
    assert_close(r.micro.precision, mi_p, &format!("{tag} micro.precision"));
    assert_close(r.micro.recall, mi_r, &format!("{tag} micro.recall"));
    assert_close(r.micro.f1, mi_f1, &format!("{tag} micro.f1"));
    assert_close(r.r#macro.precision, ma_p, &format!("{tag} macro.precision"));
    assert_close(r.r#macro.recall, ma_r, &format!("{tag} macro.recall"));
    assert_close(r.r#macro.f1, ma_f1, &format!("{tag} macro.f1"));
    assert_close(r.exact_match, exact, &format!("{tag} exact_match"));
}

#[test]
fn civil_comments_multilabel_sigmoid_thresholds_when_configured() {
    let civil_dir = std::env::var("TURBO_CIVIL_COMMENTS_DIR").ok();
    let bundle_dir = std::env::var("TURBO_CIVIL_BUNDLE_DIR").ok();
    let (civil_dir, bundle_dir) = match (civil_dir, bundle_dir) {
        (None, None) => return,
        (Some(c), Some(b)) => (c, b),
        (c, b) => panic!(
            "both TURBO_CIVIL_COMMENTS_DIR ({:?}) and TURBO_CIVIL_BUNDLE_DIR ({:?}) must be set together",
            c.is_some(),
            b.is_some()
        ),
    };
    let cache_dir = std::env::var("TURBO_CIVIL_CACHE_DIR").ok();
    let device = match std::env::var("TURBO_CIVIL_DEVICE").as_deref() {
        Ok("cpu") | Err(_) => Device::Cpu,
        Ok(b) => Device::Backend(b.into()),
    };

    let bundle_path = Path::new(&bundle_dir);
    let bundle = Bundle::open(bundle_path).unwrap();
    let tokenizer = Tokenizer::load(&bundle).unwrap();
    let m = &bundle.manifest.model;
    let model = ModelRef { id: m.id.clone(), revision: m.revision.clone(), sha256: bundle.manifest_sha256.clone() };
    let mut encoder = Encoder::open(bundle_path, &EncoderOptions { device, ..Default::default() }).unwrap();

    let text = std::fs::read_to_string(Path::new(&civil_dir).join("all_data_with_identities.csv")).unwrap();
    let mut records = Vec::new();
    let (mut record, mut field, mut quoted) = (Vec::new(), String::new(), false);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if quoted {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                '"' => quoted = false,
                _ => field.push(c),
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => quoted = true,
            ',' => record.push(std::mem::take(&mut field)),
            '\r' if chars.peek() == Some(&'\n') => {}
            '\n' => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    let h = &records[0];
    let col = |name: &str| h.iter().position(|c| c == name).unwrap();
    let text_col = col("comment_text");
    let split_col = col("split");
    let label_names = ["insult", "obscene", "threat", "identity_attack", "severe_toxicity"];
    let label_cols: Vec<usize> = label_names.iter().map(|n| col(n)).collect();
    let (mut train_t, mut val_t, mut test_t) = (Vec::new(), Vec::new(), Vec::new());
    let (mut train_y, mut val_y, mut test_y) = (Vec::new(), Vec::new(), Vec::new());
    for r in &records[1..] {
        let row: Vec<f64> = label_cols
            .iter()
            .map(|&c| if r[c].is_empty() { 0.0 } else { r[c].parse::<f64>().unwrap().clamp(0.0, 1.0) })
            .collect();
        match r[split_col].as_str() {
            "train" => {
                train_t.push(r[text_col].clone());
                train_y.push(row);
            }
            "val" => {
                val_t.push(r[text_col].clone());
                val_y.push(row);
            }
            "test" => {
                test_t.push(r[text_col].clone());
                test_y.push(row);
            }
            s => panic!("unknown split {s}"),
        }
    }
    let cache_path = cache_dir.as_deref().map(Path::new);
    let train_v = embed_split(&mut encoder, &tokenizer, &model, &train_t, cache_path);
    let val_v = embed_split(&mut encoder, &tokenizer, &model, &val_t, cache_path);
    let test_v = embed_split(&mut encoder, &tokenizer, &model, &test_t, cache_path);

    let gold_sets = |y: &[Vec<f64>]| -> Vec<BTreeSet<usize>> {
        y.iter().map(|r| r.iter().enumerate().filter(|(_, v)| **v >= 0.5).map(|(i, _)| i).collect()).collect()
    };
    let (val_gold, test_gold) = (gold_sets(&val_y), gold_sets(&test_y));
    let ts = TagSet::new("civil_comments", &label_names);
    let hard_train: Vec<Vec<f64>> =
        train_y.iter().map(|r| r.iter().map(|&v| if v >= 0.5 { 1.0 } else { 0.0 }).collect()).collect();
    let trainer = DenseClassifierTrainer {
        architecture: Architecture::Logistic,
        output: OutputKind::Sigmoid,
        params: AdamParameters {
            seed: 42,
            learning_rate: 0.01,
            weight_decay: 1e-5,
            batch_size: 256,
            epochs: 8,
            patience: None,
        },
    };
    let fit = trainer.fit(&train_v, &hard_train, ts, None).unwrap();
    assert_eq!(fit.model.encoder(), &model);
    let cal = ThresholdCalibrator::default();
    let tuned = fit.model.clone().calibrate_thresholds(&val_v, &val_gold, &cal).unwrap();
    let val_def = fit.model.evaluate_multilabel(&val_v, &val_gold).unwrap();
    let val_cal = tuned.evaluate_multilabel(&val_v, &val_gold).unwrap();
    let test_def = fit.model.evaluate_multilabel(&test_v, &test_gold).unwrap();
    let test_cal = tuned.evaluate_multilabel(&test_v, &test_gold).unwrap();
    assert!(val_cal.r#macro.f1 > val_def.r#macro.f1 && test_cal.r#macro.f1 > test_def.r#macro.f1);

    match model.id.as_str() {
        "sentence-transformers/all-MiniLM-L6-v2" => {
            for (got, want) in tuned.thresholds().iter().zip([0.1928, 0.0429, 0.0248, 0.1319, 0.5000]) {
                assert_close(*got, want, "MiniLM threshold");
            }
            check_report(&val_def, 0.6845, 0.1135, 0.1947, 0.8424, 0.2397, 0.2666, 0.9228, "MiniLM val 0.5");
            check_report(&val_cal, 0.3496, 0.4862, 0.4068, 0.4230, 0.4775, 0.4409, 0.8634, "MiniLM val cal");
            check_report(&test_def, 0.6788, 0.1103, 0.1898, 0.8468, 0.0401, 0.0681, 0.9226, "MiniLM test 0.5");
            check_report(&test_cal, 0.3452, 0.4714, 0.3985, 0.4220, 0.2735, 0.2390, 0.8628, "MiniLM test cal");
        }
        "BAAI/bge-small-en-v1.5" => {
            for (got, want) in tuned.thresholds().iter().zip([0.2018, 0.0476, 0.0186, 0.1369, 0.5000]) {
                assert_close(*got, want, "bge-small threshold");
            }
            check_report(&val_def, 0.7602, 0.1101, 0.1923, 0.8706, 0.2377, 0.2642, 0.9234, "bge val 0.5");
            check_report(&val_cal, 0.3899, 0.4568, 0.4207, 0.4439, 0.4602, 0.4472, 0.8780, "bge val cal");
            check_report(&test_def, 0.7308, 0.1043, 0.1825, 0.8689, 0.0364, 0.0624, 0.9232, "bge test 0.5");
            check_report(&test_cal, 0.3856, 0.4475, 0.4142, 0.4321, 0.2571, 0.2401, 0.8777, "bge test cal");
        }
        _ => {}
    }
}
