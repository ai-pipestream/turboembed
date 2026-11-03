//! MASC 3.0.0 MPQA polarity head training, evaluation, and paired bootstrap on
//! `answerdotai/ModernBERT-base` vs `all-MiniLM-L6-v2`, `LexiconSentiment`, and
//! the majority baseline (#1732).

use std::fs;
use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};
use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Polarity, Sentences, Vectors};
use turbo_doccat::PolarityVectorScorer;
use turbo_doccat::sentiment::{
    LexiconSentiment, PolarityExample, PolarityFitConfig, PolarityMetrics, PolarityVectorHead, blend_polarity_scores,
    evaluate_predictions, majority_baseline, vader_to_polarity_scores,
};
use turbo_trainer::rng::Rng;
use turbo_zoo::{Sha256, Sha512};

use super::make_doc_with_sentences;

const BOOTSTRAP_RESAMPLES: usize = 10_000;
const BOOTSTRAP_SEED: u64 = 0x5eed_1732_0321_0001;

pub struct EncoderMascEval {
    pub model: ModelRef,
    pub dim: usize,
    pub centroids_preds: Vec<Polarity>,
    pub linear_preds: Vec<Polarity>,
    pub polarity_head: PolarityVectorHead,
    pub polarity_preds: Vec<Polarity>,
    pub hybrid_weight: f64,
    pub hybrid_preds: Vec<Polarity>,
    pub fit_config: PolarityFitConfig,
}

pub fn eval_encoder_on_masc(
    bundle_path: &Path,
    use_cuda: bool,
    m_train: &[(String, usize)],
    m_test: &[(String, usize)],
    lexicon: &LexiconSentiment,
) -> EncoderMascEval {
    let bundle_path = turbo_pins::require(bundle_path);
    let o = VectorOptions {
        encoder: EncoderOptions {
            device: if use_cuda { Device::Backend("cuda".into()) } else { Device::Cpu },
            max_seq: 64,
            ..EncoderOptions::default()
        },
        ..VectorOptions::default()
    };
    let vec_ann = Arc::new(
        VectorAnnotator::open(&bundle_path, &o).unwrap_or_else(|e| panic!("bundle {}: {e}", bundle_path.display())),
    );
    let dim = vec_ann.dim();
    let model = vec_ann.model().clone();
    let sub = SubwordAnnotator::open(&bundle_path).unwrap();
    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(sub)
        .add(SentenceVectors(Arc::clone(&vec_ann)))
        .build()
        .unwrap();

    let embed_slice = |slice: &[(String, usize)]| -> Vec<(Vec<f32>, Tag)> {
        let mut docs: Vec<Document> =
            slice.iter().map(|(t, _)| make_doc_with_sentences(t, &[(0, t.len() as u32)])).collect();
        for r in pipe.run(&mut docs) {
            r.unwrap();
        }
        docs.iter()
            .zip(slice)
            .map(|(d, (_, l))| {
                let v = d.get::<Vectors<Sentences>>().unwrap().row(Ix::new(0)).to_vec();
                (v, Tag(*l as u16))
            })
            .collect()
    };

    let train_vecs = embed_slice(m_train);
    let test_vecs = embed_slice(m_test);

    let pos_vecs: Vec<&[f32]> = train_vecs.iter().filter(|(_, t)| *t == Tag(0)).map(|(v, _)| v.as_slice()).collect();
    let neg_vecs: Vec<&[f32]> = train_vecs.iter().filter(|(_, t)| *t == Tag(1)).map(|(v, _)| v.as_slice()).collect();
    let tags = TagSet::new("sentiment", &["positive", "negative"]);

    let centroids = LabelCentroids::from_vector_examples(
        model.clone(),
        dim,
        "sentiment",
        &[("positive", &pos_vecs[..]), ("negative", &neg_vecs[..])],
        CentroidNorm::L2,
        0.1,
    )
    .unwrap();

    let linear =
        ClassifierHead::fit_single_label(model.clone(), dim, Arc::clone(&tags), &train_vecs, 1e-4, 300, 1.5).unwrap();

    let train_examples: Vec<PolarityExample> = train_vecs
        .iter()
        .map(|(v, t)| {
            PolarityExample::new(v.clone(), if *t == Tag(0) { Polarity::Positive } else { Polarity::Negative })
        })
        .collect();

    // Tune learning rate and hybrid weight on a 4:1 train/dev split inside m_train.
    let dev_split = (train_examples.len() * 4) / 5;
    let (inner_train, inner_dev) = train_examples.split_at(dev_split);
    let mut best_cfg = PolarityFitConfig {
        epochs: 60,
        batch_size: 32,
        learning_rate: 0.02,
        weight_decay: 1e-3,
        warmup_steps: 2,
        clip_norm: 5.0,
        patience: 12,
        seed: 0x504F_4C41_5249_5459,
    };
    let mut best_dev_score = -1.0f64;
    let mut best_dev_head: Option<PolarityVectorHead> = None;
    for &lr in &[0.005, 0.01, 0.02, 0.05] {
        for &wd in &[1e-4, 1e-3, 1e-2, 5e-2] {
            let cfg = PolarityFitConfig { learning_rate: lr, weight_decay: wd, ..best_cfg };
            let (cand, rep) =
                PolarityVectorHead::fit_with_dev(inner_train, inner_dev, &cfg, Some(model.clone())).unwrap();
            let dev_acc = rep.best_dev_accuracy.unwrap_or(0.0);
            let dev_f1 = rep.best_dev_macro_f1.unwrap_or(0.0);
            let score = dev_acc + dev_f1;
            if score > best_dev_score + 1e-12 {
                best_dev_score = score;
                best_cfg = PolarityFitConfig { epochs: rep.epochs_run, ..cfg };
                best_dev_head = Some(cand);
            }
        }
    }

    let dev_head_ref = best_dev_head.as_ref().unwrap();
    let mut best_hybrid_w = 0.50f64;
    let mut best_hybrid_dev_ok = 0usize;
    for w_step in 1..=19 {
        let w = w_step as f64 * 0.05;
        let mut ok = 0usize;
        for (i, ex) in inner_dev.iter().enumerate() {
            let (text, _) = &m_train[dev_split + i];
            let (_, lex_s) = vader_to_polarity_scores(lexicon.score_text(text));
            let head_s = dev_head_ref.score_polarity(&ex.vector).unwrap();
            let (_, blended) = blend_polarity_scores(lex_s, head_s, w).unwrap();
            let pred = if blended.positive >= blended.negative { Polarity::Positive } else { Polarity::Negative };
            if pred == ex.polarity {
                ok += 1;
            }
        }
        if ok > best_hybrid_dev_ok {
            best_hybrid_dev_ok = ok;
            best_hybrid_w = w;
        }
    }
    let polarity_head = PolarityVectorHead::fit(&train_examples, &best_cfg, Some(model.clone())).unwrap();

    let mut probs = [0.0f64; 2];
    let mut centroids_preds = Vec::with_capacity(m_test.len());
    let mut linear_preds = Vec::with_capacity(m_test.len());
    let mut polarity_preds = Vec::with_capacity(m_test.len());
    let mut hybrid_preds = Vec::with_capacity(m_test.len());

    for (i, (text, _)) in m_test.iter().enumerate() {
        let (v, _) = &test_vecs[i];
        let c_tag = centroids.score_single_label(v, &mut probs).unwrap().0;
        centroids_preds.push(if c_tag == Tag(0) { Polarity::Positive } else { Polarity::Negative });

        let l_tag = linear.score(v, &mut probs).unwrap().0;
        linear_preds.push(if l_tag == Tag(0) { Polarity::Positive } else { Polarity::Negative });

        let p_scores = polarity_head.score_polarity(v).unwrap();
        polarity_preds.push(if p_scores.positive >= p_scores.negative {
            Polarity::Positive
        } else {
            Polarity::Negative
        });

        let (_, lex_s) = vader_to_polarity_scores(lexicon.score_text(text));
        let (_, h_scores) = blend_polarity_scores(lex_s, p_scores, best_hybrid_w).unwrap();
        hybrid_preds.push(if h_scores.positive >= h_scores.negative { Polarity::Positive } else { Polarity::Negative });
    }

    EncoderMascEval {
        model,
        dim: dim as usize,
        centroids_preds,
        linear_preds,
        polarity_head,
        polarity_preds,
        hybrid_weight: best_hybrid_w,
        hybrid_preds,
        fit_config: best_cfg,
    }
}

pub fn paired_bootstrap_polarity(gold: &[Polarity], cand: &[Polarity], base: &[Polarity]) -> Value {
    let n = gold.len();
    let cand_m = evaluate_predictions(gold, cand).unwrap();
    let base_m = evaluate_predictions(gold, base).unwrap();

    let mut rng = Rng::new(BOOTSTRAP_SEED);
    let mut acc_diffs = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    let mut f1_diffs = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    let mut g_sample = Vec::with_capacity(n);
    let mut c_sample = Vec::with_capacity(n);
    let mut b_sample = Vec::with_capacity(n);

    for _ in 0..BOOTSTRAP_RESAMPLES {
        g_sample.clear();
        c_sample.clear();
        b_sample.clear();
        for _ in 0..n {
            let idx = (rng.next_u64() as usize) % n;
            g_sample.push(gold[idx]);
            c_sample.push(cand[idx]);
            b_sample.push(base[idx]);
        }
        let cm = evaluate_predictions(&g_sample, &c_sample).unwrap();
        let bm = evaluate_predictions(&g_sample, &b_sample).unwrap();
        acc_diffs.push(cm.accuracy - bm.accuracy);
        f1_diffs.push(cm.macro_f1 - bm.macro_f1);
    }
    acc_diffs.sort_by(f64::total_cmp);
    f1_diffs.sort_by(f64::total_cmp);
    let lo = (BOOTSTRAP_RESAMPLES as f64 * 0.025) as usize;
    let hi = (BOOTSTRAP_RESAMPLES as f64 * 0.975) as usize;
    let acc_p = acc_diffs.iter().filter(|&&d| d <= 0.0).count() as f64 / BOOTSTRAP_RESAMPLES as f64;
    let f1_p = f1_diffs.iter().filter(|&&d| d <= 0.0).count() as f64 / BOOTSTRAP_RESAMPLES as f64;

    json!({
        "resamples": BOOTSTRAP_RESAMPLES,
        "accuracy": {
            "candidate": cand_m.accuracy,
            "baseline": base_m.accuracy,
            "diff": cand_m.accuracy - base_m.accuracy,
            "ci95": [acc_diffs[lo], acc_diffs[hi]],
            "p_value_one_sided": acc_p,
        },
        "macro_f1": {
            "candidate": cand_m.macro_f1,
            "baseline": base_m.macro_f1,
            "diff": cand_m.macro_f1 - base_m.macro_f1,
            "ci95": [f1_diffs[lo], f1_diffs[hi]],
            "p_value_one_sided": f1_p,
        },
    })
}

fn metrics_to_json(m: &PolarityMetrics, total: usize) -> Value {
    let correct = (m.accuracy * total as f64).round() as usize;
    json!({
        "correct": correct,
        "total": total,
        "accuracy": m.accuracy,
        "macro_f1": m.macro_f1,
        "positive": {
            "precision": m.class(Polarity::Positive).precision,
            "recall": m.class(Polarity::Positive).recall,
            "f1": m.class(Polarity::Positive).f1,
            "support": m.class(Polarity::Positive).support,
        },
        "negative": {
            "precision": m.class(Polarity::Negative).precision,
            "recall": m.class(Polarity::Negative).recall,
            "f1": m.class(Polarity::Negative).f1,
            "support": m.class(Polarity::Negative).support,
        },
    })
}

pub fn write_modernbert_polarity_receipt(
    m_train: &[(String, usize)],
    m_test: &[(String, usize)],
    lex_preds: &[Polarity],
    modern: &EncoderMascEval,
    minilm: Option<&EncoderMascEval>,
    head_out: Option<&Path>,
    receipt_out: Option<&Path>,
) {
    let train_gold: Vec<Polarity> =
        m_train.iter().map(|(_, l)| if *l == 0 { Polarity::Positive } else { Polarity::Negative }).collect();
    let test_gold: Vec<Polarity> =
        m_test.iter().map(|(_, l)| if *l == 0 { Polarity::Positive } else { Polarity::Negative }).collect();
    let (maj_label, maj_m) = majority_baseline(&train_gold, &test_gold).unwrap();
    let maj_preds = vec![maj_label; test_gold.len()];

    let head_bytes = modern.polarity_head.to_bytes();
    let head_sha256 = Sha256::of(&head_bytes).to_string();
    let head_sha512 = Sha512::of(&head_bytes).to_string();

    if let Some(p) = head_out {
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(p, &head_bytes).unwrap();
    }

    let n = test_gold.len();
    let lex_m = evaluate_predictions(&test_gold, lex_preds).unwrap();
    let mod_cent_m = evaluate_predictions(&test_gold, &modern.centroids_preds).unwrap();
    let mod_lin_m = evaluate_predictions(&test_gold, &modern.linear_preds).unwrap();
    let mod_pol_m = evaluate_predictions(&test_gold, &modern.polarity_preds).unwrap();
    let mod_hyb_m = evaluate_predictions(&test_gold, &modern.hybrid_preds).unwrap();

    let mut bootstrap = serde_json::Map::new();
    bootstrap.insert(
        "modernbert_polarity_head_vs_majority".into(),
        paired_bootstrap_polarity(&test_gold, &modern.polarity_preds, &maj_preds),
    );
    bootstrap.insert(
        "modernbert_classifier_head_vs_majority".into(),
        paired_bootstrap_polarity(&test_gold, &modern.linear_preds, &maj_preds),
    );
    bootstrap.insert(
        "modernbert_hybrid_vs_majority".into(),
        paired_bootstrap_polarity(&test_gold, &modern.hybrid_preds, &maj_preds),
    );
    bootstrap.insert(
        "modernbert_hybrid_vs_lexicon".into(),
        paired_bootstrap_polarity(&test_gold, &modern.hybrid_preds, lex_preds),
    );
    bootstrap.insert(
        "modernbert_polarity_head_vs_lexicon".into(),
        paired_bootstrap_polarity(&test_gold, &modern.polarity_preds, lex_preds),
    );

    let minilm_json = minilm.map(|ml| {
        bootstrap.insert(
            "modernbert_polarity_head_vs_minilm_polarity_head".into(),
            paired_bootstrap_polarity(&test_gold, &modern.polarity_preds, &ml.polarity_preds),
        );
        bootstrap.insert(
            "modernbert_classifier_head_vs_minilm_classifier_head".into(),
            paired_bootstrap_polarity(&test_gold, &modern.linear_preds, &ml.linear_preds),
        );
        bootstrap.insert(
            "modernbert_hybrid_vs_minilm_hybrid".into(),
            paired_bootstrap_polarity(&test_gold, &modern.hybrid_preds, &ml.hybrid_preds),
        );
        json!({
            "encoder": {
                "id": ml.model.id,
                "revision": ml.model.revision,
                "manifest_sha256": ml.model.sha256,
                "dim": ml.dim,
            },
            "centroids": metrics_to_json(&evaluate_predictions(&test_gold, &ml.centroids_preds).unwrap(), n),
            "classifier_head": metrics_to_json(&evaluate_predictions(&test_gold, &ml.linear_preds).unwrap(), n),
            "polarity_vector_head_learned": metrics_to_json(&evaluate_predictions(&test_gold, &ml.polarity_preds).unwrap(), n),
            "polarity_hybrid": {
                "head_weight": ml.hybrid_weight,
                "metrics": metrics_to_json(&evaluate_predictions(&test_gold, &ml.hybrid_preds).unwrap(), n),
            },
        })
    });

    if let Some(out_path) = receipt_out {
        let record = json!({
            "corpus": {
                "name": "MASC 3.0.0 MPQA subjective polarity spans",
                "license": "CC-BY-3.0-US",
                "train_spans": m_train.len(),
                "test_spans": m_test.len(),
            },
            "encoder": {
                "id": modern.model.id,
                "revision": modern.model.revision,
                "manifest_sha256": modern.model.sha256,
                "dim": modern.dim,
            },
            "polarity_vector_head_artifact": {
                "id": "turbo-en-masc-3.0.0-modernbert-base-polarity-head-1",
                "format": "TPVH v1",
                "trainer": "turbo-trainer (AdamW, linear::forward, linear::backward, xent::loss, Rng)",
                "fit_config": {
                    "epochs": modern.fit_config.epochs,
                    "batch_size": modern.fit_config.batch_size,
                    "learning_rate": modern.fit_config.learning_rate,
                    "weight_decay": modern.fit_config.weight_decay,
                    "warmup_steps": modern.fit_config.warmup_steps,
                    "clip_norm": modern.fit_config.clip_norm,
                    "seed": modern.fit_config.seed,
                },
                "size_bytes": head_bytes.len(),
                "sha256": head_sha256,
                "sha512": head_sha512,
            },
            "baselines": {
                "majority": {
                    "predicted_class": maj_label.to_string(),
                    "metrics": metrics_to_json(&maj_m, n),
                },
                "lexicon_vader": metrics_to_json(&lex_m, n),
            },
            "modernbert_base": {
                "centroids": metrics_to_json(&mod_cent_m, n),
                "classifier_head": metrics_to_json(&mod_lin_m, n),
                "polarity_vector_head_learned": metrics_to_json(&mod_pol_m, n),
                "polarity_hybrid": {
                    "head_weight": modern.hybrid_weight,
                    "metrics": metrics_to_json(&mod_hyb_m, n),
                },
            },
            "legacy_all_minilm_l6_v2": minilm_json,
            "paired_bootstrap": bootstrap,
        });
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(out_path, serde_json::to_string_pretty(&record).unwrap() + "\n").unwrap();
    }
}
