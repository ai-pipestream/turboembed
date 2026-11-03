use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Value, json};
use turbo_analysis::Annotator;
use turbo_analysis_embed::aspect::{LabeledAbsaDoc, LearnedAspectSentiment};
use turbo_analysis_embed::{
    ClassifierHead, Device, EncoderOptions, ScoringHead, SubwordAnnotator, VectorAnnotator, VectorOptions,
};
use turbo_analysis_layers::Subwords;
use turbo_doccat::sentiment::aspect::RuleAspectSentiment;
use turbo_trainer::rng::Rng;

use super::corpus::{CorpusDoc, load};
use super::metrics::{self, Metrics};
use super::release;

struct Prepared {
    corpus: CorpusDoc,
    subwords: Subwords,
}

pub fn run() {
    let root = PathBuf::from(std::env::var_os("TURBO_MASC_ROOT").expect("TURBO_MASC_ROOT names MASC's data directory"));
    let bundle = turbo_pins::require(PathBuf::from(
        std::env::var_os("TURBO_MINILM_BUNDLE").expect("TURBO_MINILM_BUNDLE names the MiniLM bundle"),
    ));
    let corpus = load(&root);
    assert!(corpus.len() >= 20, "expected at least 20 usable MASC documents, got {}", corpus.len());
    let release_dir = std::env::var_os("TURBO_ASPECT_HEAD");
    assert!(
        release_dir.is_some() != std::env::var_os("TURBO_ASPECT_OUT").is_some(),
        "choose release verification OR explicit training output"
    );
    if let Some(dir) = &release_dir {
        for name in ["aspect-bio-1.thead", "aspect-polarity-1.thead"] {
            turbo_pins::require(Path::new(dir).join(name));
        }
        release::validate_population(&corpus, corpus.len() * 4 / 5).unwrap();
    }
    let subwords = SubwordAnnotator::open(&bundle).unwrap();
    let options = VectorOptions { encoder: EncoderOptions::default(), ..VectorOptions::default() };
    let vectors = Arc::new(VectorAnnotator::open(&bundle, &options).unwrap());
    assert_eq!(vectors.dim(), 384);
    let prepared: Vec<Prepared> = corpus
        .into_iter()
        .map(|corpus| Prepared { subwords: subwords.pieces(&corpus.view).unwrap(), corpus })
        .collect();
    let split = prepared.len() * 4 / 5;
    assert!(split > 0 && split < prepared.len());
    let learned = if let Some(dir) = &release_dir {
        load_heads(Path::new(&dir), vectors)
    } else {
        train(&prepared[..split], vectors)
    };
    if let Some(dir) = std::env::var_os("TURBO_ASPECT_OUT") {
        save_heads(Path::new(&dir), &learned);
    }
    println!(
        "MASC 3.0.0 MPQA: {} documents ({} train, {} held out), {} train aspects, {} held-out aspects",
        prepared.len(),
        split,
        prepared.len() - split,
        prepared[..split].iter().map(|d| d.corpus.aspects.len()).sum::<usize>(),
        prepared[split..].iter().map(|d| d.corpus.aspects.len()).sum::<usize>()
    );
    println!(
        "held-out files: {}",
        prepared[split..].iter().map(|d| d.corpus.id.as_str()).collect::<Vec<_>>().join(", ")
    );
    let train_metrics = evaluate(&prepared[..split], &learned);
    report("LearnedAspectSentiment train", &train_metrics);
    let learned_metrics = evaluate(&prepared[split..], &learned);
    let rule = RuleAspectSentiment::new();
    let rule_metrics = evaluate(&prepared[split..], &rule);
    if release_dir.is_some() {
        let expected = release::expected();
        release::validate_metrics(&learned_metrics, &expected["config"]["evaluation"]).unwrap();
        release::validate_metrics(&rule_metrics, &expected["rules"]).unwrap();
    }
    report("LearnedAspectSentiment", &learned_metrics);
    report("RuleAspectSentiment", &rule_metrics);
    println!(
        "RESULT learned_aspect_f1={:.6} learned_polarity_accuracy={:.6} rule_aspect_f1={:.6} rule_polarity_accuracy={:.6}",
        learned_metrics.f1(),
        learned_metrics.polarity_accuracy(),
        rule_metrics.f1(),
        rule_metrics.polarity_accuracy()
    );
}

fn train(docs: &[Prepared], vectors: Arc<VectorAnnotator>) -> LearnedAspectSentiment {
    let labeled: Vec<LabeledAbsaDoc> = docs
        .iter()
        .map(|d| LabeledAbsaDoc {
            subwords: &d.subwords,
            sentences: &d.corpus.sentences,
            tokens: &d.corpus.tokens,
            aspects: &d.corpus.aspects,
            opinions: &d.corpus.opinions,
        })
        .collect();
    let fitted = LearnedAspectSentiment::fit_balanced(vectors.clone(), &["general"], &labeled, 0.01, 400, 0.1).unwrap();
    LearnedAspectSentiment::new(vectors, classifier(fitted.bio_head), classifier(fitted.polarity_head), None)
        .with_bio_threshold(0.4)
}

fn classifier(head: ScoringHead) -> Arc<ClassifierHead> {
    let ScoringHead::Classifier(head) = head else { panic!("fitted heads must be classifiers") };
    head
}

fn load_heads(dir: &Path, vectors: Arc<VectorAnnotator>) -> LearnedAspectSentiment {
    let (bio, polarity, threshold) = release::load(dir, vectors.model()).unwrap();
    LearnedAspectSentiment::new(vectors, Arc::new(bio), Arc::new(polarity), None).with_bio_threshold(threshold)
}

fn save_heads(dir: &Path, learned: &LearnedAspectSentiment) {
    std::fs::create_dir_all(dir).unwrap();
    classifier(learned.bio_head.clone()).save(&dir.join("aspect-bio-1.thead")).unwrap();
    classifier(learned.polarity_head.clone()).save(&dir.join("aspect-polarity-1.thead")).unwrap();
}

fn evaluate<A>(docs: &[Prepared], annotator: &A) -> Metrics
where
    A: AspectEval,
{
    let mut total = metrics::empty();
    for doc in docs {
        let aspects = annotator.aspects(doc);
        metrics::assert_aligned(&doc.corpus, &aspects);
        total.add(&metrics::score(&doc.corpus, &aspects));
    }
    total
}

trait AspectEval {
    fn aspects(&self, doc: &Prepared) -> turbo_analysis_layers::Aspects;
}

impl AspectEval for LearnedAspectSentiment {
    fn aspects(&self, doc: &Prepared) -> turbo_analysis_layers::Aspects {
        let (_, aspects) = self
            .annotate((&doc.corpus.view, &doc.corpus.sentences, &doc.corpus.tokens, &doc.subwords, None, None, None))
            .unwrap();
        aspects
    }
}

impl AspectEval for RuleAspectSentiment {
    fn aspects(&self, doc: &Prepared) -> turbo_analysis_layers::Aspects {
        let (_, aspects) =
            self.annotate((&doc.corpus.view, &doc.corpus.sentences, &doc.corpus.tokens, None, None, None)).unwrap();
        aspects
    }
}

fn report(name: &str, m: &Metrics) {
    println!(
        "{name}: aspect term TP={} FP={} FN={} F1={:.4}; polarity {}/{} accuracy={:.4}",
        m.aspect_tp,
        m.aspect_fp,
        m.aspect_fn,
        m.f1(),
        m.polarity_correct,
        m.polarity_total,
        m.polarity_accuracy()
    );
}

pub fn run_modernbert() {
    let root = PathBuf::from(std::env::var_os("TURBO_MASC_ROOT").expect("TURBO_MASC_ROOT names MASC's data directory"));
    let bundle = turbo_pins::require(PathBuf::from(
        std::env::var_os("TURBO_MODERNBERT_BUNDLE").expect("TURBO_MODERNBERT_BUNDLE names the ModernBERT-base bundle"),
    ));
    let corpus = load(&root);
    assert!(corpus.len() >= 20, "expected at least 20 usable MASC documents, got {}", corpus.len());
    let use_cuda = std::env::var("TURBO_TEST_REQUIRE_CUDA").as_deref() == Ok("1");
    let subwords = SubwordAnnotator::open(&bundle).unwrap();
    let options = VectorOptions {
        encoder: EncoderOptions {
            device: if use_cuda { Device::Backend("cuda".into()) } else { Device::Cpu },
            max_seq: 512,
            ..EncoderOptions::default()
        },
        ..VectorOptions::default()
    };
    let vectors = Arc::new(VectorAnnotator::open(&bundle, &options).unwrap());
    assert_eq!(vectors.dim(), 768);
    let model = vectors.model().clone();
    let pop_sha = release::population_digest(&corpus);
    let prepared: Vec<Prepared> = corpus
        .into_iter()
        .map(|corpus| Prepared { subwords: subwords.pieces(&corpus.view).unwrap(), corpus })
        .collect();
    let split = prepared.len() * 4 / 5;
    assert!(split > 0 && split < prepared.len());
    let learned = train(&prepared[..split], vectors);
    let rule = RuleAspectSentiment::new();

    let train_metrics = evaluate(&prepared[..split], &learned);
    let learned_per_doc = evaluate_per_doc(&prepared[split..], &learned);
    let rule_per_doc = evaluate_per_doc(&prepared[split..], &rule);
    let maj_per_doc: Vec<Metrics> = prepared[split..]
        .iter()
        .map(|d| metrics::score(&d.corpus, &turbo_analysis_layers::Aspects::default()))
        .collect();

    let mut learned_metrics = metrics::empty();
    for m in &learned_per_doc {
        learned_metrics.add(m);
    }
    let mut rule_metrics = metrics::empty();
    for m in &rule_per_doc {
        rule_metrics.add(m);
    }

    report("ModernBERT LearnedAspectSentiment train", &train_metrics);
    report("ModernBERT LearnedAspectSentiment held-out", &learned_metrics);
    report("RuleAspectSentiment held-out", &rule_metrics);

    let bs_vs_rule = paired_bootstrap_aspect(&learned_per_doc, &rule_per_doc);
    let bs_vs_maj = paired_bootstrap_aspect(&learned_per_doc, &maj_per_doc);

    if let Some(out_path) = std::env::var_os("TURBO_ASPECT_RECEIPT_OUT").map(PathBuf::from) {
        let record = json!({
            "corpus": {
                "name": "MASC 3.0.0 MPQA aspect-opinion spans",
                "license": "CC-BY-3.0-US",
                "documents_total": prepared.len(),
                "documents_train": split,
                "documents_test": prepared.len() - split,
                "train_aspects": prepared[..split].iter().map(|d| d.corpus.aspects.len()).sum::<usize>(),
                "test_aspects": prepared[split..].iter().map(|d| d.corpus.aspects.len()).sum::<usize>(),
                "population_sha256": pop_sha,
            },
            "encoder": {
                "id": model.id,
                "revision": model.revision,
                "manifest_sha256": model.sha256,
                "dim": 768,
            },
            "trainer": "turbo-trainer (ClassifierHead::fit_single_label via linear::forward, linear::backward, xent::loss)",
            "held_out": {
                "learned_modernbert_base": metrics_json(&learned_metrics),
                "rule_aspect_sentiment": metrics_json(&rule_metrics),
                "majority_baseline_no_aspects": {
                    "aspect_tp": 0,
                    "aspect_fp": 0,
                    "aspect_fn": maj_per_doc.iter().map(|m| m.aspect_fn).sum::<usize>(),
                    "aspect_f1": 0.0,
                },
                "legacy_minilm_l6_v2_reference": release::expected()["config"]["evaluation"],
            },
            "paired_bootstrap": {
                "learned_modernbert_vs_rule": bs_vs_rule,
                "learned_modernbert_vs_majority": bs_vs_maj,
            },
            "default_selection": {
                "default_extractor": "RuleAspectSentiment",
                "reason": "Held-out exact aspect-span F1 remains near zero on 56 MASC training documents; RuleAspectSentiment stays the default extractor and no CLASS_PERMISSIVE aspect head is published in zoo.json.",
            },
        });
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&out_path, serde_json::to_string_pretty(&record).unwrap() + "\n").unwrap();
    }
}

fn evaluate_per_doc<A: AspectEval>(docs: &[Prepared], annotator: &A) -> Vec<Metrics> {
    docs.iter()
        .map(|doc| {
            let aspects = annotator.aspects(doc);
            metrics::assert_aligned(&doc.corpus, &aspects);
            metrics::score(&doc.corpus, &aspects)
        })
        .collect()
}

fn metrics_json(m: &Metrics) -> Value {
    json!({
        "aspect_tp": m.aspect_tp,
        "aspect_fp": m.aspect_fp,
        "aspect_fn": m.aspect_fn,
        "aspect_f1": m.f1(),
        "polarity_correct": m.polarity_correct,
        "polarity_total": m.polarity_total,
        "polarity_accuracy": m.polarity_accuracy(),
    })
}

fn paired_bootstrap_aspect(cand: &[Metrics], base: &[Metrics]) -> Value {
    const RESAMPLES: usize = 10_000;
    let n = cand.len();
    let mut cand_tot = metrics::empty();
    let mut base_tot = metrics::empty();
    for i in 0..n {
        cand_tot.add(&cand[i]);
        base_tot.add(&base[i]);
    }
    let mut rng = Rng::new(0x5eed_1732_0924_0001);
    let mut diffs = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        let mut c = metrics::empty();
        let mut b = metrics::empty();
        for _ in 0..n {
            let idx = (rng.next_u64() as usize) % n;
            c.add(&cand[idx]);
            b.add(&base[idx]);
        }
        diffs.push(c.f1() - b.f1());
    }
    diffs.sort_by(f64::total_cmp);
    let lo = (RESAMPLES as f64 * 0.025) as usize;
    let hi = (RESAMPLES as f64 * 0.975) as usize;
    let p_val = diffs.iter().filter(|&&d| d <= 0.0).count() as f64 / RESAMPLES as f64;
    json!({
        "resamples": RESAMPLES,
        "aspect_f1": {
            "candidate": cand_tot.f1(),
            "baseline": base_tot.f1(),
            "diff": cand_tot.f1() - base_tot.f1(),
            "ci95": [diffs[lo], diffs[hi]],
            "p_value_one_sided": p_val,
        },
    })
}
