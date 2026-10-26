use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_analysis::{Annotator, Document, Span, Spans, TagSet, View, ViewId};
use turbo_analysis_embed::vectors::Input;
use turbo_analysis_embed::{ClassifierHead, SubwordAnnotator, VectorAnnotator, VectorOptions};
use turbo_analysis_layers::{Sentences, Subwords, Tokens, Vectors};
use turbo_doccat::sentiment::pragmatics::PragmaticsAnnotator;
use turbo_tokenize::uax29::UntypedWordTokenizer;

use super::corpus::{Response, load};
use super::metrics::{Counts, best_threshold, report, score};
use super::release;
use super::split::Partition;

struct Prepared {
    response: Response,
    view: View,
    sentences: Sentences,
    tokens: Tokens,
    subwords: Subwords,
    vectors: Option<Vectors<Sentences>>,
}

pub fn run() {
    let csv = PathBuf::from(std::env::var_os("TURBO_CSC_CSV").expect("TURBO_CSC_CSV names CSC data_full.csv"));
    let bundle = PathBuf::from(std::env::var_os("TURBO_MINILM_BUNDLE").expect("TURBO_MINILM_BUNDLE names MiniLM"));
    let release_dir = std::env::var_os("TURBO_PRAGMATICS_HEAD");
    let output_dir = std::env::var_os("TURBO_PRAGMATICS_OUT");
    assert!(release_dir.is_some() != output_dir.is_some(), "choose release verification OR explicit training output");
    if release_dir.is_some() {
        release::validate_source(&csv).unwrap();
    }
    let responses = load(&csv);
    if release_dir.is_some() {
        release::validate_population(&responses).unwrap();
    }
    release::validate_disjoint_inputs(&responses).unwrap();
    assert!(responses.len() >= 1_000, "expected aggregated CSC responses, got {}", responses.len());
    let subwords = SubwordAnnotator::open(&bundle).unwrap();
    let vectors = Arc::new(VectorAnnotator::open(&bundle, &VectorOptions::default()).unwrap());
    let mut prepared: Vec<Prepared> = responses.into_iter().map(|r| prepare(r, &subwords)).collect();
    embed(&mut prepared, &vectors);
    let part = |want| prepared.iter().filter(|p| p.response.partition == want).collect::<Vec<&Prepared>>();
    let (fit_rows, dev, test) = (part(Partition::Train), part(Partition::Dev), part(Partition::Test));
    assert!(!fit_rows.is_empty() && !dev.is_empty() && !test.is_empty());
    let (head, head_threshold, combined_threshold) = if let Some(dir) = &release_dir {
        release::load(Path::new(dir), vectors.model()).unwrap()
    } else {
        let head = fit(&fit_rows, &vectors);
        let head_threshold = best_threshold(&head_scores(&dev, &head), &labels(&dev));
        let combined_threshold = best_threshold(&combined_scores(&dev, &head), &labels(&dev));
        (head, head_threshold, combined_threshold)
    };
    println!(
        "CSC: {} responses, {} train, {} development, {} test; positive test={}/{}",
        prepared.len(),
        fit_rows.len(),
        dev.len(),
        test.len(),
        labels(&test).iter().filter(|&&x| x).count(),
        test.len()
    );
    let labels = labels(&test);
    let rule_counts = score(&rule_scores(&test), &labels, 0.5);
    let head_counts = score(&head_scores(&test, &head), &labels, head_threshold);
    let combined_counts = score(&combined_scores(&test, &head), &labels, combined_threshold);
    let all_sarcastic = score(&vec![1.0; labels.len()], &labels, 0.5);
    let all_literal = score(&vec![0.0; labels.len()], &labels, 0.5);
    let counts = [rule_counts, head_counts, combined_counts, all_sarcastic, all_literal];
    if release_dir.is_some() {
        let expected = release::expected();
        for (name, counts) in EVALUATIONS.into_iter().zip(counts) {
            release::validate_metrics(counts, &expected["config"]["evaluation"][name]).unwrap();
        }
    }
    report("rules", 0.5, rule_counts);
    report("head", head_threshold, head_counts);
    report("combined", combined_threshold, combined_counts);
    report("all_sarcastic baseline", 0.5, all_sarcastic);
    report("all_literal baseline", 0.5, all_literal);
    println!(
        "RESULT rules_precision={:.6} rules_recall={:.6} rules_f1={:.6} head_precision={:.6} head_recall={:.6} head_f1={:.6} combined_precision={:.6} combined_recall={:.6} combined_f1={:.6}",
        rule_counts.precision(),
        rule_counts.recall(),
        rule_counts.f1(),
        head_counts.precision(),
        head_counts.recall(),
        head_counts.f1(),
        combined_counts.precision(),
        combined_counts.recall(),
        combined_counts.f1()
    );
    if let Some(dir) = output_dir {
        save(Path::new(&dir), &head, head_threshold, combined_threshold, counts);
    }
}

fn prepare(response: Response, subwords: &SubwordAnnotator) -> Prepared {
    let document = Document::new(response.text.clone());
    let view = document.get::<View>().unwrap().clone();
    let mut sent_spans = Spans::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, response.text.len() as u32));
    let sentences = Sentences { spans: sent_spans };
    let (tokens,) = UntypedWordTokenizer::default().annotate((&view, Some(&sentences))).unwrap();
    let subwords = subwords.pieces(&view).unwrap();
    Prepared { response, view, sentences, tokens, subwords, vectors: None }
}

fn embed(docs: &mut [Prepared], vectors: &VectorAnnotator) {
    for chunk in docs.chunks_mut(64) {
        let inputs: Vec<Input> = chunk
            .iter()
            .map(|p| Input { subwords: &p.subwords, sentences: Some(&p.sentences.spans), chunks: None, levels: None })
            .collect();
        let outputs = vectors.embed("pragmatics-csc", &inputs);
        for (p, output) in chunk.iter_mut().zip(outputs) {
            p.vectors = Some(output.unwrap().sentences.unwrap());
        }
    }
}

fn fit(docs: &[&Prepared], vectors: &Arc<VectorAnnotator>) -> ClassifierHead {
    let tags = TagSet::new("pragmatics", &["literal", "sarcastic"]);
    let examples: Vec<(&[f32], turbo_analysis::Tag)> = docs
        .iter()
        .map(|p| {
            let tag = if p.response.sarcastic { turbo_analysis::Tag(1) } else { turbo_analysis::Tag(0) };
            (p.vectors.as_ref().unwrap().row(turbo_analysis::Ix::new(0)), tag)
        })
        .collect();
    ClassifierHead::fit_single_label(vectors.model().clone(), vectors.dim(), tags, &examples, 0.001, 200, 0.1).unwrap()
}

fn labels(docs: &[&Prepared]) -> Vec<bool> {
    docs.iter().map(|p| p.response.sarcastic).collect()
}

fn head_scores(docs: &[&Prepared], head: &ClassifierHead) -> Vec<f64> {
    docs.iter()
        .map(|p| head.score_pragmatics_vector(p.vectors.as_ref().unwrap().row(turbo_analysis::Ix::new(0))).unwrap())
        .collect()
}

fn rule_scores(docs: &[&Prepared]) -> Vec<f64> {
    let rules = PragmaticsAnnotator::new();
    docs.iter()
        .map(|p| rules.annotate((&p.view, &p.sentences, &p.tokens, None, None, None, None)).unwrap().0.score[0])
        .collect()
}

fn combined_scores(docs: &[&Prepared], head: &ClassifierHead) -> Vec<f64> {
    let combined = PragmaticsAnnotator::new().with_vector_scorer(Arc::new(head.clone()));
    docs.iter()
        .map(|p| {
            combined
                .annotate((&p.view, &p.sentences, &p.tokens, None, None, None, Some(p.vectors.as_ref().unwrap())))
                .unwrap()
                .0
                .score[0]
        })
        .collect()
}

/// Held-out evaluations in the frozen configuration, the last two being
/// the constant all-sarcastic and all-literal (majority) baselines.
const EVALUATIONS: [&str; 5] = ["rules", "head", "combined", "all_sarcastic", "all_literal"];

fn save(dir: &Path, head: &ClassifierHead, head_threshold: f64, combined_threshold: f64, counts: [Counts; 5]) {
    std::fs::create_dir_all(dir).unwrap();
    head.save(&dir.join("pragmatics-1.thead")).unwrap();
    let config = serde_json::json!({
        "format": "turbo-pragmatics-1",
        "encoder": "sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "head": "pragmatics-1.thead",
        "dataset": "CoPsyN/CSC@f11c92b8d54686f64cf1d92766cef07201efb986",
        "dataset_notice": "Trained on the Conversational Sarcasm Corpus by Hyewon Jang, Bettina Braun, and Diego Frassinelli. MIT License, Copyright (c) 2024 CoPsyN.",
        "head_threshold": head_threshold,
        "combined_threshold": combined_threshold,
        "rule_threshold": 0.5,
        "training": { "l2_reg": 0.001, "epochs": 200, "learning_rate": 0.1, "split": "scenario groups (context word Jaccard >= 0.4), smallest context id modulo 5, repeated responses in one partition" },
        "combined_path": "PragmaticsAnnotator rule signals plus this embedding head",
        "evaluation": EVALUATIONS.into_iter().zip(counts).map(|(name, c)| (name.to_owned(), metric_json(c))).collect::<serde_json::Map<_, _>>()
    });
    std::fs::write(dir.join("pragmatics-1.json"), serde_json::to_string_pretty(&config).unwrap()).unwrap();
}

fn metric_json(c: Counts) -> serde_json::Value {
    serde_json::json!({
        "tp": c.tp, "fp": c.fp, "tn": c.tn, "fn": c.r#fn,
        "precision": c.precision(), "recall": c.recall(), "f1": c.f1(), "accuracy": c.accuracy()
    })
}
