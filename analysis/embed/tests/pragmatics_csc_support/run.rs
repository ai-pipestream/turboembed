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
    let responses = load(&csv);
    assert!(responses.len() >= 1_000, "expected aggregated CSC responses, got {}", responses.len());
    let subwords = SubwordAnnotator::open(&bundle).unwrap();
    let vectors = Arc::new(VectorAnnotator::open(&bundle, &VectorOptions::default()).unwrap());
    let mut prepared: Vec<Prepared> = responses.into_iter().map(|r| prepare(r, &subwords)).collect();
    embed(&mut prepared, &vectors);
    let train = &prepared[..];
    let dev: Vec<&Prepared> = prepared.iter().filter(|p| p.response.context % 5 == 1).collect();
    let test: Vec<&Prepared> = prepared.iter().filter(|p| p.response.context % 5 == 0).collect();
    let fit_rows: Vec<&Prepared> = train.iter().filter(|p| p.response.context % 5 > 1).collect();
    assert!(!fit_rows.is_empty() && !dev.is_empty() && !test.is_empty());
    let head = if let Some(dir) = std::env::var_os("TURBO_PRAGMATICS_HEAD") {
        ClassifierHead::load(&Path::new(&dir).join("pragmatics-1.thead")).unwrap()
    } else {
        fit(&fit_rows, &vectors)
    };
    let (head_threshold, combined_threshold) = if let Some(dir) = std::env::var_os("TURBO_PRAGMATICS_HEAD") {
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(Path::new(&dir).join("pragmatics-1.json")).unwrap()).unwrap();
        (config["head_threshold"].as_f64().unwrap(), config["combined_threshold"].as_f64().unwrap())
    } else {
        (best_threshold(&head_scores(&dev, &head), &labels(&dev)), 0.5)
    };
    let combined_threshold = if std::env::var_os("TURBO_PRAGMATICS_HEAD").is_some() {
        combined_threshold
    } else {
        best_threshold(&combined_scores(&dev, &head), &labels(&dev))
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
    report("rules", 0.5, rule_counts);
    report("head", head_threshold, head_counts);
    report("combined", combined_threshold, combined_counts);
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
    if let Some(dir) = std::env::var_os("TURBO_PRAGMATICS_OUT") {
        save(Path::new(&dir), &head, head_threshold, combined_threshold, [rule_counts, head_counts, combined_counts]);
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

fn save(dir: &Path, head: &ClassifierHead, head_threshold: f64, combined_threshold: f64, counts: [Counts; 3]) {
    std::fs::create_dir_all(dir).unwrap();
    head.save(&dir.join("pragmatics-1.thead")).unwrap();
    let [rules, head_score, combined] = counts;
    let config = serde_json::json!({
        "format": "turbo-pragmatics-1",
        "encoder": "sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "head": "pragmatics-1.thead",
        "dataset": "CoPsyN/CSC@f11c92b8d54686f64cf1d92766cef07201efb986",
        "head_threshold": head_threshold,
        "combined_threshold": combined_threshold,
        "rule_threshold": 0.5,
        "training": { "l2_reg": 0.001, "epochs": 200, "learning_rate": 0.1, "split": "context modulo 5" },
        "evaluation": {
            "rules": metric_json(rules),
            "head": metric_json(head_score),
            "combined": metric_json(combined),
            "default_combination": "PragmaticsAnnotator rule signals plus this embedding head"
        }
    });
    std::fs::write(dir.join("pragmatics-1.json"), serde_json::to_string_pretty(&config).unwrap()).unwrap();
}

fn metric_json(c: Counts) -> serde_json::Value {
    serde_json::json!({
        "tp": c.tp, "fp": c.fp, "tn": c.tn, "fn": c.r#fn,
        "precision": c.precision(), "recall": c.recall(), "f1": c.f1()
    })
}
