use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_analysis::Annotator;
use turbo_analysis_embed::aspect::{LabeledAbsaDoc, LearnedAspectSentiment};
use turbo_analysis_embed::{
    ClassifierHead, EncoderOptions, ScoringHead, SubwordAnnotator, VectorAnnotator, VectorOptions,
};
use turbo_analysis_layers::Subwords;
use turbo_doccat::sentiment::aspect::RuleAspectSentiment;

use super::corpus::{CorpusDoc, load};
use super::metrics::{self, Metrics};
use super::release;

struct Prepared {
    corpus: CorpusDoc,
    subwords: Subwords,
}

pub fn run() {
    let root = PathBuf::from(std::env::var_os("TURBO_MASC_ROOT").expect("TURBO_MASC_ROOT names MASC's data directory"));
    let bundle =
        PathBuf::from(std::env::var_os("TURBO_MINILM_BUNDLE").expect("TURBO_MINILM_BUNDLE names the MiniLM bundle"));
    let corpus = load(&root);
    assert!(corpus.len() >= 20, "expected at least 20 usable MASC documents, got {}", corpus.len());
    let release_dir = std::env::var_os("TURBO_ASPECT_HEAD");
    assert!(
        release_dir.is_some() != std::env::var_os("TURBO_ASPECT_OUT").is_some(),
        "choose release verification OR explicit training output"
    );
    if release_dir.is_some() {
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
