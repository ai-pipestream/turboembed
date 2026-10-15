//! How often the rule splitter agrees with OpenNLP's maxent sentence
//! detector, for the languages that have a published model (German,
//! English, Portuguese), on texts in that language:
//!
//! - the golden inputs (`analysis/goldens/data/inputs`): for English the
//!   `en` and `edge` sets, which the English model's goldens also run on;
//!   for German `multi/de.txt` and `edge/german.txt`; for Portuguese
//!   `multi/pt.txt`;
//! - the texts of the conformance suites in that language (their expected
//!   sentences play no part here).
//!
//! Agreement is counted three ways: texts split identically, sentences
//! with the same span in both, and the boundaries (the start of every
//! sentence after the first) the two place alike, as precision and recall
//! of the rule splitter's boundaries against the model's. The numbers are
//! asserted so that a change to either detector shows here, and
//! docs/segmentation.md reports them. Without the models this test says so
//! and passes.

mod common;

use std::collections::BTreeSet;

use turbo_analysis::Span;
use turbo_goldens::data_dir;
use turbo_goldens::sentences::sentence_suite;
use turbo_maxent::ModelFile;
use turbo_sentdetect::{RuleSplitter, SentenceDetector, SentenceModel, WhitespaceMode};

#[derive(Debug, Default, PartialEq, Eq)]
struct Agreement {
    texts: usize,
    identical: usize,
    maxent_sentences: usize,
    rule_sentences: usize,
    same_spans: usize,
    maxent_boundaries: usize,
    rule_boundaries: usize,
    same_boundaries: usize,
}

impl Agreement {
    fn add(&mut self, maxent: &[Span], rules: &[Span]) {
        self.texts += 1;
        self.identical += usize::from(maxent == rules);
        self.maxent_sentences += maxent.len();
        self.rule_sentences += rules.len();
        let (m, r): (BTreeSet<_>, BTreeSet<_>) = (maxent.iter().collect(), rules.iter().collect());
        self.same_spans += m.intersection(&r).count();
        let starts = |s: &[Span]| s.iter().skip(1).map(|s| s.start).collect::<BTreeSet<_>>();
        let (m, r) = (starts(maxent), starts(rules));
        self.maxent_boundaries += m.len();
        self.rule_boundaries += r.len();
        self.same_boundaries += m.intersection(&r).count();
    }

    fn report(&self, what: &str) {
        let pct = |a: usize, b: usize| if b == 0 { 100.0 } else { 100.0 * a as f64 / b as f64 };
        eprintln!(
            "{what}: {} texts, {} identical ({:.1}%); sentences maxent {} rules {} same {}; boundaries maxent {} rules {} \
             same {} (precision {:.1}%, recall {:.1}%)",
            self.texts,
            self.identical,
            pct(self.identical, self.texts),
            self.maxent_sentences,
            self.rule_sentences,
            self.same_spans,
            self.maxent_boundaries,
            self.rule_boundaries,
            self.same_boundaries,
            pct(self.same_boundaries, self.rule_boundaries),
            pct(self.same_boundaries, self.maxent_boundaries),
        );
    }
}

fn maxent(file: &str) -> Option<SentenceDetector> {
    let path = common::published_model(file)?;
    let model = SentenceModel::open(ModelFile::read(&path).unwrap()).unwrap();
    Some(SentenceDetector::new(model, WhitespaceMode::Unicode))
}

fn agreement(lang: &str, detector: &SentenceDetector, texts: impl IntoIterator<Item = String>) -> Agreement {
    let rules = RuleSplitter::new(lang);
    let mut a = Agreement::default();
    let (mut m, mut r) = (Vec::new(), Vec::new());
    for text in texts {
        m.clear();
        r.clear();
        detector.detect(&text, &mut m);
        rules.split(&text, &mut r);
        let m: Vec<Span> = m.iter().map(|&(s, _)| s).collect();
        a.add(&m, &r);
    }
    a
}

fn inputs(paths: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for p in paths {
        let path = data_dir().join("inputs").join(p);
        if path.is_dir() {
            let mut files: Vec<_> = std::fs::read_dir(&path).unwrap().map(|e| e.unwrap().path()).collect();
            files.sort();
            out.extend(files.iter().map(|f| std::fs::read_to_string(f).unwrap()));
        } else {
            out.push(std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())));
        }
    }
    out
}

fn conformance_texts(lang: &str) -> Vec<String> {
    let mut out = Vec::new();
    for suite in ["pragmatic-segmenter", "pysbd"] {
        out.extend(sentence_suite(suite).into_iter().filter(|c| c.lang == lang).map(|c| c.text));
    }
    out
}

#[test]
fn english() {
    let Some(d) = maxent("opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin") else { return };
    let golden = agreement("en", &d, inputs(&["en", "edge"]));
    golden.report("en, golden inputs");
    let conformance = agreement("en", &d, conformance_texts("en"));
    conformance.report("en, conformance texts");
    assert_eq!(
        golden,
        Agreement {
            texts: 45,
            identical: 31,
            maxent_sentences: 125,
            rule_sentences: 123,
            same_spans: 93,
            maxent_boundaries: 82,
            rule_boundaries: 80,
            same_boundaries: 69
        }
    );
    assert_eq!(
        conformance,
        Agreement {
            texts: 482,
            identical: 255,
            maxent_sentences: 3253,
            rule_sentences: 2679,
            same_spans: 1770,
            maxent_boundaries: 2771,
            rule_boundaries: 2197,
            same_boundaries: 1950
        }
    );
}

#[test]
fn german() {
    let Some(d) = maxent("opennlp-de-ud-gsd-sentence-1.3-2.5.4.bin") else { return };
    let golden = agreement("de", &d, inputs(&["multi/de.txt", "edge/german.txt"]));
    golden.report("de, golden inputs");
    let conformance = agreement("de", &d, conformance_texts("de"));
    conformance.report("de, conformance texts");
    assert_eq!(
        golden,
        Agreement {
            texts: 2,
            identical: 1,
            maxent_sentences: 7,
            rule_sentences: 6,
            same_spans: 5,
            maxent_boundaries: 5,
            rule_boundaries: 4,
            same_boundaries: 4
        }
    );
    assert_eq!(
        conformance,
        Agreement {
            texts: 72,
            identical: 42,
            maxent_sentences: 194,
            rule_sentences: 179,
            same_spans: 100,
            maxent_boundaries: 122,
            rule_boundaries: 107,
            same_boundaries: 76
        }
    );
}

#[test]
fn portuguese() {
    let Some(d) = maxent("opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin") else { return };
    let golden = agreement("pt", &d, inputs(&["multi/pt.txt"]));
    golden.report("pt, golden inputs");
    assert_eq!(
        golden,
        Agreement {
            texts: 1,
            identical: 1,
            maxent_sentences: 5,
            rule_sentences: 5,
            same_spans: 5,
            maxent_boundaries: 4,
            rule_boundaries: 4,
            same_boundaries: 4
        }
    );
    assert!(conformance_texts("pt").is_empty());
}
