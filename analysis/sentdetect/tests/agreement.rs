//! Native rule segmentation counts on the retained input and external suites.
//! These are the rule-side corpus assertions formerly paired with a model.

use turbo_goldens::{data_dir, sentences::sentence_suite};
use turbo_sentdetect::RuleSplitter;

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

fn counts(lang: &str, texts: &[String]) -> (usize, usize, usize) {
    let splitter = RuleSplitter::new(lang);
    let mut sentences = 0;
    let mut boundaries = 0;
    for text in texts {
        let mut spans = Vec::new();
        splitter.split(text, &mut spans);
        sentences += spans.len();
        boundaries += spans.len().saturating_sub(1);
    }
    (texts.len(), sentences, boundaries)
}

#[test]
fn english() {
    assert_eq!(counts("en", &inputs(&["en", "edge"])), (45, 119, 76));
    assert_eq!(counts("en", &conformance_texts("en")), (482, 2649, 2167));
}

#[test]
fn german() {
    assert_eq!(counts("de", &inputs(&["multi/de.txt", "edge/german.txt"])), (2, 6, 4));
    assert_eq!(counts("de", &conformance_texts("de")), (72, 175, 103));
}

#[test]
fn portuguese() {
    assert_eq!(counts("pt", &inputs(&["multi/pt.txt"])), (1, 5, 4));
    assert!(conformance_texts("pt").is_empty());
}
