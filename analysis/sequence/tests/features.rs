//! The features of every generator the test descriptors use, and of
//! `DefaultPOSContextGenerator`, against the JVM; and the errors of
//! malformed descriptors (the goldens of `FeatureGolden.java` and
//! `DescriptorErrorGolden.java` in `analysis/goldens-tool`).

mod common;

use std::sync::Arc;

use common::{DirResources, escaped, featuregen, layer, lossless_escaped, sentences, strs, u32s};
use turbo_maxent::Dictionary;
use turbo_sequence::featuregen::descriptor;
use turbo_sequence::postag::DefaultPosContext;
use turbo_sequence::{Features, PatternMode};

fn diff(what: &str, k: usize, i: usize, expected: &[String], actual: &[String]) -> Option<String> {
    (expected != actual)
        .then(|| format!("{what} sentence {k} token {i}:\n  expected {expected:?}\n  actual   {actual:?}"))
}

/// A features golden: its sentences, and each token's features, escaped
/// as `Features::escaped` writes them.
struct Expected {
    sentences: Vec<common::Sentence>,
    features: Vec<Vec<Vec<String>>>,
}

fn read(run: &str) -> Expected {
    let g = common::read("features", run, "sequence", "sentences");
    let sentences = sentences(&g);
    let mut features: Vec<Vec<Vec<String>>> = sentences.iter().map(|s| vec![Vec::new(); s.tokens.len()]).collect();
    let l = layer(&g, "turbo:features");
    let (sentence, token) = (u32s(l, "sentence"), u32s(l, "token"));
    for (row, feature) in strs(l, "feature").iter().enumerate() {
        features[sentence[row] as usize][token[row] as usize].push(lossless_escaped(feature));
    }
    Expected { sentences, features }
}

fn check_descriptor(name: &str) {
    let expected = read(name);
    let xml = std::fs::read(featuregen().join(format!("{name}.xml"))).unwrap();
    let resources = DirResources(featuregen());
    let mut g = descriptor::build(&xml, Some(&resources), PatternMode::Utf16Units).unwrap();
    g.clear_adaptive_data();
    let mut out = Features::new();
    let mut failures = Vec::new();
    for (k, s) in expected.sentences.iter().enumerate() {
        let tokens: Vec<&str> = s.tokens.iter().map(String::as_str).collect();
        let outcomes: Vec<&str> = s.outcomes.iter().map(String::as_str).collect();
        g.start_sequence();
        for (i, expected) in expected.features[k].iter().enumerate() {
            out.clear();
            g.create_features(&mut out, &tokens, i, &outcomes);
            failures.extend(diff(name, k, i, expected, &escaped(&out)));
        }
        // Java updates the adaptive data after each sentence.
        g.update_adaptive_data(&tokens, &outcomes);
    }
    assert!(failures.is_empty(), "{} tokens differ:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn pos_default_descriptor() {
    check_descriptor("pos-default");
}

#[test]
fn ner_default_descriptor() {
    check_descriptor("ner-default");
}

#[test]
fn every_factory() {
    check_descriptor("everything");
}

#[test]
fn nested_and_cached_generators() {
    check_descriptor("nesting");
}

/// The copies of OpenNLP's default descriptors the crate builds in are the
/// files the goldens were made from.
#[test]
fn built_in_descriptors_are_the_golden_ones() {
    let read = |name: &str| std::fs::read_to_string(featuregen().join(name)).unwrap();
    assert_eq!(turbo_sequence::postag::DEFAULT_DESCRIPTOR, read("pos-default.xml"));
    assert_eq!(turbo_sequence::namefind::DEFAULT_DESCRIPTOR, read("ner-default.xml"));
}

fn check_default_pos(run: &str, dictionary: Option<Arc<Dictionary>>) {
    let expected = read(run);
    let mut cg = DefaultPosContext::new(dictionary);
    let mut out = Features::new();
    let mut failures = Vec::new();
    for (k, s) in expected.sentences.iter().enumerate() {
        let tokens: Vec<&str> = s.tokens.iter().map(String::as_str).collect();
        let outcomes: Vec<&str> = s.outcomes.iter().map(String::as_str).collect();
        for (i, expected) in expected.features[k].iter().enumerate() {
            out.clear();
            cg.features(&mut out, i, &tokens, &outcomes);
            failures.extend(diff(run, k, i, expected, &escaped(&out)));
        }
    }
    assert!(failures.is_empty(), "{} tokens differ:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn default_pos_context() {
    check_default_pos("default-pos-context", None);
}

#[test]
fn default_pos_context_with_dictionary() {
    let xml = std::fs::read(featuregen().join("names.dictionary")).unwrap();
    let dictionary = Arc::new(Dictionary::read(&xml).unwrap());
    check_default_pos("default-pos-context+names-dictionary", Some(dictionary));
}

struct NoResources;

impl descriptor::Resources for NoResources {
    fn resource(&self, _: &str) -> turbo_sequence::Result<Option<std::borrow::Cow<'_, [u8]>>> {
        Ok(None)
    }
}

/// Every malformed descriptor fails, with Java's message where the
/// descriptor itself is at fault.
#[test]
fn descriptor_errors() {
    let mut failures = Vec::new();
    let goldens = common::read_run("features", "errors", "descriptor-errors");
    for (case, g) in &goldens {
        let xml = g.input_text(turbo_goldens::data_dir()).unwrap();
        let Some(error) = &g.header.error else {
            failures.push(format!("{case}: Java built it"));
            continue;
        };
        // Java was given a resource manager that has nothing.
        match descriptor::build(xml.as_bytes(), Some(&NoResources), PatternMode::Utf16Units) {
            Ok(_) => failures.push(format!("{case}: built, Java threw {} {}", error.class, error.message)),
            Err(e) => {
                let actual = e.to_string();
                if !actual.contains(&error.message) {
                    failures.push(format!("{case}: expected {:?}, got {actual:?}", error.message));
                }
            }
        }
    }
    assert_eq!(goldens.len(), 16);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
