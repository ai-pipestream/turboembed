//! The features of every generator the test descriptors use, and of
//! `DefaultPOSContextGenerator`, against the JVM; and the errors of
//! malformed descriptors.

mod common;

use std::sync::Arc;

use common::{DirResources, body, escaped, field, read_dump, unescape};
use turbo_maxent::Dictionary;
use turbo_sequence::featuregen::descriptor;
use turbo_sequence::postag::DefaultPosContext;
use turbo_sequence::{Features, PatternMode};

fn diff(what: &str, k: usize, i: usize, expected: &[String], actual: &[String]) -> Option<String> {
    (expected != actual)
        .then(|| format!("{what} sentence {k} token {i}:\n  expected {expected:?}\n  actual   {actual:?}"))
}

fn check_descriptor(name: &str) {
    let (file, sentences) = read_dump(&format!("expected/features-{name}.txt"));
    assert_eq!(file, format!("{name}.xml"));
    let xml = std::fs::read(common::golden().join("descriptors").join(&file)).unwrap();
    let resources = DirResources(common::golden().join("resources"));
    let mut g = descriptor::build(&xml, Some(&resources), PatternMode::Utf16Units).unwrap();
    g.clear_adaptive_data();
    let mut out = Features::new();
    let mut failures = Vec::new();
    for (k, s) in sentences.iter().enumerate() {
        let tokens: Vec<&str> = s.tokens.iter().map(String::as_str).collect();
        let outcomes: Vec<&str> = s.outcomes.iter().map(String::as_str).collect();
        g.start_sequence();
        for (i, expected) in s.features.iter().enumerate() {
            out.clear();
            g.create_features(&mut out, &tokens, i, &outcomes);
            failures.extend(diff(name, k, i, expected, &escaped(&out)));
        }
        assert!(s.update);
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
/// files the dumps were made from.
#[test]
fn built_in_descriptors_are_the_golden_ones() {
    let read = |name: &str| std::fs::read_to_string(common::golden().join("descriptors").join(name)).unwrap();
    assert_eq!(turbo_sequence::postag::DEFAULT_DESCRIPTOR, read("pos-default.xml"));
    assert_eq!(turbo_sequence::namefind::DEFAULT_DESCRIPTOR, read("ner-default.xml"));
}

fn check_default_pos(file: &str, dictionary: Option<Arc<Dictionary>>) {
    let (_, sentences) = read_dump(file);
    let mut cg = DefaultPosContext::new(dictionary);
    let mut out = Features::new();
    let mut failures = Vec::new();
    for (k, s) in sentences.iter().enumerate() {
        let tokens: Vec<&str> = s.tokens.iter().map(String::as_str).collect();
        let outcomes: Vec<&str> = s.outcomes.iter().map(String::as_str).collect();
        for (i, expected) in s.features.iter().enumerate() {
            out.clear();
            cg.features(&mut out, i, &tokens, &outcomes);
            failures.extend(diff(file, k, i, expected, &escaped(&out)));
        }
    }
    assert!(failures.is_empty(), "{} tokens differ:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn default_pos_context() {
    check_default_pos("expected/features-default-pos.txt", None);
}

#[test]
fn default_pos_context_with_dictionary() {
    let xml = std::fs::read(common::golden().join("resources/names.dictionary")).unwrap();
    let dictionary = Arc::new(Dictionary::read(&xml).unwrap());
    check_default_pos("expected/features-default-pos-dictionary.txt", Some(dictionary));
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
    let text = common::read("expected/descriptor-errors.txt");
    let mut lines = body(&text, "descriptor-errors");
    let mut failures = Vec::new();
    let mut count = 0;
    while let Some(line) = lines.next() {
        let case = field(line, "case").to_string();
        let xml = unescape(field(lines.next().unwrap(), "xml"));
        let result = unescape(field(lines.next().unwrap(), "result"));
        let (_, expected) = result.split_once(": ").unwrap();
        count += 1;
        // Java was given a resource manager that has nothing.
        match descriptor::build(xml.as_bytes(), Some(&NoResources), PatternMode::Utf16Units) {
            Ok(_) => failures.push(format!("{case}: built, Java threw {result}")),
            Err(e) => {
                let actual = e.to_string();
                if !actual.contains(expected) {
                    failures.push(format!("{case}: expected {expected:?}, got {actual:?}"));
                }
            }
        }
    }
    assert_eq!(count, 16);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
