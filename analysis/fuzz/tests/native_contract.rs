//! Native fuzz admission must reject retired archives before resource lookup.

use std::path::PathBuf;

use turbo_fuzz::{Resources, Unavailable, runs};

#[test]
fn retired_runs_are_rejected_before_resource_lookup() {
    let absent = PathBuf::from("this-resource-directory-must-not-exist");
    assert!(!absent.exists());
    let resources = Resources { models: absent.clone(), data: absent.clone(), workspace: absent };
    for (annotator, run) in [
        ("sentdetect", "en-abbrev"),
        ("sentdetect", "arbitrary-model"),
        ("tokenize", "tokenizer-abbreviations"),
        ("tokenize", "arbitrary-model"),
        ("postag", "pos-fixture"),
        ("lemmatize", "lemma-fixture-rust"),
        ("lemmatize-dictionary", "lemmatizer-dictionary"),
        ("chunk", "chunker-fixture"),
        ("langdetect", "langdetect-fixture-rust"),
    ] {
        assert!(
            matches!(runs::open(annotator, run, &resources), Err(Unavailable::NotCovered(_))),
            "{annotator}/{run} reached resource lookup instead of retirement admission"
        );
    }
}

#[test]
fn native_registry_keeps_each_boundary_family() {
    let runs: Vec<_> = runs::native_runs().collect();
    let unique: std::collections::BTreeSet<_> = runs.iter().copied().collect();
    assert_eq!(runs.len(), unique.len());
    for (family, expected) in [
        ("sentdetect", 3),
        ("tokenize", 5),
        // The normalizers left the fuzz registry; their boundary contracts are
        // `analysis/normalize/tests/native_projection.rs`.
        ("normalize", 0),
        ("pii", 10),
        ("numbers", 2),
        ("money", 2),
        ("money-convert", 2),
        ("quantity", 2),
        ("temporal", 2),
        ("unigram", 3),
        ("lattice", 6),
        ("subwords", 1),
    ] {
        assert_eq!(runs.iter().filter(|(a, _)| *a == family).count(), expected, "{family}");
    }
    assert!(runs.iter().all(|(a, r)| !runs::is_retired(a, r)));
}

#[test]
fn native_unicode_boundaries() {
    use turbo_analysis::View;
    use turbo_goldens::invariants::{check_document, check_spans};
    let resources = Resources::new("this-resource-directory-must-not-exist");
    let text = "😀 Zoë  e\u{301} 東京.\n\nEmail a@example.org costs 12 EUR.";
    for (family, name) in runs::native_runs().filter(|(family, _)| {
        matches!(*family, "sentdetect" | "tokenize" | "pii" | "numbers" | "money" | "quantity" | "temporal")
    }) {
        let run = runs::open(family, name, &resources).unwrap();
        let doc = run.annotate(text, None).unwrap();
        let (layers, violations) = check_document(&doc);
        assert!(layers > 0, "{family}/{name}: no native layer checked");
        assert!(violations.is_empty(), "{family}/{name}: {violations:?}");
        let view = doc.get::<View>().unwrap();
        let starts: Vec<_> = view.text().char_indices().map(|(i, _)| i as u32).collect();
        let ends: Vec<_> = starts.iter().skip(1).copied().chain([view.text().len() as u32]).collect();
        let violations = check_spans(&doc, "source scalars", view.id(), &starts, &ends);
        assert!(violations.is_empty(), "{family}/{name}: {violations:?}");
    }
    assert!(matches!(runs::open("normalize", "whitespace", &resources), Err(Unavailable::NotCovered(_))));
}
