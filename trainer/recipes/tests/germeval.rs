//! Authored German-style fixtures; no corpus text is packaged.
use std::fs;
use std::path::PathBuf;
use turbo_trainer_recipes::{head, read_corpus};

#[test]
fn german_recipe_keeps_both_layers_and_compound_subtypes() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("germeval-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let sample = "# source\n1\tDie\tO\tO\n2\tBerliner\tB-ORG\tB-LOCderiv\n3\tGruppe\tI-ORG\tO\n4\t.\tO\tO\t\n\n";
    for f in ["NER-de-train.tsv", "NER-de-dev.tsv", "NER-de-test.tsv"] {
        fs::write(root.join(f), sample).unwrap();
    }
    let outer = read_corpus("germeval2014-outer", &root, true).expect("German outer recipe");
    let inner = read_corpus("germeval2014-inner", &root, true).expect("German inner recipe");
    assert_eq!(outer.splits.train[0].words, ["Die", "Berliner", "Gruppe", "."]);
    assert_eq!(outer.splits.train[0].labels, ["O", "B-ORG", "I-ORG", "O"]);
    assert_eq!(inner.splits.train[0].labels, ["O", "B-LOCderiv", "O", "O"]);
    assert_eq!(outer.record.spdx, "CC-BY-4.0");
    assert_eq!(outer.record.files, inner.record.files);
    assert_ne!(head("germeval2014-outer", outer.record.task), head("germeval2014-inner", inner.record.task));
    assert!(!outer.record.recorded);
    assert!(read_corpus("germeval2014-outer", &root, false).is_err());
    for bad in [
        sample.replace("3\tGruppe", "9\tGruppe"),
        sample.replace("B-LOCderiv", "B-NEW"),
        sample.replace("B-LOCderiv", "I-LOCderiv"),
        sample.replace("I-ORG", "I-PER"),
        sample.replace("\tO\tO\t\n", "\tO\tO\tnot-empty\n"),
        sample.replace("2\tBerliner", "2\t"),
        sample.replace("3\tGruppe", "# interrupted\n3\tGruppe"),
    ] {
        fs::write(root.join("NER-de-test.tsv"), bad).unwrap();
        assert!(read_corpus("germeval2014-outer", &root, true).is_err());
        assert!(read_corpus("germeval2014-inner", &root, true).is_err());
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn fixed_schema_covers_unobserved_training_labels_without_reading_held_out_labels() {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("germeval-schema-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    for file in ["NER-de-train.tsv", "NER-de-dev.tsv", "NER-de-test.tsv"] {
        fs::write(root.join(file), "1\tann\tB-PER\tO\n2\tlee\tI-PER\tO\n\n").unwrap();
    }
    let corpus = read_corpus("germeval2014-outer", &root, true).unwrap();
    fs::remove_dir_all(root).unwrap();
    let labels = corpus.labels();
    assert_eq!(labels.len(), 25, "the declared 12-type schema has O and both BIO labels for each type");
    assert_eq!(labels[0], "O");
    assert!(labels.iter().any(|label| label == "I-ORGderiv"));
    let mut changed = corpus.clone();
    changed.splits.train.clear();
    changed.splits.dev.clear();
    changed.splits.test.clear();
    assert_eq!(labels, changed.labels(), "fixed vocabulary does not depend on any observed split");
}
