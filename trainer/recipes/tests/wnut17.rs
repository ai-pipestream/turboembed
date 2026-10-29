//! Native WNUT recipe contract, using project-authored text only.
use std::fs;
use turbo::manifest::{Decode, LicenseClass};
use turbo_trainer_recipes::{
    read_corpus,
    run::{HeadOptions, spec},
};

#[test]
fn wnut_is_a_separate_permissive_crf_corpus_with_pinned_splits() {
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("wnut-recipe-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    for (file, text) in [
        ("wnut17train.conll", "Ánn\tB-person\nLee\tI-person\nvisits\tO\nOslo\tB-location\n\n"),
        ("emerging.dev.conll", "Bob\tB-person\nvisits\tO\nOslo\tB-location\n\n"),
        ("emerging.test.annotated", "Zoe\tB-person\nvisits\tO\nOslo\tB-location\n\n"),
    ] {
        fs::write(dir.join(file), text).unwrap();
    }
    let corpus = read_corpus("wnut17", &dir, true).expect("WNUT recipe registered");
    assert!(!corpus.record.recorded);
    assert_eq!(corpus.record.spdx, "CC-BY-4.0");
    assert_eq!(corpus.record.files.len(), 3);
    assert_eq!(corpus.splits.train[0].words, ["Ánn", "Lee", "visits", "Oslo"]);
    assert_eq!(corpus.splits.dev[0].words[0], "Bob");
    assert_eq!(corpus.splits.test[0].words[0], "Zoe");
    let head = spec(&corpus, &HeadOptions { crf: true, ..HeadOptions::default() }).unwrap();
    assert_eq!(head.decode, Decode::Crf);
    assert_eq!(head.license_class, Some(LicenseClass::Permissive));
    let error = read_corpus("wnut17", &dir, false).unwrap_err().to_string();
    assert!(error.contains("recorded release"), "{error}");
    for invalid in [
        "word\tB-unknown\n",
        "word\tB-person,B-location\n",
        "word\tI-person\n",
        "a\tB-person\nb\tI-location\n",
        "\tO\n",
    ] {
        fs::write(dir.join("emerging.dev.conll"), invalid).unwrap();
        assert!(read_corpus("wnut17", &dir, true).is_err(), "{invalid}");
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn publisher_tab_only_separators_do_not_become_empty_tokens() {
    let rows = turbo_trainer_recipes::wnut17::parse("fixture", b"Ann\tB-person\n\t\nBob\tB-person\n\t\n").unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].words, ["Ann"]);
    assert_eq!(rows[1].words, ["Bob"]);
}
