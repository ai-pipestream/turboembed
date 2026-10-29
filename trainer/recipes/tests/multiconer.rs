//! The MultiCoNER v2 reader and its pins, on small files written here.
//! With TURBO_TEST_MULTICONER_DIR set to a copy of `multiconer2023/`,
//! the recorded English files are read and counted too.

use std::fs;
use std::path::PathBuf;

use turbo_trainer_recipes::corpus::Task;
use turbo_trainer_recipes::multiconer::{COARSE, coarse, parse, read};

const TEXT: &str = "\n# id a\tdomain=en\neli _ _ B-OtherPER\nlilly _ _ I-OtherPER\nfounded _ _ O\n\n\n# id b\tdomain=en\nparis _ _ B-HumanSettlement\n";

#[test]
fn sentences_are_read_with_their_labels() {
    let got = parse("t", TEXT.as_bytes()).unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].words, ["eli", "lilly", "founded"]);
    assert_eq!(got[0].labels, ["B-OtherPER", "I-OtherPER", "O"]);
    assert_eq!(got[1].labels, ["B-HumanSettlement"]);
}

#[test]
fn malformed_lines_are_refused_with_their_place() {
    for (text, line) in [
        ("# id a\nword _ B-X\n", "t:2"),
        ("# id a\nword _ _ X-Y\n", "t:2"),
        ("# id a\nword _ _ B-\n", "t:2"),
        ("word _ _ O\n", "t:1"),
        ("# id a\nok _ _ O\n x _ _ O\n", "t:3"),
    ] {
        let e = parse("t", text.as_bytes()).unwrap_err();
        assert!(e.0.starts_with(line), "{text:?}: {e}");
    }
    assert!(parse("t", b"# id a\n\xff _ _ O\n").unwrap_err().0.contains("UTF-8"));
}

#[test]
fn the_coarse_groups_cover_33_types_once() {
    let mut all: Vec<&str> = COARSE.iter().flat_map(|(_, f)| f.iter().copied()).collect();
    assert_eq!(all.len(), 33);
    all.sort();
    all.dedup();
    assert_eq!(all.len(), 33);
    assert_eq!(coarse("Medication/Vaccine"), "Medical");
    assert_eq!(coarse("Unlisted"), "Unlisted");
}

#[test]
fn files_that_are_not_the_release_are_refused_unless_asked_for() {
    let dir = std::env::temp_dir().join(format!("turbo-multiconer-{}", std::process::id()));
    fs::create_dir_all(dir.join("EN-English")).unwrap();
    for f in ["en_train", "en_dev", "en_test"] {
        fs::write(dir.join(format!("EN-English/{f}.conll")), TEXT).unwrap();
    }
    let e = read(&dir, false).unwrap_err();
    assert!(e.0.contains("not the recorded release") && e.0.contains("--unrecorded-corpus"), "{e}");
    let c = read(&dir, true).unwrap();
    assert!(!c.record.recorded);
    assert_eq!(c.record.task, Task::Bio);
    assert_eq!((c.record.train.sentences, c.record.train.words), (2, 4));
    assert_eq!(c.labels(), ["O", "B-HumanSettlement", "B-OtherPER", "I-OtherPER"]);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_recorded_release_reads_as_pinned() {
    let Some(dir) = std::env::var_os("TURBO_TEST_MULTICONER_DIR").map(PathBuf::from) else { return };
    let c = read(&dir, false).unwrap();
    assert!(c.record.recorded);
    let n = |x: turbo_trainer_recipes::corpus::Counts| (x.sentences, x.words);
    assert_eq!(n(c.record.train), (16_778, 253_011));
    assert_eq!(n(c.record.dev), (871, 13_323));
    assert_eq!(n(c.record.test), (249_980, 3_773_671));
    assert_eq!(c.labels().len(), 67);
}
