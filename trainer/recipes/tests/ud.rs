//! The UD treebank reader, its pins and its tag corpora, on small files
//! written here. With TURBO_TEST_UD_EWT_DIR set to a copy of
//! UD_English-EWT at r2.18, the recorded files are read and counted too.

use std::fs;
use std::path::PathBuf;

use turbo_trainer_recipes::corpus::Task;
use turbo_trainer_recipes::run::{HeadOptions, spec};
use turbo_trainer_recipes::ud::{Column, EN_EWT, find, parse, read, read_treebank};
use turbo_trainer_recipes::{corpora, read_corpus};

const TEXT: &str = "# sent_id = a-1\n# text = Vamos al mar.\n\
1\tVamos\tir\tVERB\tVBP\tMood=Ind\t0\troot\t_\t_\n\
2-3\tal\t_\t_\t_\t_\t_\t_\t_\t_\n\
2\ta\ta\tADP\tIN\t_\t4\tcase\t_\t_\n\
3\tel\tel\tDET\tDT\tDefinite=Def\t4\tdet\t_\t_\n\
3.1\tx\tx\tX\t_\t_\t_\t_\t4:dep\t_\n\
4\tmar\tmar\tNOUN\tNN\tNumber=Sing\t1\tobl\t_\tSpaceAfter=No\n\
5\t.\t.\tPUNCT\t.\t_\t1\tpunct\t_\t_\n\
\n\
# sent_id = a-2\n\
1\tYes\tyes\tINTJ\tUH\t_\t0\troot\t_\t_\n\n";

#[test]
fn syntactic_words_are_read_with_every_column() {
    let got = parse("t", TEXT.as_bytes()).unwrap();
    assert_eq!(got.len(), 2);
    let s = &got[0];
    assert_eq!(s.words, ["Vamos", "a", "el", "mar", "."], "the multiword line and the empty node are not words");
    assert_eq!(s.lemmas, ["ir", "a", "el", "mar", "."]);
    assert_eq!(s.upos, ["VERB", "ADP", "DET", "NOUN", "PUNCT"]);
    assert_eq!(s.xpos, ["VBP", "IN", "DT", "NN", "."]);
    assert_eq!(s.feats, ["Mood=Ind", "_", "Definite=Def", "Number=Sing", "_"]);
    assert_eq!(s.heads, [0, 4, 4, 1, 1]);
    assert_eq!(s.rels, ["root", "case", "det", "obl", "punct"]);
    assert_eq!(got[1].words, ["Yes"]);
}

#[test]
fn a_head_outside_the_sentence_is_refused_with_its_place() {
    for head in ["2", "x", "-1"] {
        let text = format!("# sent_id = b\n1\tNo\tno\tINTJ\tUH\t_\t{head}\troot\t_\t_\n");
        let e = parse("t", text.as_bytes()).unwrap_err();
        assert!(e.0.starts_with("t:2:") && e.0.contains("head"), "{head}: {e}");
    }
    assert!(parse("t", b"1\t\xff\n").unwrap_err().0.contains("UTF-8"));
}

#[test]
fn word_ids_out_of_order_repeated_zero_or_not_numbers_are_refused() {
    let row = |id: &str, form: &str, head: &str| format!("{id}\t{form}\t{form}\tX\tX\t_\t{head}\tdep\t_\t_\n");
    // Reordered: C, A, B with heads that only mean the right tree by ID.
    let reordered = row("3", "C", "0") + &row("1", "A", "3") + &row("2", "B", "1");
    for (case, text) in [
        ("reordered", reordered),
        ("repeated", row("1", "A", "0") + &row("1", "B", "1")),
        ("zero", row("0", "A", "0") + &row("1", "B", "0")),
        ("not a number", row("1", "A", "0") + &row("b", "B", "1")),
    ] {
        let e = parse("t", format!("# sent_id = c\n{text}\n").as_bytes()).err();
        let e = e.unwrap_or_else(|| panic!("{case} was read"));
        assert!(e.0.starts_with("t:") && e.0.contains("word ID"), "{case}: {e}");
    }
    // Multiword tokens and empty nodes around ordered IDs still read.
    assert_eq!(parse("t", TEXT.as_bytes()).unwrap()[0].heads, [0, 4, 4, 1, 1]);
}

#[test]
fn each_column_is_a_corpus_id_with_its_own_head() {
    assert_eq!(find("ud-en-ewt-upos"), Some((EN_EWT, Column::Upos)));
    assert_eq!(find("ud-en-ewt-feats"), Some((EN_EWT, Column::Feats)));
    assert_eq!(find("ud-en-ewt-lemma"), None);
    assert_eq!(find("ud-en-ewtupos"), None);
    let ids = corpora();
    for id in ["masc-pos", "multiconer2-en", "ud-en-ewt-upos", "ud-en-ewt-xpos", "ud-en-ewt-feats"] {
        assert!(ids.iter().any(|i| i == id), "{id} in {ids:?}");
    }
    let e = read_corpus("ud-en-ewt-deps", std::path::Path::new("."), false).unwrap_err();
    assert!(e.0.contains("unknown corpus") && e.0.contains("ud-en-ewt-upos"), "{e}");
}

#[test]
fn files_that_are_not_the_release_are_refused_unless_asked_for() {
    let dir = std::env::temp_dir().join(format!("turbo-ud-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    for pin in [EN_EWT.train, EN_EWT.dev, EN_EWT.test] {
        fs::write(dir.join(pin.path), TEXT).unwrap();
    }
    let e = read(&EN_EWT, Column::Upos, &dir, false).unwrap_err();
    assert!(e.0.contains("not the recorded release") && e.0.contains("--unrecorded-corpus"), "{e}");
    let c = read(&EN_EWT, Column::Feats, &dir, true).unwrap();
    assert!(!c.record.recorded);
    assert_eq!(c.record.id, "ud-en-ewt-feats");
    assert_eq!(c.record.task, Task::Tags);
    assert_eq!(c.record.spdx, "CC-BY-SA-4.0");
    assert_eq!((c.record.train.sentences, c.record.train.words), (2, 6));
    assert_eq!(c.splits.dev[0].labels, ["Mood=Ind", "_", "Definite=Def", "Number=Sing", "_"]);
    assert_eq!(c.record.files.len(), 3);

    let s = spec(&c, &HeadOptions::default()).unwrap();
    assert_eq!((s.name.as_str(), s.layer.as_str()), ("feats", "turboembed:feats"));
    assert_eq!(s.license_class, Some(turbo::manifest::LicenseClass::ShareAlike));
    let u = spec(&read(&EN_EWT, Column::Upos, &dir, true).unwrap(), &HeadOptions::default()).unwrap();
    assert_eq!((u.name.as_str(), u.layer.as_str()), ("upos", "turboembed:pos"));
    assert_eq!(u.labels, ["ADP", "DET", "INTJ", "NOUN", "PUNCT", "VERB"]);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_recorded_release_reads_with_its_counts() {
    let Some(dir) = std::env::var_os("TURBO_TEST_UD_EWT_DIR").map(PathBuf::from) else { return };
    let (splits, files, recorded) = read_treebank(&EN_EWT, &dir, false).unwrap();
    assert!(recorded);
    assert_eq!(files.len(), 3);
    let count = |p: &[turbo_trainer_recipes::ud::UdSentence]| (p.len(), p.iter().map(|s| s.words.len()).sum::<usize>());
    assert_eq!(count(&splits.train), TRAIN_COUNTS);
    assert_eq!(count(&splits.dev), DEV_COUNTS);
    assert_eq!(count(&splits.test), TEST_COUNTS);
    let c = read(&EN_EWT, Column::Upos, &dir, false).unwrap();
    assert_eq!(c.labels().len(), 17, "the UPOS tag set");
}

const TRAIN_COUNTS: (usize, usize) = (12_544, 204_578);
const DEV_COUNTS: (usize, usize) = (2_001, 25_148);
const TEST_COUNTS: (usize, usize) = (2_077, 25_094);
