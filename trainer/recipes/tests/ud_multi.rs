use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_trainer_recipes::receipt::Receipt;
use turbo_trainer_recipes::run::{HeadOptions, spec};
use turbo_trainer_recipes::ud::{CA_ANCORA, Column, ES_ANCORA, Treebank};
use turbo_trainer_recipes::ud_multi::{find_column, read, sample_counts};

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-ud-multi-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn sentence(bank: &str, index: usize) -> String {
    let word = format!("{bank}{index}");
    format!("# sent_id = {bank}-{index}\n1\t{word}\t{word}\tNOUN\tNN\tNumber=Sing\t0\troot\t_\t_\n\n")
}

fn write_bank(dir: &Path, bank: &Treebank, train: usize) {
    for (pin, count) in [(bank.train, train), (bank.dev, 1), (bank.test, 1)] {
        let text = (0..count).map(|i| sentence(bank.id, i)).collect::<String>();
        fs::write(dir.join(pin.path), text).unwrap();
    }
}

fn corpora(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    write_bank(dir, &ES_ANCORA, 3);
    write_bank(dir, &CA_ANCORA, 1);
}

#[test]
fn temperature_counts_preserve_the_total_and_balance_small_banks() {
    assert_eq!(sample_counts(&[100, 10, 1], 1.0).unwrap(), [100, 10, 1]);
    assert_eq!(sample_counts(&[100, 10, 1], 0.5).unwrap(), [78, 25, 8]);
    assert_eq!(sample_counts(&[100, 10, 1], 0.0).unwrap(), [37, 37, 37]);
    assert_eq!(sample_counts(&[3, 1], 0.0).unwrap(), [2, 2]);
    for sample in [-0.1, 1.1, f64::NAN] {
        assert!(sample_counts(&[3, 1], sample).is_err());
    }
    assert!(sample_counts(&[], 0.5).is_err());
    assert!(sample_counts(&[3, 0], 0.5).is_err());
}

#[test]
fn multi_treebank_read_prefixes_xpos_and_only_resamples_training() {
    let dir = scratch("read");
    corpora(&dir);
    let multi = read("ud-es-ancora,ud-ca-ancora", Column::Xpos, &dir, true, 0.0).unwrap();
    assert_eq!(multi.languages, ["es", "ca"]);
    assert_eq!(multi.train_languages, [Some(0), Some(0), Some(1), Some(1)]);
    assert_eq!(multi.dev_languages, [Some(0), Some(1)]);
    assert_eq!(multi.test_languages, [Some(0), Some(1)]);
    let corpus = multi.corpus;
    assert_eq!(corpus.record.id, "ud-es-ancora-xpos,ud-ca-ancora-xpos");
    assert_eq!(find_column(&corpus.record.id), Some(Column::Xpos));
    assert_eq!(corpus.record.files.len(), 6);
    assert_eq!(corpus.record.train.sentences, 4);
    assert_eq!(corpus.record.dev.sentences, 2);
    assert_eq!(corpus.record.test.sentences, 2);
    assert!(corpus.record.split.contains("--sample 0"));
    let prefixes: Vec<&str> = corpus.splits.train.iter().map(|s| s.labels[0].as_str()).collect();
    assert_eq!(prefixes, ["ud-es-ancora:NN", "ud-es-ancora:NN", "ud-ca-ancora:NN", "ud-ca-ancora:NN"]);
    assert_eq!(corpus.splits.dev[0].labels[0], "ud-es-ancora:NN");
    assert_eq!(corpus.splits.dev[1].labels[0], "ud-ca-ancora:NN");
    assert_eq!(corpus.splits.test[0].labels[0], "ud-es-ancora:NN");
    assert_eq!(corpus.splits.test[1].labels[0], "ud-ca-ancora:NN");
    let head = spec(&corpus, &HeadOptions::default()).unwrap();
    assert_eq!(head.name, "xpos");
    assert_eq!(head.labels, ["ud-ca-ancora:NN", "ud-es-ancora:NN"]);
    let error = read("ud-es-ancora,ud-es-ancora", Column::Upos, &dir, true, 1.0).unwrap_err();
    assert!(error.0.contains("listed twice"), "{error}");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn the_cli_trains_a_multi_treebank_head_and_records_sampling() {
    let dir = scratch("cli-data");
    let out = scratch("cli-out");
    corpora(&dir);
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let output = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
        .args([
            "train",
            "--corpus",
            "ud-es-ancora,ud-ca-ancora",
            "--dir",
            dir.to_str().unwrap(),
            "--bundle",
            bundle.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--unrecorded-corpus",
            "--column",
            "xpos",
            "--sample",
            "0",
            "--epochs",
            "1",
            "--batch",
            "2",
            "--threads",
            "1",
            "--commit",
            "test",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let receipt: Receipt = serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt.config.unwrap().sample, Some(0.0));
    assert_eq!(receipt.corpus.train.sentences, 4);
    assert_eq!(receipt.corpus.dev.sentences, 2);
    assert_eq!(receipt.corpus.test.sentences, 2);
    let head = receipt.head.unwrap();
    assert_eq!(head.name, "xpos");
    assert_eq!(head.labels, ["ud-ca-ancora:NN", "ud-es-ancora:NN"]);
    let language = head.language.unwrap();
    assert_eq!(language.languages, ["es", "ca"]);
    assert_eq!(language.dim, 16);
    fs::remove_dir_all(dir).unwrap();
    fs::remove_dir_all(out).unwrap();
}

#[test]
fn the_cli_trains_a_multi_treebank_parser() {
    let dir = scratch("parse-data");
    let out = scratch("parse-out");
    corpora(&dir);
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let output = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
        .args([
            "train-parse",
            "--corpus",
            "ud-es-ancora,ud-ca-ancora",
            "--dir",
            dir.to_str().unwrap(),
            "--bundle",
            bundle.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--unrecorded-corpus",
            "--sample",
            "0",
            "--epochs",
            "1",
            "--batch",
            "2",
            "--threads",
            "1",
            "--commit",
            "test",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let receipt: Receipt = serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt.config.unwrap().sample, Some(0.0));
    assert_eq!(receipt.corpus.train.sentences, 4);
    assert_eq!(receipt.corpus.dev.sentences, 2);
    assert_eq!(receipt.corpus.test.sentences, 2);
    let head = receipt.head.unwrap();
    assert_eq!(head.name, "deps");
    assert_eq!(head.kind, turbo::manifest::HeadKind::BiaffineParser);
    let language = head.language.unwrap();
    assert_eq!(language.languages, ["ud-es-ancora", "ud-ca-ancora"]);
    assert_eq!(language.dim, 16);
    fs::remove_dir_all(dir).unwrap();
    fs::remove_dir_all(out).unwrap();
}
