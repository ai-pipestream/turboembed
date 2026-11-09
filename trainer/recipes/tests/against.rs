//! `--against <run dir>` on `train` and `train-coref` (#1828): a run
//! scores an earlier run's head on the same test split, records that
//! head's score as its own run gave it, the paired bootstrap of the new
//! head against it and the files read, and refuses a head that would read
//! other features before training starts.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_trainer_recipes::receipt::Receipt;

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-against-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).args(args).output().unwrap()
}

fn receipt(out: &Path) -> Receipt {
    serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap()
}

/// A tagging corpus where a name follows "met" and "in" names a place.
fn tag_corpus(dir: &Path) {
    let names = ["ann", "bob", "eve", "max"];
    let places = ["paris", "rome", "oslo"];
    let mut text = String::new();
    for i in 0..48 {
        let (n, p) = (names[i % names.len()], places[i % places.len()]);
        text += &format!(
            "# id {i}\tdomain=en\nthey _ _ O\nmet _ _ O\n{n} _ _ B-OtherPER\nin _ _ O\n{p} _ _ B-HumanSettlement\n\n"
        );
    }
    fs::create_dir_all(dir.join("EN-English")).unwrap();
    for f in ["en_train", "en_dev", "en_test"] {
        fs::write(dir.join(format!("EN-English/{f}.conll")), &text).unwrap();
    }
}

/// LitBank-shaped documents where identical word forms corefer.
fn coref_corpus(dir: &Path) {
    let conll = dir.join("coref/conll");
    fs::create_dir_all(&conll).unwrap();
    for i in 0..6 {
        let doc = format!("doc_{i:02}_brat");
        let row = |tok: usize, word: &str, coref: &str| {
            format!("{doc}\t0\t{tok}\t{word}\tNNP\t_\t_\t_\t_\t_\t_\t_\t{coref}\n")
        };
        let mut s = format!("#begin document ({doc}); part 0\n");
        for verb in ["saw", "met", "left"] {
            s += &(row(0, "Paris", "(0)") + &row(1, verb, "_") + &row(2, "London", "(1)") + &row(3, ".", "_") + "\n");
        }
        s += "#end document\n";
        fs::write(conll.join(format!("{doc}.conll")), s).unwrap();
    }
}

#[test]
fn a_tagger_run_scores_an_earlier_head_beside_its_own() {
    let data = scratch("tag-data");
    tag_corpus(&data);
    let bundle = tiny();
    let train = |out: &Path, extra: &[&str]| {
        let mut args = vec![
            "train",
            "--corpus",
            "multiconer2-en",
            "--dir",
            data.to_str().unwrap(),
            "--bundle",
            bundle.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--unrecorded-corpus",
            "--crf",
            "--epochs",
            "4",
            "--batch",
            "8",
            "--lr",
            "0.01",
            "--commit",
            "test",
        ];
        args.extend_from_slice(extra);
        run(&args)
    };
    let first = scratch("tag-hash");
    let o = train(&first, &["--hash-embed", "4", "--hash-rows", "16,8,8,8"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let r = receipt(&first);
    let plain = scratch("tag-plain");
    let o = train(&plain, &["--against", first.to_str().unwrap()]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let p = receipt(&plain);
    let vs = p.eval.against.as_ref().unwrap();
    assert!(vs.name.ends_with("-tag-hash (Crf)"), "{}", vs.name);
    assert_eq!(&vs.baseline, r.eval.model.as_ref().unwrap(), "the prior head scores as its own run did");
    let (mine, theirs) = (p.eval.model.as_ref().unwrap().score(), r.eval.model.as_ref().unwrap().score());
    assert_eq!((vs.comparison.model, vs.comparison.baseline), (mine, theirs));
    let names: Vec<&str> = vs.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(names, ["head.json", "head.safetensors", "receipt.json"]);
    let refused = scratch("tag-refused");
    let o = train(&refused, &["--pooling", "mean", "--against", first.to_str().unwrap()]);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success() && err.contains("word pooling"), "{err}");
    assert!(!refused.exists());
    for d in [data, first, plain] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn a_coref_run_scores_an_earlier_ranker_beside_its_own() {
    let data = scratch("coref-data");
    coref_corpus(&data);
    let bundle = tiny();
    let train = |out: &Path, extra: &[&str]| {
        let mut args = vec![
            "train-coref",
            "--corpus",
            "litbank",
            "--dir",
            data.to_str().unwrap(),
            "--bundle",
            bundle.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--unrecorded-corpus",
            "--epochs",
            "4",
            "--batch",
            "2",
            "--lr",
            "0.02",
            "--commit",
            "test",
        ];
        args.extend_from_slice(extra);
        run(&args)
    };
    let first = scratch("coref-first");
    let shape = ["--mlp", "16", "--max-antecedents", "16", "--max-words", "8", "--width-dim", "4"];
    let o = train(&first, &shape);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let r = receipt(&first);
    let wider = scratch("coref-wider");
    let against = ["--against", first.to_str().unwrap()];
    let o = train(
        &wider,
        &[&["--max-words", "6", "--width-dim", "8", "--mlp", "24", "--max-antecedents", "8"][..], &against].concat(),
    );
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let w = receipt(&wider);
    let vs = w.eval.against.as_ref().unwrap();
    assert!(vs.name.ends_with("-coref-first (max words 8, width 4, mlp 16, antecedents 16)"), "{}", vs.name);
    assert_eq!(&vs.baseline, r.eval.model.as_ref().unwrap(), "the prior head scores as its own run did");
    let (mine, theirs) = (w.eval.model.as_ref().unwrap().score(), r.eval.model.as_ref().unwrap().score());
    assert_eq!((vs.comparison.model, vs.comparison.baseline), (mine, theirs));
    let names: Vec<&str> = vs.files.iter().map(|f| f.path.as_str()).collect();
    assert_eq!(names, ["head.json", "head.safetensors", "receipt.json"]);
    let refused = scratch("coref-refused");
    let o = train(&refused, &[&["--pooling", "mean"][..], &against].concat());
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!o.status.success() && err.contains("word pooling"), "{err}");
    assert!(!refused.exists());
    for d in [data, first, wider] {
        fs::remove_dir_all(d).unwrap();
    }
}
