//! `export --corpus litbank` and `peer --column coref` (#1825): the ten
//! test works out as one CoNLL file each, and a peer's prediction file
//! back in, scored by document against the same works.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_trainer_recipes::eval::Metrics;
use turbo_trainer_recipes::litbank::parse_many;
use turbo_trainer_recipes::receipt::Receipt;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-litbank-peer-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

fn row(doc: &str, tok: usize, word: &str, coref: &str) -> String {
    format!("{doc}\t0\t{tok}\t{word}\tNNP\t_\t_\t_\t_\t_\t_\t_\t{coref}\n")
}

/// A document where `Paris` (cluster 0) and `London` (cluster 1) corefer
/// across three sentences.
fn doc_conll(doc: &str) -> String {
    let mut s = format!("#begin document ({doc}); part 0\n");
    for verb in ["saw", "met", "left"] {
        s += &row(doc, 0, "Paris", "(0)");
        s += &row(doc, 1, verb, "_");
        s += &row(doc, 2, "London", "(1)");
        s += &row(doc, 3, ".", "_");
        s += "\n";
    }
    s + "#end document\n"
}

fn corpus(dir: &Path, count: usize) {
    let conll = dir.join("coref/conll");
    fs::create_dir_all(&conll).unwrap();
    for i in 0..count {
        let name = format!("doc_{i:02}_brat");
        fs::write(conll.join(format!("{name}.conll")), doc_conll(&name)).unwrap();
    }
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).args(args).output().unwrap()
}

#[test]
fn parse_many_reads_a_peers_documents() {
    let two = format!("{}{}", doc_conll("a"), doc_conll("b"));
    let docs = parse_many("p.conll", two.as_bytes()).unwrap();
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[1].id, "b");
    assert_eq!(docs[0].clusters, [vec![(0, 0), (4, 4), (8, 8)], vec![(2, 2), (6, 6), (10, 10)]]);
    let unclosed = two.strip_suffix("#end document\n").unwrap().to_owned();
    assert!(parse_many("p.conll", unclosed.as_bytes()).unwrap_err().0.contains("unclosed"));
    let dup = format!("{}{}", doc_conll("a"), doc_conll("a"));
    assert!(parse_many("p.conll", dup.as_bytes()).unwrap_err().0.contains("twice"));
    assert!(parse_many("p.conll", b"no documents here\n").unwrap_err().0.contains("before the first"));
}

#[test]
fn export_and_peer_coref_score_a_perfect_and_a_singleton_peer() {
    let data = scratch("corpus");
    corpus(&data, 10); // 8 train, 1 dev, 1 test (doc_09_brat)
    let out = scratch("export");
    let o = run(&[
        "export",
        "--corpus",
        "litbank",
        "--unrecorded-corpus",
        "--dir",
        data.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let exported = out.join("doc_09_brat.conll");
    let text = fs::read_to_string(&exported).unwrap();
    assert!(text.starts_with("#begin document (doc_09_brat); part 0\n"));
    assert!(text.contains("doc_09_brat\t0\t0\tParis\t"));
    assert!(text.ends_with("#end document\n"));
    assert!(text.contains("\t_\n"), "the coref column is blank");

    // A peer that returns the gold chains scores 1.0 and beats singletons.
    let preds = scratch("preds");
    fs::create_dir_all(&preds).unwrap();
    fs::write(preds.join("perfect.conll"), doc_conll("doc_09_brat")).unwrap();
    let pout = scratch("peer-out");
    let o = run(&[
        "peer",
        "--corpus",
        "litbank",
        "--unrecorded-corpus",
        "--dir",
        data.to_str().unwrap(),
        "--peer",
        "perfect-0.0",
        "--predictions",
        preds.join("perfect.conll").to_str().unwrap(),
        "--column",
        "coref",
        "--out",
        pout.to_str().unwrap(),
        "--commit",
        "test",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let r: Receipt = serde_json::from_slice(&fs::read(pout.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(r.corpus.id, "litbank");
    assert_eq!(r.peer.as_ref().unwrap().name, "perfect-0.0");
    let Metrics::Coref { conll_f1, .. } = r.eval.model.as_ref().unwrap() else { panic!() };
    assert_eq!(*conll_f1, 1.0);
    let base = r.eval.baseline.score();
    assert!(base < 1.0, "singletons under a perfect peer: {base}");
    assert!(r.eval.comparison.as_ref().unwrap().delta > 0.0);

    // A peer that links nothing has no clusters and scores 0, under the
    // singleton baseline.
    let mut bare = doc_conll("doc_09_brat");
    bare = bare.replace("(0)", "_").replace("(1)", "_");
    fs::write(preds.join("bare.conll"), bare).unwrap();
    let bout = scratch("peer-bare");
    let o = run(&[
        "peer",
        "--corpus",
        "litbank",
        "--unrecorded-corpus",
        "--dir",
        data.to_str().unwrap(),
        "--peer",
        "bare-0.0",
        "--predictions",
        preds.join("bare.conll").to_str().unwrap(),
        "--column",
        "coref",
        "--out",
        bout.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let r: Receipt = serde_json::from_slice(&fs::read(bout.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(r.eval.model.as_ref().unwrap().score(), 0.0);
    assert!(r.eval.baseline.score() > 0.0);

    // A peer file with other words or without a test work is refused.
    let wrong = doc_conll("doc_09_brat").replace("Paris", "London");
    fs::write(preds.join("wrong.conll"), wrong).unwrap();
    let wout = scratch("peer-wrong");
    let o = run(&[
        "peer",
        "--corpus",
        "litbank",
        "--unrecorded-corpus",
        "--dir",
        data.to_str().unwrap(),
        "--peer",
        "w",
        "--predictions",
        preds.join("wrong.conll").to_str().unwrap(),
        "--column",
        "coref",
        "--out",
        wout.to_str().unwrap(),
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("other words"), "{}", String::from_utf8_lossy(&o.stderr));
    let missing = doc_conll("doc_08_brat");
    fs::write(preds.join("missing.conll"), missing).unwrap();
    let mout = scratch("peer-missing");
    let o = run(&[
        "peer",
        "--corpus",
        "litbank",
        "--unrecorded-corpus",
        "--dir",
        data.to_str().unwrap(),
        "--peer",
        "m",
        "--predictions",
        preds.join("missing.conll").to_str().unwrap(),
        "--column",
        "coref",
        "--out",
        mout.to_str().unwrap(),
    ]);
    assert!(!o.status.success());
    let e = String::from_utf8_lossy(&o.stderr);
    assert!(e.contains("no prediction for test work"), "{e}");
    for d in [data, out, preds, pout, bout, wout, mout] {
        let _ = fs::remove_dir_all(d);
    }
}
