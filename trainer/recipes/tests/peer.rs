//! A peer's tags: the test words out as CoNLL-U, a peer's CoNLL-U back
//! in, word for word, and `export` / `peer` end to end on a small
//! treebank in UD EWT's layout.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_trainer_recipes::corpus::Sentence;
use turbo_trainer_recipes::peer::{Column, apply, conllu, form, parse_map, read};
use turbo_trainer_recipes::receipt::Receipt;
use turbo_trainer_recipes::ud::EN_EWT;

fn sentence(words: &[&str], labels: &[&str]) -> Sentence {
    Sentence {
        words: words.iter().map(|w| (*w).to_owned()).collect(),
        labels: labels.iter().map(|l| (*l).to_owned()).collect(),
    }
}

fn gold() -> Vec<Sentence> {
    vec![sentence(&["Dogs", "bark", "."], &["NNS", "VBP", "."]), sentence(&["intervie=\nw", "D. C."], &["NN", "NNP"])]
}

#[test]
fn the_words_go_out_with_no_labels_and_come_back_with_the_peers() {
    let text = conllu(&gold());
    assert_eq!(
        text,
        "# sent_id = 1\n1\tDogs\t_\t_\t_\t_\t_\t_\t_\t_\n2\tbark\t_\t_\t_\t_\t_\t_\t_\t_\n3\t.\t_\t_\t_\t_\t_\t_\t_\t_\n\n\
         # sent_id = 2\n1\tintervie= w\t_\t_\t_\t_\t_\t_\t_\t_\n2\tD. C.\t_\t_\t_\t_\t_\t_\t_\t_\n\n"
    );
    assert_eq!(form("a\tb\r\nc"), "a b  c");
    let peer = "# sent_id = 1\n1\tDogs\t_\tNOUN\tNNS\t_\t_\t_\t_\t_\n2\tbark\t_\tVERB\tVB\t_\t_\t_\t_\t_\n\
                3\t.\t_\tPUNCT\t.\t_\t_\t_\t_\t_\n\n\
                1\tintervie= w\t_\tNOUN\tNN\t_\t_\t_\t_\t_\n2\tD. C.\t_\tPROPN\tNNP\t_\t_\t_\t_\t_\n\n";
    assert_eq!(read("p", peer.as_bytes(), Column::Xpos, &gold()).unwrap(), [vec!["NNS", "VB", "."], vec!["NN", "NNP"]]);
    assert_eq!(read("p", peer.as_bytes(), Column::Upos, &gold()).unwrap()[1], ["NOUN", "PROPN"]);
}

#[test]
fn other_words_or_sentences_are_refused() {
    let one = "1\tDogs\t_\t_\tNNS\t_\t_\t_\t_\t_\n2\tbark\t_\t_\tVB\t_\t_\t_\t_\t_\n3\t.\t_\t_\t.\t_\t_\t_\t_\t_\n\n";
    let e = read("p", one.as_bytes(), Column::Xpos, &gold()).unwrap_err();
    assert!(e.0.contains("1 sentences, the split 2"), "{e}");
    let other = one.replace("bark", "barks");
    let e = read("p", other.as_bytes(), Column::Xpos, &gold()).unwrap_err();
    assert!(e.0.contains("sentence 1 has words"), "{e}");
    let three = format!("{one}{one}{one}");
    let e = read("p", three.as_bytes(), Column::Xpos, &gold()[..1]).unwrap_err();
    assert!(e.0.contains("more than the 1 sentences"), "{e}");
    assert!("lemma".parse::<Column>().is_err());
}

#[test]
fn a_map_rewrites_tag_spellings_only_as_named() {
    let map = parse_map("-LRB-=(,HYPH=-").unwrap();
    let mut tags = vec![vec!["-LRB-".to_owned(), "NN".to_owned(), "HYPH".to_owned()]];
    apply(&map, &mut tags);
    assert_eq!(tags, [["(", "NN", "-"]]);
    for bad in ["A", "A=", "=B", "A=B,A=C"] {
        assert!(parse_map(bad).is_err(), "{bad}");
    }
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-peer-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).args(args).output().unwrap()
}

#[test]
fn export_and_peer_end_to_end() {
    let data = scratch("data");
    fs::create_dir_all(&data).unwrap();
    let text =
        "# sent_id = a\n1\tDogs\tdog\tNOUN\tNNS\t_\t2\tnsubj\t_\t_\n2\tbark\tbark\tVERB\tVBP\t_\t0\troot\t_\t_\n\n";
    for pin in [EN_EWT.train, EN_EWT.dev, EN_EWT.test] {
        fs::write(data.join(pin.path), text).unwrap();
    }
    let d = data.to_str().unwrap();
    let words = data.join("words.conllu");
    let o = run(&[
        "export",
        "--corpus",
        "ud-en-ewt-xpos",
        "--dir",
        d,
        "--out",
        words.to_str().unwrap(),
        "--unrecorded-corpus",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        fs::read_to_string(&words).unwrap(),
        "# sent_id = 1\n1\tDogs\t_\t_\t_\t_\t_\t_\t_\t_\n2\tbark\t_\t_\t_\t_\t_\t_\t_\t_\n\n"
    );
    let again = run(&[
        "export",
        "--corpus",
        "ud-en-ewt-xpos",
        "--dir",
        d,
        "--out",
        words.to_str().unwrap(),
        "--unrecorded-corpus",
    ]);
    assert!(!again.status.success(), "export never writes over a file");

    let pred = data.join("peer.conllu");
    fs::write(&pred, "1\tDogs\t_\tNOUN\tNNS\t_\t_\t_\t_\t_\n2\tbark\t_\tVERB\tVB\t_\t_\t_\t_\t_\n\n").unwrap();
    let out = scratch("out");
    let args = [
        "peer",
        "--corpus",
        "ud-en-ewt-xpos",
        "--dir",
        d,
        "--peer",
        "toy-1",
        "--predictions",
        pred.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--unrecorded-corpus",
        "--commit",
        "test",
        "--map",
        "VB=VBP",
    ];
    let o = run(&args);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let r: Receipt = serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
    let p = r.peer.as_ref().unwrap();
    assert_eq!((p.name.as_str(), p.column), ("toy-1", Column::Xpos));
    assert_eq!(p.predictions.sha256, turbo::bundle::sha256_hex(&fs::read(&pred).unwrap()));
    assert!(r.head.is_none() && r.encoder.is_none());
    assert_eq!(p.map.get("VB").map(String::as_str), Some("VBP"));
    // The peer's VB for "bark" is read as VBP, the gold tag.
    assert_eq!(r.eval.model.as_ref().unwrap().score(), 1.0);
    assert_eq!(r.eval.baseline.score(), 1.0);
    for p in [data, out] {
        fs::remove_dir_all(p).unwrap();
    }
}

#[test]
fn a_peer_run_that_fails_leaves_no_directory() {
    let out = scratch("bio");
    let o = run(&[
        "peer",
        "--corpus",
        "multiconer2-en",
        "--dir",
        "x",
        "--peer",
        "p",
        "--predictions",
        "x",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(!o.status.success());
    assert!(!Path::new(&out).exists());
}
