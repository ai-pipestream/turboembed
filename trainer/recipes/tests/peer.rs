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
use turbo_trainer_recipes::ud_banks::ES_ANCORA;

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
    assert!("lemmas".parse::<Column>().is_err());
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

#[test]
fn feats_lemmas_and_parses_are_read_from_their_columns() {
    let g = vec![sentence(&["Dogs", "bark"], &["", ""])];
    let peer = "1\tDogs\tdog\tNOUN\tNNS\tNumber=Plur\t2\tnsubj\t_\t_\n\
                2\tbark\tbark\tVERB\tVBP\t_\t0\troot\t_\t_\n\n";
    assert_eq!(read("p", peer.as_bytes(), Column::Feats, &g).unwrap(), [["Number=Plur", "_"]]);
    assert_eq!(read("p", peer.as_bytes(), Column::Lemma, &g).unwrap(), [["dog", "bark"]]);
    assert_eq!(read("p", peer.as_bytes(), Column::Deps, &g).unwrap(), [["2 nsubj", "0 root"]]);
    assert!(Column::Deps.treebank() && Column::Lemma.treebank() && !Column::Feats.treebank());
}

/// A treebank's lemmas and parses scored end to end: the peer gets one
/// lemma and one head wrong, and the receipt carries LAS, UAS and the
/// treebank's baselines.
#[test]
fn lemma_and_deps_peers_score_against_the_treebank() {
    let data = scratch("tb");
    fs::create_dir_all(&data).unwrap();
    let text = "1\tPerros\tperro\tNOUN\t_\t_\t2\tnsubj\t_\t_\n2\tladran\tladrar\tVERB\t_\t_\t0\troot\t_\t_\n\
                3\tfuerte\tfuerte\tADV\t_\t_\t2\tadvmod\t_\t_\n\n";
    for pin in [ES_ANCORA.train, ES_ANCORA.dev, ES_ANCORA.test] {
        let path = data.join(pin.path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
    let d = data.to_str().unwrap();
    let words = data.join("words.conllu");
    let o = run(&[
        "export",
        "--corpus",
        "ud-es-ancora",
        "--dir",
        d,
        "--out",
        words.to_str().unwrap(),
        "--unrecorded-corpus",
    ]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(fs::read_to_string(&words).unwrap().contains("2\tladran\t_\t_"));

    let pred = data.join("peer.conllu");
    fs::write(
        &pred,
        "1\tPerros\tperro\t_\t_\t_\t2\tnsubj\t_\t_\n2\tladran\tladran\t_\t_\t_\t0\troot\t_\t_\n\
         3\tfuerte\tfuerte\t_\t_\t_\t1\tadvmod\t_\t_\n\n",
    )
    .unwrap();
    let score = |column: &str| {
        let out = scratch(column);
        let o = run(&[
            "peer",
            "--corpus",
            "ud-es-ancora",
            "--dir",
            d,
            "--peer",
            "toy",
            "--predictions",
            pred.to_str().unwrap(),
            "--column",
            column,
            "--out",
            out.to_str().unwrap(),
            "--unrecorded-corpus",
        ]);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let r: Receipt = serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
        fs::remove_dir_all(out).unwrap();
        r
    };
    let lemma = score("lemma");
    assert_eq!(lemma.corpus.id, "ud-es-ancora-lemma");
    assert!((lemma.eval.model.as_ref().unwrap().score() - 2.0 / 3.0).abs() < 1e-12);
    assert!(lemma.eval.uas.is_none());
    let deps = score("deps");
    assert_eq!(deps.corpus.id, "ud-es-ancora-deps");
    assert!((deps.eval.model.as_ref().unwrap().score() - 2.0 / 3.0).abs() < 1e-12);
    let uas = deps.eval.uas.unwrap();
    assert!((uas.model - 2.0 / 3.0).abs() < 1e-12);
    // The training words' own attachments, so the baseline is right.
    assert_eq!((deps.eval.baseline.score(), uas.baseline), (1.0, 1.0));

    let out = scratch("tag-as-deps");
    let o = run(&[
        "peer",
        "--corpus",
        "ud-es-ancora-upos",
        "--dir",
        d,
        "--peer",
        "toy",
        "--predictions",
        pred.to_str().unwrap(),
        "--column",
        "deps",
        "--out",
        out.to_str().unwrap(),
        "--unrecorded-corpus",
    ]);
    assert!(!o.status.success(), "deps takes a treebank id, not a column corpus");
    assert!(!out.exists());
    let o = run(&[
        "peer",
        "--corpus",
        "ud-es-ancora",
        "--dir",
        d,
        "--peer",
        "toy",
        "--predictions",
        pred.to_str().unwrap(),
        "--column",
        "feats",
        "--out",
        out.to_str().unwrap(),
        "--unrecorded-corpus",
    ]);
    assert!(String::from_utf8_lossy(&o.stderr).contains("reads the tag corpus ud-es-ancora-feats"));
    assert!(!out.exists());
    fs::remove_dir_all(data).unwrap();
}
