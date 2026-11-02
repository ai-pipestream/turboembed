//! The token split head on a UD treebank: the reader that keeps
//! multiword token lines, its labels and baseline, and `train-split`
//! end to end on the small BERT in testdata and a small treebank
//! written here, with the head it writes read back as the core reads
//! it and its words decoded through the core's splitter.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use turbo::head::TokenSplitter;
use turbo::manifest::{HeadKind, HeadSpec};
use turbo_trainer_recipes::receipt::Receipt;
use turbo_trainer_recipes::split_labels::{candidates, choose};
use turbo_trainer_recipes::ud::EN_EWT;
use turbo_trainer_recipes::ud_split::{
    TokenSentence, baseline, gold_label, labelled, mine_labels, parse_tokens, sentences,
};

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-ud-split-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

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
fn tokens_are_read_with_their_words() {
    let got = parse_tokens("t", TEXT.as_bytes()).unwrap();
    assert_eq!(got.len(), 2);
    let s = &got[0];
    assert_eq!(s.tokens, ["Vamos", "al", "mar", "."], "the multiword line is a token, the empty node is not");
    assert_eq!(s.words, [vec!["Vamos"], vec!["a", "el"], vec!["mar"], vec!["."]]);
    assert_eq!(got[1].tokens, ["Yes"]);
    assert_eq!(got[1].words, [vec!["Yes"]]);
}

#[test]
fn ranges_and_ids_out_of_place_are_refused_with_their_place() {
    let row = |id: &str, form: &str| format!("{id}\t{form}\t{form}\tX\tX\t_\t0\tdep\t_\t_\n");
    for (case, text, want) in [
        ("unclosed", row("1-2", "al") + &row("1", "a") + "\n", "ends inside a multiword token"),
        ("inside", row("1-2", "al") + &row("2-3", "x"), "opens inside a multiword token"),
        ("misplaced", row("2-3", "al"), "where a token at 1 is due"),
        ("one word range", row("1-1", "a"), "not a range over two or more words"),
        ("out of order", row("1", "a") + &row("1", "b"), "word ID"),
    ] {
        let e = parse_tokens("t", text.as_bytes()).unwrap_err();
        assert!(e.0.starts_with("t:") && e.0.contains(want), "{case}: {e}");
    }
    assert!(parse_tokens("t", b"1\t\xff\n").unwrap_err().0.contains("UTF-8"));
    assert!(parse_tokens("t", b"1\n").unwrap_err().0.contains("no form column"));
}

fn token_sentence(tokens: &[(&str, &[&str])]) -> TokenSentence {
    TokenSentence {
        tokens: tokens.iter().map(|(t, _)| (*t).to_owned()).collect(),
        words: tokens.iter().map(|(_, ws)| ws.iter().map(|w| (*w).to_owned()).collect()).collect(),
    }
}

/// The words a label gives a token, through the core's own parser and
/// edit trees: the independent reading of what a label means.
fn applied(label: &str, token: &str) -> Vec<String> {
    turbo::head::split::parse(label).unwrap().iter().map(|t| t.apply(token).unwrap()).collect()
}

#[test]
fn labels_are_mined_by_frequency_and_give_the_words_back() {
    assert_eq!(gold_label("mar", &["mar".to_owned()]), "_");
    let train = [
        token_sentence(&[("Vamos", &["Vamos"]), ("del", &["de", "el"]), ("al", &["a", "el"])]),
        token_sentence(&[("del", &["de", "el"]), ("della", &["di", "la"])]),
        token_sentence(&[("del", &["de", "el"]), ("al", &["a", "el"])]),
    ];
    let labels = mine_labels(&train, &choose(&train), 2);
    assert_eq!(labels.len(), 3, "`della` comes up once, under the frequency: {labels:?}");
    assert_eq!(labels[0], "_");
    assert_eq!(applied(&labels[1], "del"), ["de", "el"], "the commonest split first");
    assert_eq!(applied(&labels[2], "al"), ["a", "el"]);
    assert_eq!(gold_label("del", &["de".to_owned(), "el".to_owned()]), "de\tel", "the gold is the words");
}

/// A clitic is cut from its verb by its own length, so the verbs that
/// take it share one label, which serves a verb training never saw;
/// `del` and `al`, which rewrite, keep named trees.
#[test]
fn a_clitic_split_takes_the_label_that_serves_the_most_tokens() {
    let w = |ws: &[&str]| ws.iter().map(|w| (*w).to_owned()).collect::<Vec<_>>();
    let c = candidates("hacerse", &w(&["hacer", "se"]));
    assert!(c.contains(&"(h 5 ()) (t 2 ())".to_owned()), "{c:?}");
    assert!(c.contains(&"(m 0 2 () (s 'se' '')) (t 2 ())".to_owned()), "{c:?}");
    assert!(c.contains(&"(m 0 2 () (s 'se' '')) (m 5 0 (s 'hacer' '') ())".to_owned()), "{c:?}");
    assert!(candidates("mar", &w(&["mar"])).is_empty());
    assert!(candidates("x", &w(&["a"; 9])).is_empty());
    let train = [
        token_sentence(&[("hacerse", &["hacer", "se"]), ("ponerse", &["poner", "se"]), ("del", &["de", "el"])]),
        token_sentence(&[("irse", &["ir", "se"]), ("quedarse", &["quedar", "se"]), ("al", &["a", "el"])]),
        token_sentence(&[("darle", &["dar", "le"]), ("verle", &["ver", "le"]), ("del", &["de", "el"])]),
    ];
    let chosen = choose(&train);
    let se = "(m 0 2 () (s 'se' '')) (t 2 ())";
    for t in ["hacerse", "ponerse", "irse", "quedarse"] {
        assert_eq!(chosen[&(t.to_owned(), w(&[&t[..t.len() - 2], "se"]))], se, "{t}");
    }
    // `del` shares its cuts with `irse` (two first, two last); `al`
    // rewrites, so its mined tree stands, over a cut that serves no
    // other token.
    assert_eq!(chosen[&("del".to_owned(), w(&["de", "el"]))], "(h 2 ()) (t 2 ())");
    assert_eq!(chosen[&("al".to_owned(), w(&["a", "el"]))], "(m 0 1 () (s 'l' '')) (m 1 0 (s 'a' 'e') ())");
    let labels = mine_labels(&train, &chosen, 2);
    let le = "(m 0 2 () (s 'le' '')) (t 2 ())";
    assert_eq!(labels[..2], ["_", se], "four tokens take the `se` label: {labels:?}");
    assert_eq!(labels.len(), 4, "`del` twice, `le` twice, `al` once: {labels:?}");
    assert!(labels.contains(&le.to_owned()) && labels.contains(&"(h 2 ()) (t 2 ())".to_owned()), "{labels:?}");
    assert_eq!(applied(se, "verse"), ["ver", "se"], "a verb training never saw");
    assert_eq!(applied(se, "sentirse"), ["sentir", "se"]);
    // Dev sentences take the head's labels: a split training never saw
    // takes the head label that gives its words, one the head lacks
    // stays its gold words.
    let dev = [token_sentence(&[("verse", &["ver", "se"]), ("al", &["a", "el"]), ("della", &["di", "la"])])];
    let got = labelled(&dev, &chosen, &labels);
    assert_eq!(got[0].labels, [se, "a\tel", "di\tla"], "`al` came up once, under the frequency");
}

#[test]
fn the_baseline_keeps_an_unseen_token_whole() {
    let train = [
        token_sentence(&[("del", &["de", "el"]), ("x", &["x", "a"])]),
        token_sentence(&[("del", &["de", "el"]), ("x", &["x", "b"])]),
    ];
    let test = [token_sentence(&[("del", &["de", "el"]), ("never", &["never"]), ("x", &["x", "a"])])];
    let got = baseline(&train, &test);
    assert_eq!(got[0][0], "de\tel");
    assert_eq!(got[0][1], "_", "an unseen token keeps whole");
    assert_eq!(got[0][2], "x\ta", "a tie goes to the smaller label");
    let gold = sentences(&test);
    assert_eq!(gold[0].words, ["del", "never", "x"]);
    assert_eq!(gold[0].labels[1], "_");
}

/// 48 sentences of one shape: a verb, a multiword token (`al` or
/// `del`), a noun and a full stop.
fn treebank() -> String {
    let (verbs, nouns) = (["vamos", "viene", "habla"], ["mar", "sol", "rio"]);
    let mut text = String::new();
    for i in 0..48 {
        let (v, n) = (verbs[i % 3], nouns[i % 3]);
        let (token, first) = if i % 2 == 0 { ("al", "a") } else { ("del", "de") };
        text += &format!(
            "# sent_id = {i}\n1\t{v}\t{v}\tVERB\tV\t_\t0\troot\t_\t_\n2-3\t{token}\t_\t_\t_\t_\t_\t_\t_\t_\n\
             2\t{first}\t{first}\tADP\tIN\t_\t4\tcase\t_\t_\n3\tel\tel\tDET\tDT\t_\t4\tdet\t_\t_\n\
             4\t{n}\t{n}\tNOUN\tNN\t_\t1\tobl\t_\t_\n5\t.\t.\tPUNCT\t.\t_\t1\tpunct\t_\t_\n\n"
        );
    }
    text
}

fn write_treebank(dir: &Path) {
    fs::create_dir_all(dir).unwrap();
    for pin in [EN_EWT.train, EN_EWT.dev, EN_EWT.test] {
        fs::write(dir.join(pin.path), treebank()).unwrap();
    }
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).args(args).output().unwrap()
}

fn train(data: &Path, out: &Path) -> Receipt {
    let bundle = tiny();
    let args = [
        "train-split",
        "--corpus",
        "ud-en-ewt",
        "--dir",
        data.to_str().unwrap(),
        "--bundle",
        bundle.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
        "--unrecorded-corpus",
        "--epochs",
        "3",
        "--batch",
        "8",
        "--lr",
        "0.01",
        "--commit",
        "test",
    ];
    let o = run(&args);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap()
}

/// The tensors of a safetensors file, by name.
fn read_safetensors(bytes: &[u8]) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data = &bytes[8 + n..];
    header
        .into_iter()
        .filter(|(k, _)| k != "__metadata__")
        .map(|(k, v)| {
            let shape = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
            let o = v["data_offsets"].as_array().unwrap();
            let (a, b) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
            (k, shape, data[a..b].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
        })
        .collect()
}

#[test]
fn train_split_writes_a_splitter_the_core_loads() {
    let data = scratch("data");
    write_treebank(&data);
    let out = scratch("out");
    let r = train(&data, &out);
    assert_eq!(r.corpus.id, "ud-en-ewt-split");
    assert_eq!(r.corpus.spdx, "CC-BY-SA-4.0");
    assert_eq!((r.corpus.train.sentences, r.corpus.train.words), (48, 192), "sentences and tokens");
    let head = r.head.clone().unwrap();
    assert_eq!(
        (head.kind, head.name.as_str(), head.layer.as_str()),
        (HeadKind::TokenSplit, "words", "turboembed:words")
    );
    assert_eq!(head.license_class, Some(turbo::manifest::LicenseClass::ShareAlike));
    assert_eq!(head.labels[0], "_");
    assert_eq!(head.labels.len(), 3, "keep, and one label for `al` and one for `del`: {:?}", head.labels);
    assert_eq!(r.eval.baseline.score(), 1.0, "every test token's split is its only training one");
    let cmp = r.eval.comparison.as_ref().unwrap();
    assert_eq!((cmp.model, cmp.baseline), (r.eval.model.as_ref().unwrap().score(), r.eval.baseline.score()));

    let t = read_safetensors(&fs::read(out.join("head.safetensors")).unwrap());
    let mut got: Vec<(String, Vec<u64>)> = t.iter().map(|(k, s, _)| (k.clone(), s.clone())).collect();
    let mut want: Vec<(String, Vec<u64>)> = turbo::head::split::tensors(&head, 32)
        .into_iter()
        .map(|(s, shape)| (format!("heads.words.{s}"), shape))
        .collect();
    got.sort();
    want.sort();
    assert_eq!(got, want);
    let get = |s: &str| t.iter().find(|(k, _, _)| *k == format!("heads.words.{s}")).map(|(_, _, v)| v.clone());
    let splitter = TokenSplitter::from_tensors(&head, 32, &get).unwrap();
    // The head's own labels, decoded by the core, give each token its
    // gold words, and `_` keeps a token whole.
    let words_of = |token: &str| -> Vec<Vec<String>> {
        (0..head.labels.len() as u32).filter_map(|l| splitter.words(&[l], &[token]).ok()).flatten().collect()
    };
    assert!(words_of("al").contains(&vec!["a".to_owned(), "el".to_owned()]), "{:?}", words_of("al"));
    assert!(words_of("del").contains(&vec!["de".to_owned(), "el".to_owned()]), "{:?}", words_of("del"));
    assert_eq!(splitter.words(&[0], &["mar"]).unwrap(), [vec!["mar".to_owned()]]);
    let stored: HeadSpec = serde_json::from_slice(&fs::read(out.join("head.json")).unwrap()).unwrap();
    assert_eq!(stored, head);

    let again = scratch("out-b");
    let r2 = train(&data, &again);
    assert_eq!(fs::read(again.join("head.safetensors")).unwrap(), fs::read(out.join("head.safetensors")).unwrap());
    assert_eq!(r2.eval, r.eval, "one seed, one splitter");
    for d in [data, out, again] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn train_split_refuses_tag_only_settings_and_unknown_treebanks() {
    let o = run(&["train-split", "--corpus", "ud-xx", "--dir", "x", "--bundle", "x", "--out", "x"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown treebank"));
    let out = scratch("out-crf");
    let o = run(&[
        "train-split",
        "--corpus",
        "ud-en-ewt",
        "--dir",
        "x",
        "--bundle",
        "x",
        "--out",
        out.to_str().unwrap(),
        "--crf",
    ]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--crf"));
    assert!(!out.exists(), "a refused run leaves no directory");
}
