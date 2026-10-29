//! Lemma and dependency heads on a UD treebank: their labels, baselines
//! and UAS, and `train-lemma` / `train-parse` end to end on the small
//! BERT in testdata and a small treebank written here, with the heads
//! they write read back as the core reads them.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use turbo::head::{BiaffineParser, EditTreeLemmatizer};
use turbo::manifest::{HeadKind, HeadSpec};
use turbo_trainer_recipes::corpus::Sentence;
use turbo_trainer_recipes::receipt::Receipt;
use turbo_trainer_recipes::ud::{EN_EWT, UdSentence, parse};
use turbo_trainer_recipes::ud_heads::{UdHead, deps_baseline, deps_label, lemma_baseline, uas};

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-ud-heads-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

/// 48 sentences of one shape: a pronoun, a verb in -ed or -s, a plural
/// noun and a full stop, the verb the root.
fn treebank() -> String {
    let (subjects, verbs, nouns) =
        (["they", "we", "you"], ["walk", "paint", "wash", "kick"], ["dogs", "cars", "walls"]);
    let mut text = String::new();
    for i in 0..48 {
        let (s, v, n) = (subjects[i % 3], verbs[i % 4], nouns[i % 3]);
        let (form, feats) = if i % 2 == 0 { (format!("{v}ed"), "Tense=Past") } else { (format!("{v}s"), "Tense=Pres") };
        let lemma = &n[..n.len() - 1];
        text += &format!(
            "# sent_id = {i}\n1\t{s}\t{s}\tPRON\tPRP\t_\t2\tnsubj\t_\t_\n2\t{form}\t{v}\tVERB\tVBD\t{feats}\t0\troot\t_\t_\n\
             3\t{n}\t{lemma}\tNOUN\tNNS\tNumber=Plur\t2\tobj\t_\t_\n4\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_\n\n"
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

fn sentence(words: &[&str], lemmas: &[&str], heads: &[u32], rels: &[&str]) -> UdSentence {
    let s = |v: &[&str]| v.iter().map(|x| (*x).to_owned()).collect::<Vec<_>>();
    UdSentence { words: s(words), lemmas: s(lemmas), heads: heads.to_vec(), rels: s(rels), ..UdSentence::default() }
}

#[test]
fn labels_baselines_and_uas() {
    let train = [sentence(&["Dogs", "ran", "."], &["dog", "run", "."], &[2, 0, 2], &["nsubj", "root", "punct"])];
    let test = [sentence(&["Dogs", "barked", "."], &["dog", "bark", "."], &[2, 0, 2], &["nsubj", "root", "punct"])];
    assert_eq!(UdHead::Deps.labels(&test[0]), ["2 nsubj", "0 root", "2 punct"]);
    assert_eq!(UdHead::Lemma.labels(&test[0]), ["dog", "bark", "."]);
    assert_eq!(deps_label(0, "root"), "0 root");
    // A seen word takes its training lemma; an unseen one is its own.
    assert_eq!(lemma_baseline(&train, &test), [["dog", "barked", "."]]);
    // Each seen word takes its training attachment; "barked" is unseen
    // and takes the commonest overall, ties to the smaller ("-1 punct"
    // before "1 nsubj" and "root root").
    assert_eq!(deps_baseline(&train, &test), [["2 nsubj", "1 punct", "2 punct"]]);
    let gold: Vec<Sentence> = UdHead::Deps.sentences(&test);
    let base = deps_baseline(&train, &test);
    assert!((uas(&gold, &base) - 2.0 / 3.0).abs() < 1e-12, "all but the verb's head");
    // A distance past the sentence or onto the word itself moves to the
    // nearest other word.
    let long = [sentence(&["a", "b", "c"], &["a", "b", "c"], &[3, 3, 0], &["x", "y", "root"])];
    let short = [sentence(&["b", "a"], &["b", "a"], &[0, 0], &["root", "root"])];
    assert_eq!(deps_baseline(&long, &short), [["2 y", "1 x"]], "a at 2: +2 is past the end, 2 is itself, so 1");
    assert_eq!(uas(&gold, &[UdHead::Deps.labels(&test[0])]), 1.0);
    assert_eq!(uas(&[], &[]), 0.0);
}

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).args(args).output().unwrap()
}

fn train(command: &str, data: &Path, out: &Path, extra: &[&str]) -> Receipt {
    let bundle = tiny();
    let mut args = vec![
        command,
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
    args.extend_from_slice(extra);
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

/// The head's tensors match the shapes `want` gives, and the getter the
/// core's `from_tensors` reads.
fn check_tensors(out: &Path, head: &HeadSpec, want: Vec<(String, Vec<u64>)>) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let t = read_safetensors(&fs::read(out.join("head.safetensors")).unwrap());
    let mut got: Vec<(String, Vec<u64>)> = t.iter().map(|(k, s, _)| (k.clone(), s.clone())).collect();
    let mut want: Vec<(String, Vec<u64>)> =
        want.into_iter().map(|(s, shape)| (format!("{}.{s}", head.tensor_prefix), shape)).collect();
    got.sort();
    want.sort();
    assert_eq!(got, want);
    t
}

#[test]
fn train_lemma_writes_a_lemmatizer_the_core_loads() {
    let data = scratch("data-lemma");
    write_treebank(&data);
    let out = scratch("out-lemma");
    let r = train("train-lemma", &data, &out, &["--min-tree-freq", "1"]);
    assert_eq!(r.corpus.id, "ud-en-ewt-lemma");
    assert_eq!(r.corpus.spdx, "CC-BY-SA-4.0");
    assert_eq!((r.corpus.train.sentences, r.corpus.train.words), (48, 192));
    let head = r.head.clone().unwrap();
    assert_eq!((head.kind, head.layer.as_str()), (HeadKind::EditTreeLemmatizer, "turboembed:lemmas"));
    assert_eq!(head.license_class, Some(turbo::manifest::LicenseClass::ShareAlike));
    assert!(r.eval.uas.is_none());
    let cmp = r.eval.comparison.as_ref().unwrap();
    assert_eq!((cmp.model, cmp.baseline), (r.eval.model.as_ref().unwrap().score(), r.eval.baseline.score()));
    let t = check_tensors(&out, &head, turbo::head::lemmatizer::tensors(&head, 32));
    let get = |s: &str| t.iter().find(|(k, _, _)| *k == format!("heads.lemma.{s}")).map(|(_, _, v)| v.clone());
    EditTreeLemmatizer::from_tensors(&head, 32, &get).unwrap();
    let stored: HeadSpec = serde_json::from_slice(&fs::read(out.join("head.json")).unwrap()).unwrap();
    assert_eq!(stored, head);
    for d in [data, out] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn train_parse_writes_a_parser_the_core_loads_and_scores_uas() {
    let data = scratch("data-parse");
    write_treebank(&data);
    let out = scratch("out-parse");
    let r = train("train-parse", &data, &out, &["--arc", "8", "--rel", "4"]);
    assert_eq!(r.corpus.id, "ud-en-ewt-deps");
    let head = r.head.clone().unwrap();
    assert_eq!((head.kind, head.layer.as_str()), (HeadKind::BiaffineParser, "turboembed:dependencies"));
    assert_eq!(head.labels, ["nsubj", "obj", "punct", "root"]);
    assert_eq!(head.biaffine, Some(turbo::manifest::Biaffine { arc: 8, rel: 4 }));
    let u = r.eval.uas.unwrap();
    // Every word's attachment is the same in training and test.
    assert_eq!(u.baseline, 1.0);
    assert!(u.model >= r.eval.model.as_ref().unwrap().score(), "UAS is at least LAS");
    let t = check_tensors(&out, &head, turbo::head::parser::tensors(&head, 32));
    let get = |s: &str| t.iter().find(|(k, _, _)| *k == format!("heads.deps.{s}")).map(|(_, _, v)| v.clone());
    BiaffineParser::from_tensors(&head, 32, &get).unwrap();

    let again = scratch("out-parse-b");
    let r2 = train("train-parse", &data, &again, &["--arc", "8", "--rel", "4"]);
    assert_eq!(fs::read(again.join("head.safetensors")).unwrap(), fs::read(out.join("head.safetensors")).unwrap());
    assert_eq!(r2.eval, r.eval, "one seed, one parser");
    for d in [data, out, again] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn ud_commands_refuse_tag_only_settings_and_unknown_treebanks() {
    let o = run(&["train-parse", "--corpus", "ud-xx", "--dir", "x", "--bundle", "x", "--out", "x"]);
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown treebank"));
    let out = scratch("out-crf");
    let o = run(&[
        "train-lemma",
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

#[test]
fn the_treebank_text_here_parses_as_written() {
    let s = parse("t", treebank().as_bytes()).unwrap();
    assert_eq!(s.len(), 48);
    assert_eq!(s[0].words, ["they", "walked", "dogs", "."]);
    assert_eq!(s[0].lemmas, ["they", "walk", "dog", "."]);
}

#[test]
fn the_recorded_release_baselines() {
    let Some(dir) = std::env::var_os("TURBO_TEST_UD_EWT_DIR").map(PathBuf::from) else { return };
    let (s, _, recorded) = turbo_trainer_recipes::ud::read_treebank(&EN_EWT, &dir, false).unwrap();
    assert!(recorded);
    let score = |head: UdHead, predicted: &[Vec<String>]| {
        let gold = head.sentences(&s.test);
        let seen = turbo_trainer_recipes::eval::Seen::of(&head.sentences(&s.train));
        turbo_trainer_recipes::eval::evaluate(turbo_trainer_recipes::corpus::Task::Tags, &gold, predicted, &seen, None)
            .score()
    };
    let lemma = score(UdHead::Lemma, &lemma_baseline(&s.train, &s.test));
    let deps = deps_baseline(&s.train, &s.test);
    let las = score(UdHead::Deps, &deps);
    let uas = uas(&UdHead::Deps.sentences(&s.test), &deps);
    assert_eq!((round(lemma), round(las), round(uas)), RECORDED, "lemma accuracy, LAS, UAS");
}

fn round(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

const RECORDED: (f64, f64, f64) = (0.9503, 0.3322, 0.3707);

/// The metadata of a safetensors file.
fn metadata(bytes: &[u8]) -> Value {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    header["__metadata__"].clone()
}

#[test]
fn a_ud_run_stopped_midway_keeps_its_best_head_so_far() {
    let data = scratch("stop-data");
    write_treebank(&data);
    let bundle = tiny();
    for (command, extra) in [("train-lemma", ["--min-tree-freq", "1"]), ("train-parse", ["--arc", "8"])] {
        let out = scratch(&format!("stop-{command}"));
        let mut args = vec![command, "--corpus", "ud-en-ewt", "--dir", data.to_str().unwrap()];
        args.extend(["--bundle", bundle.to_str().unwrap(), "--out", out.to_str().unwrap(), "--unrecorded-corpus"]);
        args.extend(["--epochs", "100000", "--patience", "0", "--batch", "8", "--commit", "test"]);
        args.extend(extra);
        let mut child = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
            .args(&args)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let start = std::time::Instant::now();
        while !out.join("head.safetensors").is_file() {
            assert!(start.elapsed().as_secs() < 120, "{command}: no checkpoint within two minutes");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        child.kill().unwrap();
        child.wait().unwrap();
        // What is left is whole: the spec, and a head the core loads that
        // names the epoch and dev score it is from; no receipt.
        let head: HeadSpec = serde_json::from_slice(&fs::read(out.join("head.json")).unwrap()).unwrap();
        let bytes = fs::read(out.join("head.safetensors")).unwrap();
        let m = metadata(&bytes);
        let _: u64 = m["best_epoch"].as_str().unwrap().parse().unwrap();
        let dev: f64 = m["dev"].as_str().unwrap().parse().unwrap();
        assert!((0.0..=1.0).contains(&dev), "{command}: dev {dev}");
        let t = read_safetensors(&bytes);
        let prefix = head.tensor_prefix.clone();
        let get = |s: &str| t.iter().find(|(k, _, _)| *k == format!("{prefix}.{s}")).map(|(_, _, v)| v.clone());
        match head.kind {
            HeadKind::EditTreeLemmatizer => drop(EditTreeLemmatizer::from_tensors(&head, 32, &get).unwrap()),
            HeadKind::BiaffineParser => drop(BiaffineParser::from_tensors(&head, 32, &get).unwrap()),
            k => panic!("{command}: a {k:?} head"),
        }
        assert!(!out.join("receipt.json").exists(), "{command} did not finish");
        fs::remove_dir_all(out).unwrap();
    }
    fs::remove_dir_all(data).unwrap();
}
