//! `turbo-train-head` end to end on the small BERT in testdata and a
//! small corpus in MultiCoNER's format: a baseline run and two training
//! runs, their receipts, and the head they write, read back as the core
//! reads a head.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;
use turbo::head::TokenTagger;
use turbo::head::tagger::tensors;
use turbo::manifest::HeadSpec;
use turbo_trainer_recipes::receipt::{Receipt, SCHEMA};

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-train-head-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

/// A corpus where a name follows "met" and "in" names a place.
fn corpus(dir: &Path) {
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

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).args(args).output().unwrap()
}

fn train(data: &Path, out: &Path) -> Receipt {
    train_with(data, out, &[])
}

fn train_with(data: &Path, out: &Path, extra: &[&str]) -> Receipt {
    let bundle = tiny();
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
    let o = run(&args);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap()
}

/// The tensors of a safetensors file, by name: shape and values.
fn read_safetensors(bytes: &[u8]) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    assert_eq!((8 + n) % 8, 0, "the data starts aligned");
    let header: serde_json::Map<String, Value> = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data = &bytes[8 + n..];
    header
        .into_iter()
        .map(|(k, v)| {
            assert_eq!(v["dtype"], "F32");
            let shape = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
            let o = v["data_offsets"].as_array().unwrap();
            let (a, b) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
            let vals = data[a..b].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            (k, shape, vals)
        })
        .collect()
}

#[test]
fn a_training_run_writes_a_head_the_core_loads_and_a_receipt() {
    let data = scratch("data");
    corpus(&data);
    let out = scratch("out-a");
    let r = train(&data, &out);
    assert_eq!(r.schema, SCHEMA);
    assert_eq!(r.commit, "test");
    assert!(!r.corpus.recorded);
    let enc = r.encoder.as_ref().unwrap();
    assert_eq!((enc.backend.as_str(), enc.hidden), ("cpu", 32));
    let head = r.head.clone().unwrap();
    assert_eq!(head.layer, "turboembed:entities");
    assert_eq!(head.labels, ["O", "B-HumanSettlement", "B-OtherPER"]);
    let training = r.training.as_ref().unwrap();
    assert!(!training.dev.is_empty() && training.best_epoch < training.dev.len());
    let model = r.eval.model.as_ref().unwrap().score();
    let cmp = r.eval.comparison.as_ref().unwrap();
    assert_eq!((cmp.model, cmp.baseline), (model, r.eval.baseline.score()));
    assert_eq!(r.outputs.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(), ["head.safetensors", "head.json"]);

    let bytes = fs::read(out.join("head.safetensors")).unwrap();
    assert_eq!(r.outputs[0].sha256, turbo::bundle::sha256_hex(&bytes));
    let stored: HeadSpec = serde_json::from_slice(&fs::read(out.join("head.json")).unwrap()).unwrap();
    assert_eq!(stored, head);
    let t = read_safetensors(&bytes);
    let want: Vec<(String, Vec<u64>)> =
        tensors(&head, 32).into_iter().map(|(s, shape)| (format!("heads.ner.{s}"), shape)).collect();
    let mut got: Vec<(String, Vec<u64>)> = t.iter().map(|(k, s, _)| (k.clone(), s.clone())).collect();
    let mut want_sorted = want.clone();
    got.sort();
    want_sorted.sort();
    assert_eq!(got, want_sorted);
    let get = |s: &str| t.iter().find(|(k, _, _)| *k == format!("heads.ner.{s}")).map(|(_, _, v)| v.clone());
    TokenTagger::from_tensors(&head, 32, &get).unwrap();

    let again = scratch("out-b");
    let r2 = train(&data, &again);
    assert_eq!(fs::read(again.join("head.safetensors")).unwrap(), bytes, "one seed, one head");
    assert_eq!(r2.eval, r.eval);
    for d in [data, out, again] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn a_hashed_head_trains_with_its_tables_and_the_core_reads_its_forms() {
    let data = scratch("data-hash");
    corpus(&data);
    let out = scratch("out-hash");
    let r = train_with(&data, &out, &["--hash-embed", "4", "--hash-rows", "16,8,8,8", "--tolerance", "0.01"]);
    let head = r.head.clone().unwrap();
    assert_eq!(head.hash_embed, Some(turbo::manifest::HashEmbed { dim: 4, rows: [16, 8, 8, 8] }));
    assert_eq!(r.config.as_ref().unwrap().tolerance, 0.01);
    let t = read_safetensors(&fs::read(out.join("head.safetensors")).unwrap());
    for f in ["norm", "prefix", "suffix", "shape"] {
        assert!(t.iter().any(|(k, _, _)| *k == format!("heads.ner.hash.{f}")), "{f}");
    }
    let get = |s: &str| t.iter().find(|(k, _, _)| *k == format!("heads.ner.{s}")).map(|(_, _, v)| v.clone());
    assert!(TokenTagger::from_tensors(&head, 32, &get).unwrap().reads_forms());
    let bad = run(&[
        "train",
        "--corpus",
        "multiconer2-en",
        "--dir",
        "x",
        "--bundle",
        "x",
        "--out",
        "x",
        "--hash-rows",
        "1,2",
    ]);
    assert!(!bad.status.success());
    for d in [data, out] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn a_baseline_run_needs_no_encoder_and_refuses_an_existing_out() {
    let data = scratch("base-data");
    corpus(&data);
    let out = scratch("base-out");
    let args =
        ["baseline", "--corpus", "multiconer2-en", "--dir", data.to_str().unwrap(), "--out", out.to_str().unwrap()];
    let o = run(&args);
    assert!(!o.status.success() && String::from_utf8_lossy(&o.stderr).contains("--unrecorded-corpus"));
    let mut args = args.to_vec();
    args.push("--unrecorded-corpus");
    assert!(run(&args).status.success());
    let r: Receipt = serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
    assert!(r.encoder.is_none() && r.eval.model.is_none());
    assert!((r.eval.baseline.score() - 1.0).abs() < 1e-12, "every word's label is fixed in this corpus");
    let o = run(&args);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("already there"));
    for d in [data, out] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn unknown_commands_corpora_and_arguments_are_refused() {
    for (args, msg) in [
        (&["fit", "--corpus", "masc-pos", "--dir", "."][..], "unknown command"),
        (&["check", "--corpus", "conll2003", "--dir", "."][..], "unknown corpus"),
        (&["check", "--corpus", "masc-pos", "--dir", ".", "--dropout", "0.1"][..], "unknown arguments"),
        (&["check", "--dir", "."][..], "--corpus is required"),
    ] {
        let o = run(args);
        assert_eq!(o.status.code(), Some(2), "{args:?}");
        assert!(String::from_utf8_lossy(&o.stderr).contains(msg), "{args:?}: {}", String::from_utf8_lossy(&o.stderr));
    }
}
