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
use turbo::manifest::WordPooling;
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer_recipes::receipt::{Receipt, SCHEMA};
use turbo_trainer_recipes::run::examples;

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
        .filter(|(k, _)| k != "__metadata__")
        .map(|(k, v)| {
            assert_eq!(v["dtype"], "F32");
            let shape = v["shape"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
            let o = v["data_offsets"].as_array().unwrap();
            let (a, b) = (o[0].as_u64().unwrap() as usize, o[1].as_u64().unwrap() as usize);
            let vals = data[a..b].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            (k, shape, vals)
        })
        .collect()
}

/// The `__metadata__` of a safetensors file.
fn metadata(bytes: &[u8]) -> Value {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    header["__metadata__"].clone()
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
    assert_eq!((enc.backend.as_str(), enc.hidden, enc.layers), ("cpu", 32, Some(2)));
    // The data's 48 sentences of five words, train and dev both read.
    assert_eq!(r.timing_s.feature_words, 2 * 48 * 5);
    assert!(r.timing_s.words_per_s().is_some_and(|w| w > 0.0));
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

/// Runs of `args` started together, each given `out`: exactly one
/// succeeds, every other is refused for the directory, and what the
/// winner wrote is whole and its receipt's.
fn race(args: &[&str], out: &Path, runs: usize) -> Receipt {
    let children: Vec<_> = (0..runs)
        .map(|_| {
            Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
                .args(args)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    let outs: Vec<_> = children.into_iter().map(|c| c.wait_with_output().unwrap()).collect();
    let won = outs.iter().filter(|o| o.status.success()).count();
    assert_eq!(
        won,
        1,
        "{:?}",
        outs.iter().map(|o| String::from_utf8_lossy(&o.stderr).into_owned()).collect::<Vec<_>>()
    );
    for o in outs.iter().filter(|o| !o.status.success()) {
        assert_eq!(o.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&o.stderr).contains("already there"), "{}", String::from_utf8_lossy(&o.stderr));
    }
    let r: Receipt = serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
    let mut listed: Vec<String> = r.outputs.iter().map(|f| f.path.clone()).collect();
    for f in &r.outputs {
        let bytes = fs::read(out.join(&f.path)).unwrap();
        assert_eq!((bytes.len() as u64, turbo::bundle::sha256_hex(&bytes)), (f.size, f.sha256.clone()), "{}", f.path);
    }
    listed.push("receipt.json".into());
    listed.sort();
    let mut on_disk: Vec<String> =
        fs::read_dir(out).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    on_disk.sort();
    assert_eq!(on_disk, listed, "only the winner's files");
    r
}

#[test]
fn of_runs_racing_for_one_out_exactly_one_writes() {
    let data = scratch("race-data");
    corpus(&data);
    let out = scratch("race-baseline");
    let (d, o) = (data.to_str().unwrap(), out.to_str().unwrap());
    let base = ["baseline", "--corpus", "multiconer2-en", "--dir", d, "--out", o, "--unrecorded-corpus"];
    race(&base, &out, 6);
    let first = out;
    let out = scratch("race-train");
    let o = out.to_str().unwrap();
    let bundle = tiny();
    let b = bundle.to_str().unwrap();
    let train = [
        "train",
        "--corpus",
        "multiconer2-en",
        "--dir",
        d,
        "--bundle",
        b,
        "--out",
        o,
        "--unrecorded-corpus",
        "--epochs",
        "2",
        "--commit",
        "test",
    ];
    let r = race(&train, &out, 3);
    assert!(r.head.is_some());
    // A run refused after its claim leaves no directory behind.
    let gone = scratch("race-refused");
    let g = gone.to_str().unwrap();
    let bad = run(&["train", "--corpus", "multiconer2-en", "--dir", d, "--bundle", "/nonexistent", "--out", g]);
    assert_eq!(bad.status.code(), Some(2));
    assert!(!gone.exists());
    for p in [data, first, out] {
        fs::remove_dir_all(p).unwrap();
    }
}

#[test]
fn a_run_stopped_midway_keeps_its_best_head_so_far() {
    let data = scratch("stop-data");
    corpus(&data);
    let out = scratch("stop-out");
    let (d, o, b) = (data.to_str().unwrap(), out.to_str().unwrap(), tiny());
    let args = [
        "train",
        "--corpus",
        "multiconer2-en",
        "--dir",
        d,
        "--bundle",
        b.to_str().unwrap(),
        "--out",
        o,
        "--unrecorded-corpus",
        "--epochs",
        "100000",
        "--patience",
        "0",
        "--commit",
        "test",
    ];
    let mut child = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    // The dev examples the run scores, read as it reads them.
    let corpus = turbo_trainer_recipes::read_corpus("multiconer2-en", &data, true).unwrap();
    let mut enc = WordEncoder::open(&b, &EncoderOptions::default()).unwrap();
    let mut head: Option<HeadSpec> = None;
    let mut dev = Vec::new();
    // Each read of the checkpoint, while the run replaces it and after
    // it is killed, is one generation: weights that score on dev exactly
    // the score its metadata gives, beside the epoch they are from.
    let mut check = |bytes: &[u8]| -> u64 {
        let h = head.get_or_insert_with(|| {
            let h: HeadSpec = serde_json::from_slice(&fs::read(out.join("head.json")).unwrap()).unwrap();
            dev = examples(&mut enc, &corpus.splits.dev, &h.labels, WordPooling::First).unwrap().0;
            h
        });
        let m = metadata(bytes);
        let epoch: u64 = m["best_epoch"].as_str().unwrap().parse().unwrap();
        let want: f64 = m["dev"].as_str().unwrap().parse().unwrap();
        let t = read_safetensors(bytes);
        let get = |s: &str| t.iter().find(|(k, _, _)| *k == format!("heads.ner.{s}")).map(|(_, _, v)| v.clone());
        let tagger = TokenTagger::from_tensors(h, 32, &get).unwrap();
        assert_eq!(turbo_trainer::fit::score(&tagger, &dev).unwrap(), want, "epoch {epoch}: weights and score agree");
        epoch
    };
    let start = std::time::Instant::now();
    while !out.join("head.safetensors").is_file() {
        assert!(start.elapsed().as_secs() < 120, "no checkpoint within two minutes");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let mut epochs = Vec::new();
    while start.elapsed().as_millis() < 1500 && epochs.len() < 40 {
        epochs.push(check(&fs::read(out.join("head.safetensors")).unwrap()));
    }
    child.kill().unwrap();
    child.wait().unwrap();
    // What is left is whole: the spec, and one generation of the head; no
    // receipt and nothing else naming an epoch, since the run did not finish.
    let last = check(&fs::read(out.join("head.safetensors")).unwrap());
    assert!(epochs.windows(2).all(|w| w[0] <= w[1]) && epochs.last() <= Some(&last), "{epochs:?} then {last}");
    assert!(!out.join("receipt.json").exists() && !out.join("checkpoint.json").exists());
    fs::remove_dir_all(data).unwrap();
    fs::remove_dir_all(out).unwrap();
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

/// A one-word sentence of `word` labelled `label` in MultiCoNER's format.
fn sentence(i: usize, word: &str, label: &str) -> String {
    format!("# id {i}\tdomain=en\n{word} _ _ {label}\n\n")
}

#[test]
fn the_baseline_learns_from_the_sentences_the_model_is_given() {
    // The second training sentence alone teaches "x" its name; --max-train
    // 1 withholds it from the model, so it is withheld from the baseline.
    let data = scratch("max-train-data");
    fs::create_dir_all(data.join("EN-English")).unwrap();
    let train = sentence(0, "a", "O") + &sentence(1, "x", "B-OtherPER");
    let test = sentence(0, "x", "B-OtherPER");
    fs::write(data.join("EN-English/en_train.conll"), train).unwrap();
    for f in ["en_dev", "en_test"] {
        fs::write(data.join(format!("EN-English/{f}.conll")), &test).unwrap();
    }
    let d = data.to_str().unwrap();
    let bundle = tiny();
    let train_out = |out: &Path, max: &str| {
        let o = out.to_str().unwrap();
        let args = [
            "train",
            "--corpus",
            "multiconer2-en",
            "--dir",
            d,
            "--bundle",
            bundle.to_str().unwrap(),
            "--out",
            o,
            "--unrecorded-corpus",
            "--epochs",
            "1",
            "--max-train",
            max,
        ];
        let r = run(&args);
        assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
        serde_json::from_slice::<Receipt>(&fs::read(out.join("receipt.json")).unwrap()).unwrap()
    };
    let one = scratch("max-train-1");
    let r = train_out(&one, "1");
    // Given only "a O", the baseline can only say O: no span is found.
    assert_eq!(r.eval.baseline.score(), 0.0);
    let cmp = r.eval.comparison.as_ref().unwrap();
    assert_eq!(cmp.baseline, 0.0, "the comparison pairs the model with that baseline");
    // Given both sentences, it names "x".
    let all = scratch("max-train-all");
    assert_eq!(train_out(&all, "0").eval.baseline.score(), 1.0);
    for d in [data, one, all] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn a_bilstm_head_waits_longer_before_it_stops() {
    let data = scratch("patience-data");
    corpus(&data);
    let (d, b) = (data.to_str().unwrap(), tiny());
    let patience = |out: &Path, extra: &[&str]| {
        let mut args = vec![
            "train",
            "--corpus",
            "multiconer2-en",
            "--dir",
            d,
            "--bundle",
            b.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
            "--unrecorded-corpus",
            "--epochs",
            "1",
        ];
        args.extend(extra);
        let o = run(&args);
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let r: Receipt = serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
        r.config.unwrap().patience
    };
    let outs = [scratch("patience-linear"), scratch("patience-bilstm"), scratch("patience-given")];
    assert_eq!(patience(&outs[0], &[]), turbo_trainer::Config::default().patience);
    assert_eq!(patience(&outs[1], &["--bilstm", "4x1"]), 15);
    assert_eq!(patience(&outs[2], &["--bilstm", "4x1", "--patience", "3"]), 3);
    for p in outs.into_iter().chain([data]) {
        fs::remove_dir_all(p).unwrap();
    }
}
