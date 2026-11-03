//! The Re-DocRED reader, the relation scoring, and `train-relation` end
//! to end on the small BERT in testdata over documents written here in
//! the release's format.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};
use turbo::head::RelationClassifier;
use turbo::head::relation::{Mention, tensors};
use turbo::manifest::{HeadSpec, LicenseClass, NO_RELATION};
use turbo_trainer_recipes::corpus::Task;
use turbo_trainer_recipes::eval::Metrics;
use turbo_trainer_recipes::receipt::Receipt;
use turbo_trainer_recipes::redocred::{self, Document, ENTITY_TYPES, parse, read};
use turbo_trainer_recipes::redocred_eval::{RelationOptions, baseline, bootstrap, evaluate, spec};

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-redocred-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

fn mention(sent: usize, a: usize, b: usize, kind: &str) -> Value {
    json!({"name": "x", "pos": [a, b], "sent_id": sent, "type": kind, "global_pos": [a, a], "index": "0_0"})
}

/// A document: "<per> works for <org> ." then "<org> is in <loc> ." then
/// "<per> left .", the person mentioned twice; the person works for the
/// organisation (P108), which is in the place (P17).
fn doc(i: usize) -> Value {
    let (p, o, l) = (["ann", "bob", "eve"][i % 3], ["acme", "zeta"][i % 2], ["paris", "oslo", "rome", "kyiv"][i % 4]);
    json!({
        "title": format!("doc {i}"),
        "sents": [[p, "works", "for", o, "."], [o, "is", "in", l, "."], [p, "left", "."]],
        "vertexSet": [
            [mention(0, 0, 1, "PER"), mention(2, 0, 1, "PER")],
            [mention(0, 3, 4, "ORG"), mention(1, 0, 1, "ORG")],
            [mention(1, 3, 4, "LOC")],
        ],
        "labels": [
            {"r": "P108", "h": 0, "t": 1, "evidence": [0]},
            {"r": "P17", "h": 1, "t": 2, "evidence": [1]},
        ],
    })
}

fn corpus(dir: &Path, docs: usize) {
    fs::create_dir_all(dir.join("data")).unwrap();
    let all: Vec<Value> = (0..docs).map(doc).collect();
    for f in ["train", "dev", "test"] {
        fs::write(dir.join(format!("data/{f}_revised.json")), serde_json::to_vec(&all).unwrap()).unwrap();
    }
    fs::write(dir.join("LICENSE"), "MIT").unwrap();
}

#[test]
fn documents_join_their_sentences_and_mentions_take_document_words() {
    let got = parse("t", &serde_json::to_vec(&[doc(0)]).unwrap()).unwrap();
    let d = &got[0];
    assert_eq!(d.words.len(), 13);
    assert_eq!(d.words[5..9], ["acme", "is", "in", "paris"]);
    let m = |first, last, entity| Mention { first, last, entity };
    assert_eq!(d.mentions, [m(0, 0, 0), m(10, 10, 0), m(3, 3, 1), m(5, 5, 1), m(8, 8, 2)]);
    assert_eq!(d.types, ["PER", "ORG", "LOC"]);
    assert_eq!(d.relations, [(0, 1, "P108".to_owned()), (1, 2, "P17".to_owned())]);
}

#[test]
fn malformed_documents_are_refused_with_their_place() {
    let bad = |edit: &dyn Fn(&mut Value)| {
        let mut d = doc(0);
        edit(&mut d);
        parse("t", &serde_json::to_vec(&[doc(1), d]).unwrap()).unwrap_err().0
    };
    for e in [
        bad(&|d| d["vertexSet"][2][0]["pos"] = json!([3, 6])),
        bad(&|d| d["vertexSet"][2][0]["sent_id"] = json!(3)),
        bad(&|d| d["vertexSet"][2][0]["pos"] = json!([2, 2])),
        bad(&|d| d["vertexSet"][2] = json!([])),
        bad(&|d| d["vertexSet"][2][0]["type"] = json!("")),
        bad(&|d| d["sents"][2][1] = json!("")),
        bad(&|d| d["labels"][0]["t"] = json!(0)),
        bad(&|d| d["labels"][0]["t"] = json!(3)),
        bad(&|d| d["labels"][1] = d["labels"][0].clone()),
        bad(&|d| d["labels"][0]["r"] = json!("")),
    ] {
        assert!(e.starts_with("t: document 1: "), "{e}");
    }
    assert!(parse("t", b"[{\"title\": 1}]").unwrap_err().0.starts_with("t: "));
}

#[test]
fn files_that_are_not_the_release_are_refused_unless_asked_for() {
    let dir = scratch("pins");
    corpus(&dir, 4);
    let e = read(&dir, false).unwrap_err();
    assert!(e.0.contains("not the recorded release") && e.0.contains("--unrecorded-corpus"), "{e}");
    let c = read(&dir, true).unwrap();
    assert!(!c.record.recorded);
    assert_eq!(c.record.task, Task::Relations);
    assert_eq!(c.record.files.len(), 4);
    assert_eq!((c.record.train.sentences, c.record.train.words), (4, 52));
    assert_eq!(c.labels(), [NO_RELATION, "P108", "P17"]);
    assert_eq!(c.record.class().unwrap(), LicenseClass::ShareAlike);
    let s = spec(&c, &RelationOptions::default()).unwrap();
    assert_eq!(s.relation.as_ref().unwrap().entity_labels, ENTITY_TYPES);
    assert_eq!((s.name.as_str(), s.layer.as_str()), (redocred::HEAD, redocred::LAYER));
    let _ = fs::remove_dir_all(&dir);
}

fn gold(relations: &[(u32, u32, &str)]) -> Document {
    Document {
        title: String::new(),
        words: Vec::new(),
        mentions: Vec::new(),
        types: Vec::new(),
        relations: relations.iter().map(|&(h, t, r)| (h, t, r.to_owned())).collect(),
    }
}

#[test]
fn relations_score_by_exact_triples_micro_and_per_label() {
    let test = [gold(&[(0, 1, "P108"), (1, 2, "P17")]), gold(&[(0, 1, "P108"), (0, 1, "P27")])];
    let t = |h, t, r: &str| (h, t, r.to_owned());
    // Right, reversed, right, a label training never had missed.
    let predicted = vec![vec![t(0, 1, "P108"), t(2, 1, "P17")], vec![t(0, 1, "P108")]];
    let Metrics::Relations { all, per_type } = evaluate(&test, &predicted) else { panic!() };
    assert_eq!((all.correct, all.found, all.gold), (2, 3, 4));
    assert!((all.f1 - 4.0 / 7.0).abs() < 1e-12);
    assert_eq!((per_type["P108"].correct, per_type["P108"].f1), (2, 1.0));
    assert_eq!((per_type["P17"].found, per_type["P17"].gold, per_type["P17"].f1), (1, 1, 0.0));
    assert_eq!((per_type["P27"].gold, per_type["P27"].f1), (1, 0.0));
    let base = baseline(&test);
    assert_eq!(evaluate(&test, &base).score(), 0.0);
    let c = bootstrap(&test, &predicted, &base, 200, 7);
    assert_eq!((c.model, c.baseline), (all.f1, 0.0));
    assert!(c.low <= c.delta && c.delta <= c.high, "{c:?}");
    assert_eq!(c, bootstrap(&test, &predicted, &base, 200, 7));
}

/// The tensors of a safetensors file of F32s: name, shape, values.
fn read_safetensors(bytes: &[u8]) -> Vec<(String, Vec<u64>, Vec<f32>)> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
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

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn train(data: &Path, out: &Path, extra: &[&str]) -> std::process::Output {
    let mut args = vec!["--epochs", "30", "--lr", "0.01"];
    args.extend(extra);
    run(data, out, &args)
}

fn run(data: &Path, out: &Path, extra: &[&str]) -> std::process::Output {
    let (data, out, bundle) = (data.to_str().unwrap(), out.to_str().unwrap(), tiny());
    let mut args = vec!["train-relation", "--corpus", "re-docred", "--dir", data, "--bundle"];
    args.extend([bundle.to_str().unwrap(), "--out", out, "--unrecorded-corpus", "--commit", "test"]);
    args.extend(["--mlp", "16", "--type-dim", "4", "--batch", "4"]);
    args.extend(extra);
    Command::new(env!("CARGO_BIN_EXE_turbo-train-head")).args(&args).output().unwrap()
}

#[test]
fn a_relation_run_writes_a_head_the_core_reads_and_a_receipt() {
    let data = scratch("train-data");
    corpus(&data, 12);
    let out = scratch("train-out");
    let o = train(&data, &out, &[]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let r: Receipt = serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(r.corpus.task, Task::Relations);
    let model = r.eval.model.as_ref().unwrap();
    assert!(matches!(model, Metrics::Relations { .. }));
    assert!(model.score() > 0.9, "{model:?}");
    assert_eq!(r.eval.baseline.score(), 0.0);
    let loss = &r.training.as_ref().unwrap().train_loss;
    assert!(loss.last().unwrap() < &(loss[0] / 2.0), "{loss:?}");

    // The head as written reads back as the core's classifier.
    let spec: HeadSpec = serde_json::from_slice(&fs::read(out.join("head.json")).unwrap()).unwrap();
    assert_eq!(&spec, r.head.as_ref().unwrap());
    let hidden = r.encoder.as_ref().unwrap().hidden;
    let st = read_safetensors(&fs::read(out.join("head.safetensors")).unwrap());
    for (name, shape) in tensors(&spec, hidden) {
        let t = st.iter().find(|t| t.0 == format!("{}.{name}", spec.tensor_prefix)).unwrap();
        assert_eq!(t.1, shape, "{name}");
    }
    let get = |s: &str| st.iter().find(|t| t.0 == format!("{}.{s}", spec.tensor_prefix)).map(|t| t.2.clone());
    let head = RelationClassifier::from_tensors(&spec, hidden, &get).unwrap();
    assert_eq!(head.labels(), [NO_RELATION, "P108", "P17"]);

    // The same command gives the same head, byte for byte.
    let again = scratch("train-again");
    assert!(train(&data, &again, &[]).status.success());
    assert_eq!(fs::read(out.join("head.safetensors")).unwrap(), fs::read(again.join("head.safetensors")).unwrap());
    let refused = train(&data, &scratch("train-crf"), &["--crf"]);
    assert!(String::from_utf8_lossy(&refused.stderr).contains("neither --crf"));

    // Without --lr, a relation head trains at its own default, which the
    // receipt records.
    assert_eq!(r.config.as_ref().unwrap().lr, 0.01);
    let plain = scratch("train-plain");
    assert!(run(&data, &plain, &["--epochs", "1"]).status.success());
    let p: Receipt = serde_json::from_slice(&fs::read(plain.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(p.config.unwrap().lr, 3e-4);
    for d in [data, out, again, plain] {
        let _ = fs::remove_dir_all(d);
    }
}
