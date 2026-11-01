//! CPU training, held-out zero-shot evaluation, bundle assembly and reload
//! agreement for `train-open-span` (`HEAD_OPEN_SPAN`, #1735).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo::head::OpenSpanMatcher;
use turbo::manifest::{Decode, HeadKind, HeadSpec, LicenseClass};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer_recipes::assemble::assemble_bundle;
use turbo_trainer_recipes::corpus::Sentence;
use turbo_trainer_recipes::open_span::{dictionary_baseline, entity_types, filter_sentences, predict};
use turbo_trainer_recipes::{eval, read_corpus, receipt::Receipt};

fn train_open_span(data: &Path, bundle: &Path, out: &Path, hold_out: Option<&str>) -> Receipt {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"));
    cmd.args(["train-open-span", "--corpus", "wnut17", "--dir"])
        .arg(data)
        .arg("--bundle")
        .arg(bundle)
        .arg("--out")
        .arg(out)
        .args([
            "--unrecorded-corpus",
            "--max-words",
            "4",
            "--width-dim",
            "16",
            "--threads",
            "2",
            "--epochs",
            "3",
            "--batch",
            "4",
            "--commit",
            "fixture",
        ]);
    if let Some(h) = hold_out {
        cmd.args(["--hold-out", h]);
    }
    let result = cmd.output().unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap()
}

fn load_open_span(out: &Path) -> OpenSpanMatcher {
    let spec: HeadSpec = serde_json::from_slice(&fs::read(out.join("head.json")).unwrap()).unwrap();
    assert_eq!(spec.kind, HeadKind::OpenSpan);
    assert_eq!(spec.decode, Decode::SpansFlat);
    assert_eq!(spec.license_class, Some(LicenseClass::Permissive));
    let bytes = fs::read(out.join("head.safetensors")).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data = &bytes[8 + n..];
    OpenSpanMatcher::from_tensors(&spec, 32, &|name| {
        let offsets = header.get(format!("heads.open_ner.{name}"))?.get("data_offsets")?.as_array()?;
        let a = offsets[0].as_u64()? as usize;
        let b = offsets[1].as_u64()? as usize;
        Some(data[a..b].as_chunks::<4>().0.iter().map(|v| f32::from_le_bytes(*v)).collect())
    })
    .unwrap()
}

#[test]
fn open_span_hold_out_round_trip_and_bundle_assembly() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("open-span-train-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let data = dir.join("data");
    fs::create_dir(&data).unwrap();
    let mut training = String::new();
    for kind in turbo_trainer_recipes::wnut17::TYPES {
        for _ in 0..2 {
            training.push_str(&format!("ann\tB-{kind}\nlee\tI-{kind}\nin\tO\nparis\tO\n\t\n"));
        }
    }
    training.push_str("visit\tO\noslo\tB-location\ntoday\tO\n\t\n");
    fs::write(data.join("wnut17train.conll"), &training).unwrap();
    fs::write(data.join("emerging.dev.conll"), "bob\tB-person\nlee\tI-person\nin\tO\nrome\tB-location\n\n").unwrap();
    let test_file = data.join("emerging.test.annotated");
    fs::write(&test_file, "eve\tB-person\nin\tO\nparis\tB-location\n\n").unwrap();

    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let a = dir.join("a");
    let first = train_open_span(&data, &bundle, &a, Some("location"));
    assert!(!first.corpus.recorded);
    let head_spec = first.head.as_ref().unwrap();
    assert_eq!(head_spec.kind, HeadKind::OpenSpan);
    assert_eq!(head_spec.labels.len(), 5);
    assert!(!head_spec.labels.iter().any(|l| l == "location"));

    let matcher = load_open_span(&a);
    let corpus = read_corpus("wnut17", &data, true).unwrap();
    let all_types = entity_types(&corpus).unwrap();
    let seen_types: Vec<String> = all_types.iter().filter(|t| t.as_str() != "location").cloned().collect();
    let mut encoder = WordEncoder::open(&bundle, &EncoderOptions::default()).unwrap();
    let predicted = predict(&mut encoder, &matcher, &corpus.splits.test, &all_types).unwrap();
    let train_filtered = filter_sentences(&corpus.splits.train, &seen_types);
    let scores =
        eval::evaluate(corpus.record.task, &corpus.splits.test, &predicted, &eval::Seen::of(&train_filtered), None);
    assert_eq!(first.eval.model.as_ref().unwrap(), &scores);
    assert!(first.eval.dictionary.is_some());
    let os = first.eval.open_span.as_ref().unwrap();
    assert_eq!(os.seen_types, seen_types);
    assert_eq!(os.held_out_types, ["location"]);
    assert!(os.held_out_model.is_some());
    assert!(os.held_out_baseline.is_some());
    assert!(os.held_out_dictionary.is_some());

    // Assemble refuses an unrecorded corpus run until recorded is true.
    let assembled_dir = dir.join("assembled");
    fs::create_dir(&assembled_dir).unwrap();
    assert!(assemble_bundle(&bundle, &[&a], &assembled_dir).unwrap_err().0.contains("corpus.recorded is false"));
    let mut rec_ok = first.clone();
    rec_ok.corpus.recorded = true;
    fs::write(a.join("receipt.json"), serde_json::to_vec_pretty(&rec_ok).unwrap()).unwrap();
    let assembled = assemble_bundle(&bundle, &[&a], &assembled_dir).unwrap();
    assert_eq!(assembled.license_tier, LicenseClass::Permissive);
    assert!(assembled.manifest.heads.iter().any(|h| h.name == "open_ner" && h.kind == HeadKind::OpenSpan));

    // Changing test gold does not alter training loss, dev scores, or saved weights.
    fs::write(&test_file, "eve\tO\nin\tO\nparis\tO\n\n").unwrap();
    let b = dir.join("b");
    let second = train_open_span(&data, &bundle, &b, Some("location"));
    assert_eq!(first.training.as_ref().unwrap().dev, second.training.as_ref().unwrap().dev);
    assert_eq!(fs::read(a.join("head.safetensors")).unwrap(), fs::read(b.join("head.safetensors")).unwrap(),);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn dictionary_baseline_matches_multi_word_and_hyphenated_labels() {
    let sents = vec![Sentence {
        words: vec!["the".into(), "creative".into(), "work".into(), "and".into(), "person".into()],
        labels: vec!["O".into(), "B-creative-work".into(), "I-creative-work".into(), "O".into(), "B-person".into()],
    }];
    let labels = vec!["person".to_owned(), "creative-work".to_owned(), "location".to_owned()];
    let pred = dictionary_baseline(&sents, &labels);
    assert_eq!(pred[0], sents[0].labels);
}

#[test]
fn committed_open_span_receipts_preserve_dictionary_and_holdout_evidence_and_name_no_machine() {
    let records = Path::new(env!("CARGO_MANIFEST_DIR")).join("records/open-span-1735");
    for (file, held_out) in [("wnut17.json", &[][..]), ("wnut17-holdout-location.json", &["location"][..])] {
        let p = records.join(file);
        let raw = fs::read_to_string(&p).unwrap();
        assert!(!raw.contains("/work/"), "{file} contains host path");
        let raw_val: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(raw_val.get("host").is_none(), "{file} contains host field");
        let r: Receipt = serde_json::from_str(&raw).unwrap();
        assert!((1..=r.training.as_ref().unwrap().dev.len()).contains(&r.training.as_ref().unwrap().best_epoch));
        assert!(r.eval.model.as_ref().unwrap().score() > r.eval.baseline.score());
        assert!(r.eval.model.as_ref().unwrap().score() > r.eval.dictionary.as_ref().unwrap().score());
        let os = r.eval.open_span.as_ref().unwrap();
        assert_eq!(os.held_out_types, held_out);
        if !held_out.is_empty() {
            let hm = os.held_out_model.as_ref().unwrap().score();
            let hb = os.held_out_baseline.as_ref().unwrap().score();
            let hd = os.held_out_dictionary.as_ref().unwrap().score();
            assert!(hm > hb && hm > hd, "zero-shot held-out {hm} must beat majority {hb} and dictionary {hd}");
        }
    }
}

#[test]
fn open_span_trains_on_ud_treebank_propn_runs_and_selects_requested_cache_layer() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("open-span-ud-cache-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let ud_data = dir.join("ud");
    fs::create_dir(&ud_data).unwrap();
    let conllu = concat!(
        "# text = Ann Lee met Bob in Paris\n",
        "1\tAnn\tAnn\tPROPN\tNNP\t_\t3\tnsubj\t_\t_\n",
        "2\tLee\tLee\tPROPN\tNNP\t_\t1\tflat\t_\t_\n",
        "3\tmet\tmeet\tVERB\tVBD\t_\t0\troot\t_\t_\n",
        "4\tBob\tBob\tPROPN\tNNP\t_\t3\tobj\t_\t_\n",
        "5\tin\tin\tADP\tIN\t_\t6\tcase\t_\t_\n",
        "6\tParis\tParis\tPROPN\tNNP\t_\t3\tobl\t_\t_\n\n",
    );
    for f in ["en_ewt-ud-train.conllu", "en_ewt-ud-dev.conllu", "en_ewt-ud-test.conllu"] {
        fs::write(ud_data.join(f), conllu).unwrap();
    }
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let ud_out = dir.join("ud-out");
    let res = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
        .args(["train-open-span", "--corpus", "ud-en-ewt", "--dir"])
        .arg(&ud_data)
        .arg("--bundle")
        .arg(&bundle)
        .arg("--out")
        .arg(&ud_out)
        .args([
            "--unrecorded-corpus",
            "--max-words",
            "4",
            "--width-dim",
            "16",
            "--epochs",
            "2",
            "--batch",
            "2",
            "--commit",
            "ud-test",
        ])
        .output()
        .unwrap();
    assert!(res.status.success(), "{}", String::from_utf8_lossy(&res.stderr));
    let ud_receipt: Receipt = serde_json::from_slice(&fs::read(ud_out.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(ud_receipt.corpus.id, "ud-en-ewt");
    assert_eq!(ud_receipt.head.as_ref().unwrap().labels, ["proper noun"]);
    assert_eq!(ud_receipt.eval.open_span.as_ref().unwrap().seen_types, ["proper noun"]);

    // Multi-layer cache with --layer selection.
    let cache_dir = dir.join("cache");
    let cache_res = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
        .args(["cache", "--corpus", "ud-en-ewt-upos", "--dir"])
        .arg(&ud_data)
        .arg("--bundle")
        .arg(&bundle)
        .arg("--out")
        .arg(&cache_dir)
        .args(["--unrecorded-corpus", "--layers", "2", "--commit", "cache-test"])
        .output()
        .unwrap();
    assert!(cache_res.status.success(), "{}", String::from_utf8_lossy(&cache_res.stderr));

    for layer in ["1", "2"] {
        let out_l = dir.join(format!("layer-{layer}"));
        let r = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
            .args(["train-open-span", "--corpus", "ud-en-ewt", "--dir"])
            .arg(&ud_data)
            .arg("--bundle")
            .arg(&bundle)
            .arg("--cache")
            .arg(&cache_dir)
            .args(["--layer", layer, "--out"])
            .arg(&out_l)
            .args(["--unrecorded-corpus", "--max-words", "4", "--width-dim", "16", "--epochs", "2", "--commit", "l"])
            .output()
            .unwrap();
        assert!(r.status.success(), "layer {layer}: {}", String::from_utf8_lossy(&r.stderr));
    }

    // Requesting a layer not in the cache fails cleanly.
    let bad_l = dir.join("layer-99");
    let r_bad = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
        .args(["train-open-span", "--corpus", "ud-en-ewt", "--dir"])
        .arg(&ud_data)
        .arg("--bundle")
        .arg(&bundle)
        .arg("--cache")
        .arg(&cache_dir)
        .args(["--layer", "99", "--out"])
        .arg(&bad_l)
        .args(["--unrecorded-corpus", "--epochs", "1", "--commit", "l"])
        .output()
        .unwrap();
    assert!(!r_bad.status.success());
    assert!(String::from_utf8_lossy(&r_bad.stderr).contains("no layer 99"));

    fs::remove_dir_all(dir).unwrap();
}
