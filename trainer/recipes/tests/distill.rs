//! End-to-end tests for `turbo-train-head distill-head` (#1520): teacher
//! loading, soft-target student distillation into a fast linear head that
//! loads directly next to the encoder bundle, accuracy margin gate
//! verification, end-to-end throughput receipt generation, and gate/argument
//! refusals.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo::head::TokenTagger;
use turbo::safetensors::{Dtype, File};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer_recipes::distill::{DistillReceipt, SCHEMA, load_head};
use turbo_trainer_recipes::read_corpus;
use turbo_trainer_recipes::run::predict;

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-train-head-distill-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

fn corpus(dir: &Path) {
    let names = ["ann", "bob", "eve", "max"];
    let places = ["paris", "rome", "oslo"];
    let mut text = String::new();
    for i in 0..32 {
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

#[test]
fn distill_head_writes_verified_student_bundle_and_deterministic_outputs() {
    let data = scratch("data");
    corpus(&data);
    let teacher_out = scratch("teacher");
    let student_out = scratch("student");
    let student_again = scratch("student-again");
    let bundle = tiny();
    let (d, b, t_dir, s_dir) = (
        data.to_str().unwrap(),
        bundle.to_str().unwrap(),
        teacher_out.to_str().unwrap(),
        student_out.to_str().unwrap(),
    );

    // 1. Train a teacher head with a BiLSTM adapter and CRF over the 32-wide encoder.
    let train_teacher = run(&[
        "train",
        "--corpus",
        "multiconer2-en",
        "--dir",
        d,
        "--bundle",
        b,
        "--out",
        t_dir,
        "--unrecorded-corpus",
        "--bilstm",
        "8x1",
        "--crf",
        "--epochs",
        "20",
        "--batch",
        "8",
        "--lr",
        "0.05",
        "--commit",
        "test-sha",
    ]);
    assert!(train_teacher.status.success(), "{}", String::from_utf8_lossy(&train_teacher.stderr));
    let teacher_bytes = fs::read(teacher_out.join("head.safetensors")).unwrap();
    let teacher_sha = turbo::bundle::sha256_hex(&teacher_bytes);

    // 2. Distil the BiLSTM+CRF teacher into a linear student head (no BiLSTM, no CRF) on the bundle.
    let distill_args = [
        "distill-head",
        "--corpus",
        "multiconer2-en",
        "--dir",
        d,
        "--bundle",
        b,
        "--student-bundle",
        b,
        "--teacher",
        t_dir,
        "--out",
        s_dir,
        "--unrecorded-corpus",
        "--temperature",
        "2.0",
        "--alpha",
        "0.5",
        "--max-accuracy-drop",
        "0.05",
        "--epochs",
        "25",
        "--batch",
        "8",
        "--lr",
        "0.05",
        "--commit",
        "test-sha",
    ];
    let dist = run(&distill_args);
    assert!(dist.status.success(), "{}", String::from_utf8_lossy(&dist.stderr));

    let receipt: DistillReceipt = serde_json::from_slice(&fs::read(student_out.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt.schema, SCHEMA);
    assert_eq!(receipt.commit, "test-sha");
    assert_eq!((receipt.teacher_hidden, receipt.student_hidden), (32, 32));
    assert_eq!(receipt.teacher_sha256, teacher_sha);
    assert_eq!((receipt.temperature, receipt.alpha), (2.0, 0.5));
    assert!(receipt.within_margin, "{receipt:?}");
    assert!(receipt.accuracy_drop <= receipt.max_accuracy_drop);
    assert_eq!(receipt.throughput.backend, "cpu");
    assert!(!receipt.throughput.device.is_empty());
    assert!(receipt.throughput.teacher_tokens_per_sec > 0.0);
    assert!(receipt.throughput.student_tokens_per_sec > 0.0);
    assert!(receipt.throughput.speedup_ratio > 0.0);

    // Verify the written `head.safetensors` + `head.json` loads directly next to the
    // encoder bundle via `TokenTagger::from_safetensors(spec, enc.hidden(), &file)`
    // and runs end-to-end prediction over the corpus test split.
    let student_bytes = fs::read(student_out.join("head.safetensors")).unwrap();
    assert_eq!(receipt.student_sha256, turbo::bundle::sha256_hex(&student_bytes));
    let mut enc = WordEncoder::open(&bundle, &EncoderOptions::default()).unwrap();
    let (loaded_spec, loaded_student, loaded_sha) = load_head(&student_out, enc.hidden()).unwrap();
    assert_eq!(loaded_sha, receipt.student_sha256);
    assert_eq!(loaded_spec, receipt.student_head);
    let st_file = File::parse("head.safetensors", &student_bytes).unwrap();
    let core_tagger = TokenTagger::from_tensors(&loaded_spec, enc.hidden(), &|s| {
        let key = format!("{}.{s}", loaded_spec.tensor_prefix);
        st_file
            .get(&key, Dtype::F32, 0)
            .ok()
            .or_else(|| st_file.get(&key, Dtype::F32, 1).ok().or_else(|| st_file.get(&key, Dtype::F32, 2).ok()))
            .map(|t| t.f32s())
    })
    .unwrap();
    assert_eq!(core_tagger.hidden(), enc.hidden());
    assert_eq!(loaded_student.to_token_tagger().unwrap().hidden(), enc.hidden());
    let parsed_corpus = read_corpus("multiconer2-en", &data, true).unwrap();
    let preds = predict(&mut enc, &core_tagger, &parsed_corpus.splits.test).unwrap();
    assert_eq!(preds.len(), parsed_corpus.splits.test.len());

    // 3. Determinism: a second run with the same seed produces identical student bytes and accuracies.
    let mut args2 = distill_args.to_vec();
    args2[12] = student_again.to_str().unwrap();
    let dist2 = run(&args2);
    assert!(dist2.status.success(), "{}", String::from_utf8_lossy(&dist2.stderr));
    let r2: DistillReceipt = serde_json::from_slice(&fs::read(student_again.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(r2.student_sha256, receipt.student_sha256);
    assert_eq!(r2.train_loss, receipt.train_loss);
    assert_eq!(r2.val_scores, receipt.val_scores);
    assert_eq!(r2.teacher_test_accuracy, receipt.teacher_test_accuracy);
    assert_eq!(r2.student_test_accuracy, receipt.student_test_accuracy);

    // 4. Gate failure and invalid option refusals leave no output directory behind.
    let gate_fail = scratch("gate-fail");
    let mut fail_args = distill_args.to_vec();
    fail_args[12] = gate_fail.to_str().unwrap();
    fail_args.extend(["--min-speedup", "1000000.0"]);
    let bad_gate = run(&fail_args);
    assert_eq!(bad_gate.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&bad_gate.stderr).contains("distill gate failed"));
    assert!(!gate_fail.exists());

    let bad_temp = scratch("bad-temp");
    let mut temp_args = distill_args.to_vec();
    temp_args[12] = bad_temp.to_str().unwrap();
    temp_args[15] = "0.0";
    let res_temp = run(&temp_args);
    assert_eq!(res_temp.status.code(), Some(2));
    assert!(!bad_temp.exists());

    for p in [data, teacher_out, student_out, student_again] {
        fs::remove_dir_all(p).unwrap();
    }
}
