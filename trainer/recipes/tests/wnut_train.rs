//! Real CPU training and persisted inference on authored WNUT-format text.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo::head::TokenTagger;
use turbo::manifest::{Decode, HeadSpec, LicenseClass};
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_trainer_recipes::{eval, read_corpus, receipt::Receipt, run};

fn train(data: &Path, bundle: &Path, out: &Path) -> Receipt {
    let result = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
        .args(["train", "--corpus", "wnut17", "--dir"])
        .arg(data)
        .arg("--bundle")
        .arg(bundle)
        .arg("--out")
        .arg(out)
        .args([
            "--unrecorded-corpus",
            "--crf",
            "--threads",
            "2",
            "--epochs",
            "3",
            "--batch",
            "4",
            "--commit",
            "fixture",
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap()
}

fn load(out: &Path) -> TokenTagger {
    let spec: HeadSpec = serde_json::from_slice(&fs::read(out.join("head.json")).unwrap()).unwrap();
    assert_eq!(spec.decode, Decode::Crf);
    assert_eq!(spec.license_class, Some(LicenseClass::Permissive));
    let bytes = fs::read(out.join("head.safetensors")).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data = &bytes[8 + n..];
    TokenTagger::from_tensors(&spec, 32, &|name| {
        let offsets = header.get(format!("heads.ner.{name}"))?.get("data_offsets")?.as_array()?;
        let a = offsets[0].as_u64()? as usize;
        let b = offsets[1].as_u64()? as usize;
        Some(data[a..b].as_chunks::<4>().0.iter().map(|v| f32::from_le_bytes(*v)).collect())
    })
    .unwrap()
}

#[test]
fn cpu_crf_round_trip_and_test_labels_do_not_select_weights() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("wnut-train-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let data = dir.join("data");
    fs::create_dir(&data).unwrap();
    let mut training = String::new();
    for kind in turbo_trainer_recipes::wnut17::TYPES {
        for _ in 0..2 {
            training.push_str(&format!("ann\tB-{kind}\nlee\tI-{kind}\nin\tO\noslo\tB-location\n\t\n"));
        }
    }
    fs::write(data.join("wnut17train.conll"), &training).unwrap();
    fs::write(data.join("emerging.dev.conll"), "bob\tB-person\nlee\tI-person\nin\tO\nrome\tB-location\n\n").unwrap();
    let test_file = data.join("emerging.test.annotated");
    fs::write(&test_file, "eve\tB-person\nin\tO\nparis\tB-location\n\n").unwrap();
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let a = dir.join("a");
    let first = train(&data, &bundle, &a);
    assert!(!first.corpus.recorded, "authored fixture is never a release receipt");
    assert_eq!(first.corpus.train.sentences, 12);
    assert_eq!(first.corpus.dev.sentences, 1);
    assert_eq!(first.corpus.test.sentences, 1);
    assert_eq!(first.config.as_ref().unwrap().dev_left_out, 0);
    assert_eq!(first.timing_s.feature_words, 52);
    assert_eq!(first.head.as_ref().unwrap().labels.len(), 13);
    let tagger = load(&a);
    let corpus = read_corpus("wnut17", &data, true).unwrap();
    let mut encoder = WordEncoder::open(&bundle, &EncoderOptions::default()).unwrap();
    let predicted = run::predict(&mut encoder, &tagger, &corpus.splits.test).unwrap();
    let scores = eval::evaluate(
        corpus.record.task,
        &corpus.splits.test,
        &predicted,
        &eval::Seen::of(&corpus.splits.train),
        None,
    );
    assert_eq!(first.eval.model.as_ref().unwrap(), &scores);

    fs::write(&test_file, "eve\tO\nin\tO\nparis\tO\n\n").unwrap();
    let b = dir.join("b");
    let second = train(&data, &bundle, &b);
    assert_ne!(first.corpus.files, second.corpus.files, "changed test labels are recorded");
    assert_eq!(first.training.as_ref().unwrap().dev, second.training.as_ref().unwrap().dev);
    assert_eq!(
        fs::read(a.join("head.safetensors")).unwrap(),
        fs::read(b.join("head.safetensors")).unwrap(),
        "test gold cannot select weights or checkpoint"
    );
    fs::remove_dir_all(dir).unwrap();
}
