//! Actual CPU inference from a persisted authored WNUT head and tiny encoder.
//! Bundle construction here is a test fixture, not a replacement assembler.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{Value, json};
use turbo::bundle::{Bundle, sha256_hex};
use turbo_analysis_embed::EncoderOptions;
use turbo_trainer_recipes::{ner_peer, ner_predict, read_corpus};

struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn corpus(dir: &Path) {
    fs::create_dir(dir).unwrap();
    let mut train = String::new();
    for kind in turbo_trainer_recipes::wnut17::TYPES {
        train.push_str(&format!("Ánn\tB-{kind}\n李\tI-{kind}\nvisits\tO\nOslo\tB-location\n\n"));
    }
    fs::write(dir.join("wnut17train.conll"), train).unwrap();
    let held_out = "Zoë\tB-person\n李\tI-person\n🦀\tO\nOslo\tB-location\n\ne\u{301}\tO\n東京\tB-location\n\n";
    for name in ["emerging.dev.conll", "emerging.test.annotated"] {
        fs::write(dir.join(name), held_out).unwrap();
    }
}

fn trained(dir: &Path, data: &Path) {
    let result = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
        .args(["train", "--corpus", "wnut17", "--dir"])
        .arg(data)
        .arg("--bundle")
        .arg(tiny())
        .arg("--out")
        .arg(dir)
        .args([
            "--unrecorded-corpus",
            "--crf",
            "--threads",
            "2",
            "--epochs",
            "2",
            "--batch",
            "4",
            "--commit",
            "fixture",
        ])
        .output()
        .unwrap();
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
}

fn fixture_bundle(run: &Path, out: &Path) -> Value {
    fs::create_dir(out).unwrap();
    let source = tiny();
    let mut manifest: Value = serde_json::from_slice(&fs::read(source.join("manifest.json")).unwrap()).unwrap();
    manifest["model"]["license_class"] = json!("CLASS_PERMISSIVE");
    for file in manifest["files"].as_array().unwrap() {
        let relative = file["path"].as_str().unwrap();
        let to = out.join(relative);
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::copy(source.join(relative), to).unwrap();
    }
    let weights = fs::read(run.join("head.safetensors")).unwrap();
    fs::write(out.join("head.safetensors"), &weights).unwrap();
    let mut head: Value = serde_json::from_slice(&fs::read(run.join("head.json")).unwrap()).unwrap();
    head["artifact"] = json!("ner-head");
    manifest["heads"] = json!([head]);
    let mut artifact = manifest["artifacts"][0].clone();
    artifact["name"] = json!("ner-head");
    artifact["files"] = json!(["head.safetensors"]);
    manifest["artifacts"].as_array_mut().unwrap().push(artifact);
    manifest["files"].as_array_mut().unwrap().push(json!({
        "path": "head.safetensors", "size": weights.len(), "sha256": sha256_hex(&weights)
    }));
    write_manifest(out, &manifest);
    Bundle::open(out).unwrap();
    manifest
}

fn write_manifest(dir: &Path, manifest: &Value) {
    fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(manifest).unwrap()).unwrap();
}

#[test]
fn persisted_native_predictions_preserve_unicode_and_refuse_mismatched_inputs() {
    let root = Fixture(PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("ner-predict-{}", std::process::id())));
    fs::create_dir(&root.0).unwrap();
    let data = root.0.join("corpus");
    corpus(&data);
    let run = root.0.join("run");
    trained(&run, &data);
    let bundle = root.0.join("bundle");
    let manifest = fixture_bundle(&run, &bundle);
    let gold = read_corpus("wnut17", &data, true).unwrap();
    let options = EncoderOptions::default();
    let first = ner_predict::predict(&gold, &bundle, "ner", &options, "fixture").unwrap();
    let second = ner_predict::predict(&gold, &bundle, "ner", &options, "fixture").unwrap();
    assert_eq!(first.record, second.record, "the prediction receipt is deterministic");
    assert_eq!(first.record.bundle_manifest_sha256, Bundle::open(&bundle).unwrap().manifest_sha256);
    assert_eq!(first.record.predictions.sha256, sha256_hex(first.bio.as_bytes()));
    assert_eq!(first.record.predictions.size, first.bio.len() as u64);
    assert_eq!(first.record.predictions.path, "predictions.bio");
    for head in ["ner", "missing"] {
        let destination = root.0.join(format!("cli-{head}"));
        let result = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
            .args(["predict-ner", "--corpus", "wnut17", "--unrecorded-corpus", "--dir"])
            .arg(&data)
            .arg("--bundle")
            .arg(&bundle)
            .args(["--head", head, "--commit", "fixture", "--out"])
            .arg(&destination)
            .output()
            .unwrap();
        if head == "ner" {
            assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
            assert_eq!(fs::read_to_string(destination.join("predictions.bio")).unwrap(), first.bio);
            let record: ner_predict::PredictionRecord =
                serde_json::from_slice(&fs::read(destination.join("prediction.json")).unwrap()).unwrap();
            assert_eq!(record, first.record);
        } else {
            assert!(!result.status.success(), "unknown head must fail");
            assert!(!destination.exists(), "refused CLI prediction leaves no output directory");
        }
    }
    let refused = root.0.join("pooling-override");
    let result = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
        .args(["predict-ner", "--corpus", "wnut17", "--unrecorded-corpus", "--dir"])
        .arg(&data)
        .arg("--bundle")
        .arg(&bundle)
        .args(["--head", "ner", "--pooling", "mean", "--out"])
        .arg(&refused)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("pooling"));
    assert!(!refused.exists());
    assert_eq!(first.bio, second.bio, "fixed persisted weights give deterministic BIO predictions");
    let predictions = ner_peer::read("native.bio", first.bio.as_bytes(), &gold.splits.test).unwrap();
    ner_peer::validate("wnut17", &predictions).unwrap();
    let words: Vec<&str> =
        first.bio.lines().filter(|line| !line.is_empty()).map(|line| line.split_once('\t').unwrap().0).collect();
    assert_eq!(words, ["Zoë", "李", "🦀", "Oslo", "e\u{301}", "東京"]);
    assert!(ner_predict::predict(&gold, &bundle, "missing", &options, "fixture").is_err());

    // A separate held-out split reaches the minimum paired sentence count.
    let comparison_data = root.0.join("comparison-corpus");
    corpus(&comparison_data);
    let held_out = fs::read_to_string(data.join("emerging.test.annotated")).unwrap();
    fs::write(comparison_data.join("emerging.test.annotated"), held_out.repeat(15)).unwrap();
    let comparison_gold = read_corpus("wnut17", &comparison_data, true).unwrap();
    assert_eq!(comparison_gold.splits.test.len(), 30);
    let prediction = ner_predict::predict(&comparison_gold, &bundle, "ner", &options, "fixture").unwrap();
    let candidate_path = root.0.join("candidate.bio");
    let peer_path = root.0.join("peer.bio");
    let receipt_path = root.0.join("candidate.json");
    fs::write(&candidate_path, &prediction.bio).unwrap();
    fs::write(&peer_path, &prediction.bio).unwrap();
    fs::write(&receipt_path, serde_json::to_vec(&prediction.record).unwrap()).unwrap();
    let compare_cli = |destination: &Path| {
        Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
            .args(["compare-ner", "--corpus", "wnut17", "--unrecorded-corpus", "--dir"])
            .arg(&comparison_data)
            .arg("--predictions")
            .arg(&candidate_path)
            .arg("--prediction-record")
            .arg(&receipt_path)
            .arg("--peer-predictions")
            .arg(&peer_path)
            .args(["--peer", "fixture", "--types", "person,location", "--out"])
            .arg(destination)
            .output()
            .unwrap()
    };
    let comparison_out = root.0.join("comparison-out");
    let result = compare_cli(&comparison_out);
    assert!(result.status.success(), "{}", String::from_utf8_lossy(&result.stderr));
    let receipt: Value = serde_json::from_slice(&fs::read(comparison_out.join("comparison.json")).unwrap()).unwrap();
    assert_eq!(receipt["comparison"]["candidate_vs_peer"]["delta"], 0.0);
    assert_eq!(receipt["candidate"], serde_json::to_value(&prediction.record).unwrap());
    assert_eq!(receipt["comparison"]["candidate_file"]["sha256"], prediction.record.predictions.sha256);
    assert_eq!(receipt["comparison"]["candidate"]["full_metrics"]["spans"]["all"]["gold"], 45);
    fs::write(&candidate_path, format!("{}\n", prediction.bio)).unwrap();
    let refused_out = root.0.join("comparison-refused");
    let refused = compare_cli(&refused_out);
    assert!(!refused.status.success(), "modified bytes must fail their receipt check");
    assert!(String::from_utf8_lossy(&refused.stderr).contains("receipt does not match"));
    assert!(!refused_out.exists(), "refused comparison leaves no output directory");

    let mut wrong = manifest.clone();
    for label in wrong["heads"][0]["labels"].as_array_mut().unwrap() {
        if let Some(text) = label.as_str() {
            *label = json!(text.replace("person", "unknown-type"));
        }
    }
    write_manifest(&bundle, &wrong);
    assert!(ner_predict::predict(&gold, &bundle, "ner", &options, "fixture").is_err());

    let mut non_bio = manifest.clone();
    non_bio["heads"][0]["scheme"] = json!("SCHEME_TAGS");
    non_bio["heads"][0]["decode"] = json!("DECODE_ARGMAX");
    write_manifest(&bundle, &non_bio);
    assert!(ner_predict::predict(&gold, &bundle, "ner", &options, "fixture").is_err());

    write_manifest(&bundle, &manifest);
    let path = bundle.join("head.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(path, bytes).unwrap();
    assert!(
        Bundle::open(&bundle).unwrap().read_verified("head.safetensors").is_err(),
        "persisted weight tampering fails the bundle digest gate"
    );
    assert!(ner_predict::predict(&gold, &bundle, "ner", &options, "fixture").is_err());
}
