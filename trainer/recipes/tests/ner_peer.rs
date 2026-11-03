//! The real CLI scores authored BIO predictions against the held-out words.
//! These are protocol tests, not spaCy or Stanza accuracy records.
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use turbo_trainer_recipes::receipt::Receipt;

#[test]
fn bio_export_and_peer_preserve_words_boundaries_and_exact_span_scores() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("ner-peer-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    for name in ["wnut17train.conll", "emerging.dev.conll", "emerging.test.annotated"] {
        fs::write(dir.join(name), "Ánn\tB-person\nLee\tI-person\nin\tO\nOslo\tB-location\n\n").unwrap();
    }
    let words = dir.join("words.tsv");
    let export = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
        .args(["export", "--corpus", "wnut17", "--dir"])
        .arg(&dir)
        .arg("--out")
        .arg(&words)
        .arg("--unrecorded-corpus")
        .output()
        .unwrap();
    assert!(export.status.success(), "{}", String::from_utf8_lossy(&export.stderr));
    assert_eq!(fs::read_to_string(&words).unwrap(), "Ánn\t_\nLee\t_\nin\t_\nOslo\t_\n\n");
    let pred = dir.join("predictions.tsv");
    let valid = "Ánn\tB-PERSON\nLee\tI-PERSON\nin\tO\nOslo\tO\n\n";
    for (i, text) in [
        valid.to_owned(),
        valid.replace("Ánn", "Ann"),
        valid.replace("Lee\tI-PERSON\n", ""),
        valid.replace("Ánn\tB-PERSON", "Ánn\tI-PERSON"),
        valid.replace("I-PERSON", "I-OTHER"),
        valid.replace("Oslo\tO", "Oslo\tB-UNKNOWN"),
        valid.replace("Lee\tI-PERSON", "\nLee\tI-PERSON"),
        format!("{valid}extra\tO\n"),
        valid.replace("Oslo\tO", "Oslo\tO\textra"),
    ]
    .iter()
    .enumerate()
    {
        fs::write(&pred, text).unwrap();
        let out = dir.join(format!("result{i}"));
        let result = Command::new(env!("CARGO_BIN_EXE_turbo-train-head"))
            .args(["peer", "--corpus", "wnut17", "--dir"])
            .arg(&dir)
            .args(["--peer", "authored-fixture", "--column", "bio", "--map", "B-PERSON=B-person,I-PERSON=I-person"])
            .arg("--predictions")
            .arg(&pred)
            .arg("--out")
            .arg(&out)
            .arg("--unrecorded-corpus")
            .output()
            .unwrap();
        assert_eq!(result.status.success(), i == 0, "case {i}: {}", String::from_utf8_lossy(&result.stderr));
        if i == 0 {
            let receipt: Receipt = serde_json::from_slice(&fs::read(out.join("receipt.json")).unwrap()).unwrap();
            assert_eq!(receipt.eval.baseline.score(), 1.0);
            assert!((receipt.eval.model.unwrap().score() - 2.0 / 3.0).abs() < 1e-10);
            assert_eq!(receipt.peer.unwrap().predictions.sha256, turbo::bundle::sha256_hex(text.as_bytes()));
        } else {
            assert!(!out.exists(), "refused predictions leave no success receipt");
        }
    }
    fs::remove_dir_all(dir).unwrap();
}
