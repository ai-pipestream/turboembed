use turbo_analysis::ModelRef;
use turbo_langdetect::NgramModel;

#[test]
fn flores_command_loads_a_model_and_scores_every_row() {
    let dir = std::env::temp_dir().join(format!("langid-flores-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model = NgramModel::new(
        vec!["eng".into()],
        vec![0.0; 16],
        vec![0.0],
        (1, 3),
        16,
        ModelRef { id: "test".into(), revision: "1".into(), sha256: "0".into() },
    )
    .unwrap();
    let model_path = dir.join("model");
    let gold_path = dir.join("gold.tsv");
    std::fs::write(&model_path, model.to_bytes()).unwrap();
    std::fs::write(&gold_path, "eng\thello\ndeu\twelt\n").unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-langid-flores"))
        .arg(&model_path)
        .arg(&gold_path)
        .output()
        .unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let receipt: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(receipt["texts"], 2);
    assert_eq!(receipt["correct"], 1);
    assert_eq!(receipt["accuracy"], 0.5);
    assert_eq!(receipt["revision_claim"], turbo_train_recipes::flores::FLORES_PLUS_REVISION);
    assert!(receipt["elapsed_seconds"].as_f64().unwrap() >= 0.0);
    assert!(receipt["texts_per_second"].as_f64().unwrap() > 0.0);
    std::fs::remove_dir_all(dir).unwrap();
}
