use serde_json::{Value, json};
use turbo::TURBO_PRECISION_MODEL;
use turbo::bundle::sha256_hex;

use super::*;

#[test]
fn questions_read_squad_layout_in_order() {
    let text = r#"{"version":"v2.0","data":[{"title":"A","paragraphs":[
        {"context":"Paris is in France.","qas":[{"id":"1","question":"Where is Paris?","answers":[],"is_impossible":false},
                                            {"id":"2","question":"Who?","answers":[],"is_impossible":true}]},
        {"context":"Second.","qas":[{"id":"3","question":"What?","answers":[]}]}]}]}"#;
    let q = questions(text).unwrap();
    assert_eq!(q.len(), 3);
    assert_eq!(q[0], ("Where is Paris?".into(), "Paris is in France.".into()));
    assert_eq!(q[2], ("What?".into(), "Second.".into()));
    assert!(questions(r#"{"data":[{"paragraphs":[{"qas":[]}]}]}"#).is_err());
    assert!(questions("[]").is_err());
}

#[test]
fn bins_start_at_their_bounds() {
    assert_eq!([0, 127, 128, 255, 256, 383, 384, 5000].map(bin_of), [0, 0, 1, 1, 2, 2, 3, 3]);
}

/// Each committed record parses and passes `check`, and is filed
/// under the name `file_name` gives it.
#[test]
fn committed_records_pass_their_check() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(QA_DIR);
    let Ok(entries) = std::fs::read_dir(&dir) else { return };
    for e in entries {
        let path = e.unwrap().path();
        let r: QaRecord = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        check(&r).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(path.file_name().unwrap().to_str().unwrap(), file_name(&r));
    }
}

fn tiny_qa_bundle(dir: &Path) -> PathBuf {
    let bundle = dir.join("bundle");
    std::fs::create_dir_all(bundle.join("weights")).unwrap();
    std::fs::create_dir_all(bundle.join("reference")).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let tiny = root.join("testdata/tiny-bert-bundle");
    let reference = root.join("testdata/tiny-qa-reference");
    let spec: Value = serde_json::from_slice(&std::fs::read(reference.join("cases.json")).unwrap()).unwrap();
    let mut manifest: Value = serde_json::from_slice(&std::fs::read(tiny.join("manifest.json")).unwrap()).unwrap();
    manifest.as_object_mut().unwrap().remove("embed");
    manifest["model"]["id"] = "turbo/test-tiny-qa".into();
    manifest["task"] = "TASK_EXTRACTIVE_QA".into();
    manifest["qa"] = spec["qa"].clone();
    manifest["classifier"] = json!({"kind":"CLASSIFIER_TOKEN","labels":["start","end"]});
    manifest["artifacts"][0]["files"] = json!(["weights/model.safetensors", "weights/head.safetensors"]);
    manifest["artifacts"][0]["tensor_names"]["classifier_weight"] = "qa_outputs.weight".into();
    manifest["artifacts"][0]["tensor_names"]["classifier_bias"] = "qa_outputs.bias".into();
    let mut cases = spec["cases"].clone();
    for c in cases.as_array_mut().unwrap() {
        c["prompt_role"] = "PROMPT_NONE".into();
    }
    manifest["reference"]["cases"] = cases;
    manifest["reference"]["produced_by"]["tool"] = "transformers".into();
    let mut files = Vec::new();
    for (src, dest) in [
        (tiny.join("tokenizer.json"), "tokenizer.json"),
        (tiny.join("weights/model.safetensors"), "weights/model.safetensors"),
        (reference.join("head.safetensors"), "weights/head.safetensors"),
        (reference.join("reference.safetensors"), "reference/reference.safetensors"),
    ] {
        let b = std::fs::read(src).unwrap();
        std::fs::write(bundle.join(dest), &b).unwrap();
        files.push(json!({"path": dest, "size": b.len(), "sha256": sha256_hex(&b)}));
    }
    manifest["files"] = files.into();
    std::fs::write(bundle.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    bundle
}

#[test]
fn qa_measures_bins_without_a_cpu_gpu_clock() {
    let scratch =
        turbo_scratch::Scratch::at(std::env::temp_dir().join(format!("turbo-bench-qa-{}", std::process::id())));
    let bundle = tiny_qa_bundle(&scratch);
    let questions_path = scratch.join("squad.json");
    let long_context = vec!["Paris"; 140].join(" ");
    let squad = json!({
        "data": [{
            "paragraphs": [
                {"context": "Paris is in France.", "qas": [{"question": "Where is Paris?"}]},
                {"context": long_context, "qas": [{"question": "Where?"}]}
            ]
        }]
    });
    std::fs::write(&questions_path, serde_json::to_vec(&squad).unwrap()).unwrap();

    let plan = QaPlan {
        bundle,
        device: "cpu".into(),
        precision: TURBO_PRECISION_MODEL,
        batch: 1,
        max_batch: 0,
        questions: questions_path,
        per_bin: 1,
        warmup: 1,
        iterations: 2,
    };
    let p = Provenance { top: String::new(), commit: "b".repeat(40), pushed_to: vec!["origin/main".into()] };

    let cpu_record = measure(&plan, &p, "2026-10-22T00:00:00Z".into()).unwrap();
    assert_eq!(cpu_record.bins.len(), 2);
    assert_eq!(cpu_record.device.clock, None);
    check(&cpu_record).unwrap();

    if let Some(_turn) = crate::gpu_clock::tests::levelzero_turn("qa_tests.rs") {
        let mut gpu_plan = plan.clone();
        gpu_plan.device = "levelzero".into();
        let gpu_record = measure(&gpu_plan, &p, "2026-10-22T00:00:00Z".into()).unwrap();
        assert_eq!(gpu_record.device.backend, "levelzero");
        crate::gpu_clock::tests::assert_device_clock(gpu_record.device.clock);
        check(&gpu_record).unwrap();
    }

    std::fs::remove_dir_all(&scratch).unwrap();
}
