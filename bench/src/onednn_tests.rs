use super::*;
use std::path::{Path, PathBuf};
use std::ptr;

use serde_json::{Value, json};
use turbo::bundle::sha256_hex;
use turbo::{TURBO_PRECISION_EXACT, TURBO_PRECISION_MODEL, turbo_text};

use crate::git::Provenance;
use crate::measure::{self, Plan, RowKind};
use crate::qa::{self, QaPlan};
use crate::ranges::{self, RangePlan};
use crate::rerank::{self, RerankPlan};

const SAMPLE_WARNING: &str = "levelzero device 0: oneDNN not used, the backend's own kernels run the linear \
                              layers: levelzero: oneDNN: cannot open libturbo_onednn.so on the library path \
                              (libdnnl.so.3: cannot open shared object file: No such file or directory)";

#[test]
fn prints_one_line_with_logged_reason_when_onednn_cannot_open() {
    let mut out = Vec::new();
    let warnings = vec![
        "levelzero: building the encoder's kernels failed; building without the flag".to_owned(),
        SAMPLE_WARNING.to_owned(),
        SAMPLE_WARNING.to_owned(),
    ];
    write_fallback(&mut out, "levelzero", "", &warnings);
    let text = String::from_utf8(out).unwrap();
    assert_eq!(text.lines().count(), 1);
    assert_eq!(text, format!("turbo-bench: {SAMPLE_WARNING}\n"));
}

#[test]
fn collapses_multiline_logged_reason_onto_one_stderr_line() {
    let mut out = Vec::new();
    let multiline = "levelzero device 1: oneDNN not used, the backend's own kernels run the linear layers: \
                     sycl open failed\nmissing oneAPI runtime\n";
    write_fallback(&mut out, "levelzero", "", &[multiline.to_owned()]);
    let text = String::from_utf8(out).unwrap();
    assert_eq!(text.lines().count(), 1);
    assert_eq!(
        text,
        "turbo-bench: levelzero device 1: oneDNN not used, the backend's own kernels run the linear layers: sycl \
         open failed; missing oneAPI runtime\n"
    );
}

#[test]
fn stays_silent_when_onednn_runs_or_own_was_forced_or_backend_differs() {
    let warnings = vec![SAMPLE_WARNING.to_owned()];
    for (backend, choices, ws) in [
        ("levelzero", "linear=onednn;forced=", warnings.as_slice()),
        ("levelzero", "linear=onednn;forced=linear", warnings.as_slice()),
        ("levelzero", "linear=own;forced=linear", warnings.as_slice()),
        ("levelzero", "linear=own", warnings.as_slice()),
        ("levelzero", "", &[][..]),
        ("levelzero", "", &["levelzero: building without the flag".to_owned()][..]),
        ("cpu", "", warnings.as_slice()),
        ("cuda", "", warnings.as_slice()),
        ("metal", "", warnings.as_slice()),
        ("npu", "", warnings.as_slice()),
    ] {
        let mut out = Vec::new();
        write_fallback(&mut out, backend, choices, ws);
        assert!(out.is_empty(), "expected silence for backend={backend}, choices={choices}: {out:?}");
    }
}

#[test]
fn runtime_log_callback_and_out_of_order_handle_drop_keep_warnings_intact() {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let session = {
        let rt = crate::api::Runtime::create().unwrap();
        assert!(rt.warnings().is_empty());
        let cpu = rt.find("cpu").unwrap();
        let ctx = rt.context(cpu).unwrap();
        let tok = rt.tokenizer(&bundle).unwrap();
        assert!(tok.info().unwrap().vocab_size > 0);
        drop(rt);
        let model = ctx.load(&bundle).unwrap();
        drop(ctx);
        let s = model.session(1, 8, TURBO_PRECISION_EXACT).unwrap();
        drop(model);
        s
    };
    assert_eq!(session.info().unwrap().max_batch, 1);
    drop(session);

    let sink: std::sync::Arc<std::sync::Mutex<Vec<String>>> = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let user_data = std::sync::Arc::as_ptr(&sink).cast_mut().cast();
    for (level, msg) in [(0u32, "err"), (1, SAMPLE_WARNING), (2, "info"), (3, "debug")] {
        let t = turbo_text { ptr: msg.as_ptr().cast(), len: msg.len() as u64 };
        unsafe { crate::api::collect_warning(user_data, level, t) };
    }
    unsafe { crate::api::collect_warning(user_data, 1, turbo_text { ptr: ptr::null(), len: 0 }) };
    let recorded = sink.lock().unwrap().clone();
    assert_eq!(recorded, vec!["err".to_owned(), SAMPLE_WARNING.to_owned(), String::new()]);
}

fn tiny_qa_bundle(dir: &Path) -> PathBuf {
    let bundle = dir.join("qa-bundle");
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

fn tiny_rerank_bundle(dir: &Path) -> PathBuf {
    let bundle = dir.join("rerank-bundle");
    std::fs::create_dir_all(bundle.join("weights")).unwrap();
    std::fs::create_dir_all(bundle.join("reference")).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let tiny = root.join("testdata/tiny-bert-bundle");
    let reference = root.join("testdata/tiny-rerank-reference");
    let mut manifest: Value = serde_json::from_slice(&std::fs::read(tiny.join("manifest.json")).unwrap()).unwrap();
    manifest.as_object_mut().unwrap().remove("embed");
    manifest["task"] = "TASK_RERANK".into();
    manifest["rerank"] = json!({"max_seq": 32, "max_batch": 4, "query_tokens": 8, "pair_template": "PAIR_BERT"});
    manifest["classifier"] = json!({"kind": "CLASSIFIER_SEQUENCE", "labels": ["LABEL_0"]});
    manifest["artifacts"][0]["files"] = json!(["weights/model.safetensors", "weights/head.safetensors"]);
    let names = manifest["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    for (k, v) in [
        ("pooler_weight", "bert.pooler.dense.weight"),
        ("pooler_bias", "bert.pooler.dense.bias"),
        ("classifier_weight", "classifier.weight"),
        ("classifier_bias", "classifier.bias"),
    ] {
        names.insert(k.into(), v.into());
    }
    manifest["reference"]["cases"] =
        serde_json::from_slice(&std::fs::read(reference.join("cases.json")).unwrap()).unwrap();
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
fn record_ranges_qa_and_rerank_each_emit_at_most_one_onednn_fallback_line() {
    let scratch =
        turbo_scratch::Scratch::at(std::env::temp_dir().join(format!("turbo-bench-onednn-{}", std::process::id())));
    let tiny_bert = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let qa_bundle = tiny_qa_bundle(&scratch);
    let rerank_bundle = tiny_rerank_bundle(&scratch);
    let questions_path = scratch.join("squad.json");
    let long_context = vec!["Paris"; 140].join(" ");
    let squad = json!({
        "data": [{
            "paragraphs": [
                {"context": "Paris is in France.", "qas": [{"question": "Where is Paris?"}]},
                {"context": "Berlin is in Germany.", "qas": [{"question": "Where is Berlin?"}]},
                {"context": long_context, "qas": [{"question": "Where?"}]}
            ]
        }]
    });
    std::fs::write(&questions_path, serde_json::to_vec(&squad).unwrap()).unwrap();
    let prov = Provenance { top: String::new(), commit: "a".repeat(40), pushed_to: vec!["origin/main".into()] };

    let record_plan = Plan {
        bundle: tiny_bert.clone(),
        device: "cpu".into(),
        precision: TURBO_PRECISION_MODEL,
        batch: Some(2),
        seq: Some(8),
        rows: RowKind::Mixed,
        warmup: 0,
        iterations: 1,
    };
    let ranges_plan = RangePlan {
        bundle: tiny_bert,
        device: "cpu".into(),
        precision: TURBO_PRECISION_MODEL,
        batch: Some(2),
        seq: Some(8),
        chunk: 3,
        warmup: 0,
        iterations: 1,
    };
    let qa_plan = QaPlan {
        bundle: qa_bundle,
        device: "cpu".into(),
        precision: TURBO_PRECISION_MODEL,
        batch: 1,
        max_batch: 0,
        questions: questions_path.clone(),
        per_bin: 1,
        warmup: 0,
        iterations: 1,
    };
    let rerank_plan = RerankPlan {
        bundle: rerank_bundle,
        device: "cpu".into(),
        precision: TURBO_PRECISION_MODEL,
        questions: questions_path,
        requests: 1,
        passages: 2,
        warmup: 0,
        iterations: 1,
        tei: None,
    };

    let want = format!("turbo-bench: {SAMPLE_WARNING}\n");
    for run in [
        Box::new(|| {
            measure::measure(&record_plan).unwrap();
        }) as Box<dyn Fn()>,
        Box::new(|| {
            ranges::measure(&ranges_plan, &prov, "2026-10-22T00:00:00Z".into()).unwrap();
        }),
        Box::new(|| {
            let r = qa::measure(&qa_plan, &prov, "2026-10-22T00:00:00Z".into()).unwrap();
            assert_eq!(r.bins.len(), 2);
        }),
        Box::new(|| {
            rerank::measure(&rerank_plan, &prov, "2026-10-22T00:00:00Z".into()).unwrap();
        }),
    ] {
        let (_, cpu_cap) = capture_fallback(None, &run);
        assert_eq!(cpu_cap.calls.len(), 1);
        assert_eq!(cpu_cap.calls[0].0, "cpu");
        assert!(cpu_cap.calls[0].2.is_empty());
        assert!(cpu_cap.stderr.is_empty());

        let (_, fallback_cap) = capture_fallback(Some(("levelzero", "", vec![SAMPLE_WARNING.to_owned()])), &run);
        assert_eq!(fallback_cap.calls.len(), 1);
        assert_eq!(fallback_cap.stderr, want);

        let (_, onednn_cap) =
            capture_fallback(Some(("levelzero", "linear=onednn;forced=", vec![SAMPLE_WARNING.to_owned()])), &run);
        assert!(onednn_cap.stderr.is_empty());
    }
}
