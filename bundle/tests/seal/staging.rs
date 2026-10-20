use super::*;

/// The files a host IR conversion leaves, plus a reference copied from a
/// bundle the pinned image already sealed. No container is run.
fn staged_ir(dir: &Path) -> (Recipe, PathBuf, PathBuf) {
    let recipe = tiny_recipe(dir);
    let r = Recipe::load(&recipe).unwrap();
    let bundle = dir.join("bundle");
    fs::create_dir_all(bundle.join("openvino")).unwrap();
    fs::write(bundle.join("openvino/model.xml"), b"<ir>").unwrap();
    fs::write(bundle.join("openvino/model.bin"), b"<weights>").unwrap();
    fs::write(bundle.join("openvino/report.json"), serde_json::to_vec(&reported_ir()).unwrap()).unwrap();
    fs::create_dir_all(bundle.join("reference")).unwrap();
    fs::copy(
        root().join("testdata/tiny-bert-reference/reference.safetensors"),
        bundle.join("reference/reference.safetensors"),
    )
    .unwrap();
    let container = r.str_at("/reference/produced_by/container").unwrap();
    fs::write(
        bundle.join("reference/report.json"),
        serde_json::to_vec(&json!({
            "tool": "sentence-transformers",
            "tool_version": "6.1.0",
            "container": container,
            "args": ["--device", "cpu"]
        }))
        .unwrap(),
    )
    .unwrap();
    let upstream = upstream(dir);
    (r, upstream, bundle)
}

#[test]
fn a_staged_ir_seals_from_the_image_the_report_names() {
    let d = scratch("staged-ir");
    let (mut r, upstream, bundle) = staged_ir(&d);
    seal::seal_staged(&mut r, &upstream, &bundle).expect("sealed from the staged IR");
    let m: Value = serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    let names: Vec<&str> = m["artifacts"].as_array().unwrap().iter().map(|a| a["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["weights-f32", "onnx-f32", "openvino-f16"], "the conversions that were not staged are omitted");
    let ir = m["artifacts"].as_array().unwrap().iter().find(|a| a["name"] == "openvino-f16").unwrap();
    assert_eq!(ir["produced_by"]["container"], CONTAINER);
    assert_eq!(ir["produced_by"]["reproducible"], json!(true));
    assert_eq!(ir["produced_by"]["tool"], "openvino.save_model");
    assert_eq!(
        ir["produced_by"]["args"],
        json!([
            "onnx/model.onnx",
            "openvino/model.xml",
            "openvino/model.bin",
            "seq=128",
            "batch=1",
            "max_opset=11",
            "compress_to_fp16=True",
            "output_precision=FP16"
        ])
    );
    let paths: Vec<&str> = m["files"].as_array().unwrap().iter().map(|f| f["path"].as_str().unwrap()).collect();
    assert!(!paths.contains(&"calibration/texts.jsonl"), "the HEF's texts are not carried when the HEF is omitted");
    assert!(!bundle.join("openvino/report.json").exists(), "the report is not a bundle file");
    assert!(!bundle.join("reference/report.json").exists());
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn a_staged_ir_refuses_a_host_container() {
    let d = scratch("staged-ir-host");
    let (mut r, upstream, bundle) = staged_ir(&d);
    let mut report: Value = serde_json::from_slice(&fs::read(bundle.join("openvino/report.json")).unwrap()).unwrap();
    report["container"] = json!("host");
    fs::write(bundle.join("openvino/report.json"), serde_json::to_vec(&report).unwrap()).unwrap();
    let e = seal::seal_staged(&mut r, &upstream, &bundle).unwrap_err();
    assert!(e.contains("openvino-f16") && e.contains("no container"), "{e}");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn a_staged_reference_refuses_a_host_container() {
    let d = scratch("staged-ref-host");
    let (mut r, upstream, bundle) = staged_ir(&d);
    let mut report: Value = serde_json::from_slice(&fs::read(bundle.join("reference/report.json")).unwrap()).unwrap();
    report["container"] = json!("host");
    fs::write(bundle.join("reference/report.json"), serde_json::to_vec(&report).unwrap()).unwrap();
    let e = seal::seal_staged(&mut r, &upstream, &bundle).unwrap_err();
    assert!(e.contains("reference") && e.contains("no container"), "{e}");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn a_staged_ir_refuses_a_report_that_omits_reproducible() {
    let d = scratch("staged-ir-repro");
    let (mut r, upstream, bundle) = staged_ir(&d);
    let mut report: Value = serde_json::from_slice(&fs::read(bundle.join("openvino/report.json")).unwrap()).unwrap();
    report.as_object_mut().unwrap().remove("reproducible");
    fs::write(bundle.join("openvino/report.json"), serde_json::to_vec(&report).unwrap()).unwrap();
    let e = seal::seal_staged(&mut r, &upstream, &bundle).unwrap_err();
    assert!(e.contains("reproducible"), "{e}");
    fs::remove_dir_all(d).unwrap();
}

/// What the embeddings cut reports, in the form onnx_to_openvino_ir.py writes it.
fn reported_embeddings() -> Value {
    json!({
        "tool": "openvino.save_model",
        "tool_version": "2026.3.0",
        "container": CONTAINER,
        "reproducible": true,
        "settings": [
            "seq=128",
            "batch=1",
            "max_opset=11",
            "compress_to_fp16=True",
            "output_precision=FP16",
            "cut=embeddings",
            "heads=12",
            "masked=-100.0",
            "calibration_texts=180",
            "onnx_cut_max_abs_diff=3.34e-06",
            "cut_max_abs_diff=0.012"
        ],
    })
}

#[test]
fn both_openvino_irs_bind_their_own_reports() {
    let d = scratch("both-irs");
    let (mut r, upstream, bundle) = staged_ir(&d);
    fs::write(bundle.join("openvino/embeddings.xml"), b"<emb-ir>").unwrap();
    fs::write(bundle.join("openvino/embeddings.bin"), b"<emb-weights>").unwrap();
    fs::write(bundle.join("openvino/embeddings.xml.report.json"), serde_json::to_vec(&reported_embeddings()).unwrap())
        .unwrap();
    seal::seal_staged(&mut r, &upstream, &bundle).expect("both irs, each with its own report");
    let m: Value = serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    let args = |name: &str| -> Vec<String> {
        m["artifacts"].as_array().unwrap().iter().find(|a| a["name"] == name).unwrap()["produced_by"]["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect()
    };
    let token = args("openvino-f16");
    let emb = args("openvino-embeddings-f16");
    assert!(!token.iter().any(|s| s.contains("cut=")), "the token-id IR kept its own report: {token:?}");
    assert!(emb.iter().any(|s| s == "cut=embeddings"), "{emb:?}");
    assert!(emb.iter().any(|s| s == "cut_max_abs_diff=0.012"), "{emb:?}");
    assert!(!bundle.join("openvino/report.json").exists());
    assert!(!bundle.join("openvino/embeddings.xml.report.json").exists());
    fs::remove_dir_all(d).unwrap();

    // The directory report.json is the token-id receipt. Without
    // embeddings.xml.report.json the embeddings artifact would bind it,
    // and that is refused.
    let d = scratch("both-irs-shared-report");
    let (mut r, upstream, bundle) = staged_ir(&d);
    fs::write(bundle.join("openvino/embeddings.xml"), b"<emb-ir>").unwrap();
    fs::write(bundle.join("openvino/embeddings.bin"), b"<emb-weights>").unwrap();
    let e = seal::seal_staged(&mut r, &upstream, &bundle).unwrap_err();
    assert!(e.contains("openvino-embeddings-f16") && e.contains("cut=embeddings") && e.contains("report.json"), "{e}");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn a_staged_ir_without_its_report_or_its_reference_is_refused() {
    let d = scratch("staged-no-report");
    let (mut r, upstream, bundle) = staged_ir(&d);
    fs::remove_file(bundle.join("openvino/report.json")).unwrap();
    let e = seal::seal_staged(&mut r, &upstream, &bundle).unwrap_err();
    assert!(e.contains("no report"), "{e}");
    fs::remove_dir_all(d).unwrap();

    let d = scratch("staged-partial");
    let (mut r, upstream, bundle) = staged_ir(&d);
    fs::remove_file(bundle.join("openvino/model.bin")).unwrap();
    let e = seal::seal_staged(&mut r, &upstream, &bundle).unwrap_err();
    assert!(e.contains("model.bin"), "{e}");
    fs::remove_dir_all(d).unwrap();

    let d = scratch("staged-no-reference");
    let (mut r, upstream, bundle) = staged_ir(&d);
    fs::remove_file(bundle.join("reference/reference.safetensors")).unwrap();
    let e = seal::seal_staged(&mut r, &upstream, &bundle).unwrap_err();
    assert!(e.contains("reference/reference.safetensors"), "{e}");
    fs::remove_dir_all(d).unwrap();
}
