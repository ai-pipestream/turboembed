use super::*;

#[test]
fn a_sealed_bundle_loads_through_the_core() {
    let (bundle, out) = sealed("loads", |_| {});
    out.expect("sealed and verified");
    let m: Value = serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    let pb = &m["reference"]["produced_by"];
    assert_eq!(pb["container"], CONTAINER);
    assert_eq!(pb["tool_version"], "6.1.0", "from the run, not the recipe");
    let paths: Vec<&str> = m["files"].as_array().unwrap().iter().map(|f| f["path"].as_str().unwrap()).collect();
    assert_eq!(
        paths,
        [
            "calibration/texts.jsonl",
            "hailo/model-hailo10h-s128.hef",
            "hailo/model-hailo8-s128.hef",
            "onnx/model-f16.onnx",
            "onnx/model.onnx",
            "openvino/embeddings.bin",
            "openvino/embeddings.xml",
            "openvino/model.bin",
            "openvino/model.xml",
            "reference/reference.safetensors",
            "tokenizer.json",
            "weights/model.safetensors"
        ]
    );
    // The converted artifact's produced_by is the run's, beside what the
    // recipe named.
    let f16 = m["artifacts"].as_array().unwrap().iter().find(|a| a["name"] == "onnx-f16").unwrap();
    assert_eq!(
        f16["produced_by"],
        json!({
            "tool": "onnxconverter-common",
            "tool_version": "1.16.0 (onnx 1.23.0)",
            "container": CONTAINER,
            "from": "onnx-f32",
            "args": ["onnx/model.onnx", "onnx/model-f16.onnx", "keep_io_types=True", "max_finite_val=10000.0", "min_positive_val=1e-07", "float_casts_into_f16_ops=FLOAT16"],
            "reproducible": true
        })
    );
    let ir = m["artifacts"].as_array().unwrap().iter().find(|a| a["name"] == "openvino-f16").unwrap();
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
    assert_eq!(ir["produced_by"]["tool"], "openvino.save_model");
    assert_eq!(ir["backends"], json!(["npu"]));
    // The HEF's produced_by names its container and calibration texts,
    // which are in the bundle, with the compile's settings from the run.
    let hef = m["artifacts"].as_array().unwrap().iter().find(|a| a["name"] == "hef-hailo10h-s128").unwrap();
    let pb = &hef["produced_by"];
    assert_eq!(pb["tool"], "hailo-dataflow-compiler");
    assert!(pb["container"].as_str().unwrap().starts_with("turbo-hailo-dfc@sha256:"));
    assert_eq!(pb["inputs"], json!(["calibration/texts.jsonl"]));
    assert_eq!(pb["args"], json!(["onnx/model.onnx", "hailo/model-hailo10h-s128.hef", "hw_arch=hailo10h", "seq=128"]));
    // The loader opens it as a machine would.
    seal::verify(&bundle).unwrap();
    let tok = bundle.join("tokenizer.json");
    let tok_entry = m["files"].as_array().unwrap().iter().find(|f| f["path"] == "tokenizer.json").unwrap();
    assert_eq!(
        tok_entry["sha256"].as_str().unwrap(),
        turbo::bundle::sha256_hex(&fs::read(tok).unwrap()),
        "hashes are computed, not copied from the recipe"
    );
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

/// The backends the core runs an artifact on.

#[test]
fn the_recipe_carries_upstreams_onnx_export_for_reference_programs_only() {
    let r = Recipe::load(&root().join("bundle/recipes/all-minilm-l6-v2.json")).unwrap();
    let arts = r.manifest["artifacts"].as_array().unwrap();
    let onnx: Vec<&Value> = arts.iter().filter(|a| a["format"] == "FORMAT_ONNX").collect();
    assert_eq!(onnx.len(), 2, "upstream's ONNX artifact and its F16 copy");
    assert_eq!(onnx[0]["name"], "onnx-f32");
    assert_eq!(onnx[0]["files"], json!(["onnx/model.onnx"]));
    assert_eq!(onnx[0]["backends"], json!([]), "no backend loads it");
    assert!(onnx[0].get("produced_by").is_none(), "the upstream file, unchanged");
    assert_eq!(arts[0]["format"], "FORMAT_SAFETENSORS", "manifest order: the weights come first");
    let up = r.upstream.iter().find(|u| u.path == "onnx/model.onnx").expect("fetched from upstream");
    assert_eq!(up.to.as_deref(), Some("onnx/model.onnx"), "and carried at the path the artifact names");
    assert!(seal::named_paths(&r.manifest).unwrap().contains("onnx/model.onnx"));

    // The F16 copy: made from onnx-f32 in the reference container, for
    // programs that build F16 only from a strongly typed graph.
    assert_eq!(onnx[1]["name"], "onnx-f16");
    assert_eq!(onnx[1]["compute_dtype"], "DTYPE_F16");
    assert_eq!(onnx[1]["backends"], json!([]), "no backend loads it");
    assert_eq!(onnx[1]["produced_by"], json!({ "from": "onnx-f32" }), "the rest comes from the run");
    assert!(r.upstream.iter().all(|u| u.to.as_deref() != Some("onnx/model-f16.onnx")), "not fetched");
    let c = convert::conversions(&r).unwrap();
    assert_eq!(
        c[0],
        convert::Conversion {
            name: "onnx-f16".into(),
            file: "onnx/model-f16.onnx".into(),
            from: "onnx-f32".into(),
            from_file: "onnx/model.onnx".into(),
            from_upstream: false,
            file2: None,
            script: convert::ONNX_F16,
            container: None,
            inputs: vec![],
            args: vec![],
        }
    );
}

/// The static-shape OpenVINO IR the npu backend loads: two files, made
/// from the upstream export in the reference container.
#[test]
fn an_openvino_ir_is_two_files_at_a_fixed_shape() {
    let r = Recipe::load(&root().join("bundle/recipes/all-minilm-l6-v2.json")).unwrap();
    let c = convert::conversions(&r).unwrap();
    let ir = c.iter().find(|c| c.name == "openvino-f16").expect("the recipe converts an IR");
    assert_eq!(
        ir,
        &convert::Conversion {
            name: "openvino-f16".into(),
            file: "openvino/model.xml".into(),
            file2: Some("openvino/model.bin".into()),
            from: "onnx-f32".into(),
            from_file: "onnx/model.onnx".into(),
            from_upstream: false,
            script: convert::ONNX_TO_OPENVINO_IR,
            container: None,
            inputs: vec![],
            args: vec!["--seq".into(), "128".into(), "--batch".into(), "1".into(), "--max-opset".into(), "11".into()],
        }
    );
    let cut = c.iter().find(|c| c.name == "openvino-embeddings-f16").expect("the recipe converts an embeddings IR");
    assert_eq!(cut.file, "openvino/embeddings.xml");
    assert_eq!(cut.file2.as_deref(), Some("openvino/embeddings.bin"));
    let want: Vec<String> = [
        "--seq",
        "128",
        "--batch",
        "1",
        "--max-opset",
        "11",
        "--cut",
        "embeddings",
        "--heads",
        "12",
        "--tokenizer",
        "/bundle/tokenizer.json",
        "--calibration",
        "/bundle/calibration/texts.jsonl",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(cut.args, want);
    let d = scratch("ir-conversion");
    let write = |edit: &dyn Fn(&mut Value)| {
        let mut r: Value = serde_json::from_slice(&fs::read(tiny_recipe(&d)).unwrap()).unwrap();
        edit(
            r["manifest"]["artifacts"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .find(|a| a["name"] == "openvino-f16")
                .unwrap(),
        );
        let p = d.join("edited.json");
        fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
        Recipe::load(&p).map(|_| ())
    };
    write(&|_| {}).unwrap();
    let e = write(&|a| {
        a.as_object_mut().unwrap().remove("fixed_seq");
    })
    .unwrap_err();
    assert!(e.contains("gives fixed_seq"), "{e}");
    let e = write(&|a| a["files"] = json!(["openvino/model.xml"])).unwrap_err();
    assert!(e.contains("two files"), "{e}");
    let e = write(&|a| a["compute_dtype"] = json!("DTYPE_F32")).unwrap_err();
    assert!(e.contains("DTYPE_F16 OpenVINO IR"), "{e}");
    fs::remove_dir_all(d).unwrap();
}
