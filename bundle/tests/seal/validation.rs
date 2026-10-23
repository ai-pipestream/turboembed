use super::*;

#[test]
fn only_an_f16_onnx_file_from_the_upstream_one_is_made() {
    let d = scratch("conversions");
    let write = |edit: &dyn Fn(&mut Value)| {
        let mut r: Value = serde_json::from_slice(&fs::read(tiny_recipe(&d)).unwrap()).unwrap();
        let arts = r["manifest"]["artifacts"].as_array_mut().unwrap();
        edit(arts.iter_mut().find(|a| a["name"] == "onnx-f16").unwrap());
        let p = d.join("edited.json");
        fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
        Recipe::load(&p).map(|_| ())
    };
    write(&|_| {}).unwrap();
    let e = write(&|a| a["produced_by"]["tool"] = json!("typed by hand")).unwrap_err();
    assert!(e.contains("names only from"), "{e}");
    let e = write(&|a| a["produced_by"]["from"] = json!("onnx-f64")).unwrap_err();
    assert!(e.contains("names no artifact"), "{e}");
    let e = write(&|a| a["compute_dtype"] = json!("DTYPE_BF16")).unwrap_err();
    assert!(e.contains("only a DTYPE_F16 FORMAT_ONNX file"), "{e}");
    let e = write(&|a| a["produced_by"]["from"] = json!("weights-f32")).unwrap_err();
    assert!(e.contains("only a DTYPE_F16 FORMAT_ONNX file"), "{e}");
    let e = write(&|a| a["files"] = json!(["onnx/a.onnx", "onnx/b.onnx"])).unwrap_err();
    assert!(e.contains("a converted artifact is one file"), "{e}");
    fs::remove_dir_all(d).unwrap();
}

/// Raw weights the tool writes from a PyTorch checkpoint: produced_by
/// names the checkpoint among the upstream files, which is not carried.
#[test]
fn raw_weights_come_from_an_upstream_checkpoint_the_bundle_does_not_carry() {
    let d = scratch("weights-conversion");
    let write = |edit: &dyn Fn(&mut Value)| {
        let mut r: Value = serde_json::from_slice(&fs::read(tiny_recipe(&d)).unwrap()).unwrap();
        r["upstream"].as_array_mut().unwrap().retain(|u| u["path"] != "model.safetensors");
        r["upstream"].as_array_mut().unwrap().push(json!({ "path": "pytorch_model.bin" }));
        let arts = r["manifest"]["artifacts"].as_array_mut().unwrap();
        let w = arts.iter_mut().find(|a| a["name"] == "weights-f32").unwrap();
        w["produced_by"] = json!({ "upstream": "pytorch_model.bin" });
        edit(&mut r);
        let p = d.join("edited.json");
        fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
        Recipe::load(&p).map(|r| convert::conversions(&r).unwrap())
    };
    let c = write(&|_| {}).unwrap();
    let w = c.iter().find(|c| c.name == "weights-f32").unwrap();
    assert_eq!(
        (w.from.as_str(), w.from_file.as_str(), w.from_upstream, w.script),
        ("", "pytorch_model.bin", true, convert::BIN_TO_SAFETENSORS)
    );
    let e = write(&|r| r["manifest"]["artifacts"][0]["produced_by"]["from"] = json!("onnx-f32")).unwrap_err();
    assert!(e.contains("names only upstream"), "{e}");
    let e = write(&|r| r["manifest"]["artifacts"][0]["produced_by"]["upstream"] = json!("model.bin")).unwrap_err();
    assert!(e.contains("is not in the recipe's upstream files"), "{e}");
    let e = write(&|r| r["upstream"][2]["to"] = json!("weights/pytorch_model.bin")).unwrap_err();
    assert!(e.contains("not carried"), "{e}");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn a_conversion_that_did_not_run_is_not_sealed() {
    let (bundle, out) = sealed("unconverted", |_| {});
    out.unwrap();
    let r = Recipe::load(&bundle.parent().unwrap().join("recipe.json")).unwrap();
    let pb = reference::produced_by(&reported(), CONTAINER).unwrap();
    let e = seal::seal(&r, &bundle, pb, vec![]).unwrap_err();
    assert!(
        e.contains(
            "the recipe converts [\"onnx-f16\", \"openvino-f16\", \"openvino-embeddings-f16\", \"hef-hailo10h-s128\", \
             \"hef-hailo8-s128\"], and the runs made []"
        ),
        "{e}"
    );
    // A second run that gave other bytes is recorded as such.
    let c = &convert::conversions(&r).unwrap()[0];
    assert_eq!(convert::produced_by(&reported_f16(), CONTAINER, c, false).unwrap()["reproducible"], false);
    assert!(convert::produced_by(&json!({ "tool": "x", "tool_version": "1" }), CONTAINER, c, true).is_err());
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

#[test]
fn the_onnx_file_is_copied_and_sealed_and_no_backend_chooses_it() {
    let (bundle, out) = sealed("onnx", |_| {});
    out.expect("sealed and verified");
    assert_eq!(fs::read(bundle.join("onnx/model.onnx")).unwrap(), ONNX, "copied byte for byte");
    let b = turbo::bundle::Bundle::open(&bundle).unwrap();
    let f = b.manifest.file("onnx/model.onnx");
    assert_eq!((f.size, f.sha256.as_str()), (ONNX.len() as u64, turbo::bundle::sha256_hex(ONNX).as_str()));
    assert_eq!(b.read_verified("onnx/model.onnx").unwrap(), ONNX);
    let i = b.manifest.artifacts.iter().position(|a| a.format == turbo::manifest::Format::Onnx).unwrap();
    assert_eq!(b.manifest.artifacts[i].files, ["onnx/model.onnx"]);
    assert!(b.manifest.artifacts[i].backends.is_empty());
    for backend in BACKENDS {
        // Not even a backend that says it loads every format.
        assert_ne!(turbo::model::choose(&b.manifest, backend, u32::MAX, "any").unwrap(), i, "{backend}");
    }
    let e = turbo::model::choose(&b.manifest, "openvino", u32::MAX, "any").unwrap_err();
    assert!(e.message.contains("onnx-f32: backends [] has no openvino"), "{}", e.message);

    // A changed ONNX file fails verification like any other.
    fs::write(bundle.join("onnx/model.onnx"), b"another graph").unwrap();
    let e = seal::verify(&bundle).unwrap_err();
    assert!(e.contains("onnx/model.onnx"), "{e}");
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

#[test]
fn a_recipe_without_the_onnx_upstream_file_is_not_sealed() {
    let (bundle, out) = sealed("no-onnx", |r| r.upstream.retain(|u| u.path != "onnx/model.onnx"));
    let e = out.unwrap_err();
    assert!(e.contains("onnx/model.onnx"), "{e}");
    assert!(!bundle.join("manifest.json").exists());
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

#[test]
fn a_reference_whose_ids_differ_is_refused() {
    // Another case text: the reference file's ids no longer match what the
    // core's tokenizer gives, which is loader rule 5.
    let (bundle, out) = sealed("ids", |r| {
        r.manifest["reference"]["cases"][1]["text"] = json!("The quick brown fox jumps over the lazy cat.");
    });
    let e = out.unwrap_err();
    assert!(e.contains("case") || e.contains("ids"), "{e}");
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

#[test]
fn a_reference_that_is_not_normalized_is_refused() {
    let (bundle, out) = sealed("norm", |_| {});
    out.unwrap();
    // Scale every vector: the ids still match, the norms no longer do.
    let path = bundle.join("reference/reference.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let [a, b] = [0, 1].map(|i| header["embeddings"]["data_offsets"][i].as_u64().unwrap() as usize + 8 + n);
    for c in bytes[a..b].as_chunks_mut::<4>().0 {
        let v = f32::from_le_bytes(*c) * 2.0;
        c.copy_from_slice(&v.to_le_bytes());
    }
    fs::write(&path, &bytes).unwrap();
    let r = Recipe::load(&bundle.parent().unwrap().join("recipe.json")).unwrap();
    let made = converted(&r, &bundle);
    let e = seal::seal(&r, &bundle, reference::produced_by(&reported(), CONTAINER).unwrap(), made).unwrap_err();
    assert!(e.contains("norm"), "{e}");
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

#[test]
fn verify_refuses_a_changed_file_and_an_unlisted_one() {
    let (bundle, out) = sealed("verify", |_| {});
    out.unwrap();
    let w = bundle.join("weights/model.safetensors");
    let mut bytes = fs::read(&w).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&w, &bytes).unwrap();
    let e = seal::verify(&bundle).unwrap_err();
    assert!(e.contains("weights/model.safetensors") && e.contains("SHA-256"), "{e}");

    bytes[last] ^= 1;
    fs::write(&w, &bytes).unwrap();
    seal::verify(&bundle).unwrap();
    fs::write(bundle.join("notes.txt"), "x").unwrap();
    let e = seal::verify(&bundle).unwrap_err();
    assert!(e.contains("notes.txt"), "{e}");
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

#[test]
fn recipes_are_checked() {
    let d = scratch("recipes");
    let write = |edit: &dyn Fn(&mut Value)| {
        let mut r: Value = serde_json::from_slice(&fs::read(tiny_recipe(&d)).unwrap()).unwrap();
        edit(&mut r);
        let p = d.join("edited.json");
        fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
        Recipe::load(&p).map(|_| ())
    };
    write(&|_| {}).unwrap();
    let e = write(&|r| r["manifest"]["files"] = json!([])).unwrap_err();
    assert!(e.contains("files"), "{e}");
    let e = write(&|r| r["manifest"]["model"]["source"]["commit"] = json!("main")).unwrap_err();
    assert!(e.contains("commit"), "{e}");
    let e = write(&|r| r["upstream"][0]["to"] = json!("../tokenizer.json")).unwrap_err();
    assert!(e.contains("relative"), "{e}");
    let e = write(&|r| r["upstream"][0]["sha256"] = json!("ABC")).unwrap_err();
    assert!(e.contains("sha256"), "{e}");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn the_container_is_pinned_by_content() {
    assert!(reference::check_pinned(CONTAINER).is_ok());
    for bad in ["turbo-reference", "turbo-reference:latest", "turbo-reference@sha256:abc", "x@sha256:ABCDEF"] {
        assert!(reference::check_pinned(bad).is_err(), "{bad}");
    }
}

#[test]
fn the_container_gets_the_cases_with_their_prefixes() {
    let d = scratch("cases");
    let r = Recipe::load(&tiny_recipe(&d)).unwrap();
    let c = reference::cases(&r).unwrap();
    assert_eq!((c["max_seq"].as_u64(), c["max_batch"].as_u64()), (Some(64), Some(64)));
    let cases = c["cases"].as_array().unwrap();
    assert_eq!(cases.len(), r.manifest["reference"]["cases"].as_array().unwrap().len());
    assert_eq!(cases[5], json!({ "text": "how do I reset a password", "prefix": "query: " }));
    assert_eq!(cases[6]["prefix"], "", "the recipe has no document prefix");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn produced_by_comes_from_the_run() {
    let pb = reference::produced_by(&reported(), CONTAINER).unwrap();
    assert_eq!(pb["reproducible"], false);
    assert!(reference::produced_by(&json!({ "tool": "x", "args": [] }), CONTAINER).is_err());
    assert!(reference::produced_by(&json!({ "tool": "x", "tool_version": "1", "args": [1] }), CONTAINER).is_err());
}

#[test]
fn upstream_files_are_fetched_at_the_commit() {
    assert_eq!(
        turbo_bundle::fetch::url(
            "https://huggingface.co/org/model/",
            "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            "1_Pooling/config.json"
        ),
        "https://huggingface.co/org/model/resolve/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/1_Pooling/config.json"
    );
}

#[test]
fn a_hef_is_compiled_from_the_export_in_a_pinned_container() {
    let d = scratch("hef");
    let write = |edit: &dyn Fn(&mut Value)| {
        let mut r: Value = serde_json::from_slice(&fs::read(tiny_recipe(&d)).unwrap()).unwrap();
        let arts = r["manifest"]["artifacts"].as_array_mut().unwrap();
        edit(arts.iter_mut().find(|a| a["name"] == "hef-hailo10h-s128").unwrap());
        let p = d.join("edited.json");
        fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
        Recipe::load(&p).map(|r| convert::conversions(&r).unwrap())
    };
    // Absent, compiler_optimization is the compiler's default.
    let c = write(&|a| {
        a["produced_by"].as_object_mut().unwrap().remove("compiler_optimization");
    })
    .unwrap();
    let hef = c.iter().find(|c| c.name == "hef-hailo10h-s128").unwrap();
    assert_eq!(hef.script, convert::HEF_COMPILE);
    assert_eq!(hef.inputs, ["calibration/texts.jsonl"]);
    assert_eq!(
        hef.args,
        [
            "--tokenizer",
            "/bundle/tokenizer.json",
            "--calibration",
            "/bundle/calibration/texts.jsonl",
            "--target",
            "hailo10h",
            "--seq",
            "128",
            "--heads",
            "4"
        ]
    );
    // compiler_optimization "max" is passed on.
    let c = write(&|a| a["produced_by"]["compiler_optimization"] = json!("max")).unwrap();
    let hef = c.iter().find(|c| c.name == "hef-hailo10h-s128").unwrap();
    assert_eq!(hef.args[10..], ["--compiler-optimization", "max"]);
    let refused = |edit: &dyn Fn(&mut Value), says: &str| {
        let e = write(edit).unwrap_err();
        assert!(e.contains(says), "{says}: {e}");
    };
    refused(&|a| a["produced_by"]["tool"] = json!("typed by hand"), "names from, container and inputs");
    refused(&|a| a["produced_by"]["container"] = json!("turbo-hailo-dfc:latest"), "sha256");
    refused(&|a| a["produced_by"]["inputs"] = json!([]), "one file of calibration texts");
    refused(&|a| a["produced_by"]["from"] = json!("onnx-f16"), "export with no compute_dtype");
    refused(&|a| a["produced_by"]["from"] = json!("weights-f32"), "export with no compute_dtype");
    refused(&|a| a["compute_dtype"] = json!("DTYPE_F16"), "DTYPE_I8 HEF");
    refused(&|a| a["graph_input"] = json!("INPUT_TOKEN_IDS"), "DTYPE_I8 HEF");
    refused(&|a| a["fixed_batch"] = json!(4), "fixed_batch 1");
    refused(&|a| a["target"] = json!(""), "names its target");
    refused(&|a| a["produced_by"]["compiler_optimization"] = json!("fast"), "\"max\" or absent");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn a_recipe_file_that_changed_is_not_staged() {
    let d = scratch("local");
    let r = Recipe::load(&tiny_recipe(&d)).unwrap();
    let f = d.join(&r.local[0].path);
    let mut bytes = fs::read(&f).unwrap();
    bytes.extend_from_slice(b"{\"text\": \"one more\"}\n");
    fs::write(&f, bytes).unwrap();
    let e = seal::stage(&r, &upstream(&d), &d.join("bundle")).unwrap_err();
    assert!(e.contains("the recipe pins"), "{e}");
    fs::remove_dir_all(d).unwrap();
}

/// The pin is the LF bytes in the commit. A Windows checkout that
/// rewrote CR LF hashes to the other value and make refuses it.
#[test]
fn the_calibration_pin_is_the_committed_lf_bytes() {
    let path = root().join("bundle/recipes/all-minilm-l6-v2.calibration.jsonl");
    let bytes = fs::read(&path).unwrap();
    assert!(!bytes.contains(&b'\r'), "the committed calibration file is LF");
    let r = Recipe::load(&root().join("bundle/recipes/all-minilm-l6-v2.json")).unwrap();
    let pin = &r.local.iter().find(|l| l.path.ends_with(".jsonl")).unwrap().sha256;
    assert_eq!(turbo::bundle::sha256_hex(&bytes), *pin);
    let mut crlf = Vec::with_capacity(bytes.len() + bytes.iter().filter(|b| **b == b'\n').count());
    for b in &bytes {
        if *b == b'\n' {
            crlf.push(b'\r');
        }
        crlf.push(*b);
    }
    assert_eq!(turbo::bundle::sha256_hex(&crlf), "9e6ac4eb099b4934d86adb2af92fdc3d5817b84ecd9d7bf1445d1b832d37d940");
}
