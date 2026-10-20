use super::*;

// Rule 6

#[test]
fn no_artifact_for_the_cpu_is_no_artifact_and_says_why() {
    let e = with("no-cpu", |m| m["artifacts"][0]["backends"] = json!(["cuda", "metal"]));
    assert!(e.is(BUNDLE_NO_ARTIFACT, "weights-f32: backends [\"cuda\", \"metal\"] has no cpu"), "{e:?}");
    let e = with("target", |m| m["artifacts"][0]["target"] = json!("rtx4080"));
    assert!(e.is(BUNDLE_NO_ARTIFACT, "target rtx4080 is not this device's"), "{e:?}");
}

/// An artifact listing cpu that this build cannot hand the cpu backend,
/// ahead of the weights.
fn openvino(f: &mut Fixture) {
    fs::create_dir_all(f.dir.join("openvino")).unwrap();
    fs::write(f.dir.join("openvino/model.xml"), b"<net/>").unwrap();
    fs::write(f.dir.join("openvino/model.bin"), b"\0\0\0\0").unwrap();
    f.list("openvino/model.xml");
    f.list("openvino/model.bin");
    let ir = json!({
        "name": "openvino-f16",
        "format": "FORMAT_OPENVINO_IR",
        "files": ["openvino/model.xml", "openvino/model.bin"],
        "backends": ["openvino", "cpu"],
        "compute_dtype": "DTYPE_F16",
        "graph_input": "INPUT_TOKEN_IDS",
        "graph_output": "OUTPUT_HIDDEN_STATES",
        "fixed_seq": 128,
        "fixed_batch": 1
    });
    f.manifest["artifacts"].as_array_mut().unwrap().insert(0, ir);
}

#[test]
fn the_first_artifact_the_device_can_load_is_chosen() {
    let mut f = Fixture::model("order");
    openvino(&mut f);
    // A second weights artifact for the cpu, after the first.
    f.weights("weights/other.safetensors", &tiny_weights(1));
    let mut other = f.manifest["artifacts"][1].clone();
    other["name"] = json!("weights-other");
    other["files"] = json!(["weights/other.safetensors"]);
    f.manifest["artifacts"].as_array_mut().unwrap().push(other);
    let info = f.load().unwrap().info();
    assert_eq!(field(&info.artifact_sha256), f.sha256("weights/model.safetensors"));
    assert_eq!(info.dtype, TURBO_DTYPE_F32, "the chosen artifact's dtype, not the one skipped");

    // Only what cannot be loaded: each is named with why.
    let mut f = Fixture::model("order-none");
    openvino(&mut f);
    f.manifest["artifacts"][1]["backends"] = json!(["cuda"]);
    let e = refused(&f);
    assert!(
        e.is(BUNDLE_NO_ARTIFACT, "openvino-f16: FORMAT_OPENVINO_IR is not a format the cpu backend loads"),
        "{e:?}"
    );
    assert!(e.message.contains("weights-f32: backends"), "{e:?}");
}

/// An OpenVINO IR is two files, the xml then its weights, with the
/// compute dtype its conversion fixed.
#[test]
fn an_openvino_ir_without_its_two_files_or_its_dtype_is_invalid() {
    let mut f = Fixture::model("ir-one-file");
    openvino(&mut f);
    f.manifest["artifacts"][0]["files"] = json!(["openvino/model.xml"]);
    let e = refused(&f);
    assert!(e.is(BUNDLE_INVALID, "artifacts[0].files: an OpenVINO IR is two files"), "{e:?}");
    let mut f = Fixture::model("ir-no-dtype");
    openvino(&mut f);
    f.manifest["artifacts"][0].as_object_mut().unwrap().remove("compute_dtype");
    let e = refused(&f);
    assert!(e.is(BUNDLE_INVALID, "artifacts[0].compute_dtype: required for a compiled OpenVINO IR"), "{e:?}");
}

/// A compiled OpenVINO IR names the static frame it was built for, the
/// same way a HEF does. Absent and zero are the same refusal.
#[test]
fn an_openvino_ir_without_a_fixed_shape_is_invalid() {
    for field in ["fixed_seq", "fixed_batch"] {
        let mut f = Fixture::model(&format!("ir-no-{field}"));
        openvino(&mut f);
        f.manifest["artifacts"][0].as_object_mut().unwrap().remove(field);
        let e = refused(&f);
        assert!(e.is(BUNDLE_INVALID, &format!("artifacts[0].{field}: required for a compiled OpenVINO IR")), "{e:?}");
        let mut f = Fixture::model(&format!("ir-zero-{field}"));
        openvino(&mut f);
        f.manifest["artifacts"][0][field] = json!(0);
        let e = refused(&f);
        assert!(e.is(BUNDLE_INVALID, &format!("artifacts[0].{field}: required for a compiled OpenVINO IR")), "{e:?}");
    }
}

#[test]
fn raw_weights_that_start_at_embeddings_or_fix_a_dtype_are_invalid() {
    let e = with("embeddings", |m| {
        m["artifacts"][0]["graph_input"] = json!("INPUT_EMBEDDINGS");
        m["artifacts"][0]["host_weights"] = json!("weights-f32");
    });
    assert!(e.is(BUNDLE_INVALID, "artifacts[0].graph_input: raw weights start at INPUT_TOKEN_IDS"), "{e:?}");
    // What raw weights compute in is the session's precision to say.
    let e = with("compute-dtype", |m| m["artifacts"][0]["compute_dtype"] = json!("DTYPE_F16"));
    assert!(e.is(BUNDLE_INVALID, "artifacts[0].compute_dtype: fixed by a compilation"), "{e:?}");
}

// Rule 6 and 7: a compiled HEF
