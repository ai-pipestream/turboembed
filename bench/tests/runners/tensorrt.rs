use super::*;

fn trt(work: &Path) -> TensorRt {
    TensorRt {
        image: format!("nvcr.io/nvidia/tensorrt@sha256:{DIGEST}"),
        trtexec: "trtexec".into(),
        inputs: vec!["input_ids".into(), "attention_mask".into(), "token_type_ids".into()],
        input_dtype: "int64".into(),
        warmup_ms: 1000,
        work: work.to_owned(),
        binary: None,
    }
}

#[test]
fn trtexec_is_run_with_every_setting_on_its_command_line() {
    let rows = Rows {
        kind: RowKind::Mixed,
        batch: 2,
        seq: 3,
        ids: vec![0; 6],
        mask: vec![0; 6],
        types: vec![0; 6],
        cases: vec![0, 1],
    };
    let t = trt(Path::new("/tmp"));
    let (graph, flags) = tensorrt::precision(TURBO_DTYPE_F32).unwrap();
    assert_eq!(graph, None, "F32: the upstream graph");
    let a = tensorrt::run_argv(&t, Path::new("/b"), Path::new("/w"), "onnx/model.onnx", 0, &rows, 200, &flags);
    let want = strings(&[
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--gpus",
        "device=0",
        "--mount",
        "type=bind,src=/b,dst=/bundle,readonly",
        "--mount",
        "type=bind,src=/w,dst=/work,readonly",
        &t.image,
        "trtexec",
        "--onnx=/bundle/onnx/model.onnx",
        "--shapes=input_ids:2x3,attention_mask:2x3,token_type_ids:2x3",
        "--loadInputs=input_ids:/work/input_ids.bin,attention_mask:/work/attention_mask.bin,token_type_ids:/work/token_type_ids.bin",
        "--warmUp=1000",
        "--iterations=200",
        "--duration=0",
        "--percentile=99",
        "--noTF32",
    ]);
    assert_eq!(a, want);
    // TensorRT 11 has no --fp16 or --bf16: F16 and BF16 build strongly
    // typed from the graph converted to that dtype, as TensorRT 10 can.
    assert_eq!(tensorrt::precision(TURBO_DTYPE_F16).unwrap(), (Some(Dtype::F16), strings(&["--stronglyTyped"])));
    assert_eq!(tensorrt::precision(TURBO_DTYPE_BF16).unwrap(), (Some(Dtype::Bf16), strings(&["--stronglyTyped"])));
    assert!(tensorrt::precision(8).is_err());
}

#[test]
fn the_machines_trtexec_is_pinned_by_its_bytes_and_run_in_place_of_the_container() {
    let d = upstream_dir("trtexec-native-pin");
    let bin = d.join("trtexec");
    std::fs::write(&bin, b"an engine builder").unwrap();
    let pin = tensorrt::native_pin(&bin).unwrap();
    assert_eq!(pin, format!("trtexec@sha256:{}", turbo::bundle::sha256_hex(b"an engine builder")));
    assert!(turbo::record::pinned(&pin).is_some(), "{pin} is pinned as a record reads it");
    assert!(tensorrt::native_pin(&d.join("absent")).unwrap_err().contains("--tensorrt-bin"));

    let rows = Rows {
        kind: RowKind::Mixed,
        batch: 2,
        seq: 3,
        ids: vec![0; 6],
        mask: vec![0; 6],
        types: vec![0; 6],
        cases: vec![0, 1],
    };
    let mut t = trt(Path::new("/tmp"));
    t.image = String::new();
    t.binary = Some(bin.clone());
    let (_, flags) = tensorrt::precision(TURBO_DTYPE_F16).unwrap();
    let a = tensorrt::native_argv(
        &t,
        tensorrt::TRTEXEC_BIN,
        Path::new("/b"),
        Path::new("/w"),
        "onnx/model-f16.onnx",
        1,
        &rows,
        200,
        &flags,
    );
    // No docker: the binary, the GPU by its ordinal, and the same flags on
    // the directories as they are.
    let want = strings(&[
        "<trtexec-bin>",
        "--device=1",
        "--onnx=/b/onnx/model-f16.onnx",
        "--shapes=input_ids:2x3,attention_mask:2x3,token_type_ids:2x3",
        "--loadInputs=input_ids:/w/input_ids.bin,attention_mask:/w/attention_mask.bin,token_type_ids:/w/token_type_ids.bin",
        "--warmUp=1000",
        "--iterations=200",
        "--duration=0",
        "--percentile=99",
        "--stronglyTyped",
    ]);
    assert_eq!(a, want);
    let in_container =
        tensorrt::run_argv(&t, Path::new("/b"), Path::new("/w"), "onnx/model-f16.onnx", 1, &rows, 200, &flags);
    assert!(in_container.contains(&want[3]), "the same shapes");
    assert_eq!(&in_container[in_container.len() - 5..], &want[5..], "the same timing and precision flags");
}

#[test]
fn trtexec_inputs_are_the_rows_in_the_exports_element_type() {
    assert_eq!(tensorrt::input_bytes(&[1, -2], "int32").unwrap(), [1, 0, 0, 0, 0xfe, 0xff, 0xff, 0xff]);
    assert_eq!(tensorrt::input_bytes(&[1], "int64").unwrap(), [1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(tensorrt::input_bytes(&[-1], "int64").unwrap(), [0xff; 8]);
    assert!(tensorrt::input_bytes(&[1], "fp32").is_err());
}

#[test]
fn trtexec_input_names_are_plain_names() {
    let names = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    tensorrt::check_inputs(&names(&["input_ids", "attention_mask", "token_type_ids"])).unwrap();
    tensorrt::check_inputs(&names(&["onnx::Input.0"])).unwrap_err();
    tensorrt::check_inputs(&names(&["input.1", "Mask_2"])).unwrap();
    for bad in [".", "..", "", "../x", "a/b", "a,b", "a:b", "a b", "-x"] {
        let e = tensorrt::check_inputs(&names(&[bad])).unwrap_err();
        assert!(e.contains("is not an input name of [A-Za-z0-9_.]+"), "{bad:?}: {e}");
    }
}

#[test]
fn trtexec_output_gives_its_version_and_summary() {
    let s = tensorrt::parse(TRTEXEC_OUT).unwrap();
    assert_eq!(s.version, "10.3.0");
    assert_eq!(s.queries, 200);
    assert_eq!(s.qps, 2699.06);
    assert_eq!((s.latency_median, s.latency_p99), (0.372559, 0.412598), "Latency, not H2D or D2H Latency");
    assert_eq!((s.compute_median, s.compute_p99), (0.348389, 0.385986));
}

#[test]
fn trtexec_output_without_a_pass_or_a_figure_is_refused() {
    let failed = TRTEXEC_OUT.replace("&&&& PASSED", "&&&& FAILED");
    assert!(tensorrt::parse(&failed).unwrap_err().contains("did not report &&&& PASSED"));
    let no_p99 = TRTEXEC_OUT.replace(", percentile(99%) = 0.412598 ms", "");
    assert!(tensorrt::parse(&no_p99).unwrap_err().contains("has no percentile(99%)"));
    let no_version = TRTEXEC_OUT.replace("TensorRT version: 10.3.0", "Format: ONNX");
    assert!(tensorrt::parse(&no_version).unwrap_err().contains("no TensorRT version line"));
    let no_trace = TRTEXEC_OUT.replace("Timing trace has", "Timing");
    assert!(tensorrt::parse(&no_trace).unwrap_err().contains("no Timing trace line"));
}

#[test]
fn a_bundle_without_onnx_is_recorded_as_trtexec_not_run() {
    let m = cpu_measurement();
    let work = scratch("trt-work");
    let r = tensorrt::run(&trt(&work), m, 0, 10).unwrap();
    assert_eq!(r.name, "tensorrt");
    assert_eq!(r.role, "kernel");
    assert!(r.measured.is_none());
    assert_eq!(
        r.not_run.as_deref(),
        Some("the bundle carries no FORMAT_ONNX artifact for trtexec to build an engine from")
    );
    assert!(r.commands.is_empty(), "nothing ran");
    assert_eq!(std::fs::read_dir(&*work).unwrap().count(), 0, "no input was written");
    let mut unpinned = trt(&work);
    unpinned.image = "nvcr.io/nvidia/tensorrt:24.08-py3".into();
    assert!(tensorrt::run(&unpinned, m, 0, 10).unwrap_err().contains("is not pinned"));
    // The machine's trtexec: pinned by its bytes, and the same reference
    // that says why it did not run, with no docker in it.
    let bin = work.join("trtexec");
    std::fs::write(&bin, b"an engine builder").unwrap();
    let mut native = trt(&work);
    native.image = String::new();
    native.binary = Some(bin.clone());
    let r = tensorrt::run(&native, m, 0, 10).unwrap();
    assert_eq!(r.pinned, tensorrt::native_pin(&bin).unwrap());
    assert!(r.procedure.contains("the machine's own (--tensorrt-bin)"), "{}", r.procedure);
    assert!(r.measured.is_none() && r.not_run.is_some() && r.commands.is_empty());
    native.binary = Some(work.join("absent"));
    assert!(tensorrt::run(&native, m, 0, 10).unwrap_err().contains("--tensorrt-bin"));
}

/// trtexec's output when TensorRT refuses the graph, in the form its
/// samples print it (logger.cpp tags errors `[E]`).
const TRTEXEC_FAILED: &str = "\
&&&& RUNNING TensorRT.trtexec [TensorRT v110201] [b3] # trtexec --onnx=/bundle/onnx/model-f16.onnx --stronglyTyped
[09/25/2026-10:00:00] [I] TensorRT version: 11.2.1
[09/25/2026-10:00:02] [E] [TRT] ITensor::getDimensions: Error Code 4: API Usage Error (/Sub: ElementWiseOperation SUB must have same input types. But they are of types Half and Float.)
[09/25/2026-10:00:02] [E] Failed to parse onnx file
[09/25/2026-10:00:02] [E] Engine set up failed
&&&& FAILED TensorRT.trtexec [TensorRT v110201] [b3] # trtexec --onnx=/bundle/onnx/model-f16.onnx --stronglyTyped
";

#[test]
fn a_program_that_fails_is_a_reason_and_docker_failing_is_an_error() {
    // A real process that prints trtexec's failure and exits 1, as trtexec
    // does in its container; docker passes the code on.
    let mut log = docker::Log::default();
    let argv = strings(&["sh", "-c", "printf '%s' \"$0\"; exit 1", TRTEXEC_FAILED]);
    let ran = log.run_program(&argv, strings(&["trtexec"]), &tensorrt::ERROR_TAGS).unwrap();
    let why = "exited with code 1: [TRT] ITensor::getDimensions: Error Code 4: API Usage Error (/Sub: \
               ElementWiseOperation SUB must have same input types. But they are of types Half and Float.)";
    assert_eq!(ran, docker::Ran::Failed(why.into()));
    assert_eq!(log.commands, [strings(&["trtexec"])], "the command is recorded");

    // Success gives the output; docker's own codes and no error line.
    let ok = log.run_program(&strings(&["sh", "-c", "echo fine"]), vec![], &tensorrt::ERROR_TAGS).unwrap();
    assert_eq!(ok, docker::Ran::Done("fine\n".into()));
    for code in [125, 126, 127] {
        let e = log.run_program(&strings(&["sh", "-c", &format!("exit {code}")]), vec![], &[]).unwrap_err();
        assert!(e.contains(&format!("exit status: {code}")), "{e}");
    }
    let bare = log.run_program(&strings(&["sh", "-c", "echo; echo last words >&2; exit 3"]), vec![], &[]).unwrap();
    assert_eq!(bare, docker::Ran::Failed("exited with code 3: last words".into()));
    let silent = log.run_program(&strings(&["sh", "-c", "exit 2"]), vec![], &[]).unwrap();
    assert_eq!(silent, docker::Ran::Failed("exited with code 2: no output".into()));
    assert_eq!(
        docker::failure(
            Some(1),
            "[ INFO ] Loading\n[ ERROR ] Device with \"GPU\" name is not registered\n",
            &openvino::ERROR_TAGS
        ),
        Some("exited with code 1: Device with \"GPU\" name is not registered".into())
    );
    assert_eq!(docker::failure(None, "killed", &[]), None, "a signal is not the program's answer");
}

// ---- the bundle's ONNX file ----

/// The MiniLM recipe's manifest, sealed over stand-in files: every path it
/// names listed with a size and hash, and the reference's produced_by as a
/// run fills it in.
fn recipe_manifest() -> turbo::manifest::Manifest {
    let r: serde_json::Value =
        serde_json::from_slice(&std::fs::read(workspace().join("bundle/recipes/all-minilm-l6-v2.json")).unwrap())
            .unwrap();
    let mut m = r["manifest"].clone();
    let run =
        serde_json::json!({ "tool": "t", "tool_version": "1", "container": "c", "args": [], "reproducible": false });
    m["reference"]["produced_by"] = run.clone();
    for a in m["artifacts"].as_array_mut().unwrap() {
        if let Some(from) = a.get("produced_by").map(|p| p["from"].clone()) {
            a["produced_by"] = run.clone();
            a["produced_by"]["from"] = from;
        }
    }
    let paths = [
        "tokenizer.json",
        "weights/model.safetensors",
        "onnx/model.onnx",
        "onnx/model-f16.onnx",
        "openvino/model.xml",
        "openvino/model.bin",
        "openvino/embeddings.xml",
        "openvino/embeddings.bin",
        "hailo/model-hailo10h-s128.hef",
        "hailo/model-hailo8-s128.hef",
        "reference/reference.safetensors",
    ];
    m["files"] = paths.iter().map(|p| serde_json::json!({ "path": p, "size": 1, "sha256": "0".repeat(64) })).collect();
    turbo::manifest::Manifest::parse(&serde_json::to_vec(&m).unwrap()).unwrap_or_else(|e| panic!("{}", e.message))
}

#[test]
fn the_runners_find_the_recipes_onnx_file_by_its_format() {
    let m = recipe_manifest();
    assert_eq!(turbo_bench::onnx::file(&m, None, "for a program").unwrap(), "onnx/model.onnx");
    assert_eq!(openvino::model_file(openvino::DEVICE_GPU, &m).unwrap(), "onnx/model.onnx");
    assert_eq!(openvino::model_file(openvino::DEVICE_NPU, &m).unwrap(), "openvino/model.xml");
    assert_eq!(turbo_bench::onnx::file(&m, Some(Dtype::F16), "for a program").unwrap(), "onnx/model-f16.onnx");
    assert_eq!(
        turbo_bench::onnx::file(&m, Some(Dtype::Bf16), "for a program").unwrap_err(),
        "the bundle carries no FORMAT_ONNX artifact in DTYPE_BF16 for a program"
    );
    let bundle = turbo::bundle::Bundle::open(&tiny_bundle()).unwrap();
    assert_eq!(
        turbo_bench::onnx::file(&bundle.manifest, None, "for a program").unwrap_err(),
        "the bundle carries no FORMAT_ONNX artifact for a program"
    );
    // The inputs both runners default to are the export's.
    assert_eq!(trt(Path::new("/w")).inputs, ["input_ids", "attention_mask", "token_type_ids"]);
    assert_eq!(ov(Path::new("/w")).inputs, trt(Path::new("/w")).inputs);
    assert_eq!(
        (trt(Path::new("/w")).input_dtype.as_str(), ov(Path::new("/w")).input_dtype.as_str()),
        ("int64", "int64")
    );
}

// ---- OpenVINO ----
