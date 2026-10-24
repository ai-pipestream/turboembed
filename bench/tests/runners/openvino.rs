use super::*;

pub(super) fn ov(work: &Path) -> OpenVino {
    OpenVino {
        image: format!("openvino/ubuntu24_dev@sha256:{DIGEST}"),
        benchmark_app: "benchmark_app".into(),
        inputs: vec!["input_ids".into(), "attention_mask".into(), "token_type_ids".into()],
        input_dtype: "int64".into(),
        device: openvino::DEVICE_GPU.into(),
        dri: "/dev/dri".into(),
        accel: "/dev/accel/accel0".into(),
        work: work.to_owned(),
        binary: None,
    }
}

#[test]
fn benchmark_app_is_run_with_every_setting_on_its_command_line() {
    let rows = Rows {
        kind: RowKind::Mixed,
        batch: 2,
        seq: 3,
        ids: vec![0; 6],
        mask: vec![0; 6],
        types: vec![0; 6],
        cases: vec![0, 1],
    };
    let o = ov(Path::new("/tmp"));
    let precision = openvino::infer_precision(openvino::DEVICE_GPU, TURBO_DTYPE_F32).unwrap();
    let a = openvino::run_argv(&o, Path::new("/b"), Path::new("/w"), "onnx/model.onnx", 993, &rows, 200, precision, 99);
    let want = strings(&[
        "docker",
        "run",
        "--rm",
        "--pull",
        "never",
        "--network",
        "none",
        "--device",
        "/dev/dri",
        "--group-add",
        "993",
        "--mount",
        "type=bind,src=/b,dst=/bundle,readonly",
        "--mount",
        "type=bind,src=/w,dst=/work,readonly",
        &o.image,
        "benchmark_app",
        "-m",
        "/bundle/onnx/model.onnx",
        "-d",
        "GPU",
        "-hint",
        "latency",
        "-api",
        "sync",
        "-nireq",
        "1",
        "-niter",
        "200",
        "-shape",
        "input_ids[2,3],attention_mask[2,3],token_type_ids[2,3]",
        "-i",
        "input_ids:/work/input_ids.bin,attention_mask:/work/attention_mask.bin,token_type_ids:/work/token_type_ids.bin",
        "-infer_precision",
        "f32",
        "-latency_percentile",
        "99",
    ]);
    assert_eq!(a, want);
    // The two runs differ in their percentile and nothing else.
    let b = openvino::run_argv(&o, Path::new("/b"), Path::new("/w"), "onnx/model.onnx", 993, &rows, 200, precision, 50);
    let differ: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
    assert_eq!((a.len(), differ), (b.len(), vec![a.len() - 1]));
    assert_eq!(openvino::PERCENTILES, [50, 99]);
    assert_eq!(openvino::infer_precision(openvino::DEVICE_GPU, TURBO_DTYPE_F16).unwrap(), "f16");
    let e = openvino::infer_precision(openvino::DEVICE_NPU, TURBO_DTYPE_BF16).unwrap_err();
    assert!(e.contains("NPU") && e.contains("no inference precision"), "{e}");
}

#[test]
fn benchmark_app_on_the_npu_is_not_given_the_gpu() {
    let rows = rows32();
    let mut o = ov(Path::new("/tmp"));
    o.device = openvino::DEVICE_NPU.into();
    o.accel = "/dev/accel/accel0".into();
    let precision = openvino::infer_precision(openvino::DEVICE_NPU, TURBO_DTYPE_F16).unwrap();
    let a =
        openvino::run_argv(&o, Path::new("/b"), Path::new("/w"), "openvino/model.xml", 44, &rows, 200, precision, 50);
    assert!(a.iter().any(|s| s == "-d"));
    assert!(a.contains(&"NPU".to_owned()));
    assert!(!a.iter().any(|s| s.contains("/dev/dri")), "the NPU container is not given the GPU nodes: {a:?}");
    assert!(a.iter().any(|s| s == "/dev/accel/accel0"));
    assert!(a.iter().any(|s| s.ends_with("openvino/model.xml")), "{a:?}");
    assert!(!a.iter().any(|s| s.ends_with("onnx/model.onnx")), "{a:?}");
    let shown = openvino::native_argv(
        &o,
        openvino::BENCHMARK_APP_BIN,
        Path::new("<bundle>"),
        Path::new("<work>"),
        "openvino/model.xml",
        &rows,
        200,
        precision,
        50,
    );
    assert_eq!(shown[0], "<benchmark-app>");
    assert!(shown.contains(&"-d".to_owned()) && shown.contains(&"NPU".to_owned()));
    assert!(!shown.iter().any(|s| s.contains("/home/") || s.contains("Users")), "{shown:?}");
    let leaked = openvino::redact_user_paths(r"failed: C:\Users\someone\bundle\onnx\model.onnx");
    assert!(!leaked.to_ascii_lowercase().contains("someone"), "{leaked}");
    assert!(leaked.contains("<home>"), "{leaked}");
}

#[test]
fn the_container_gets_the_render_nodes_group() {
    let d = scratch("dri");
    let e = openvino::render_group(&d).unwrap_err();
    assert!(e.contains("no render node"), "{e}");
    std::fs::write(d.join("card0"), "").unwrap();
    assert!(openvino::render_group(&d).is_err(), "a card node is not a render node");
    std::fs::write(d.join("renderD128"), "").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let gid = std::fs::metadata(d.join("renderD128")).unwrap().gid();
        assert_eq!(openvino::render_group(&d).unwrap(), gid);
    }
    assert!(openvino::render_group(&d.join("absent")).is_err());
}

#[test]
fn benchmark_app_output_gives_its_version_and_report() {
    let r = openvino::parse(BENCHMARK_APP_OUT, 50).unwrap();
    assert_eq!(r.version, "2025.3.0-19807-44526285f24-releases/2025/3");
    assert_eq!(r.count, 200);
    assert_eq!(r.duration_ms, 412.64);
    assert_eq!((r.latency_ms, r.average_ms), (2.03, 2.05));
    assert_eq!(r.throughput_fps, 484.68);
    let r = openvino::parse(&p99_out(), 99).unwrap();
    assert!((r.latency_ms - 0.9612).abs() < 1e-12 && (r.average_ms - 0.90433).abs() < 1e-12, "{r:?}");

    // benchmark_app's C++ form: the version as ov::Version prints it, the
    // latency block as LatencyMetrics::write_to_slog does.
    let cpp = BENCHMARK_APP_OUT
        .replace(
            "[ INFO ] Build ................................. 2025.3.0-19807-44526285f24-releases/2025/3\n[ INFO ] \n[ INFO ] Device info:",
            "[ INFO ] OpenVINO Runtime\n    Version : 2025.3.0\n    Build   : 2025.3.0-19807-44526285f24-releases/2025/3\n[ INFO ] Device info:",
        )
        .replace("Count:            200", "Count:               200")
        .replace("   Median:        2.03 ms", "   Median:           2.03 ms");
    let r = openvino::parse(&cpp, 50).unwrap();
    assert_eq!((r.version.as_str(), r.latency_ms, r.count), ("2025.3.0-19807-44526285f24-releases/2025/3", 2.03, 200));
}

#[test]
fn benchmark_app_output_without_a_figure_is_refused() {
    let e = openvino::parse(BENCHMARK_APP_OUT, 99).unwrap_err();
    assert!(e.contains("no 99 percentile: line"), "the percentile asked for, not another: {e}");
    let e = openvino::parse(&p99_out(), 50).unwrap_err();
    assert!(e.contains("no Median: line"), "{e}");
    let no_version = BENCHMARK_APP_OUT.replace("[ INFO ] OpenVINO:\n", "");
    assert!(openvino::parse(&no_version, 50).unwrap_err().contains("no Build line under OpenVINO:"));
    let no_count = BENCHMARK_APP_OUT.replace("Count:", "Counted:");
    assert!(openvino::parse(&no_count, 50).unwrap_err().contains("no Count line"));
    let no_fps = BENCHMARK_APP_OUT.replace("484.68 FPS", "484.68");
    assert!(openvino::parse(&no_fps, 50).unwrap_err().contains("no Throughput line"));
    let seconds = BENCHMARK_APP_OUT.replace("Median:        2.03 ms", "Median:        2.03 s");
    assert!(openvino::parse(&seconds, 50).unwrap_err().contains("no Median: line"));
    let failed = "[ ERROR ] Exception from src/inference/src/cpp/core.cpp:112\n";
    assert!(openvino::parse(failed, 50).is_err());
}

/// 32 padded rows of 64 tokens, as benchmark_app's static shape takes them.
pub(super) fn rows32() -> Rows {
    let n = 32 * 64;
    Rows {
        kind: RowKind::Mixed,
        batch: 32,
        seq: 64,
        ids: vec![0; n],
        mask: vec![0; n],
        types: vec![0; n],
        cases: vec![0; 32],
    }
}

#[test]
fn the_two_runs_make_one_measured_reference() {
    let p50 = openvino::parse(BENCHMARK_APP_OUT, 50).unwrap();
    let mut p99 = p50.clone();
    p99.latency_ms = 2.6;
    let image = ov(Path::new("/w")).image;
    let log = docker::Log { commands: vec![strings(&["docker", "run"]), strings(&["docker", "run"])] };
    let r = openvino::measured(&image, log.clone(), "two runs", p50.clone(), p99.clone(), &rows32()).unwrap();
    assert_eq!((r.name.as_str(), r.role.as_str()), ("openvino", "kernel"));
    assert!(turbo::record::REFERENCES.contains(&(r.name.as_str(), r.role.as_str())));
    assert_eq!(r.version, "2025.3.0-19807-44526285f24-releases/2025/3");
    assert_eq!(r.commands.len(), 2, "both runs are recorded");
    let m = r.measured.unwrap();
    assert_eq!((m.iterations, m.p50_ms, m.p99_ms, m.min_cosine), (200, 2.03, 2.6, None));
    assert_eq!(m.computed_tokens, Some(32 * 64), "every position of the static shape");
    assert!((m.rows_per_second - 200.0 * 32.0 / 0.41264).abs() < 1e-6, "{}", m.rows_per_second);
    assert!(r.procedure.contains("484.68 FPS"));

    // A batch-1 cycle: each case is its own [1, seq] request, and the
    // record's latency is the sum. The shape is not [case count, seq].
    let precision = openvino::infer_precision(openvino::DEVICE_NPU, TURBO_DTYPE_F16).unwrap();
    let reference = Reference {
        texts: vec![("a".into(), 0), ("b".into(), 0)],
        ids: vec![vec![1, 2], vec![3, 4, 5]],
        vectors: vec![vec![0.0], vec![0.0]],
    };
    let cycled = Rows::cycle(&reference, 0, 8).unwrap();
    assert_eq!((cycled.batch, cycled.cases.as_slice()), (1, &[0, 1][..]));
    let frame = cycled.frame(1);
    let argv = openvino::run_argv(
        &ov(Path::new("/tmp")),
        Path::new("/b"),
        &Rows::frame_dir(Path::new("/w"), 1),
        "openvino/model.xml",
        7,
        &frame,
        200,
        precision,
        50,
    );
    let shape = argv.iter().find(|s| s.contains("input_ids[")).unwrap();
    assert!(shape.contains("input_ids[1,8]"), "{shape}");
    assert!(!shape.contains("[2,8]"), "{shape}");
    // The frame directory is mounted at /work, so the request stays one
    // [1, seq] file and the recorded mount names the case.
    assert!(argv.iter().any(|s| s.contains("src=/w/frame-1,dst=/work")), "{argv:?}");
    assert!(argv.iter().any(|s| s.contains("input_ids:/work/input_ids.bin")), "{argv:?}");
    let one = |latency: f64, duration: f64| openvino::Report {
        version: "2026.5.0".into(),
        count: 200,
        duration_ms: duration,
        latency_ms: latency,
        average_ms: latency,
        throughput_fps: 1.0,
    };
    let pairs = [(one(1.5, 200.0), one(2.0, 210.0)), (one(2.5, 300.0), one(3.0, 310.0))];
    let summed = openvino::measured_frames(&image, log.clone(), "cycle", &pairs, &cycled).unwrap();
    let sm = summed.measured.unwrap();
    assert_eq!((sm.p50_ms, sm.p99_ms, sm.iterations), (4.0, 5.0, 200));
    assert_eq!(sm.computed_tokens, Some(2 * 8));
    assert!((sm.rows_per_second - 200.0 * 2.0 / 0.5).abs() < 1e-9, "{}", sm.rows_per_second);
    assert!(summed.procedure.contains("[1, 8]"), "{}", summed.procedure);
    assert!(!summed.procedure.contains("[2, 8]"), "{}", summed.procedure);

    let mut under = p99.clone();
    under.latency_ms = 2.0;
    let e = openvino::measured(&image, log.clone(), "", p50.clone(), under, &rows32()).unwrap_err();
    assert!(e.contains("under its median run's"), "{e}");
    let mut other = p99;
    other.count = 199;
    assert!(openvino::measured(&image, log, "", p50, other, &rows32()).unwrap_err().contains("two runs differ"));
}

#[test]
fn a_bundle_without_onnx_is_recorded_as_benchmark_app_not_run() {
    let m = cpu_measurement();
    let work = scratch("ov-work");
    let r = openvino::run(&ov(&work), m, 10).unwrap();
    assert_eq!((r.name.as_str(), r.role.as_str()), ("openvino", "kernel"));
    assert!(r.measured.is_none());
    assert_eq!(r.not_run.as_deref(), Some("the bundle carries no FORMAT_ONNX artifact for benchmark_app to compile"));
    assert!(r.commands.is_empty(), "nothing ran");
    assert_eq!(std::fs::read_dir(&*work).unwrap().count(), 0, "no input was written");
    let mut unpinned = ov(&work);
    unpinned.image = "openvino/ubuntu24_dev:2025.3.0".into();
    assert!(openvino::run(&unpinned, m, 10).unwrap_err().contains("--openvino-image"));
}

// ---- which programs a backend gets ----
