use super::*;

#[test]
fn the_fastest_measured_reference_sets_the_speed_ratio() {
    let mut slow = measured_reference(TRT);
    slow.measured.as_mut().unwrap().p50_ms *= 4.0;
    slow.measured.as_mut().unwrap().p99_ms *= 4.0;
    let r = cpu_record("fastest", vec![slow, measured_reference(TEI)]);
    assert_eq!((r.speed_ratio, r.speed_reference.as_deref()), (Some(0.5), Some(TEI)));
}

/// A record of `references` as the tool makes one for an Intel GPU on
/// Level Zero: the real CPU measurement's figures, filed as that device.
/// No Intel GPU is measured here; the rule looks only at the fields
/// changed.
fn levelzero_record(name: &str, references: Vec<ReferenceRun>) -> Record {
    let mut r = cpu_record(name, references);
    r.device.backend = "levelzero".into();
    r.device.kind = "DEVICE_GPU".into();
    r.device.name = "Intel(R) Arc(TM) B580 Graphics".into();
    r.device.vendor = "Intel(R) Corporation".into();
    r.machine.arch = "bmg-g21".into();
    reparse(&r).unwrap();
    r
}

fn levelzero_cell(f: impl FnOnce(&Cell)) {
    f(&Cell {
        arch: "bmg-g21",
        name: "Intel(R) Arc(TM) B580 Graphics",
        cpu: false,
        backend: "levelzero",
        task: TURBO_TASK_EMBED,
        precision: TURBO_PRECISION_MODEL,
        dtype: TURBO_DTYPE_F32,
        version: record::library_version(),
        os: std::env::consts::OS,
        graph_format: None,
    })
}

fn disabled(name: &str, flag: &str) -> ReferenceRun {
    let mut r = measured_reference(name);
    r.measured = None;
    r.pinned.clear();
    r.version.clear();
    r.commands.clear();
    r.procedure.clear();
    r.not_run = Some(format!("disabled on the command line ({flag})"));
    r
}

#[test]
fn a_levelzero_record_is_backed_by_a_measured_openvino_reference() {
    let r = levelzero_record("lz-openvino", vec![disabled(TEI, "--no-tei"), measured_reference(OV)]);
    assert_eq!(r.references[1].role, "kernel");
    assert_eq!(r.speed_reference.as_deref(), Some(OV));
    levelzero_cell(|cell| {
        assert!(r.is_for(cell));
        let v = verdict(&[&r], cell);
        let Verdict::Supported { benchmark, speed_ratio, .. } = v else { panic!("{v:?}") };
        assert_eq!(benchmark, record::file_name(&r).unwrap());
        assert!(benchmark.starts_with("bmg-g21.levelzero.embed.model."), "{benchmark}");
        assert_eq!(speed_ratio, 0.5);
    });
    cpu_cell(|cell| assert_eq!(verdict(&[&r], cell), Verdict::Not(NO_RECORD.into()), "not the CPU's cell"));
}

#[test]
fn a_levelzero_record_without_openvino_measured_backs_nothing() {
    let r = levelzero_record("lz-none", vec![disabled(TEI, "--no-tei"), disabled(OV, "--no-openvino")]);
    levelzero_cell(|cell| {
        let v = verdict(&[&r], cell);
        assert!(matches!(&v, Verdict::Not(w) if w.ends_with("no reference program measured")), "{v:?}");
    });
}

#[test]
fn an_unknown_reference_backs_nothing_on_levelzero() {
    let r = levelzero_record("lz-unknown", vec![disabled(TEI, "--no-tei"), measured_reference(OV)]);
    for (name, role) in [("openvino-gpu", "kernel"), ("openvino", "end_to_end"), ("onnxruntime", "kernel")] {
        let mut x = r.clone();
        x.references[1].name = name.into();
        x.references[1].role = role.into();
        x.speed_reference = Some(name.into());
        let e = reparse(&x).unwrap_err();
        assert!(e.contains("is not one of"), "{name} {role}: {e}");
    }
    // Were one read without its checks, the rule still counts only the
    // programs it knows.
    let mut x = r.clone();
    x.references[1].name = "openvino-gpu".into();
    levelzero_cell(|cell| {
        let v = verdict(&[&x], cell);
        assert!(matches!(&v, Verdict::Not(w) if w.ends_with("no reference program measured")), "{v:?}");
    });
}

#[test]
fn a_record_backs_only_its_own_cell() {
    let r = cpu_record("own-cell", vec![measured_reference(TEI)]);
    cpu_cell(|cell| {
        let other_cpu = Cell { name: "another processor", ..*cell };
        assert_eq!(verdict(&[&r], &other_cpu), Verdict::Not(NO_RECORD.into()), "a CPU is keyed on its name");
        let other_arch = Cell { arch: "rtx4080", cpu: false, ..*cell };
        assert_eq!(verdict(&[&r], &other_arch), Verdict::Not(NO_RECORD.into()));
        let other_backend = Cell { backend: "cuda", ..*cell };
        assert_eq!(verdict(&[&r], &other_backend), Verdict::Not(NO_RECORD.into()));
        let other = if std::env::consts::OS == "macos" { "linux" } else { "macos" };
        let other_os = Cell { os: other, ..*cell };
        assert_eq!(verdict(&[&r], &other_os), Verdict::Not(NO_RECORD.into()), "a machine is keyed on its OS too");
        let other_precision = Cell { precision: TURBO_PRECISION_EXACT, ..*cell };
        assert_eq!(verdict(&[&r], &other_precision), Verdict::Not(NO_RECORD.into()));
        let v = verdict(&[&r], &Cell { version: "0.2.0", ..*cell });
        assert!(
            matches!(&v, Verdict::Not(w) if w.ends_with(&format!("recorded with library {}, this is 0.2.0", record::library_version()))),
            "{v:?}"
        );
        let v = verdict(&[&r], &Cell { dtype: TURBO_DTYPE_F16, ..*cell });
        assert!(
            matches!(&v, Verdict::Not(w) if w.ends_with("computed in DTYPE_F32, the cell computes in dtype 10")),
            "{v:?}"
        );
    });
}

#[test]
fn f32_must_reach_the_cosine_and_the_absolute_difference() {
    let base = cpu_record("f32-floor", vec![measured_reference(TEI)]);
    cpu_cell(|cell| {
        let mut r = base.clone();
        r.conformance.min_cosine = 0.99989;
        let v = verdict(&[&r], cell);
        assert!(matches!(&v, Verdict::Not(w) if w.ends_with("min cosine 0.99989 is under 0.9999")), "{v:?}");
        let mut r = base.clone();
        r.conformance.max_abs_diff = 2e-4;
        let v = verdict(&[&r], cell);
        assert!(matches!(&v, Verdict::Not(w) if w.ends_with("max abs diff 2e-4 is over 1e-4")), "{v:?}");
        let mut r = base.clone();
        r.conformance.min_cosine = 0.9999;
        r.conformance.max_abs_diff = 1e-4;
        assert!(matches!(verdict(&[&r], cell), Verdict::Supported { .. }), "the floors are inclusive");
    });
}

#[test]
fn f16_and_bf16_need_only_the_cosine() {
    let base = cpu_record("half", vec![measured_reference(TEI)]);
    cpu_cell(|cell| {
        let half = Cell { dtype: TURBO_DTYPE_F16, ..*cell };
        let mut r = base.clone();
        r.compute_dtype = "DTYPE_F16".into();
        r.conformance.min_cosine = 0.9991;
        r.conformance.max_abs_diff = 3e-2;
        assert!(matches!(verdict(&[&r], &half), Verdict::Supported { .. }));
        r.conformance.min_cosine = 0.998;
        assert!(matches!(verdict(&[&r], &half), Verdict::Not(w) if w.ends_with("min cosine 0.998 is under 0.999")));
        let bf = Cell { dtype: turbo::TURBO_DTYPE_BF16, ..*cell };
        let mut r = base.clone();
        r.compute_dtype = "DTYPE_BF16".into();
        r.conformance.min_cosine = 0.9995;
        assert!(matches!(verdict(&[&r], &bf), Verdict::Supported { .. }));
    });
}

#[test]
fn the_newest_record_that_backs_the_cell_is_the_one_named() {
    let older = cpu_record("older", vec![measured_reference(TEI)]);
    let mut newer = cpu_record("newer", vec![measured_reference(TEI)]);
    newer.recorded_at = "2026-06-01T00:00:00Z".into();
    let mut newest_failing = cpu_record("newest", vec![]);
    newest_failing.recorded_at = "2026-07-01T00:00:00Z".into();
    cpu_cell(|cell| {
        let v = verdict(&[&newest_failing, &older, &newer], cell);
        assert!(
            matches!(&v, Verdict::Supported { benchmark, .. } if *benchmark == record::file_name(&newer).unwrap()),
            "{v:?}"
        );
        let v = verdict(&[&older, &newest_failing], cell);
        assert!(
            matches!(&v, Verdict::Supported { benchmark, .. } if *benchmark == record::file_name(&older).unwrap()),
            "{v:?}"
        );
        // With none backing it, the newest record's reason.
        let mut bad = older.clone();
        bad.conformance.min_cosine = 0.5;
        let v = verdict(&[&bad, &newest_failing], cell);
        assert!(matches!(&v, Verdict::Not(w) if w.starts_with(&record::file_name(&newest_failing).unwrap())), "{v:?}");
    });
}

#[test]
fn only_a_mixed_record_backs_the_cell() {
    let mixed = cpu_record("mixed-backs", vec![measured_reference(TEI)]);
    // A real dense measurement, newer, with a reference that gives it a
    // speed_ratio of its own.
    let m = dense_measurement(3, None);
    let mut fast = measured_reference(TEI);
    let at = fast.measured.as_mut().unwrap();
    let k = m.timing.p50_ms * 4.0 / at.p50_ms;
    (at.p50_ms, at.p99_ms) = (at.p50_ms * k, at.p99_ms * k);
    at.computed_tokens = Some(m.rows.padded_tokens());
    let dense = turbo_bench::record(&m, &provenance("dense-backs"), vec![fast], "2026-06-01T00:00:00Z".into()).unwrap();
    assert_eq!(dense.rows.kind, "ROWS_DENSE");
    assert!((dense.speed_ratio.unwrap() - 0.25).abs() < 1e-9, "{:?}", dense.speed_ratio);
    reparse(&dense).unwrap();
    cpu_cell(|cell| {
        assert!(dense.falls_short(cell).is_none(), "it would back the cell but for its rows");
        assert!(!dense.is_for(cell));
        let v = verdict(&[&dense, &mixed], cell);
        assert_eq!(
            v,
            Verdict::Supported {
                benchmark: record::file_name(&mixed).unwrap(),
                cosine_floor: mixed.conformance.min_cosine,
                speed_ratio: 0.5,
            }
        );
        assert_eq!(verdict(&[&dense], cell), Verdict::Not(NO_RECORD.into()));
    });
    // turbo-bench check says so too.
    let dir = scratch("dense-check");
    let name = record::file_name(&dense).unwrap();
    let path = dir.join(&name);
    std::fs::write(&path, serde_json::to_vec(&dense).unwrap()).unwrap();
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-bench"))
        .args(["check", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        String::from_utf8(o.stdout).unwrap(),
        format!("{name}: does not back SUPPORTED: rows.kind ROWS_DENSE; only ROWS_MIXED rows do\n")
    );
}

#[test]
fn the_capability_is_what_the_compiled_in_records_decide() {
    // benchmarks/records/ holds no record until a real run commits one;
    // then the C interface reports what record::decide_embedded says.
    for (name, r) in record::embedded() {
        assert!(r.is_ok(), "{name}: {r:?}");
    }
    let rt = Runtime::create().unwrap();
    let i = rt.find("cpu").unwrap();
    for p in [TURBO_PRECISION_MODEL, turbo::TURBO_PRECISION_FASTEST, TURBO_PRECISION_EXACT] {
        let cap = rt.capability(i, TURBO_TASK_EMBED, p).unwrap();
        cpu_cell(|cell| match record::decide_embedded(&Cell { precision: p, dtype: cap.dtype, ..*cell }) {
            Verdict::Supported { benchmark, cosine_floor, speed_ratio } => {
                assert_eq!(cap.status, turbo::backend::TURBO_CAP_SUPPORTED);
                assert_eq!(field(&cap.benchmark), benchmark);
                assert_eq!((cap.cosine_floor, cap.speed_ratio), (cosine_floor as f32, speed_ratio as f32));
                assert_eq!(field(&cap.reason), "");
            }
            Verdict::Not(why) => {
                assert_eq!(cap.status, turbo::backend::TURBO_CAP_EXPERIMENTAL);
                assert_eq!(field(&cap.reason), why);
                assert_eq!(field(&cap.benchmark), "");
                assert_eq!((cap.cosine_floor, cap.speed_ratio), (0.0, 0.0));
            }
        });
    }
}

/// An npu record as the tool would file one: the CPU measurement's
/// figures, the compiled frame as computed tokens, and the two settings
/// the bench writes. No NPU is measured here.
fn npu_record(name: &str, format: &str, input: &str, references: Vec<ReferenceRun>) -> Record {
    let mut r = cpu_record(name, references);
    r.device.backend = "npu".into();
    r.device.kind = "DEVICE_NPU".into();
    r.device.name = "Intel(R) AI Boost".into();
    r.machine.arch = "arl-npu".into();
    r.library.settings = vec![format!("TURBO_NPU_GRAPH_FORMAT={format}"), format!("TURBO_NPU_GRAPH_INPUT={input}")];
    r.timing.computed_tokens = Some(r.rows.batch as u64 * r.rows.seq as u64);
    r
}

fn npu_cell<'a>(format: &'a str) -> Cell<'a> {
    Cell {
        arch: "arl-npu",
        name: "Intel(R) AI Boost",
        cpu: false,
        backend: "npu",
        task: TURBO_TASK_EMBED,
        precision: TURBO_PRECISION_MODEL,
        dtype: TURBO_DTYPE_F32,
        version: record::library_version(),
        os: std::env::consts::OS,
        graph_format: Some(format),
    }
}

#[test]
fn an_npu_record_names_its_graph_format_and_refuses_a_host_gather_ratio() {
    let r = npu_record("lite", "NGRAPH_LITE", "INPUT_TOKEN_IDS", vec![measured_reference(OV)]);
    let name = record::file_name(&r).unwrap();
    assert!(name.contains(".npu.ngraph-lite."), "{name}");
    assert!(reparse(&r).is_ok());
    assert!(r.is_for(&npu_cell("NGRAPH_LITE")));
    assert!(!r.is_for(&npu_cell("NATIVE")), "a later driver advertising bit 0x1 is another cell");

    let mut bare = r.clone();
    bare.library.settings.clear();
    assert!(reparse(&bare).unwrap_err().contains("TURBO_NPU_GRAPH_FORMAT"));

    let mut packed = r.clone();
    packed.timing.computed_tokens = Some(packed.rows.live_tokens);
    assert!(
        packed.rows.live_tokens < packed.rows.batch as u64 * packed.rows.seq as u64,
        "the fixture must have padding, or this does not tell the frame from the packed count"
    );
    assert!(reparse(&packed).unwrap_err().contains("compiled frame"));

    let native = npu_record("native", "NATIVE", "INPUT_TOKEN_IDS", vec![measured_reference(OV)]);
    assert!(record::file_name(&native).unwrap().contains(".npu.native."));
    assert!(reparse(&native).is_ok());
    assert!(!native.is_for(&npu_cell("NGRAPH_LITE")));

    let embeddings = npu_record("gather", "NGRAPH_LITE", "INPUT_EMBEDDINGS", vec![measured_reference(OV)]);
    assert!(reparse(&embeddings).unwrap_err().contains("INPUT_EMBEDDINGS has no speed_ratio"));

    let mut library_only = npu_record("gather-only", "NGRAPH_LITE", "INPUT_EMBEDDINGS", vec![]);
    library_only.speed_ratio = None;
    library_only.speed_reference = None;
    assert!(reparse(&library_only).is_ok(), "{}", reparse(&library_only).unwrap_err());
}

#[test]
fn a_batch_1_mixed_cycle_counts_every_frame() {
    let mut r = npu_record("cycle", "NGRAPH_LITE", "INPUT_TOKEN_IDS", vec![measured_reference(OV)]);
    r.rows.batch = 1;
    r.rows.seq = 128;
    r.rows.cases = vec![0, 1, 3, 4];
    r.rows.live_tokens = 40;
    let positions = 4 * 128;
    r.timing.computed_tokens = Some(positions);
    r.references[0].measured.as_mut().unwrap().computed_tokens = Some(positions);
    assert!(reparse(&r).is_ok(), "{}", reparse(&r).unwrap_err());
    assert!(r.is_for(&npu_cell("NGRAPH_LITE")));

    // One frame's positions are not the pass.
    r.timing.computed_tokens = Some(128);
    assert!(reparse(&r).unwrap_err().contains("compiled frame"));

    // A wider batch still needs one case per row.
    r.rows.batch = 4;
    r.timing.computed_tokens = Some(positions);
    assert!(reparse(&r).is_ok(), "{}", reparse(&r).unwrap_err());
    r.rows.cases.push(5);
    assert!(reparse(&r).unwrap_err().contains("cases one per row"));
}
