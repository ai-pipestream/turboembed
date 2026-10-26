use super::*;

#[test]
fn every_committed_record_parses_under_its_own_name() {
    for (name, r) in embedded() {
        if let Err(e) = r {
            panic!("benchmarks/records/{name}: {e}");
        }
    }
}

#[test]
fn slugs_are_lower_case_letters_digits_and_single_dashes() {
    assert_eq!(slug("all-MiniLM-L6-v2"), "all-minilm-l6-v2");
    assert_eq!(slug("  Intel(R) Xeon(R)  CPU @ 2.10GHz "), "intel-r-xeon-r-cpu-2-10ghz");
    assert_eq!(slug("x86_64"), "x86-64");
}

#[test]
fn the_tolerances_are_the_documented_ones() {
    assert_eq!(tolerance(TURBO_DTYPE_F32), Some(Tolerance { min_cosine: 0.9999, max_abs_diff: Some(1e-4) }));
    assert_eq!(tolerance(TURBO_DTYPE_F16), Some(Tolerance { min_cosine: 0.999, max_abs_diff: None }));
    assert_eq!(tolerance(TURBO_DTYPE_BF16), tolerance(TURBO_DTYPE_F16));
    assert_eq!(tolerance(crate::TURBO_DTYPE_I32), None);
    assert_eq!(dtype_value("DTYPE_I8"), Some(crate::TURBO_DTYPE_I8));
    assert_eq!(tolerance(crate::TURBO_DTYPE_I8), Some(Tolerance { min_cosine: 0.93, max_abs_diff: None }));
}

#[test]
fn an_image_is_pinned_by_digest_under_a_plain_name() {
    let d = "0".repeat(64);
    assert_eq!(pinned(&format!("nvcr.io/nvidia/tensorrt@sha256:{d}")), Some(d.as_str()));
    assert_eq!(pinned(&format!("localhost:5000/tei@sha256:{d}")), Some(d.as_str()));
    for bad in ["-v/:/x", "", "Upper/case", "a b", "a;b", "a@b"] {
        assert_eq!(pinned(&format!("{bad}@sha256:{d}")), None, "{bad:?}");
    }
    assert_eq!(pinned("nvcr.io/nvidia/tensorrt:25.01"), None);
    assert_eq!(pinned(&format!("tei@sha256:{}", "A".repeat(64))), None);
}

#[test]
fn mixed_arl_npu_minilm_records_support_model_and_fastest() {
    fn cell<'a>(precision: u32, os: &'a str, arch: &'a str, graph_format: Option<&'a str>) -> Cell<'a> {
        Cell {
            arch,
            name: "Intel(R) AI Boost",
            cpu: false,
            backend: "npu",
            task: TURBO_TASK_EMBED,
            precision,
            dtype: TURBO_DTYPE_F16,
            version: library_version(),
            os,
            graph_format,
        }
    }
    // Cases 0 through 7, each its own [1, 128] frame, live_tokens 181.
    // conformance.rows is 16: those eight compared alone, then the
    // eight measured rows. is_for matches the windows arl-npu
    // NGRAPH_LITE cell and falls_short is none, so the cell is
    // SUPPORTED. Committed at bc15354, measured at 99c9264.
    let model = "arl-npu.npu.ngraph-lite.embed.model.all-minilm-l6-v2-da08a0f9.99c92648aa9a.json";
    let fastest = "arl-npu.npu.ngraph-lite.embed.fastest.all-minilm-l6-v2-da08a0f9.99c92648aa9a.json";
    let windows = cell(TURBO_PRECISION_MODEL, "windows", "arl-npu", Some("NGRAPH_LITE"));
    let model_rec = embedded().iter().find(|(n, _)| *n == model).unwrap().1.as_ref().unwrap();
    let fastest_rec = embedded().iter().find(|(n, _)| *n == fastest).unwrap().1.as_ref().unwrap();
    assert!(model_rec.is_for(&windows));
    assert!(fastest_rec.is_for(&cell(TURBO_PRECISION_FASTEST, "windows", "arl-npu", Some("NGRAPH_LITE"))));
    assert!(
        !model_rec.is_for(&cell(TURBO_PRECISION_MODEL, "windows", "arl-npu", None)),
        "an npu cell with no graph format matches no record"
    );
    assert!(
        !fastest_rec.is_for(&cell(TURBO_PRECISION_FASTEST, "windows", "arl-npu", None)),
        "an npu cell with no graph format matches no record"
    );
    assert_eq!(model_rec.rows.cases, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(fastest_rec.rows.cases, model_rec.rows.cases);
    assert_eq!(model_rec.rows.live_tokens, 181);
    assert_eq!((model_rec.rows.batch, model_rec.rows.seq, model_rec.conformance.rows), (1, 128, 16));
    assert_eq!(model_rec.timing.computed_tokens, Some(1024));
    assert_eq!(model_rec.device.driver_version, "0.15.21738");
    assert_eq!(fastest_rec.device.driver_version, "0.15.21738");
    assert_eq!(model_rec.library.version, "0.1.0");
    assert!(model_rec.library.commit.starts_with("99c92648aa9a"));
    assert_eq!(model_rec.falls_short(&windows), None);
    assert_eq!(model_rec.timing.p50_ms, 30.2244);
    assert_eq!(model_rec.speed_ratio, Some(1.0001455989410986));
    let model_ov = model_rec.references.iter().find(|r| r.name == "openvino").unwrap();
    assert_eq!(model_ov.measured.as_ref().unwrap().p50_ms, 30.22);
    assert_eq!(fastest_rec.timing.p50_ms, 29.9935);
    assert_eq!(fastest_rec.speed_ratio, Some(0.9911929940515533));
    let fastest_ov = fastest_rec.references.iter().find(|r| r.name == "openvino").unwrap();
    assert_eq!(fastest_ov.measured.as_ref().unwrap().p50_ms, 30.259999999999998);
    let minilm = "sentence-transformers/all-MiniLM-L6-v2";
    for (label, name, precision, rec) in
        [("model", model, TURBO_PRECISION_MODEL, model_rec), ("fastest", fastest, TURBO_PRECISION_FASTEST, fastest_rec)]
    {
        let c = cell(precision, "windows", "arl-npu", Some("NGRAPH_LITE"));
        match decide_embedded_for(&c, Some(minilm)) {
            Verdict::Supported { benchmark, cosine_floor, speed_ratio } => {
                assert_eq!(benchmark, name, "{label}");
                assert_eq!(cosine_floor, rec.conformance.min_cosine, "{label}");
                assert_eq!(speed_ratio, rec.speed_ratio.unwrap(), "{label}");
            }
            Verdict::Not(why) => panic!("{label} not supported: {why}"),
        }
    }
    // The cell is the device, not the model: the newest arl-npu mixed
    // record of the precision backs it, whichever model it measured.
    for (label, precision) in [("model", TURBO_PRECISION_MODEL), ("fastest", TURBO_PRECISION_FASTEST)] {
        match decide_embedded(&cell(precision, "windows", "arl-npu", Some("NGRAPH_LITE"))) {
            Verdict::Supported { benchmark, cosine_floor, speed_ratio } => {
                assert!(
                    benchmark.starts_with(&format!("arl-npu.npu.ngraph-lite.embed.{label}.")),
                    "{label}: {benchmark}"
                );
                let backing = embedded().iter().find(|(n, _)| *n == benchmark).unwrap().1.as_ref().unwrap();
                assert_eq!(backing.rows.kind, ROWS_MIXED, "{label}");
                assert_eq!(cosine_floor, backing.conformance.min_cosine, "{label}");
                assert_eq!(speed_ratio, backing.speed_ratio.unwrap(), "{label}");
            }
            Verdict::Not(why) => panic!("{label} not supported: {why}"),
        }
    }
    for (label, c) in [
        ("linux", cell(TURBO_PRECISION_MODEL, "linux", "arl-npu", Some("NGRAPH_LITE"))),
        ("native", cell(TURBO_PRECISION_MODEL, "windows", "arl-npu", Some("NATIVE"))),
        ("mtl", cell(TURBO_PRECISION_MODEL, "windows", "mtl-npu", Some("NGRAPH_LITE"))),
        ("exact", cell(TURBO_PRECISION_EXACT, "windows", "arl-npu", Some("NGRAPH_LITE"))),
    ] {
        match decide_embedded(&c) {
            Verdict::Not(why) => assert_eq!(why, NO_RECORD, "{label}"),
            Verdict::Supported { benchmark, .. } => panic!("{label} supported by {benchmark}"),
        }
    }
}

#[test]
fn mixed_arl_npu_bge_records_support_model_and_fastest() {
    fn cell(precision: u32) -> Cell<'static> {
        Cell {
            arch: "arl-npu",
            name: "Intel(R) AI Boost",
            cpu: false,
            backend: "npu",
            task: TURBO_TASK_EMBED,
            precision,
            dtype: TURBO_DTYPE_F16,
            version: library_version(),
            os: "windows",
            graph_format: Some("NGRAPH_LITE"),
        }
    }
    let cases = [
        (
            "BAAI/bge-small-en-v1.5",
            "arl-npu.npu.ngraph-lite.embed.model.bge-small-en-v1-5-d44a36cd.ab1d3d9984b3.json",
            "arl-npu.npu.ngraph-lite.embed.fastest.bge-small-en-v1-5-d44a36cd.ab1d3d9984b3.json",
            "arl-npu.npu.ngraph-lite.embed.model-dense.bge-small-en-v1-5-d44a36cd.ab1d3d9984b3.json",
            "arl-npu.npu.ngraph-lite.embed.fastest-dense.bge-small-en-v1-5-d44a36cd.ab1d3d9984b3.json",
        ),
        (
            "BAAI/bge-base-en-v1.5",
            "arl-npu.npu.ngraph-lite.embed.model.bge-base-en-v1-5-5e903b01.e56bb0bfad5f.json",
            "arl-npu.npu.ngraph-lite.embed.fastest.bge-base-en-v1-5-5e903b01.e56bb0bfad5f.json",
            "arl-npu.npu.ngraph-lite.embed.model-dense.bge-base-en-v1-5-5e903b01.e56bb0bfad5f.json",
            "arl-npu.npu.ngraph-lite.embed.fastest-dense.bge-base-en-v1-5-5e903b01.e56bb0bfad5f.json",
        ),
        (
            "BAAI/bge-large-en-v1.5",
            "arl-npu.npu.ngraph-lite.embed.model.bge-large-en-v1-5-7ff68f5c.de964e7a6bcf.json",
            "arl-npu.npu.ngraph-lite.embed.fastest.bge-large-en-v1-5-7ff68f5c.de964e7a6bcf.json",
            "arl-npu.npu.ngraph-lite.embed.model-dense.bge-large-en-v1-5-7ff68f5c.de964e7a6bcf.json",
            "arl-npu.npu.ngraph-lite.embed.fastest-dense.bge-large-en-v1-5-7ff68f5c.de964e7a6bcf.json",
        ),
    ];
    for (model_id, model, fastest, model_dense, fastest_dense) in cases {
        let model_rec = embedded().iter().find(|(n, _)| *n == model).unwrap().1.as_ref().unwrap();
        let fastest_rec = embedded().iter().find(|(n, _)| *n == fastest).unwrap().1.as_ref().unwrap();
        let model_dense_rec = embedded().iter().find(|(n, _)| *n == model_dense).unwrap().1.as_ref().unwrap();
        let fastest_dense_rec = embedded().iter().find(|(n, _)| *n == fastest_dense).unwrap().1.as_ref().unwrap();
        assert_eq!(model_rec.bundle.model_id, model_id);
        assert_eq!(model_rec.rows.kind, ROWS_MIXED);
        assert_eq!(fastest_rec.rows.kind, ROWS_MIXED);
        assert_eq!(model_dense_rec.rows.kind, ROWS_DENSE);
        assert_eq!(fastest_dense_rec.rows.kind, ROWS_DENSE);
        let windows_model = cell(TURBO_PRECISION_MODEL);
        let windows_fastest = cell(TURBO_PRECISION_FASTEST);
        assert!(model_rec.is_for(&windows_model));
        assert!(fastest_rec.is_for(&windows_fastest));
        assert!(!model_dense_rec.is_for(&windows_model));
        assert!(!fastest_dense_rec.is_for(&windows_fastest));
        assert_eq!(model_rec.falls_short(&windows_model), None);
        assert_eq!(fastest_rec.falls_short(&windows_fastest), None);
        assert_eq!(model_rec.rows.cases, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(model_rec.rows.live_tokens, 181);
        assert_eq!(model_rec.timing.computed_tokens, Some(1024));
        for (name, precision, rec) in
            [(model, TURBO_PRECISION_MODEL, model_rec), (fastest, TURBO_PRECISION_FASTEST, fastest_rec)]
        {
            match decide_embedded_for(&cell(precision), Some(model_id)) {
                Verdict::Supported { benchmark, cosine_floor, speed_ratio } => {
                    assert_eq!(benchmark, name, "{model_id}");
                    assert_eq!(cosine_floor, rec.conformance.min_cosine, "{model_id}");
                    assert_eq!(speed_ratio, rec.speed_ratio.unwrap(), "{model_id}");
                }
                Verdict::Not(why) => panic!("{model_id} {name} not supported: {why}"),
            }
        }
    }
}

#[test]
fn a_batch_1_cycle_covers_the_set_only_when_every_frame_was_compared() {
    let model = "arl-npu.npu.ngraph-lite.embed.model.all-minilm-l6-v2-da08a0f9.99c92648aa9a.json";
    let cell = Cell {
        arch: "arl-npu",
        name: "Intel(R) AI Boost",
        cpu: false,
        backend: "npu",
        task: TURBO_TASK_EMBED,
        precision: TURBO_PRECISION_MODEL,
        dtype: TURBO_DTYPE_F16,
        version: library_version(),
        os: "windows",
        graph_format: Some("NGRAPH_LITE"),
    };
    let full = embedded().iter().find(|(n, _)| *n == model).unwrap().1.as_ref().unwrap().clone();
    // The committed file is this record: eight fitting cases, each
    // its own [1, 128] frame, plus those eight compared alone.
    assert_eq!(full.falls_short(&cell), None);
    // Naming the cases without comparing them does not cover the set.
    let mut covered = full.clone();
    covered.conformance.rows = 9;
    assert_eq!(covered.falls_short(&cell).as_deref(), Some("rows cover 8 of 1 reference cases that fit seq"));
    // Eight frames of case 0 are still one case.
    covered.rows.cases = vec![0; 8];
    covered.conformance.rows = 16;
    assert_eq!(covered.falls_short(&cell).as_deref(), Some("rows cover 1 of 8 reference cases that fit seq"));
    // A wider batch is unchanged: distinct cases equal conformance
    // rows minus the batch, and cases.len() is that batch.
    let mut wide = full;
    wide.rows.batch = 32;
    wide.rows.cases = (0..9).cycle().take(32).collect();
    wide.conformance.rows = 41;
    wide.timing.computed_tokens = Some(u64::from(wide.rows.batch) * u64::from(wide.rows.seq));
    wide.check().unwrap();
    assert_eq!(wide.falls_short(&cell), None);
    wide.rows.cases = vec![0; 32];
    assert_eq!(wide.falls_short(&cell).as_deref(), Some("rows cover 1 of 9 reference cases that fit seq"));
}

#[test]
fn a_kernel_time_goes_with_a_measurement_and_has_ordered_figures() {
    let mut r = embedded()
        .iter()
        .filter_map(|(_, r)| r.as_ref().ok().cloned())
        .find(|r| r.references.iter().any(|x| x.name == "text-embeddings-inference" && x.measured.is_some()))
        .expect("a committed record with TEI measured");
    let at = r.references.iter().position(|x| x.name == "text-embeddings-inference").unwrap();
    let good = KernelTime {
        profiler: "nsys".into(),
        requests: 3,
        p50_ms: 1.5,
        p99_ms: 2.0,
        copy_p50_ms: 0.0,
        kernels_p50: 40,
    };
    r.references[at].kernel_time = Some(good.clone());
    assert_eq!(r.check(), Ok(()));
    let bad = [
        KernelTime { p99_ms: 1.0, ..good.clone() },
        KernelTime { p50_ms: 0.0, ..good.clone() },
        KernelTime { requests: 0, ..good.clone() },
        KernelTime { kernels_p50: 0, ..good.clone() },
        KernelTime { copy_p50_ms: -1.0, ..good.clone() },
        KernelTime { profiler: String::new(), ..good.clone() },
    ];
    for k in bad {
        r.references[at].kernel_time = Some(k.clone());
        assert!(r.check().is_err(), "{k:?}");
    }
    r.references[at].kernel_time = Some(good);
    r.references[at].measured = None;
    r.references[at].not_run = Some("not run".into());
    assert!(r.check().unwrap_err().contains("kernel_time goes with a measurement"));
}
