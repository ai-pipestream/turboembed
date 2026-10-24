//! Validation of records measured on physical Hailo-8 and Hailo-10H boards.
//! Negative cases mutate those receipts; no device or timing is mocked.
use turbo::record::{self, Cell, Record, Verdict};
use turbo::{TURBO_DTYPE_I8, TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL, TURBO_TASK_EMBED};

fn hailo_records() -> Vec<(&'static str, &'static Record)> {
    record::embedded()
        .iter()
        .filter_map(|(name, parsed)| {
            let r = parsed.as_ref().expect("committed record must parse");
            (r.device.backend == "hailo").then_some((*name, r))
        })
        .collect()
}

fn cell(r: &Record) -> Cell<'_> {
    Cell {
        arch: &r.machine.arch,
        name: &r.device.name,
        cpu: false,
        backend: "hailo",
        task: TURBO_TASK_EMBED,
        precision: match r.precision.as_str() {
            "PRECISION_MODEL" => TURBO_PRECISION_MODEL,
            "PRECISION_FASTEST" => TURBO_PRECISION_FASTEST,
            other => panic!("unmeasured precision: {other}"),
        },
        dtype: TURBO_DTYPE_I8,
        version: record::library_version(),
        os: "linux",
        graph_format: None,
    }
}

#[test]
fn real_hailo_records_back_only_the_measured_architectures_and_precisions() {
    let records = hailo_records();
    for arch in ["hailo8", "hailo10h"] {
        for precision in ["PRECISION_MODEL", "PRECISION_FASTEST"] {
            let &(name, r) = records
                .iter()
                .find(|(_, r)| r.machine.arch == arch && r.precision == precision)
                .expect("both actual boards and both supported precisions must have records");
            let c = cell(r);
            assert!(r.is_for(&c));
            assert_eq!(r.falls_short(&c), None);
            assert!(matches!(record::decide([(name, r)], &c), Verdict::Supported { .. }));
            assert_eq!((r.rows.batch, r.rows.seq), (32, 128));
            assert_eq!(r.timing.computed_tokens, Some(4096));
            assert_eq!(r.references[0].name, "hailort");
            assert_eq!(r.references[0].measured.as_ref().unwrap().computed_tokens, Some(4096));
            assert!(r.references[0].procedure.contains("not request latency tails"));
            assert!(!r.is_for(&Cell { precision: TURBO_PRECISION_EXACT, ..c }));
            assert!(!r.is_for(&Cell { arch: "hailo8l", ..c }));
        }
    }
}

#[test]
fn i8_floor_is_inclusive_and_failed_or_unmeasured_records_do_not_back_support() {
    let records = hailo_records();
    assert!(!records.is_empty());
    for (_, measured) in records {
        let c = cell(measured);
        let mut r = measured.clone();
        r.conformance.min_cosine = 0.93;
        assert_eq!(r.falls_short(&c), None);
        r.conformance.min_cosine = 0.92999;
        assert!(r.falls_short(&c).unwrap().contains("under 0.93"));
        r = measured.clone();
        r.timing.computed_tokens = Some(r.rows.live_tokens);
        assert!(r.check().unwrap_err().contains("every position"));
        r = measured.clone();
        r.references.clear();
        r.speed_ratio = None;
        r.speed_reference = None;
        assert_eq!(r.falls_short(&c).as_deref(), Some("no reference program measured"));
        r = measured.clone();
        let first = r.rows.cases[0];
        r.rows.cases.fill(first);
        assert!(r.falls_short(&c).unwrap().contains("reference cases"));
    }
}
