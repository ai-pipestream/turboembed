//! The record schema, its names and its parsing, and the rule the core
//! decides SUPPORTED by, on records made from a real measurement of the
//! CPU backend on the small bundle.

mod common;

use common::*;
use turbo::record::{self, Cell, NO_RECORD, Record, Verdict, decide};
use turbo::{TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_PRECISION_EXACT, TURBO_PRECISION_MODEL, TURBO_TASK_EMBED};
use turbo_bench::api::{Runtime, field};
use turbo_bench::measure::{Reference, Rows};

/// The cell the core asks about for the CPU device, as the runtime lists it.
fn cpu_cell(f: impl FnOnce(&Cell)) {
    let rt = Runtime::create().unwrap();
    let d = rt.device_info(rt.find("cpu").unwrap()).unwrap();
    let (arch, name) = (field(&d.arch), field(&d.name));
    let cell = Cell {
        arch: &arch,
        name: &name,
        cpu: true,
        backend: "cpu",
        task: TURBO_TASK_EMBED,
        precision: TURBO_PRECISION_MODEL,
        dtype: TURBO_DTYPE_F32,
        version: record::library_version(),
        os: std::env::consts::OS,
    };
    f(&cell)
}

fn named(r: &Record) -> (String, &Record) {
    (record::file_name(r).unwrap(), r)
}

fn verdict(records: &[&Record], cell: &Cell) -> Verdict {
    let named: Vec<(String, &Record)> = records.iter().map(|r| named(r)).collect();
    decide(named.iter().map(|(n, r)| (n.as_str(), *r)), cell)
}

fn reparse(r: &Record) -> Result<Record, String> {
    Record::parse(&record::file_name(r)?, &serde_json::to_vec(r).unwrap())
}

#[test]
fn a_record_holds_what_was_measured() {
    let m = cpu_measurement();
    let r = cpu_record("holds", vec![]);
    assert_eq!(r.device.kind, "DEVICE_CPU");
    assert_eq!(r.device.backend, "cpu");
    assert_eq!(r.machine.arch, std::env::consts::ARCH);
    assert_eq!(r.machine.host_cpu, r.device.name, "the host CPU is the device");
    assert_eq!(r.library.version, record::library_version());
    assert_eq!(r.library.build, turbo_bench::api::version());
    assert_eq!((r.precision.as_str(), r.compute_dtype.as_str()), ("PRECISION_MODEL", "DTYPE_F32"));
    assert_eq!(r.timing, m.timing);
    assert_eq!(r.conformance, m.conformance);
    assert!(r.conformance.rows > m.rows.batch, "each case alone, then every row of the timed batch");
    assert!(r.conformance.min_cosine >= 0.9999 && r.conformance.max_abs_diff <= 1e-4, "{:?}", r.conformance);
    // The rows: the reference cases in order, cycled, hashed as written.
    let bundle = turbo::bundle::Bundle::open(&tiny_bundle()).unwrap();
    let reference = Reference::read(&bundle).unwrap();
    let rows = Rows::build(&reference, 0, r.rows.batch, r.rows.seq).unwrap();
    assert_eq!(r.rows.sha256, rows.sha256());
    assert_eq!(r.rows.cases, rows.cases);
    assert_eq!(r.rows.seq as usize, reference.ids.iter().map(Vec::len).max().unwrap());
    assert_eq!(r.rows.batch, 32);
    assert_eq!(r.rows.live_tokens, reference.ids.iter().cycle().take(32).map(|r| r.len() as u64).sum::<u64>());
    assert_eq!(r.bundle.manifest_sha256, bundle.manifest_sha256);
    assert_eq!(r.bundle.model_id, "sentence-transformers/all-MiniLM-L6-v2");
    assert_eq!((r.speed_ratio, r.speed_reference.as_deref()), (None, None));
}

#[test]
fn the_rows_hash_changes_with_any_value_or_the_shape() {
    let bundle = turbo::bundle::Bundle::open(&tiny_bundle()).unwrap();
    let reference = Reference::read(&bundle).unwrap();
    let a = Rows::build(&reference, 0, 4, 64).unwrap();
    assert_eq!(a.sha256(), Rows::build(&reference, 0, 4, 64).unwrap().sha256());
    assert_ne!(a.sha256(), Rows::build(&reference, 0, 4, 63).unwrap().sha256());
    assert_ne!(a.sha256(), Rows::build(&reference, 0, 5, 64).unwrap().sha256());
    assert_ne!(a.sha256(), Rows::build(&reference, 1, 4, 64).unwrap().sha256(), "the padding is hashed");
    let mut b = a.clone();
    b.types[0] = 1;
    assert_ne!(a.sha256(), b.sha256());
    assert!(Rows::build(&reference, 0, 4, 1).is_err(), "no case is one token");
}

#[test]
fn a_record_is_named_from_its_contents_within_96_bytes() {
    let r = cpu_record("name", vec![]);
    let name = record::file_name(&r).unwrap();
    let cpu = &turbo::bundle::sha256_hex(r.device.name.as_bytes())[..8];
    let arch = std::env::consts::ARCH.replace('_', "-");
    let want = format!(
        "{arch}-{cpu}.cpu.embed.model.all-minilm-l6-v2-{}.{}.json",
        &r.bundle.manifest_sha256[..8],
        &r.library.commit[..12]
    );
    assert_eq!(name, want);
    assert!(name.len() <= record::NAME_MAX);
    // A GPU record has no processor in its name.
    let mut g = r.clone();
    g.device.kind = "DEVICE_GPU".into();
    g.device.backend = "cuda".into();
    g.machine.arch = "rtx4080".into();
    g.precision = "PRECISION_EXACT".into();
    assert!(record::file_name(&g).unwrap().starts_with("rtx4080.cuda.embed.exact.all-minilm-l6-v2-"));
    g.machine.arch = "a".repeat(31);
    g.device.backend = "b".repeat(31);
    assert!(record::file_name(&g).unwrap_err().contains("over 95"));
}

#[test]
fn a_written_record_parses_back_the_same_and_is_never_replaced() {
    let r = cpu_record("write", vec![]);
    let dir = scratch("write-out");
    let path = turbo_bench::write(&r, &dir).unwrap();
    assert_eq!(path.file_name().unwrap().to_str().unwrap(), record::file_name(&r).unwrap());
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(Record::parse(&record::file_name(&r).unwrap(), &bytes).unwrap(), r);
    let e = turbo_bench::write(&r, &dir).unwrap_err();
    assert!(e.contains("a record is never replaced"), "{e}");
}

#[test]
fn a_record_that_is_not_well_formed_is_refused() {
    let r = cpu_record("refused", vec![measured_reference(TEI)]);
    reparse(&r).unwrap();
    let name = record::file_name(&r).unwrap();
    let bytes = serde_json::to_vec(&r).unwrap();

    let e = Record::parse("other.json", &bytes).unwrap_err();
    assert!(e.contains(&format!("its contents name it {name}")), "{e}");

    let mut v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    v["typed_by_hand"] = true.into();
    assert!(Record::parse(&name, &serde_json::to_vec(&v).unwrap()).unwrap_err().contains("unknown field"));

    let refused = |f: &dyn Fn(&mut Record), want: &str| {
        let mut x = r.clone();
        f(&mut x);
        let e = reparse(&x).unwrap_err();
        assert!(e.contains(want), "{want}: {e}");
    };
    refused(&|x| x.speed_ratio = Some(0.25), "is not what the references give");
    refused(&|x| x.speed_reference = Some("b".into()), "is not what the references give");
    refused(&|x| x.timing.p50_ms = (x.timing.min_ms + x.timing.p50_ms) / 2.0, "is not what the references give");
    refused(&|x| x.timing.p99_ms = x.timing.p50_ms / 2.0, "min <= p50 <= p99 <= max");
    refused(&|x| x.timing.iterations = 0, "iterations above 0");
    refused(&|x| x.references[0].not_run = Some("both".into()), "exactly one of measured and not_run");
    refused(&|x| x.references[0].pinned = "example.invalid/a:latest".into(), "is not name@sha256");
    refused(&|x| x.references[0].version.clear(), "a measurement has iterations, a version");
    refused(&|x| x.library.pushed_to.clear(), "on no branch of origin");
    refused(&|x| x.library.commit = "HEAD".into(), "is not 40 lowercase hex");
    refused(&|x| x.bundle.artifact_sha256.make_ascii_uppercase(), "bundle.artifact_sha256");
    refused(&|x| x.rows.cases.pop().map(drop).unwrap_or(()), "cases one per row");
    refused(&|x| x.recorded_at = "yesterday".into(), "recorded_at");
    refused(&|x| x.compute_dtype = "DTYPE_F64".into(), "not a DTYPE_* value");
    refused(&|x| x.precision = "PRECISION_BEST".into(), "not a PRECISION_* value");
    refused(&|x| x.record_version = 2, "record_version 2");
    refused(&|x| x.conformance.max_abs_diff = -1e-9, "max_abs_diff finite and not negative");
    refused(&|x| x.conformance.min_cosine = -1.5, "min_cosine in [-1, 1]");
    refused(&|x| x.references[0].measured.as_mut().unwrap().min_cosine = Some(-2.0), "is not in [-1, 1]");
    refused(&|x| x.references[0].name = "a-faster-program".into(), "is not one of");
    refused(&|x| x.references[0].role = "kernel".into(), "is not one of");
    refused(&|x| x.references[0].pinned = format!("--privileged@sha256:{}", "0".repeat(64)), "is not name@sha256");
}

#[test]
fn a_record_with_no_reference_measured_does_not_back_supported() {
    let r = cpu_record("no-reference", vec![]);
    cpu_cell(|cell| {
        assert!(r.is_for(cell));
        let v = verdict(&[&r], cell);
        assert_eq!(v, Verdict::Not(format!("{}: no reference program measured", record::file_name(&r).unwrap())));
    });
    let mut disabled = measured_reference(TRT);
    disabled.measured = None;
    disabled.not_run = Some("the bundle carries no FORMAT_ONNX artifact".into());
    let r = cpu_record("not-run", vec![disabled]);
    cpu_cell(|cell| {
        assert!(matches!(verdict(&[&r], cell), Verdict::Not(w) if w.ends_with("no reference program measured")))
    });
}

#[test]
fn a_conforming_record_with_a_reference_backs_supported_with_its_numbers() {
    let r = cpu_record("supported", vec![measured_reference(TEI)]);
    let m = cpu_measurement();
    cpu_cell(|cell| {
        let v = verdict(&[&r], cell);
        let Verdict::Supported { benchmark, cosine_floor, speed_ratio } = v else { panic!("{v:?}") };
        assert_eq!(benchmark, record::file_name(&r).unwrap());
        assert_eq!(cosine_floor, m.conformance.min_cosine);
        assert_eq!(speed_ratio, 0.5, "our p50 over a reference at twice it");
        assert_eq!(r.speed_reference.as_deref(), Some(TEI));
    });
}

#[test]
fn the_fastest_measured_reference_sets_the_speed_ratio() {
    let mut slow = measured_reference(TRT);
    slow.measured.as_mut().unwrap().p50_ms *= 4.0;
    slow.measured.as_mut().unwrap().p99_ms *= 4.0;
    let r = cpu_record("fastest", vec![slow, measured_reference(TEI)]);
    assert_eq!((r.speed_ratio, r.speed_reference.as_deref()), (Some(0.5), Some(TEI)));
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
fn f16_and_bf16_need_only_the_cosine_and_int8_is_never_supported() {
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
        let mut r = base.clone();
        r.compute_dtype = "DTYPE_I8".into();
        reparse(&r).unwrap();
        let v = verdict(&[&r], cell);
        assert!(matches!(&v, Verdict::Not(w) if w.ends_with("DTYPE_I8 has no conformance floor")), "{v:?}");
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
