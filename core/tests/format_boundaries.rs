use turbo::{
    manifest::Manifest,
    record::{self, Record},
};

fn manifest() -> Vec<u8> {
    std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle/manifest.json"))
        .unwrap()
}

#[test]
fn manifest_rejects_trailing_values_and_garbage() {
    for suffix in ["{}", "null", "not json"] {
        let mut bytes = manifest();
        bytes.extend_from_slice(suffix.as_bytes());
        assert!(Manifest::parse(&bytes).is_err(), "accepted trailing {suffix}");
    }
    let mut bytes = manifest();
    bytes.extend_from_slice(b" \r\n\t");
    Manifest::parse(&bytes).unwrap();
}

#[test]
fn manifest_rejects_nul_in_c_header_strings() {
    let mut json: serde_json::Value = serde_json::from_slice(&manifest()).unwrap();
    json["model"]["id"] = "before\0after".into();
    assert!(Manifest::parse(&serde_json::to_vec(&json).unwrap()).is_err());
}

fn record() -> Record {
    record::embedded()
        .iter()
        .filter_map(|(_, r)| r.as_ref().ok())
        .find(|r| r.references.iter().any(|r| r.measured.is_some()))
        .expect("committed record with a measured reference")
        .clone()
}

#[test]
fn record_rejects_overflowed_and_underflowed_speed_ratios() {
    for (own, reference) in [(f64::MAX, f64::MIN_POSITIVE), (f64::MIN_POSITIVE, f64::MAX)] {
        let mut r = record();
        r.timing.min_ms = own;
        r.timing.max_ms = own;
        r.timing.p50_ms = own;
        r.timing.p99_ms = own;
        r.timing.mean_ms = own;
        r.references.retain(|r| r.measured.is_some());
        r.references.truncate(1);
        let m = r.references[0].measured.as_mut().unwrap();
        m.p50_ms = reference;
        m.p99_ms = reference;
        r.speed_reference = Some(r.references[0].name.clone());
        r.speed_ratio = Some(if own > reference { 1.0 } else { 0.0 });
        assert!(r.check().is_err(), "unrepresentable derived speed accepted");
    }
}

#[test]
fn record_rejects_mean_outside_observed_range() {
    let mut r = record();
    r.timing.mean_ms = r.timing.max_ms * 2.0;
    assert!(r.check().is_err());
}

#[test]
fn record_rejects_duplicate_reference_names() {
    let mut r = record();
    r.references.push(r.references[0].clone());
    assert!(r.check().is_err());
}

#[test]
fn speed_helpers_do_not_report_invalid_measurements() {
    let mut refs = record().references;
    refs.retain(|r| r.measured.is_some());
    for r in &mut refs {
        r.measured.as_mut().unwrap().p50_ms = f64::NAN;
    }
    assert!(record::fastest(&refs).is_none());
    assert_eq!(record::speed(f64::INFINITY, &record().references), (None, None));
}

#[test]
fn capability_decision_rejects_unchecked_non_finite_conformance() {
    fn cell(r: &Record) -> record::Cell<'_> {
        record::Cell {
            arch: &r.machine.arch,
            name: &r.device.name,
            cpu: r.device.kind == "DEVICE_CPU",
            backend: &r.device.backend,
            task: turbo::TURBO_TASK_EMBED,
            precision: turbo::TURBO_PRECISION_MODEL,
            dtype: turbo::TURBO_DTYPE_F16,
            version: &r.library.version,
            os: &r.machine.os,
            graph_format: Some("NGRAPH_LITE"),
        }
    }
    let mut r = record::embedded()
        .iter()
        .filter_map(|(_, r)| r.as_ref().ok())
        .find(|r| r.is_for(&cell(r)) && r.falls_short(&cell(r)).is_none())
        .unwrap()
        .clone();
    assert!(matches!(record::decide([("test", &r)], &cell(&r)), record::Verdict::Supported { .. }));
    r.conformance.min_cosine = f64::NAN;
    assert!(matches!(record::decide([("test", &r)], &cell(&r)), record::Verdict::Not(_)));
}
