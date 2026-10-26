use serde_json::{Value, json};
use turbo_bench::mlx::{self, from_receipt};

use super::*;

/// A receipt as mlx_reference.py writes it, for the CPU measurement's
/// rows, its vectors the rows' expected ones.
fn receipt(m: &turbo_bench::measure::Measurement) -> Value {
    json!({
        "version": "0.29.3",
        "libmlx_sha256": DIGEST,
        "metallib_sha256": DIGEST,
        "core_sha256": DIGEST,
        "loadavg": [2.5, 3.0, 3.5],
        "dtype": "float32",
        "batch": m.rows.batch,
        "seq": m.rows.seq,
        "times_ms": (0..30).map(|i| 2.0 + i as f64 / 10.0).collect::<Vec<_>>(),
        "vectors": m.expected,
    })
}

#[test]
fn an_mlx_receipt_for_the_measured_rows_is_a_kernel_reference() {
    let m = cpu_measurement();
    let command = strings(&["python3", "-E", "<mlx-reference>"]);
    let r = from_receipt(receipt(m).to_string().as_bytes(), m, 30, command.clone()).unwrap();
    assert_eq!((r.name.as_str(), r.role.as_str()), ("mlx", "kernel"));
    assert_eq!(r.pinned, format!("mlx@sha256:{DIGEST}"));
    assert_eq!((r.version.as_str(), r.commands.clone()), ("0.29.3", vec![command]));
    let measured = r.measured.unwrap();
    assert_eq!((measured.iterations, measured.p50_ms, measured.p99_ms), (30, 3.4, 4.9));
    assert!(measured.min_cosine.unwrap() > 1.0 - 1e-12, "its vectors are the expected ones");
    assert_eq!(measured.computed_tokens, Some(m.rows.padded_tokens()), "a padded batch: every position");
    assert!(r.procedure.contains("min 2.0000 ms, mean 3.4500 ms"), "{}", r.procedure);
    assert!(r.procedure.contains("load averages before the runs 2.50 3.00 3.50"), "{}", r.procedure);
    assert!(turbo::record::REFERENCES.contains(&("mlx", "kernel")));
}

#[test]
fn an_mlx_receipt_that_differs_from_the_run_is_refused() {
    let m = cpu_measurement();
    let source = receipt(m);
    for (key, value) in [
        ("batch", json!(m.rows.batch + 1)),
        ("seq", json!(m.rows.seq + 1)),
        ("dtype", json!("float16")),
        ("version", json!("")),
        ("libmlx_sha256", json!("not-a-hash")),
        ("metallib_sha256", json!("not-a-hash")),
        ("core_sha256", json!("")),
        ("version", json!("0.29.2")),
        ("loadavg", json!([1.0, -1.0, 1.0])),
        ("times_ms", json!([1.0, 2.0])),
        ("times_ms", json!(vec![0.0; 30])),
        ("vectors", json!([[1.0, 0.0]])),
        ("extra", json!(1)),
    ] {
        let mut changed = source.clone();
        changed[key] = value;
        assert!(from_receipt(changed.to_string().as_bytes(), m, 30, Vec::new()).is_err(), "accepted changed {key}");
    }
    assert!(from_receipt(source.to_string().as_bytes(), m, 31, Vec::new()).is_err(), "accepted 30 runs for 31");
}

#[test]
fn mlx_runs_float16_at_fastest_only() {
    assert_eq!(mlx::dtype(turbo::TURBO_PRECISION_FASTEST), "float16");
    assert_eq!(mlx::dtype(turbo::TURBO_PRECISION_MODEL), "float32");
    assert_eq!(mlx::dtype(turbo::TURBO_PRECISION_EXACT), "float32");
}
