//! Machine C gate for the Apple matched native-overhead pilot (M4 step 4).
//!
//! `make bench-apple-overhead` runs the live pilot (direct mlx-swift Metal
//! vs the `libTurboEmbed.dylib` ABI on identical MiniLM inputs/shapes) and
//! writes `testdata/receipts/bench/machine-c-metal-overhead.json`. This
//! ignored test then validates that receipt against the predeclared
//! budgets — it never invents numbers and fails when the receipt is
//! missing, was produced with `--quick`, or violates a gate.
//!
//! ```bash
//! make bench-apple-overhead
//! cargo test -p turboembed --features mlx-live --test apple_overhead_receipt \
//!   -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "macos", feature = "mlx-live"))]

use std::path::PathBuf;

use serde_json::Value;

const P50_OVERHEAD_LIMIT: f64 = 1.05;
const THROUGHPUT_FLOOR: f64 = 0.95;
const PARITY_MAX_ABS: f64 = 5e-4;
const PARITY_MAX_RMSE: f64 = 1e-4;
const FULL_GRID_CASES: usize = 18; // batch 1/8/32 × tokens 32/128/256 × full/mixed

fn workspace_root() -> PathBuf {
    if let Some(root) = option_env!("INFERSTREAM_ROOT").map(PathBuf::from) {
        if root.join("include/turboembed.h").is_file() {
            return root;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn num(value: &Value, key: &str) -> f64 {
    value
        .get(key)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("receipt missing numeric field {key}"))
}

#[test]
#[ignore = "needs macOS Metal + models/mlx/minilm + a prior `make bench-apple-overhead`"]
fn apple_overhead_receipt_meets_predeclared_budgets() {
    let path = workspace_root().join("testdata/receipts/bench/machine-c-metal-overhead.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{} missing: {e}. Run `make bench-apple-overhead` on the Machine C Metal host first — do not invent a receipt.",
            path.display()
        )
    });
    let receipt: Value = serde_json::from_str(&text).expect("receipt JSON");

    assert_eq!(
        receipt["experiment"], "apple-metal-native-overhead-pilot",
        "wrong experiment: {}",
        receipt["experiment"]
    );
    assert_eq!(
        receipt["quick"],
        Value::Bool(false),
        "receipt was produced with --quick; a formal receipt needs the full 18-case grid"
    );

    let gates = &receipt["gates"];
    assert_eq!(
        num(gates, "abi_p50_over_direct_p50_max"),
        P50_OVERHEAD_LIMIT
    );
    assert_eq!(
        num(gates, "abi_throughput_over_direct_min"),
        THROUGHPUT_FLOOR
    );
    assert_eq!(num(gates, "parity_max_abs"), PARITY_MAX_ABS);
    assert_eq!(num(gates, "parity_rmse"), PARITY_MAX_RMSE);

    let cases = receipt["cases"].as_array().expect("cases array");
    assert_eq!(
        cases.len(),
        FULL_GRID_CASES,
        "expected the full {FULL_GRID_CASES}-case grid, got {}",
        cases.len()
    );

    for case in cases {
        let name = case["case"].as_str().unwrap_or("?");
        let parity = &case["parity"];
        assert!(
            num(parity, "max_abs") <= PARITY_MAX_ABS && num(parity, "rmse") <= PARITY_MAX_RMSE,
            "{name}: parity out of band: {parity}"
        );
        assert_eq!(
            case["abi_counters_zero"],
            Value::Bool(true),
            "{name}: ABI steady-state arena allocations were not zero"
        );
        let repeats = case["repeats"].as_array().expect("repeats array");
        assert!(!repeats.is_empty(), "{name}: no timed repeats");
        for (i, repeat) in repeats.iter().enumerate() {
            let p50_ratio = num(repeat, "abi_p50_over_direct_p50");
            let tput_ratio = num(repeat, "abi_throughput_over_direct");
            assert!(
                p50_ratio <= P50_OVERHEAD_LIMIT && tput_ratio >= THROUGHPUT_FLOOR,
                "{name} repeat {i}: p50 ratio {p50_ratio:.4} (max {P50_OVERHEAD_LIMIT}), throughput ratio {tput_ratio:.4} (min {THROUGHPUT_FLOOR})"
            );
        }
    }

    assert_eq!(
        receipt["pass"],
        Value::Bool(true),
        "receipt records pass=false — re-run `make bench-apple-overhead`, do not edit the receipt"
    );
    eprintln!(
        "machine-c-metal-overhead.json: {} cases within budgets (p50 ≤ {P50_OVERHEAD_LIMIT}×, throughput ≥ {THROUGHPUT_FLOOR}×, parity ≤ {PARITY_MAX_ABS})",
        cases.len()
    );
}
