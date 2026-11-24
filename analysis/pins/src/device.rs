//! What a device test does when its backend lists no device.
//!
//! A test built with a device backend (`--features cuda`, `levelzero`,
//! `metal`, `npu`, `hailo`) that finds no device of it has run nothing on
//! a device, so it fails: a driver that stopped loading, or a device node
//! the test cannot reach, is not a passing run. A host that has no such
//! device, as a hosted CI runner, says so with
//! `TURBO_TEST_REQUIRE_<BACKEND>=0`, and the test then says it was skipped
//! and passes.

/// Whether `TURBO_TEST_REQUIRE_<BACKEND>` lets a test of `backend`
/// (`cuda`, `levelzero`, ...) skip when the backend lists no device: only
/// when it is `0`. Unset, empty or anything else, the test must fail.
pub fn device_skip_allowed(backend: &str) -> bool {
    skip_allowed(std::env::var(variable(backend)).ok().as_deref())
}

/// The variable of `backend`.
pub fn variable(backend: &str) -> String {
    format!("TURBO_TEST_REQUIRE_{}", backend.to_uppercase())
}

fn skip_allowed(value: Option<&str>) -> bool {
    value == Some("0")
}

#[cfg(test)]
mod tests {
    use super::{skip_allowed, variable};

    #[test]
    fn only_zero_lets_a_device_test_skip() {
        assert!(skip_allowed(Some("0")));
        for must_run in [None, Some(""), Some("1"), Some("false"), Some("no"), Some("00"), Some(" 0")] {
            assert!(!skip_allowed(must_run), "{must_run:?} let a device test skip");
        }
        assert_eq!(variable("levelzero"), "TURBO_TEST_REQUIRE_LEVELZERO");
        assert_eq!(variable("CUDA"), "TURBO_TEST_REQUIRE_CUDA");
    }
}
