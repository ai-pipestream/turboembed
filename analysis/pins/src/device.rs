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

/// The backend a test that takes its device from `TURBO_TEST_DEVICE` must
/// run on: the first of `backends` whose `TURBO_TEST_REQUIRE_<BACKEND>` is
/// set to anything but nothing or `0`. The variable never chooses the
/// device; [`check_test_device`] holds the device that was named to it.
pub fn device_required(backends: &[&'static str]) -> Option<&'static str> {
    backends.iter().copied().find(|b| required(std::env::var(variable(b)).ok().as_deref()))
}

fn required(value: Option<&str>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0")
}

/// Checks the device `TURBO_TEST_DEVICE` names (`cuda`, `cuda:1`, `cpu`,
/// or none when it is not set) against the backend that is `required`, if
/// one is: the test must run there, so no device, the CPU and a device of
/// another backend are each refused, with what to set.
pub fn check_test_device(named: Option<&str>, required: Option<&str>) -> Result<(), String> {
    let Some(backend) = required else { return Ok(()) };
    let var = variable(backend);
    match named {
        Some(name) if name.split(':').next() == Some(backend) => Ok(()),
        Some(name) => Err(format!("{var} requires a {backend} device, and TURBO_TEST_DEVICE={name}")),
        None => Err(format!("{var} requires a {backend} device: set TURBO_TEST_DEVICE={backend}")),
    }
}

#[cfg(test)]
mod tests {
    use super::{check_test_device, required, skip_allowed, variable};

    #[test]
    fn only_zero_lets_a_device_test_skip() {
        assert!(skip_allowed(Some("0")));
        for must_run in [None, Some(""), Some("1"), Some("false"), Some("no"), Some("00"), Some(" 0")] {
            assert!(!skip_allowed(must_run), "{must_run:?} let a device test skip");
        }
        assert_eq!(variable("levelzero"), "TURBO_TEST_REQUIRE_LEVELZERO");
        assert_eq!(variable("CUDA"), "TURBO_TEST_REQUIRE_CUDA");
    }

    #[test]
    fn anything_but_nothing_and_zero_requires_the_backend() {
        for must_run in ["1", "true", "yes", "2", " 0", "00"] {
            assert!(required(Some(must_run)), "{must_run:?} does not require the backend");
        }
        for free in [None, Some(""), Some("0")] {
            assert!(!required(free), "{free:?} requires the backend");
        }
    }

    #[test]
    fn a_required_backend_refuses_every_device_but_its_own() {
        // nothing required: any device, or none
        for named in [None, Some("cpu"), Some("cuda"), Some("levelzero:1")] {
            assert_eq!(check_test_device(named, None), Ok(()));
        }
        assert_eq!(check_test_device(Some("cuda"), Some("cuda")), Ok(()));
        assert_eq!(check_test_device(Some("cuda:1"), Some("cuda")), Ok(()));
        // the variable does not choose the device: without a name the test fails
        let unset = check_test_device(None, Some("cuda")).unwrap_err();
        assert_eq!(unset, "TURBO_TEST_REQUIRE_CUDA requires a cuda device: set TURBO_TEST_DEVICE=cuda");
        for other in ["cpu", "levelzero", "metal:0", "cudax", ""] {
            let e = check_test_device(Some(other), Some("cuda")).unwrap_err();
            assert_eq!(e, format!("TURBO_TEST_REQUIRE_CUDA requires a cuda device, and TURBO_TEST_DEVICE={other}"));
        }
    }
}
