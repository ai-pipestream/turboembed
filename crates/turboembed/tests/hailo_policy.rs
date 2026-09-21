//! Device::Hailo policy on hosts without the HailoRT provider build.
//!
//! The Hailo lane targets the Raspberry Pi AI HAT+ (Hailo-8/8L, Hailo-10H
//! later) through `native/turboembed/src/hailo.cpp`, compiled only with
//! `--features hailo` on a host with `hailo/hailort.h`. Everywhere else the
//! device must fail loud and must never serve the 8-dim mock. Live
//! on-device proofs live in `hailo_minilm.rs`.

use turboembed::{Device, Engine, Error};

#[test]
fn hailo_device_name_is_stable() {
    // The ABI name is part of the frozen surface (kebab-case token).
    assert_eq!(Device::Hailo.as_str(), "hailo");
}

#[cfg(not(feature = "hailo"))]
#[test]
fn hailo_create_fails_loud_without_feature() {
    let err = match Engine::create(Device::Hailo) {
        Ok(_) => panic!("Hailo create must fail when built without --features hailo"),
        Err(e) => e,
    };
    assert!(matches!(err, Error::Unavailable(_)), "{err:?}");
    let msg = err.to_string();
    assert!(msg.contains("hailo"), "message names the device: {msg}");
    assert!(
        msg.contains("--features hailo"),
        "message names the rebuild: {msg}"
    );
    assert!(
        msg.contains("refusing CPU fallback"),
        "message keeps the no-fallback policy: {msg}"
    );
}

#[cfg(not(feature = "hailo"))]
#[test]
fn hailo_raw_create_leaves_out_null_and_sets_thread_error() {
    use turboembed::ffi::{
        turboembed_device, turboembed_engine, turboembed_engine_create, turboembed_last_error,
        turboembed_status,
    };
    unsafe {
        let mut out: *mut turboembed_engine = std::ptr::null_mut();
        let status = turboembed_engine_create(
            turboembed_device::TURBOEMBED_DEVICE_HAILO,
            std::ptr::null(),
            &mut out,
        );
        assert_eq!(status, turboembed_status::TURBOEMBED_ERR_UNAVAILABLE);
        assert!(out.is_null(), "failed create must not yield an engine");
        let msg = turboembed_last_error(std::ptr::null());
        assert!(!msg.is_null());
        let text = std::ffi::CStr::from_ptr(msg).to_string_lossy();
        assert!(
            text.contains("hailo"),
            "thread-local error explains: {text}"
        );
    }
}

#[cfg(feature = "hailo")]
#[test]
fn hailo_create_never_falls_back_to_mock() {
    // With the feature compiled, create succeeds only when a real Hailo
    // device scanned. On the Pi this proves the NPU path; anywhere else it
    // must still fail loud (UNSUPPORTED_DEVICE), never a mock stand-in.
    match Engine::create(Device::Hailo) {
        Ok(engine) => {
            let models = engine.list_models().expect("list on hailo engine");
            for i in 0..models.len() {
                let info = models.get(i).expect("model row");
                assert_ne!(
                    info.dim, 8,
                    "hailo must never serve the 8-dim mock: {info:?}"
                );
            }
        }
        Err(e) => {
            assert!(
                matches!(e, Error::UnsupportedDevice(_) | Error::Unavailable(_)),
                "fail-loud error, not a silent substitute: {e:?}"
            );
        }
    }
}

#[test]
fn mock_engine_is_unaffected_by_the_hailo_variant() {
    let engine = Engine::create(Device::Mock).expect("mock engine");
    let models = engine.list_models().expect("list");
    assert_eq!(models.len(), 1);
    assert_eq!(models.get(0).unwrap().alias, "mock-embed");
}
