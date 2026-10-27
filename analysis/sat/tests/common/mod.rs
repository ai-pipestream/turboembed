//! What the tests share: the device and precision they run at.

use turbo_analysis_embed::{Device, EncoderOptions, Precision};

/// The device TURBO_TEST_DEVICE names (`cpu`, `select`, a device index or
/// a backend name; the CPU when it is not set) at the precision TURBO_TEST_PRECISION
/// names (model by default). With a device required (`required`) the CPU
/// is refused.
pub fn encoder() -> EncoderOptions {
    let device = match std::env::var("TURBO_TEST_DEVICE") {
        Err(std::env::VarError::NotPresent) => Device::Cpu,
        Err(e) => panic!("TURBO_TEST_DEVICE: {e}"),
        Ok(d) if d == "cpu" => Device::Cpu,
        Ok(d) if d == "select" => Device::Select,
        Ok(d) => match d.parse() {
            Ok(i) => Device::Index(i),
            Err(_) => Device::Backend(d),
        },
    };
    if let Some(b) = required() {
        assert!(!matches!(device, Device::Cpu), "{b} is required but TURBO_TEST_DEVICE names no device");
    }
    let precision = match std::env::var("TURBO_TEST_PRECISION").as_deref() {
        Err(_) | Ok("model") => Precision::Model,
        Ok("fastest") => Precision::Fastest,
        Ok("exact") => Precision::Exact,
        Ok(p) => panic!("TURBO_TEST_PRECISION={p}: model, fastest or exact"),
    };
    EncoderOptions { device, precision, ..EncoderOptions::default() }
}

/// The backend TURBO_TEST_REQUIRE_CUDA, TURBO_TEST_REQUIRE_LEVELZERO or
/// TURBO_TEST_REQUIRE_METAL requires (set, not empty and not 0), on which a device session must
/// then run.
pub fn required() -> Option<&'static str> {
    let set = |v: &str| std::env::var(v).is_ok_and(|v| !v.is_empty() && v != "0");
    [
        ("TURBO_TEST_REQUIRE_CUDA", "cuda"),
        ("TURBO_TEST_REQUIRE_LEVELZERO", "levelzero"),
        ("TURBO_TEST_REQUIRE_METAL", "metal"),
    ]
    .into_iter()
    .find(|(v, _)| set(v))
    .map(|(_, b)| b)
}
