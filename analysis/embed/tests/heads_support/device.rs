//! The device a head test runs its device half on.

use turbo_analysis_embed::Device;

/// The backends a `TURBO_TEST_REQUIRE_<BACKEND>` variable can require of
/// a head test.
const BACKENDS: [&str; 3] = ["cuda", "levelzero", "metal"];

/// The device `named` is (`cpu`, a backend, or the runtime's index of a
/// device), and None when no device is named. A `required` backend does
/// not choose the device: no name, the CPU and a device of another
/// backend are each refused, with what to set.
pub fn device_for(named: Option<&str>, required: Option<&str>) -> Result<Option<Device>, String> {
    turbo_pins::check_test_device(named, required)?;
    Ok(named.map(|name| match name.parse() {
        Ok(index) => Device::Index(index),
        Err(_) if name == "cpu" => Device::Cpu,
        Err(_) => Device::Backend(name.to_owned()),
    }))
}

/// The device `TURBO_TEST_DEVICE` names, with the name, and None when it
/// is not set. TURBO_TEST_REQUIRE_CUDA, TURBO_TEST_REQUIRE_LEVELZERO and
/// TURBO_TEST_REQUIRE_METAL do not choose it: set to anything but 0, the
/// test fails here unless the name is that backend's; unset or 0, they
/// ask nothing.
pub fn named_device() -> Option<(String, Device)> {
    let named = std::env::var("TURBO_TEST_DEVICE").ok();
    let device = device_for(named.as_deref(), turbo_pins::device_required(&BACKENDS)).unwrap_or_else(|e| panic!("{e}"));
    named.zip(device)
}
