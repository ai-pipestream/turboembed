//! The CPU backend: the host processor, through turbo_backend.h like any
//! other backend. This cut lists the device; no task runs on it yet.

use std::ffi::c_char;

use crate::backend::{TURBO_CAP_UNSUPPORTED, turbo_backend};
use crate::{TURBO_DEVICE_CPU, TURBO_DTYPE_F32, turbo_device_info, turbo_error, write_str};

pub static BACKEND: turbo_backend = turbo_backend {
    struct_size: size_of::<turbo_backend>() as u32,
    reserved: 0,
    name: c"cpu".as_ptr(),
    runtime_version: c"".as_ptr(),
    device_count,
    device_info,
    capability,
};

unsafe extern "C" fn device_count(out: *mut u32, _err: *mut turbo_error) -> i32 {
    unsafe { *out = 1 };
    0
}

unsafe extern "C" fn device_info(_ordinal: u32, out: *mut turbo_device_info, _err: *mut turbo_error) -> i32 {
    let out = unsafe { &mut *out };
    let host = Host::read();
    out.kind = TURBO_DEVICE_CPU;
    out.ordinal = 0;
    out.unified_memory = 1;
    out.memory_total = host.memory_total;
    out.memory_free = host.memory_free;
    write_str(&mut out.arch, std::env::consts::ARCH);
    write_str(&mut out.name, &host.name);
    write_str(&mut out.vendor, &host.vendor);
    write_str(&mut out.backend, "cpu");
    write_str(&mut out.runtime_version, "");
    write_str(&mut out.driver_version, "");
    0
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn capability(
    _ordinal: u32,
    _task: u32,
    _precision: u32,
    status: *mut u32,
    dtype: *mut u32,
    options_honored: *mut u32,
    reason: *mut c_char,
    reason_len: u32,
    _err: *mut turbo_error,
) -> i32 {
    unsafe {
        *status = TURBO_CAP_UNSUPPORTED;
        *dtype = TURBO_DTYPE_F32;
        *options_honored = 0;
        let r = std::slice::from_raw_parts_mut(reason, reason_len as usize);
        write_str(r, "the cpu backend has no embed kernels in this build");
    }
    0
}

/// What the operating system says about the processor and memory. A value
/// it does not report is empty or 0, which turbo.h reads as unknown.
#[derive(Default)]
struct Host {
    name: String,
    vendor: String,
    memory_total: u64,
    memory_free: u64,
}

impl Host {
    #[cfg(target_os = "linux")]
    fn read() -> Host {
        let mut h = Host::default();
        let field = |text: &str, key: &str| {
            text.lines()
                .find_map(|l| l.split_once(':').filter(|(k, _)| k.trim() == key).map(|(_, v)| v.trim().to_owned()))
        };
        if let Ok(cpu) = std::fs::read_to_string("/proc/cpuinfo") {
            // x86 names the model; Arm cores often do not, so fall back to
            // the implementer and part the kernel reports.
            h.name = field(&cpu, "model name").or_else(|| field(&cpu, "Model")).unwrap_or_default();
            h.vendor = field(&cpu, "vendor_id").or_else(|| field(&cpu, "CPU implementer")).unwrap_or_default();
        }
        if let Ok(mem) = std::fs::read_to_string("/proc/meminfo") {
            let kib = |key| field(&mem, key).and_then(|v| v.trim_end_matches("kB").trim().parse::<u64>().ok());
            h.memory_total = kib("MemTotal").map_or(0, |k| k * 1024);
            h.memory_free = kib("MemAvailable").map_or(0, |k| k * 1024);
        }
        h
    }

    #[cfg(not(target_os = "linux"))]
    fn read() -> Host {
        Host::default()
    }
}
