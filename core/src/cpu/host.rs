//! The host processor and memory, as the operating system reports them.

/// What the operating system says about the processor and memory. A value
/// it does not report is empty or 0, which turbo.h reads as unknown.
#[derive(Default)]
pub(super) struct Host {
    pub(super) name: String,
    pub(super) vendor: String,
    pub(super) memory_total: u64,
    pub(super) memory_free: u64,
}

impl Host {
    #[cfg(target_os = "linux")]
    pub(super) fn read() -> Host {
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

    /// macOS names the processor and its memory through sysctl: the free
    /// pages it counts, which leave out what the kernel would give back
    /// on demand, so memory_free is a floor.
    #[cfg(target_os = "macos")]
    pub(super) fn read() -> Host {
        use std::ffi::{c_char, c_void};
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const c_char,
                oldp: *mut c_void,
                oldlenp: *mut usize,
                newp: *mut c_void,
                newlen: usize,
            ) -> i32;
        }
        let bytes = |name: &std::ffi::CStr| -> Option<Vec<u8>> {
            let mut len = 0usize;
            let null = std::ptr::null_mut();
            if unsafe { sysctlbyname(name.as_ptr(), null, &mut len, null, 0) } != 0 || len == 0 {
                return None;
            }
            let mut buf = vec![0u8; len];
            let rc = unsafe { sysctlbyname(name.as_ptr(), buf.as_mut_ptr() as *mut c_void, &mut len, null, 0) };
            (rc == 0).then(|| {
                buf.truncate(len);
                buf
            })
        };
        let text = |name| {
            bytes(name)
                .map(|b| String::from_utf8_lossy(&b).trim_end_matches('\0').trim().to_owned())
                .unwrap_or_default()
        };
        let number = |name| {
            bytes(name).map_or(0, |b| match b.len() {
                8 => u64::from_ne_bytes(b.try_into().unwrap()),
                4 => u32::from_ne_bytes(b.try_into().unwrap()) as u64,
                _ => 0,
            })
        };
        let name = text(c"machdep.cpu.brand_string");
        // "Apple M2", "Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz": the
        // vendor leads.
        let vendor = name.split([' ', '(']).next().unwrap_or("").to_owned();
        let page = number(c"hw.pagesize");
        let free = number(c"vm.page_free_count") + number(c"vm.page_speculative_count");
        Host { name, vendor, memory_total: number(c"hw.memsize"), memory_free: free * page }
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    pub(super) fn read() -> Host {
        Host::default()
    }
}
