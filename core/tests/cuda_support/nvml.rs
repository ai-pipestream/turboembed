//! The device memory this process holds on one GPU, as NVML counts it
//! (`nvmlDeviceGetComputeRunningProcesses`): unlike the device's free
//! memory, it does not move when another process on a shared GPU, a
//! desktop or a second test run, allocates or frees.
//!
//! NVML is loaded at run time from the driver's `libnvidia-ml.so.1`, so a
//! machine without it (a Jetson, whose driver has no NVML) gets `None`
//! and the caller falls back to the device's free memory.

use std::ffi::{c_char, c_uint, c_ulonglong};

use libloading::{Library, Symbol};

/// `nvmlProcessInfo_t` as `_v2` and `_v3` lay it out.
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct ProcessInfo {
    pid: c_uint,
    used_gpu_memory: c_ulonglong,
    gpu_instance_id: c_uint,
    compute_instance_id: c_uint,
}

type Device = *mut std::ffi::c_void;

pub struct Nvml {
    lib: Library,
    device: Device,
}

impl Nvml {
    /// NVML on the GPU at `pci_bus_id` (`cudaDeviceGetPCIBusId`'s form,
    /// NUL-terminated), or None when NVML cannot be loaded or does not
    /// know the device.
    pub fn open(pci_bus_id: &[u8]) -> Option<Nvml> {
        unsafe {
            let lib = Library::new("libnvidia-ml.so.1").ok()?;
            let init: Symbol<unsafe extern "C" fn() -> c_uint> = lib.get(b"nvmlInit_v2").ok()?;
            if init() != 0 {
                return None;
            }
            let by_bus: Symbol<unsafe extern "C" fn(*const c_char, *mut Device) -> c_uint> =
                lib.get(b"nvmlDeviceGetHandleByPciBusId_v2").ok()?;
            let mut device = std::ptr::null_mut();
            if by_bus(pci_bus_id.as_ptr().cast(), &mut device) != 0 {
                return None;
            }
            Some(Nvml { lib, device })
        }
    }

    /// The bytes this process holds on the device, or None when NVML does
    /// not report it (a driver that hides per-process memory).
    pub fn own_bytes(&self) -> Option<u64> {
        unsafe {
            let f: Symbol<unsafe extern "C" fn(Device, *mut c_uint, *mut ProcessInfo) -> c_uint> =
                self.lib.get(b"nvmlDeviceGetComputeRunningProcesses_v3").ok()?;
            let mut infos = vec![ProcessInfo::default(); 256];
            let mut count = infos.len() as c_uint;
            if f(self.device, &mut count, infos.as_mut_ptr()) != 0 {
                return None;
            }
            let me = std::process::id();
            let mine = infos[..count as usize].iter().find(|p| p.pid == me)?;
            // NVML_VALUE_NOT_AVAILABLE is all ones.
            (mine.used_gpu_memory != u64::MAX).then_some(mine.used_gpu_memory)
        }
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        unsafe {
            if let Ok(shutdown) = self.lib.get::<unsafe extern "C" fn() -> c_uint>(b"nvmlShutdown") {
                shutdown();
            }
        }
    }
}
