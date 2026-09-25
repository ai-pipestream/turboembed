//! The Level Zero backend: Intel GPUs through the oneAPI Level Zero driver,
//! reached through turbo_backend.h like any other backend. The loader is
//! opened at run time, so a machine without it lists nothing. It lists
//! the GPUs, holds models in device memory, and runs embed sessions with
//! the BERT encoder in core/levelzero/encoder.cl (encoder.rs), on the
//! contexts and buffers of gpu.rs. docs/levelzero.md says how to build and
//! test it.

mod encoder;
mod gpu;
mod ze;

use std::ffi::{c_char, c_void};
use std::sync::OnceLock;

use crate::backend::{TURBO_CAP_EXPERIMENTAL, TURBO_CAP_UNSUPPORTED, refuse, turbo_backend};
use crate::status::{DEVICE_UNAVAILABLE, INVALID_ARGUMENT};
use crate::{
    TURBO_DEVICE_GPU, TURBO_DEVICE_IGPU, TURBO_DTYPE_F32, turbo_device_info, turbo_error, turbo_log_fn, write_str,
};

pub static BACKEND: turbo_backend = turbo_backend {
    struct_size: size_of::<turbo_backend>() as u32,
    reserved: 0,
    name: c"levelzero".as_ptr(),
    // Nothing is linked at build time: the loader is opened at run time,
    // and its version is each device's runtime_version.
    runtime_version: c"".as_ptr(),
    device_count,
    device_info,
    capability,
    context_create: Some(context_create),
    context_release: Some(gpu::context_release),
    buffer_alloc: Some(gpu::buffer_alloc),
    buffer_import: Some(gpu::buffer_import),
    buffer_release: Some(gpu::buffer_release),
    buffer_export: Some(gpu::buffer_export),
    model_load: Some(encoder::model_load),
    model_release: Some(encoder::model_release),
    session_create: Some(encoder::session_create),
    session_release: Some(encoder::session_release),
    embed_write: Some(encoder::embed_write),
    session_run: Some(encoder::session_run),
    buffer_read: Some(gpu::buffer_read),
};

unsafe extern "C" fn device_count(out: *mut u32, err: *mut turbo_error) -> i32 {
    match driver() {
        Ok(d) => {
            unsafe { *out = d.map_or(0, |d| d.devices.len() as u32) };
            0
        }
        Err(message) => unsafe { refuse(err, DEVICE_UNAVAILABLE, message) },
    }
}

/// The listed device at ordinal, or the status refusing it.
unsafe fn listed(ordinal: u32, err: *mut turbo_error) -> Result<(&'static Driver, &'static Device), i32> {
    let d = match driver() {
        Ok(Some(d)) => d,
        Ok(None) => return Err(unsafe { refuse(err, INVALID_ARGUMENT, "levelzero: no device is listed") }),
        Err(message) => return Err(unsafe { refuse(err, DEVICE_UNAVAILABLE, message) }),
    };
    match d.devices.get(ordinal as usize) {
        Some(dev) => Ok((d, dev)),
        None => {
            let message = format!("levelzero device {ordinal}: {} are listed", d.devices.len());
            Err(unsafe { refuse(err, INVALID_ARGUMENT, &message) })
        }
    }
}

unsafe extern "C" fn device_info(ordinal: u32, out: *mut turbo_device_info, err: *mut turbo_error) -> i32 {
    let (d, dev) = match unsafe { listed(ordinal, err) } {
        Ok(v) => v,
        Err(rc) => return rc,
    };
    let out = unsafe { &mut *out };
    out.kind = if dev.integrated { TURBO_DEVICE_IGPU } else { TURBO_DEVICE_GPU };
    out.ordinal = ordinal;
    out.unified_memory = dev.integrated as u32;
    // The size is what the driver lets a context allocate. Only sysman says
    // what is free, and it counts against a slightly larger size of its
    // own; without sysman free is unknown.
    out.memory_total = dev.memory_total;
    out.memory_free = d.memory_free(dev).map_or(0, |free| free.min(dev.memory_total));
    write_str(&mut out.arch, &dev.arch);
    write_str(&mut out.name, &dev.name);
    write_str(&mut out.vendor, &dev.vendor);
    write_str(&mut out.runtime_version, &d.loader_version);
    write_str(&mut out.driver_version, &dev.driver_version);
    0
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn capability(
    ordinal: u32,
    _task: u32,
    _precision: u32,
    status: *mut u32,
    dtype: *mut u32,
    options_honored: *mut u32,
    reason: *mut c_char,
    reason_len: u32,
    err: *mut turbo_error,
) -> i32 {
    let dev = match unsafe { listed(ordinal, err) } {
        Ok((_, dev)) => dev,
        Err(rc) => return rc,
    };
    unsafe {
        let r = std::slice::from_raw_parts_mut(reason, reason_len as usize);
        if dev.fp64 {
            *status = TURBO_CAP_EXPERIMENTAL;
            *dtype = TURBO_DTYPE_F32;
            *options_honored = EMBED_HONORED;
            write_str(r, "");
        } else {
            *status = TURBO_CAP_UNSUPPORTED;
            *dtype = 0;
            *options_honored = 0;
            write_str(r, "the encoder sums LayerNorm and L2 norms in F64, and this device has no F64");
        }
    }
    0
}

/// Every device allocation this backend has made in the process, counted
/// where it makes them. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn allocations() -> u64 {
    gpu::device_allocs_total()
}

/// The device address of the F32 copy of an F16 or BF16 model's weights,
/// once a session made it.
///
/// # Safety
/// `model` is one this backend's model_load returned, not yet released.
#[cfg(feature = "internals")]
pub(crate) unsafe fn widened(model: *mut c_void) -> Option<*const c_void> {
    unsafe { encoder::widened(model) }
}

/// Recovery from a failed append on the first listed device's queue, as
/// gpu.rs describes it. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn append_failure_recovers() -> Result<(), String> {
    let d = driver().map_err(str::to_owned)?.ok_or("no device is listed")?;
    let dev = d.devices.first().ok_or("no device is listed")?;
    gpu::append_failure_recovers(d, dev)
}

/// Fields of turbo_embed_options a run honors: normalize (4), pooling (5)
/// and output_dim (6), every value of each.
const EMBED_HONORED: u32 = 0b111000;

unsafe extern "C" fn context_create(
    ordinal: u32,
    log: turbo_log_fn,
    log_user_data: *mut c_void,
    out: *mut *mut c_void,
    err: *mut turbo_error,
) -> i32 {
    let (d, dev) = match unsafe { listed(ordinal, err) } {
        Ok(v) => v,
        Err(rc) => return rc,
    };
    unsafe { gpu::guarded(err, || gpu::create(d, dev, ordinal, log, log_user_data, out)) }
}

/// The label benchmarks are filed under, by PCI device id. A device not
/// named here is filed under its id.
fn arch(vendor_id: u32, device_id: u32) -> String {
    match (vendor_id, device_id) {
        (0x8086, 0xe223) => "b70".to_owned(),
        (0x8086, id) => format!("intel-{id:04x}"),
        (v, id) => format!("{v:04x}-{id:04x}"),
    }
}

/// Level Zero only promises that driverVersion increases. Intel's driver
/// packs major, minor and build into it; another vendor's is shown as is.
fn driver_version(vendor_id: u32, v: u32) -> String {
    match vendor_id {
        0x8086 => format!("{}.{}.{}", v >> 24, (v >> 16) & 0xff, v & 0xffff),
        _ => v.to_string(),
    }
}

fn vendor(vendor_id: u32) -> String {
    match vendor_id {
        0x8086 => "Intel".to_owned(),
        v => format!("0x{v:04x}"),
    }
}

/// The drivers and their GPUs, found once per process: the loader and
/// zeInitDrivers are process-wide, and devices do not come and go.
pub(crate) struct Driver {
    api: ze::Api,
    loader_version: String,
    devices: Vec<Device>,
}

pub(crate) struct Device {
    driver: ze::Handle,
    handle: ze::Handle,
    /// Sysman's memory modules on the device, matched to it by UUID; empty
    /// when sysman is unavailable.
    memory_modules: Vec<ze::Handle>,
    integrated: bool,
    memory_total: u64,
    /// The most local memory one work-group may have.
    max_local: u32,
    /// Whether the device computes in F64, which the encoder sums in.
    fp64: bool,
    arch: String,
    name: String,
    vendor: String,
    driver_version: String,
}

// Level Zero handles may be used from any thread; the rest is read-only.
unsafe impl Sync for Driver {}
unsafe impl Send for Driver {}

/// Ok(None) when there is no loader or no GPU driver: nothing to list, not
/// an error. Err when the driver is there and answers wrongly.
fn driver() -> Result<Option<&'static Driver>, &'static str> {
    static DRIVER: OnceLock<Result<Option<Driver>, String>> = OnceLock::new();
    match DRIVER.get_or_init(Driver::open) {
        Ok(d) => Ok(d.as_ref()),
        Err(e) => Err(e),
    }
}

impl Driver {
    fn open() -> Result<Option<Driver>, String> {
        let Some(api) = ze::Api::load()? else {
            return Ok(None);
        };
        let mut desc = ze::InitDriverTypeDesc {
            stype: ze::STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC,
            p_next: std::ptr::null(),
            flags: ze::INIT_DRIVER_TYPE_FLAG_GPU,
        };
        let mut n = 0u32;
        // No GPU driver is installed: nothing to list. Any other failure,
        // such as a user outside the render group, says why.
        match unsafe { (api.init_drivers)(&mut n, std::ptr::null_mut(), &mut desc) } {
            0 if n > 0 => {}
            0 | ze::RESULT_ERROR_UNINITIALIZED => return Ok(None),
            rc => ze::check("zeInitDrivers", rc)?,
        }
        let mut drivers = vec![std::ptr::null_mut(); n as usize];
        ze::check("zeInitDrivers", unsafe { (api.init_drivers)(&mut n, drivers.as_mut_ptr(), &mut desc) })?;
        drivers.truncate(n as usize);

        let sysman = Sysman::open(&api);
        let mut devices = Vec::new();
        for &drv in &drivers {
            let mut props = ze::DriverProperties { stype: ze::STRUCTURE_TYPE_DRIVER_PROPERTIES, ..Default::default() };
            ze::check("zeDriverGetProperties", unsafe { (api.driver_get_properties)(drv, &mut props) })?;
            for dev in ze::list("zeDeviceGet", |n, out| unsafe { (api.device_get)(drv, n, out) })? {
                let mut p = ze::DeviceProperties { stype: ze::STRUCTURE_TYPE_DEVICE_PROPERTIES, ..Default::default() };
                ze::check("zeDeviceGetProperties", unsafe { (api.device_get_properties)(dev, &mut p) })?;
                if p.kind != ze::DEVICE_TYPE_GPU {
                    continue;
                }
                let mut n = 0u32;
                let rc = unsafe { (api.device_get_memory_properties)(dev, &mut n, std::ptr::null_mut()) };
                ze::check("zeDeviceGetMemoryProperties", rc)?;
                let mut mem: Vec<ze::DeviceMemoryProperties> = (0..n)
                    .map(|_| ze::DeviceMemoryProperties {
                        stype: ze::STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES,
                        ..Default::default()
                    })
                    .collect();
                let rc = unsafe { (api.device_get_memory_properties)(dev, &mut n, mem.as_mut_ptr()) };
                ze::check("zeDeviceGetMemoryProperties", rc)?;
                mem.truncate(n as usize);
                let mut compute = ze::DeviceComputeProperties {
                    stype: ze::STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES,
                    ..Default::default()
                };
                ze::check("zeDeviceGetComputeProperties", unsafe {
                    (api.device_get_compute_properties)(dev, &mut compute)
                })?;
                let mut module = ze::DeviceModuleProperties {
                    stype: ze::STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES,
                    ..Default::default()
                };
                ze::check("zeDeviceGetModuleProperties", unsafe {
                    (api.device_get_module_properties)(dev, &mut module)
                })?;
                devices.push(Device {
                    driver: drv,
                    handle: dev,
                    memory_modules: sysman.as_ref().map_or_else(Vec::new, |s| s.memory_modules(&api, &p.uuid)),
                    integrated: p.flags & ze::DEVICE_PROPERTY_FLAG_INTEGRATED != 0,
                    memory_total: mem.iter().map(|m| m.total_size).sum(),
                    max_local: compute.max_shared_local_memory,
                    fp64: module.flags & ze::DEVICE_MODULE_FLAG_FP64 != 0,
                    arch: arch(p.vendor_id, p.device_id),
                    name: ze::string(&p.name),
                    vendor: vendor(p.vendor_id),
                    driver_version: driver_version(p.vendor_id, props.driver_version),
                });
            }
        }
        let loader_version = api.loader_version();
        Ok(Some(Driver { api, loader_version, devices }))
    }

    /// The free bytes now in the device's memory modules. None when sysman
    /// cannot say.
    fn memory_free(&self, dev: &Device) -> Option<u64> {
        let sysman = self.api.sysman.as_ref()?;
        let mut free = 0u64;
        for &m in &dev.memory_modules {
            let mut state = ze::MemState { stype: ze::ZES_STRUCTURE_TYPE_MEM_STATE, ..Default::default() };
            if unsafe { (sysman.memory_get_state)(m, &mut state) } != 0 {
                return None;
            }
            free += state.free;
        }
        (!dev.memory_modules.is_empty()).then_some(free)
    }
}

/// Sysman's view of the same devices, for free memory.
struct Sysman {
    devices: Vec<ze::Handle>,
}

impl Sysman {
    fn open(api: &ze::Api) -> Option<Sysman> {
        let s = api.sysman.as_ref()?;
        if unsafe { (s.init)(0) } != 0 {
            return None;
        }
        let drivers = ze::list("zesDriverGet", |n, out| unsafe { (s.driver_get)(n, out) }).ok()?;
        let mut devices = Vec::new();
        for drv in drivers {
            devices.extend(ze::list("zesDeviceGet", |n, out| unsafe { (s.device_get)(drv, n, out) }).ok()?);
        }
        Some(Sysman { devices })
    }

    /// The memory modules on the device with this UUID, not those on the
    /// host. Empty when sysman does not know the device.
    fn memory_modules(&self, api: &ze::Api, uuid: &[u8; 16]) -> Vec<ze::Handle> {
        let Some(s) = api.sysman.as_ref() else {
            return Vec::new();
        };
        let found = self.devices.iter().copied().find(|&d| {
            let mut p =
                ze::SysmanDeviceProperties { stype: ze::ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES, ..Default::default() };
            unsafe { (s.device_get_properties)(d, &mut p) == 0 && p.core.uuid == *uuid }
        });
        let Some(dev) = found else {
            return Vec::new();
        };
        let Ok(mems) =
            ze::list("zesDeviceEnumMemoryModules", |n, out| unsafe { (s.device_enum_memory_modules)(dev, n, out) })
        else {
            return Vec::new();
        };
        mems.into_iter()
            .filter(|&m| {
                let mut p = ze::MemProperties { stype: ze::ZES_STRUCTURE_TYPE_MEM_PROPERTIES, ..Default::default() };
                unsafe { (s.memory_get_properties)(m, &mut p) == 0 && p.location == ze::ZES_MEM_LOC_DEVICE }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_device_is_filed_under_its_label_or_its_id() {
        assert_eq!(arch(0x8086, 0xe223), "b70");
        assert_eq!(arch(0x8086, 0xe20b), "intel-e20b");
        assert_eq!(arch(0x10de, 0x2704), "10de-2704");
        assert_eq!(vendor(0x8086), "Intel");
        assert_eq!(vendor(0x10de), "0x10de");
        assert_eq!(driver_version(0x8086, 0x0103_909c), "1.3.37020");
        assert_eq!(driver_version(0x10de, 0x0103_909c), "17010844");
    }
}
