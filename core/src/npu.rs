//! The npu backend: Intel NPUs (AI Boost) through the oneAPI Level Zero
//! driver's graph extension, reached through turbo_backend.h like any
//! other backend, and distinct from `levelzero`, the Intel GPU backend.
//! The loader is opened at run time, so a machine without it lists
//! nothing. It lists the devices of type NPU whose driver carries the
//! graph extension and that answer its device probe; a device that does
//! not is never listed and nothing stands in for it. Models are OpenVINO
//! IR artifacts the driver's own compiler builds into a graph on the
//! device (npu/ir.rs, npu/graph.rs); the library carries no OpenVINO and
//! no ONNX Runtime. docs/npu.md says how to build and test it.

mod graph;
mod ir;
mod ze;

use std::ffi::{c_char, c_void};
use std::sync::OnceLock;

use crate::backend::{
    TURBO_CAP_EXPERIMENTAL, TURBO_CAP_UNSUPPORTED, TURBO_FORMAT_OPENVINO_IR, format_bit, refuse, turbo_backend,
};
use crate::status::{DEVICE_UNAVAILABLE, INVALID_ARGUMENT};
use crate::{TURBO_DEVICE_NPU, TURBO_PRECISION_EXACT, turbo_device_info, turbo_error, turbo_log_fn, write_str};

pub static BACKEND: turbo_backend = turbo_backend {
    struct_size: size_of::<turbo_backend>() as u32,
    reserved: 0,
    name: c"npu".as_ptr(),
    // Nothing is linked at build time: the loader is opened at run time,
    // and its version is each device's runtime_version.
    runtime_version: c"".as_ptr(),
    device_count,
    device_info,
    capability,
    context_create: Some(context_create),
    context_release: Some(graph::context_release),
    buffer_alloc: Some(graph::buffer_alloc),
    buffer_import: None,
    buffer_release: Some(graph::buffer_release),
    buffer_export: Some(graph::buffer_export),
    model_load: Some(graph::model_load),
    model_release: Some(graph::model_release),
    session_create: Some(graph::session_create),
    session_release: Some(graph::session_release),
    embed_write: Some(graph::embed_write),
    session_run: Some(graph::session_run),
    // Every buffer it gives has a host address.
    buffer_read: None,
    formats: format_bit(TURBO_FORMAT_OPENVINO_IR),
    reserved2: 0,
    session_create_tuned: None,
};

/// The backend's table, for the tests.
pub fn backend() -> &'static turbo_backend {
    &BACKEND
}

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
        Ok(None) => return Err(unsafe { refuse(err, INVALID_ARGUMENT, "npu: no device is listed") }),
        Err(message) => return Err(unsafe { refuse(err, DEVICE_UNAVAILABLE, message) }),
    };
    match d.devices.get(ordinal as usize) {
        Some(dev) => Ok((d, dev)),
        None => {
            let message = format!("npu device {ordinal}: {} are listed", d.devices.len());
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
    out.kind = TURBO_DEVICE_NPU;
    out.ordinal = ordinal;
    // An NPU sits beside the CPU and computes over the host's DDR.
    out.unified_memory = 1;
    out.memory_total = dev.memory_total;
    // Nothing says what of it is free right now.
    out.memory_free = 0;
    write_str(&mut out.arch, &dev.arch);
    write_str(&mut out.name, &dev.name);
    write_str(&mut out.vendor, &dev.vendor);
    write_str(&mut out.runtime_version, &d.loader_version);
    write_str(&mut out.driver_version, &dev.driver_version);
    0
}

/// Fields of turbo_embed_options a run honors: normalize (4), pooling (5)
/// and output_dim (6), every value of each.
const EMBED_HONORED: u32 = 0b111000;

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn capability(
    ordinal: u32,
    _task: u32,
    precision: u32,
    status: *mut u32,
    dtype: *mut u32,
    options_honored: *mut u32,
    reason: *mut c_char,
    reason_len: u32,
    err: *mut turbo_error,
) -> i32 {
    if let Err(rc) = unsafe { listed(ordinal, err) } {
        return rc;
    }
    unsafe {
        // The core always hands a buffer; a caller of the table that
        // does not is not written through.
        let say = |text: &str| {
            if !reason.is_null() && reason_len != 0 {
                write_str(std::slice::from_raw_parts_mut(reason, reason_len as usize), text);
            }
        };
        if precision == TURBO_PRECISION_EXACT {
            *status = TURBO_CAP_UNSUPPORTED;
            *dtype = 0;
            *options_honored = 0;
            say("EXACT asks for F32 throughout, and a compiled graph computes in the dtype its IR fixed");
        } else {
            *status = TURBO_CAP_EXPERIMENTAL;
            // No dtype is claimed before an artifact is seen: the IR's
            // compilation fixes it, model_load reads it from the
            // compiled graph, and turbo_session_get_info reports what a
            // session really resolved. Benchmark records name a dtype,
            // so a 0 here also backs no SUPPORTED claim.
            *dtype = 0;
            *options_honored = EMBED_HONORED;
            say("");
        }
    }
    0
}

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
    unsafe { graph::guarded(err, || graph::create(d, dev, ordinal, log, log_user_data, out)) }
}

/// The label benchmarks are filed under, by PCI device id. A device not
/// named here is filed under its id.
fn arch(vendor_id: u32, device_id: u32) -> String {
    match (vendor_id, device_id) {
        (0x8086, 0x7d1d) => "mtl-npu".to_owned(), // Meteor Lake, NPU 3720
        (0x8086, 0xad1d) => "arl-npu".to_owned(), // Arrow Lake, NPU 3720
        (0x8086, 0x643e) => "lnl-npu".to_owned(), // Lunar Lake, NPU 4
        (0x8086, 0xb03e) => "ptl-npu".to_owned(), // Panther Lake, NPU 5
        (0x8086, id) => format!("intel-npu-{id:04x}"),
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

/// The NPU drivers and their devices, found once per process: the loader
/// and zeInitDrivers are process-wide, and devices do not come and go.
pub(crate) struct Driver {
    api: ze::Api,
    loader_version: String,
    devices: Vec<Device>,
}

pub(crate) struct Device {
    driver: ze::Handle,
    handle: ze::Handle,
    /// The graph extension's DDI table and version, from the device's
    /// driver: being here is the probe having answered.
    ext: ze::GraphExt,
    /// The compiler in the driver, from the device's graph properties.
    compiler: ze::CompilerVersion,
    /// ze_graph_format_t bits the device's compiler takes, from the same
    /// probe: model_load refuses an IR when NGRAPH_LITE is not among
    /// them, before any compile is tried.
    formats_supported: u32,
    /// The highest OpenVINO opset the device's compiler supports; 0 when
    /// the driver does not say.
    max_opset: u32,
    memory_total: u64,
    arch: String,
    name: String,
    vendor: String,
    driver_version: String,
}

// Level Zero handles may be used from any thread; the rest is read-only.
unsafe impl Sync for Driver {}
unsafe impl Send for Driver {}
unsafe impl Sync for Device {}
unsafe impl Send for Device {}

/// Ok(None) when there is no loader or no NPU driver: nothing to list,
/// not an error. Err when a driver is there and answers wrongly, or NPU
/// hardware is there and every device had to be skipped. The result,
/// an Err too, is found once and kept for the life of the process: a
/// driver fixed underneath a running process is seen by the next one.
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
            flags: ze::INIT_DRIVER_TYPE_FLAG_NPU,
        };
        let mut n = 0u32;
        // No NPU driver is installed: nothing to list. Any other failure
        // says why.
        match unsafe { (api.init_drivers)(&mut n, std::ptr::null_mut(), &mut desc) } {
            0 if n > 0 => {}
            0 | ze::RESULT_ERROR_UNINITIALIZED => return Ok(None),
            rc => ze::check("zeInitDrivers", rc)?,
        }
        let mut drivers = vec![std::ptr::null_mut(); n as usize];
        ze::check("zeInitDrivers", unsafe { (api.init_drivers)(&mut n, drivers.as_mut_ptr(), &mut desc) })?;
        drivers.truncate(n as usize);

        let mut devices = Vec::new();
        // An NPU device seen and not listed, with why: when every one is
        // skipped nothing is listed, and that is an error with these
        // reasons, never a quiet empty list on a machine that has the
        // hardware.
        let mut skipped: Vec<String> = Vec::new();
        for &drv in &drivers {
            let mut props = ze::DriverProperties { stype: ze::STRUCTURE_TYPE_DRIVER_PROPERTIES, ..Default::default() };
            ze::check("zeDriverGetProperties", unsafe { (api.driver_get_properties)(drv, &mut props) })?;
            let mut npus = Vec::new();
            for dev in ze::list("zeDeviceGet", |n, out| unsafe { (api.device_get)(drv, n, out) })? {
                let mut p = ze::DeviceProperties { stype: ze::STRUCTURE_TYPE_DEVICE_PROPERTIES, ..Default::default() };
                ze::check("zeDeviceGetProperties", unsafe { (api.device_get_properties)(dev, &mut p) })?;
                if p.kind == ze::DEVICE_TYPE_NPU {
                    npus.push((dev, p));
                }
            }
            if npus.is_empty() {
                continue;
            }
            // The graph extension, named among the driver's extensions.
            let Some(ext_version) = graph_extension_version(&api, drv)? else {
                skipped.push(format!(
                    "{} NPU device(s) on a driver without ZE_extension_graph; the NPU driver is too old or broken",
                    npus.len()
                ));
                continue;
            };
            // Advertised and not handed out is a broken runtime, not a
            // missing one.
            let mut table = std::ptr::null_mut();
            ze::check("zeDriverGetExtensionFunctionAddress(ZE_extension_graph)", unsafe {
                (api.driver_get_extension_function_address)(drv, ze::GRAPH_EXT_NAME.as_ptr(), &mut table)
            })?;
            let ext = unsafe { ze::GraphExt::new(table as *const ze::GraphDdi, ext_version) };
            for (dev, p) in npus {
                // The probe: the device must answer for its own graph
                // properties. One that does not is not listed, nothing
                // stands in for it, and why is kept for the error below.
                let Some(get) = ext.device_get_graph_properties() else {
                    skipped.push(format!(
                        "{}: the driver's table has no pfnDeviceGetGraphProperties",
                        ze::string(&p.name)
                    ));
                    continue;
                };
                let mut gp = ze::DeviceGraphProperties {
                    stype: ze::STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES,
                    ..Default::default()
                };
                let rc = unsafe { get(dev, &mut gp) };
                if rc != 0 {
                    skipped
                        .push(format!("{}: pfnDeviceGetGraphProperties failed with 0x{rc:08x}", ze::string(&p.name)));
                    continue;
                }
                // 1.6 added pfnDeviceGetGraphProperties2. A 1.17 driver
                // fills graphExtensionVersion there; the 1.0 struct can
                // come back with that field still 0 while the compiler
                // version is real. A zero is not extension 0.0.
                let mut reported = gp.graph_extension_version;
                let mut compiler = gp.compiler_version;
                let mut formats_supported = gp.graph_formats_supported;
                let mut max_opset = gp.max_ov_opset_version_supported;
                if ext_version >= ze::version(1, 6)
                    && let Some(get2) = ext.device_get_graph_properties2()
                {
                    let mut gp2 = ze::DeviceGraphProperties2 {
                        stype: ze::STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES_2,
                        ..Default::default()
                    };
                    if unsafe { get2(dev, &mut gp2) } == 0
                        && (gp2.graph_extension_version != 0
                            || gp2.compiler_version.major != 0
                            || gp2.compiler_version.minor != 0)
                    {
                        reported = gp2.graph_extension_version;
                        compiler = gp2.compiler_version;
                        formats_supported = gp2.graph_formats_supported;
                        max_opset = gp2.max_ov_opset_version_supported;
                    }
                }
                let graph_version = ze::extension_version(ext_version, reported);
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
                devices.push(Device {
                    driver: drv,
                    handle: dev,
                    // The advertised extension version, unless the device
                    // named an older non-zero one.
                    ext: unsafe { ze::GraphExt::new(table as *const ze::GraphDdi, graph_version) },
                    compiler,
                    formats_supported,
                    max_opset,
                    memory_total: mem.iter().map(|m| m.total_size).sum(),
                    arch: arch(p.vendor_id, p.device_id),
                    name: ze::string(&p.name),
                    vendor: vendor(p.vendor_id),
                    driver_version: driver_version(p.vendor_id, props.driver_version),
                });
            }
        }
        if devices.is_empty() {
            // NPU hardware seen and every device skipped is an error
            // with the reasons; a machine without the hardware or its
            // driver lists nothing quietly.
            if !skipped.is_empty() {
                return Err(format!("npu: no device is listed: {}", skipped.join("; ")));
            }
            return Ok(None);
        }
        let loader_version = api.loader_version();
        Ok(Some(Driver { api, loader_version, devices }))
    }
}

/// ZE_extension_graph's version among the driver's extensions, None when
/// the driver does not list it.
fn graph_extension_version(api: &ze::Api, drv: ze::Handle) -> Result<Option<u32>, String> {
    let mut n = 0u32;
    ze::check("zeDriverGetExtensionProperties", unsafe {
        (api.driver_get_extension_properties)(drv, &mut n, std::ptr::null_mut())
    })?;
    let mut props: Vec<ze::DriverExtensionProperties> =
        (0..n).map(|_| ze::DriverExtensionProperties::default()).collect();
    ze::check("zeDriverGetExtensionProperties", unsafe {
        (api.driver_get_extension_properties)(drv, &mut n, props.as_mut_ptr())
    })?;
    props.truncate(n as usize);
    let name = ze::GRAPH_EXT_NAME.to_str().expect("ASCII");
    Ok(props.iter().find(|p| ze::string(&p.name) == name).map(|p| p.version))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_device_is_filed_under_its_label_or_its_id() {
        assert_eq!(arch(0x8086, 0xad1d), "arl-npu");
        assert_eq!(arch(0x8086, 0x7d1d), "mtl-npu");
        assert_eq!(arch(0x8086, 0x643e), "lnl-npu");
        assert_eq!(arch(0x8086, 0xb03e), "ptl-npu");
        assert_eq!(arch(0x8086, 0x1234), "intel-npu-1234");
        assert_eq!(arch(0x10de, 0x2704), "10de-2704");
        assert_eq!(vendor(0x8086), "Intel");
        assert_eq!(driver_version(0x8086, 0x0103_909c), "1.3.37020");
    }
}
