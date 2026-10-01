//! The part of the Level Zero C API the NPU backend calls, from
//! level_zero/ze_api.h, and the Intel graph extension from
//! intel/level-zero-npu-extensions' ze_graph_ext.h. zeInitDrivers needs a
//! 1.10 or later loader; the graph extension is reached through
//! zeDriverGetExtensionFunctionAddress, never linked.
//!
//! The graph DDI table a driver hands out is the driver's own static
//! struct, sized by the extension version the driver implements. A field
//! past that version is not read: every access goes through GraphExt,
//! which checks the advertised version first.

use std::ffi::{c_char, c_void};

pub type Handle = *mut c_void;
pub type Status = i32;

pub const RESULT_ERROR_UNINITIALIZED: Status = 0x7800_0001;
pub const RESULT_ERROR_OUT_OF_HOST_MEMORY: Status = 0x7000_0002;
pub const RESULT_ERROR_OUT_OF_DEVICE_MEMORY: Status = 0x7000_0003;

pub const STRUCTURE_TYPE_DRIVER_PROPERTIES: u32 = 0x1;
pub const STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 0x3;
pub const STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES: u32 = 0x7;
pub const STRUCTURE_TYPE_CONTEXT_DESC: u32 = 0xd;
pub const STRUCTURE_TYPE_COMMAND_QUEUE_DESC: u32 = 0xe;
pub const STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC: u32 = 0x16;
pub const STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC: u32 = 0x0002_0021;

pub const INIT_DRIVER_TYPE_FLAG_NPU: u32 = 2;
/// ze_device_type_t's ZE_DEVICE_TYPE_VPU, which the current headers name
/// the NPU: Intel AI Boost reports it.
pub const DEVICE_TYPE_NPU: u32 = 5;

pub const COMMAND_QUEUE_FLAG_IN_ORDER: u32 = 2;
pub const COMMAND_QUEUE_MODE_ASYNCHRONOUS: u32 = 2;
pub const COMMAND_QUEUE_PRIORITY_NORMAL: u32 = 0;

// ---- The graph extension (ze_graph_ext.h) ------------------------------------------

pub const GRAPH_EXT_NAME: &std::ffi::CStr = c"ZE_extension_graph";

/// ZE_MAKE_VERSION(major, minor).
pub const fn version(major: u32, minor: u32) -> u32 {
    (major << 16) | minor
}

pub const STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES: u32 = 0x1;
pub const STRUCTURE_TYPE_GRAPH_DESC_2: u32 = 0xE;
pub const STRUCTURE_TYPE_GRAPH_PROPERTIES_2: u32 = 0x10;
pub const STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3: u32 = 0xD;

/// ze_graph_format_t's ZE_GRAPH_FORMAT_NGRAPH_LITE: "ngraph lite", the
/// serialized OpenVINO IR the driver's compiler takes (NATIVE, 0x1, is a
/// pre-compiled blob, which no bundle carries yet).
pub const GRAPH_FORMAT_NGRAPH_LITE: u32 = 0x2;

pub const GRAPH_ARGUMENT_TYPE_INPUT: u32 = 0;

/// ze_graph_argument_precision_t, the ones this backend reads and writes.
pub const GRAPH_ARGUMENT_PRECISION_FP32: u32 = 0x01;
pub const GRAPH_ARGUMENT_PRECISION_FP16: u32 = 0x02;
pub const GRAPH_ARGUMENT_PRECISION_INT32: u32 = 0x05;
pub const GRAPH_ARGUMENT_PRECISION_BF16: u32 = 0x09;
pub const GRAPH_ARGUMENT_PRECISION_INT64: u32 = 0x11;
pub const GRAPH_ARGUMENT_PRECISION_UINT64: u32 = 0x10;
pub const GRAPH_ARGUMENT_PRECISION_UINT32: u32 = 0x0A;

/// ze_graph_argument_layout_t, the packed layouts the build flags name.
/// The enum is not contiguous: CHW is 0x80 and NC follows HW (0xC0), so
/// NC is 0xC1. BLOCKED (0xC8) is a device tiling the host buffers are
/// not written as.
pub const GRAPH_ARGUMENT_LAYOUT_CHW: u32 = 0x80;
pub const GRAPH_ARGUMENT_LAYOUT_NC: u32 = 0xC1;

/// ze_graph_init_stage_t.
pub const GRAPH_STAGE_COMMAND_LIST_INITIALIZE: u32 = 0x1;
pub const GRAPH_STAGE_INITIALIZE: u32 = 0x2;

pub const MAX_GRAPH_ARGUMENT_NAME: usize = 256;
pub const MAX_GRAPH_ARGUMENT_DIMENSIONS: usize = 5;
pub const MAX_GRAPH_TENSOR_NAMES: usize = 32;

#[repr(C)]
pub struct InitDriverTypeDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct ContextDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct HostMemAllocDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct CommandQueueDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub ordinal: u32,
    pub index: u32,
    pub flags: u32,
    pub mode: u32,
    pub priority: u32,
}

#[repr(C)]
pub struct DriverProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub uuid: [u8; 16],
    pub driver_version: u32,
}

pub const MAX_EXTENSION_NAME: usize = 256;

#[repr(C)]
pub struct DriverExtensionProperties {
    pub name: [c_char; MAX_EXTENSION_NAME],
    pub version: u32,
}

#[repr(C)]
pub struct DeviceProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub kind: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub flags: u32,
    pub subdevice_id: u32,
    pub core_clock_rate: u32,
    pub max_mem_alloc_size: u64,
    pub max_hardware_contexts: u32,
    pub max_command_queue_priority: u32,
    pub num_threads_per_eu: u32,
    pub physical_eu_simd_width: u32,
    pub num_eus_per_subslice: u32,
    pub num_subslices_per_slice: u32,
    pub num_slices: u32,
    pub timer_resolution: u64,
    pub timestamp_valid_bits: u32,
    pub kernel_timestamp_valid_bits: u32,
    pub uuid: [u8; 16],
    pub name: [c_char; 256],
}

#[repr(C)]
pub struct DeviceMemoryProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub flags: u32,
    pub max_clock_rate: u32,
    pub max_bus_width: u32,
    pub total_size: u64,
    pub name: [c_char; 256],
}

#[repr(C)]
pub struct ComponentVersion {
    pub component_name: [c_char; 64],
    pub spec_version: u32,
    pub major: i32,
    pub minor: i32,
    pub patch: i32,
}

/// ze_graph_compiler_version_info_t.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CompilerVersion {
    pub major: u16,
    pub minor: u16,
}

/// ze_device_graph_properties_t.
#[repr(C)]
pub struct DeviceGraphProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub graph_extension_version: u32,
    pub compiler_version: CompilerVersion,
    pub graph_formats_supported: u32,
    pub max_ov_opset_version_supported: u32,
}

/// ze_graph_desc_2_t.
#[repr(C)]
pub struct GraphDesc2 {
    pub stype: u32,
    pub p_next: *const c_void,
    pub format: u32,
    pub input_size: usize,
    pub input: *const u8,
    pub build_flags: *const c_char,
    pub flags: u32,
}

/// ze_graph_properties_2_t.
#[repr(C)]
pub struct GraphProperties2 {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub num_graph_args: u32,
    pub init_stage_required: u32,
}

/// ze_graph_argument_properties_3_t.
#[repr(C)]
pub struct GraphArgumentProperties3 {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub name: [c_char; MAX_GRAPH_ARGUMENT_NAME],
    pub kind: u32,
    pub dims: [u32; MAX_GRAPH_ARGUMENT_DIMENSIONS],
    pub network_precision: u32,
    pub network_layout: u32,
    pub device_precision: u32,
    pub device_layout: u32,
    pub quant_reverse_scale: f32,
    pub quant_zero_point: u8,
    pub dims_count: u32,
    pub debug_friendly_name: [c_char; MAX_GRAPH_ARGUMENT_NAME],
    pub associated_tensor_names: [[c_char; MAX_GRAPH_ARGUMENT_NAME]; MAX_GRAPH_TENSOR_NAMES],
    pub associated_tensor_names_count: u32,
}

// Every struct here is plain data whose all-zero value is valid.
macro_rules! zeroed_default {
    ($($t:ty),*) => {$(
        impl Default for $t {
            fn default() -> Self {
                unsafe { std::mem::zeroed() }
            }
        }
    )*};
}
zeroed_default!(
    DriverProperties,
    DriverExtensionProperties,
    DeviceProperties,
    DeviceMemoryProperties,
    ComponentVersion,
    DeviceGraphProperties,
    GraphProperties2,
    GraphArgumentProperties3
);

// ---- The graph DDI table (ze_graph_dditable_ext_t) ---------------------------------

pub type PfnGraphCreate2 = unsafe extern "C" fn(Handle, Handle, *const GraphDesc2, *mut Handle) -> Status;
pub type PfnGraphCreate3 = unsafe extern "C" fn(Handle, Handle, *const GraphDesc2, *mut Handle, *mut Handle) -> Status;
pub type PfnGraphDestroy = unsafe extern "C" fn(Handle) -> Status;
pub type PfnGraphSetArgumentValue = unsafe extern "C" fn(Handle, u32, *const c_void) -> Status;
pub type PfnAppendGraph = unsafe extern "C" fn(Handle, Handle, Handle, u32, *mut Handle) -> Status;
pub type PfnAppendGraphExecute = unsafe extern "C" fn(Handle, Handle, Handle, Handle, u32, *mut Handle) -> Status;
pub type PfnDeviceGetGraphProperties = unsafe extern "C" fn(Handle, *mut DeviceGraphProperties) -> Status;
pub type PfnGraphGetProperties2 = unsafe extern "C" fn(Handle, *mut GraphProperties2) -> Status;
pub type PfnGraphGetArgumentProperties3 = unsafe extern "C" fn(Handle, u32, *mut GraphArgumentProperties3) -> Status;
pub type PfnGraphInitialize = unsafe extern "C" fn(Handle) -> Status;
pub type PfnBuildLogGetString2 = unsafe extern "C" fn(Handle, *mut u32, *mut c_char) -> Status;
pub type PfnBuildLogDestroy = unsafe extern "C" fn(Handle) -> Status;

/// ze_graph_dditable_ext_t, field for field. Only the fields this backend
/// calls are typed; the rest are placeholders that keep every offset as
/// the header lays it out. A field is read only through GraphExt, which
/// checks the driver's advertised version covers it: the table lives in
/// the driver and ends at the driver's own header version.
#[repr(C)]
pub struct GraphDdi {
    // version 1.0
    pub pfn_create: *const c_void,
    pub pfn_destroy: Option<PfnGraphDestroy>,
    pub pfn_get_properties: *const c_void,
    pub pfn_get_argument_properties: *const c_void,
    pub pfn_set_argument_value: Option<PfnGraphSetArgumentValue>,
    pub pfn_append_graph_initialize: Option<PfnAppendGraph>,
    pub pfn_append_graph_execute: Option<PfnAppendGraphExecute>,
    pub pfn_get_native_binary: *const c_void,
    pub pfn_device_get_graph_properties: Option<PfnDeviceGetGraphProperties>,
    // version 1.1
    pub pfn_graph_get_argument_metadata: *const c_void,
    pub pfn_get_argument_properties2: *const c_void,
    // version 1.2
    pub pfn_get_argument_properties3: Option<PfnGraphGetArgumentProperties3>,
    // version 1.3
    pub pfn_query_network_create: *const c_void,
    pub pfn_query_network_destroy: *const c_void,
    pub pfn_query_network_get_supported_layers: *const c_void,
    // version 1.4
    pub pfn_build_log_get_string: *const c_void,
    // version 1.5
    pub pfn_create2: Option<PfnGraphCreate2>,
    pub pfn_query_network_create2: *const c_void,
    pub pfn_query_context_memory: *const c_void,
    // version 1.6
    pub pfn_device_get_graph_properties2: *const c_void,
    // version 1.7
    pub pfn_get_native_binary2: *const c_void,
    // version 1.8
    pub pfn_get_properties2: Option<PfnGraphGetProperties2>,
    pub pfn_graph_initialize: Option<PfnGraphInitialize>,
    // version 1.11
    pub pfn_compiler_get_supported_options: *const c_void,
    pub pfn_compiler_is_option_supported: *const c_void,
    // version 1.12
    pub pfn_create3: Option<PfnGraphCreate3>,
    pub pfn_get_properties3: *const c_void,
    pub pfn_build_log_get_string2: Option<PfnBuildLogGetString2>,
    pub pfn_build_log_destroy: Option<PfnBuildLogDestroy>,
    // version 1.15
    pub pfn_set_argument_value2: *const c_void,
    // version 1.16
    pub pfn_evict: *const c_void,
    // Versions 1.17, 1.18 and 1.19 add no pointers. Version 1.20, which
    // the published header names ZE_GRAPH_EXT_VERSION_CURRENT, appends
    // these two. This backend does not call them. They stay so the
    // table is the header's size: a C compile of that header is 33
    // pointers, with pfnEvict at index 30.
    pub pfn_get_argument_properties4: *const c_void,
    pub pfn_get_argument_names: *const c_void,
}

/// The driver's graph DDI table and the extension version it advertised:
/// a field appended after that version is never read.
#[derive(Clone, Copy)]
pub struct GraphExt {
    table: *const GraphDdi,
    pub version: u32,
}

// The table is the driver's static data, read from any thread.
unsafe impl Send for GraphExt {}
unsafe impl Sync for GraphExt {}

/// One field of the table, read only when the driver's version covers
/// it. The read projects the field's address from the raw pointer, so a
/// table that ends at an older version is never touched past its end.
macro_rules! covered {
    ($self:ident, $since:expr, $field:ident) => {
        if $self.version < $since { None } else { unsafe { std::ptr::addr_of!((*$self.table).$field).read() } }
    };
}

impl GraphExt {
    /// # Safety
    /// `table` is the pointer zeDriverGetExtensionFunctionAddress gave
    /// for ZE_extension_graph on a driver advertising `version`.
    pub unsafe fn new(table: *const GraphDdi, version: u32) -> GraphExt {
        GraphExt { table, version }
    }

    pub fn destroy(&self) -> Option<PfnGraphDestroy> {
        covered!(self, version(1, 0), pfn_destroy)
    }

    pub fn set_argument_value(&self) -> Option<PfnGraphSetArgumentValue> {
        covered!(self, version(1, 0), pfn_set_argument_value)
    }

    pub fn append_graph_initialize(&self) -> Option<PfnAppendGraph> {
        covered!(self, version(1, 0), pfn_append_graph_initialize)
    }

    pub fn append_graph_execute(&self) -> Option<PfnAppendGraphExecute> {
        covered!(self, version(1, 0), pfn_append_graph_execute)
    }

    pub fn device_get_graph_properties(&self) -> Option<PfnDeviceGetGraphProperties> {
        covered!(self, version(1, 0), pfn_device_get_graph_properties)
    }

    pub fn get_argument_properties3(&self) -> Option<PfnGraphGetArgumentProperties3> {
        covered!(self, version(1, 2), pfn_get_argument_properties3)
    }

    pub fn create2(&self) -> Option<PfnGraphCreate2> {
        covered!(self, version(1, 5), pfn_create2)
    }

    pub fn get_properties2(&self) -> Option<PfnGraphGetProperties2> {
        covered!(self, version(1, 8), pfn_get_properties2)
    }

    pub fn graph_initialize(&self) -> Option<PfnGraphInitialize> {
        covered!(self, version(1, 8), pfn_graph_initialize)
    }

    pub fn create3(&self) -> Option<PfnGraphCreate3> {
        covered!(self, version(1, 12), pfn_create3)
    }

    pub fn build_log_get_string2(&self) -> Option<PfnBuildLogGetString2> {
        covered!(self, version(1, 12), pfn_build_log_get_string2)
    }

    pub fn build_log_destroy(&self) -> Option<PfnBuildLogDestroy> {
        covered!(self, version(1, 12), pfn_build_log_destroy)
    }
}

// ---- The loader -------------------------------------------------------------------

pub struct Api {
    // Keeps the function pointers below valid.
    _lib: libloading::Library,
    pub init_drivers: unsafe extern "C" fn(*mut u32, *mut Handle, *mut InitDriverTypeDesc) -> Status,
    pub driver_get_properties: unsafe extern "C" fn(Handle, *mut DriverProperties) -> Status,
    pub driver_get_extension_properties:
        unsafe extern "C" fn(Handle, *mut u32, *mut DriverExtensionProperties) -> Status,
    pub driver_get_extension_function_address: unsafe extern "C" fn(Handle, *const c_char, *mut *mut c_void) -> Status,
    pub device_get: unsafe extern "C" fn(Handle, *mut u32, *mut Handle) -> Status,
    pub device_get_properties: unsafe extern "C" fn(Handle, *mut DeviceProperties) -> Status,
    pub device_get_memory_properties: unsafe extern "C" fn(Handle, *mut u32, *mut DeviceMemoryProperties) -> Status,
    pub context_create: unsafe extern "C" fn(Handle, *const ContextDesc, *mut Handle) -> Status,
    pub context_destroy: unsafe extern "C" fn(Handle) -> Status,
    pub mem_alloc_host: unsafe extern "C" fn(Handle, *const HostMemAllocDesc, usize, usize, *mut *mut c_void) -> Status,
    pub mem_free: unsafe extern "C" fn(Handle, *mut c_void) -> Status,
    pub command_list_create_immediate:
        unsafe extern "C" fn(Handle, Handle, *const CommandQueueDesc, *mut Handle) -> Status,
    pub command_list_destroy: unsafe extern "C" fn(Handle) -> Status,
    pub command_list_host_synchronize: unsafe extern "C" fn(Handle, u64) -> Status,
    loader_get_versions: Option<unsafe extern "C" fn(*mut usize, *mut ComponentVersion) -> Status>,
}

#[cfg(windows)]
const LOADER: &str = "ze_loader.dll";
#[cfg(not(windows))]
const LOADER: &str = "libze_loader.so.1";

impl Api {
    /// None when the loader is not installed. A loader without
    /// zeInitDrivers predates Level Zero 1.10 and is an error.
    pub fn load() -> Result<Option<Api>, String> {
        let Ok(lib) = (unsafe { libloading::Library::new(LOADER) }) else {
            return Ok(None);
        };
        let need = |name: &str| format!("npu: {LOADER} has no {name}; Level Zero 1.10 or later is needed");
        macro_rules! sym {
            ($name:literal) => {
                *unsafe { lib.get(concat!($name, "\0").as_bytes()) }.map_err(|_| need($name))?
            };
        }
        macro_rules! opt {
            ($name:literal) => {
                unsafe { lib.get(concat!($name, "\0").as_bytes()) }.ok().map(|s| *s)
            };
        }
        Ok(Some(Api {
            init_drivers: sym!("zeInitDrivers"),
            driver_get_properties: sym!("zeDriverGetProperties"),
            driver_get_extension_properties: sym!("zeDriverGetExtensionProperties"),
            driver_get_extension_function_address: sym!("zeDriverGetExtensionFunctionAddress"),
            device_get: sym!("zeDeviceGet"),
            device_get_properties: sym!("zeDeviceGetProperties"),
            device_get_memory_properties: sym!("zeDeviceGetMemoryProperties"),
            context_create: sym!("zeContextCreate"),
            context_destroy: sym!("zeContextDestroy"),
            mem_alloc_host: sym!("zeMemAllocHost"),
            mem_free: sym!("zeMemFree"),
            command_list_create_immediate: sym!("zeCommandListCreateImmediate"),
            command_list_destroy: sym!("zeCommandListDestroy"),
            command_list_host_synchronize: sym!("zeCommandListHostSynchronize"),
            loader_get_versions: opt!("zelLoaderGetVersions"),
            _lib: lib,
        }))
    }

    /// The loader's own version, "" when it does not say.
    pub fn loader_version(&self) -> String {
        let Some(f) = self.loader_get_versions else {
            return String::new();
        };
        let mut n = 0usize;
        if unsafe { f(&mut n, std::ptr::null_mut()) } != 0 {
            return String::new();
        }
        let mut v: Vec<ComponentVersion> = (0..n).map(|_| ComponentVersion::default()).collect();
        if unsafe { f(&mut n, v.as_mut_ptr()) } != 0 {
            return String::new();
        }
        v.truncate(n);
        v.iter()
            .find(|c| string(&c.component_name) == "loader")
            .map_or_else(String::new, |c| format!("{}.{}.{}", c.major, c.minor, c.patch))
    }
}

pub fn check(what: &str, rc: Status) -> Result<(), String> {
    if rc != 0 { Err(format!("npu: {what} failed with 0x{rc:08x}")) } else { Ok(()) }
}

/// The two-call pattern: ask for the count, then fill that many.
pub fn list(what: &str, mut f: impl FnMut(*mut u32, *mut Handle) -> Status) -> Result<Vec<Handle>, String> {
    let mut n = 0u32;
    check(what, f(&mut n, std::ptr::null_mut()))?;
    let mut out = vec![std::ptr::null_mut(); n as usize];
    check(what, f(&mut n, out.as_mut_ptr()))?;
    out.truncate(n as usize);
    Ok(out)
}

pub fn string(b: &[c_char]) -> String {
    crate::backend::cstr(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The sizes and offsets the C compiler gives these structs on x86_64
    // and aarch64, with the 1.28 ze_api.h and the current
    // intel/level-zero-npu-extensions ze_graph_ext.h.
    #[test]
    fn layouts_match_the_c_headers() {
        assert_eq!(size_of::<InitDriverTypeDesc>(), 24);
        assert_eq!(size_of::<ContextDesc>(), 24);
        assert_eq!(size_of::<HostMemAllocDesc>(), 24);
        assert_eq!(size_of::<CommandQueueDesc>(), 40);
        assert_eq!(std::mem::offset_of!(CommandQueueDesc, mode), 28);
        assert_eq!(size_of::<DriverProperties>(), 40);
        assert_eq!(size_of::<DriverExtensionProperties>(), 260);
        assert_eq!(size_of::<DeviceProperties>(), 368);
        assert_eq!(std::mem::offset_of!(DeviceProperties, name), 112);
        assert_eq!(size_of::<DeviceMemoryProperties>(), 296);
        assert_eq!(size_of::<CompilerVersion>(), 4);
        assert_eq!(size_of::<DeviceGraphProperties>(), 32);
        assert_eq!(std::mem::offset_of!(DeviceGraphProperties, compiler_version), 20);
        assert_eq!(size_of::<GraphDesc2>(), 56);
        assert_eq!(std::mem::offset_of!(GraphDesc2, input_size), 24);
        assert_eq!(std::mem::offset_of!(GraphDesc2, flags), 48);
        assert_eq!(size_of::<GraphProperties2>(), 24);
        assert_eq!(size_of::<GraphArgumentProperties3>(), 8776);
        assert_eq!(std::mem::offset_of!(GraphArgumentProperties3, kind), 272);
        assert_eq!(std::mem::offset_of!(GraphArgumentProperties3, dims), 276);
        assert_eq!(std::mem::offset_of!(GraphArgumentProperties3, network_precision), 296);
        assert_eq!(std::mem::offset_of!(GraphArgumentProperties3, quant_zero_point), 316);
        assert_eq!(std::mem::offset_of!(GraphArgumentProperties3, dims_count), 320);
        assert_eq!(std::mem::offset_of!(GraphArgumentProperties3, debug_friendly_name), 324);
        assert_eq!(std::mem::offset_of!(GraphArgumentProperties3, associated_tensor_names_count), 8772);
    }

    // ze_graph_dditable_ext_t's offsets, one pointer per function, in the
    // header's order: a driver whose table ends at its own version is
    // never read past it (GraphExt::field).
    #[test]
    fn the_ddi_table_lies_out_as_the_header_orders_it() {
        let p = size_of::<*const c_void>();
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_destroy), p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_set_argument_value), 4 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_append_graph_initialize), 5 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_append_graph_execute), 6 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_device_get_graph_properties), 8 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_get_argument_properties3), 11 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_create2), 16 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_get_properties2), 21 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_graph_initialize), 22 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_create3), 25 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_build_log_get_string2), 27 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_build_log_destroy), 28 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_evict), 30 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_get_argument_properties4), 31 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_get_argument_names), 32 * p);
        // ze_graph_dditable_ext_t in the published header
        // (ZE_GRAPH_EXT_VERSION_CURRENT 1.20) is 33 pointers.
        assert_eq!(size_of::<GraphDdi>(), 33 * p);
    }

    /// A table shorter than the newest version is never read past the
    /// version it came with.
    #[test]
    fn a_field_past_the_advertised_version_is_not_read() {
        // A 1.8 driver's table: everything after pfn_graph_initialize is
        // absent, and the memory past it does not belong to the table.
        let bytes = vec![0u8; std::mem::offset_of!(GraphDdi, pfn_compiler_get_supported_options)];
        let ext = unsafe { GraphExt::new(bytes.as_ptr() as *const GraphDdi, version(1, 8)) };
        assert!(ext.create3().is_none(), "1.12's create3 is past a 1.8 table");
        assert!(ext.build_log_get_string2().is_none());
        assert!(ext.create2().is_none(), "the field is covered but this driver left it NULL");
        let ext = unsafe { GraphExt::new(bytes.as_ptr() as *const GraphDdi, version(1, 7)) };
        assert!(ext.get_properties2().is_none(), "1.8's get_properties2 is past a 1.7 version");
    }
}
