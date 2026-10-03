//! The Intel graph extension (ZE_extension_graph) from
//! intel/level-zero-npu-extensions' ze_graph_ext.h. The loader, the
//! result codes, the common structs and check/list/string live in
//! crate::ze, shared with the levelzero backend. This file is the graph
//! extension only.
//!
//! The graph DDI table a driver hands out is the driver's own static
//! struct, sized by the extension version the driver implements. A field
//! past that version is not read: every access goes through GraphExt,
//! which checks the advertised version first.

use std::ffi::{c_char, c_void};

pub use crate::ze::{
    Api, COMMAND_QUEUE_FLAG_IN_ORDER, COMMAND_QUEUE_MODE_ASYNCHRONOUS, COMMAND_QUEUE_PRIORITY_NORMAL, CommandQueueDesc,
    ContextDesc, DeviceMemoryProperties, DeviceProperties, DriverExtensionProperties, DriverProperties, Handle,
    HostMemAllocDesc, InitDriverTypeDesc, RESULT_ERROR_OUT_OF_DEVICE_MEMORY, RESULT_ERROR_OUT_OF_HOST_MEMORY,
    RESULT_ERROR_UNINITIALIZED, STRUCTURE_TYPE_COMMAND_QUEUE_DESC, STRUCTURE_TYPE_CONTEXT_DESC,
    STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES, STRUCTURE_TYPE_DEVICE_PROPERTIES, STRUCTURE_TYPE_DRIVER_PROPERTIES,
    STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC, Status, string,
};

pub const INIT_DRIVER_TYPE_FLAG_NPU: u32 = 2;
/// ze_device_type_t's ZE_DEVICE_TYPE_VPU, which the current headers name
/// the NPU: Intel AI Boost reports it.
pub const DEVICE_TYPE_NPU: u32 = 5;

pub fn check(what: &str, rc: Status) -> Result<(), String> {
    crate::ze::check("npu", what, rc)
}

/// The two-call pattern: ask for the count, then fill that many.
pub fn list(what: &str, f: impl FnMut(*mut u32, *mut Handle) -> Status) -> Result<Vec<Handle>, String> {
    crate::ze::list("npu", what, f)
}

// ---- The graph extension (ze_graph_ext.h) ------------------------------------------

pub const GRAPH_EXT_NAME: &std::ffi::CStr = c"ZE_extension_graph";

/// ZE_MAKE_VERSION(major, minor).
pub const fn version(major: u32, minor: u32) -> u32 {
    (major << 16) | minor
}

/// The version that gates the DDI table.
///
/// `advertised` is what `zeDriverGetExtensionProperties` reports for
/// `ZE_extension_graph`. `reported` is `graphExtensionVersion` from the
/// device graph properties. A driver fills the compiler and leaves that
/// field 0: that is the field unwritten, not extension 0.0, and the
/// advertised version stands. A non-zero report older than the table is
/// the one used, so a field past it is not read.
pub fn extension_version(advertised: u32, reported: u32) -> u32 {
    if reported == 0 { advertised } else { reported.min(advertised) }
}

pub const STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES: u32 = 0x1;
pub const STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES_2: u32 = 0xF;
pub const STRUCTURE_TYPE_GRAPH_DESC_2: u32 = 0xE;
pub const STRUCTURE_TYPE_GRAPH_PROPERTIES_2: u32 = 0x10;
pub const STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3: u32 = 0xD;

/// ze_graph_format_t. NATIVE is a blob the driver already compiled
/// (ELF or flatbuffers). NGRAPH_LITE is the serialized OpenVINO IR.
pub const GRAPH_FORMAT_NATIVE: u32 = 0x1;
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
/// The enum is not contiguous: NCHW follows ANY (0), CHW is 0x80, and NC
/// follows HW (0xC0), so NC is 0xC1. BLOCKED (0xC8) is a device tiling
/// the host buffers are not written as.
pub const GRAPH_ARGUMENT_LAYOUT_NCHW: u32 = 0x01;
pub const GRAPH_ARGUMENT_LAYOUT_CHW: u32 = 0x80;
pub const GRAPH_ARGUMENT_LAYOUT_NC: u32 = 0xC1;

/// ze_graph_init_stage_t.
pub const GRAPH_STAGE_COMMAND_LIST_INITIALIZE: u32 = 0x1;
pub const GRAPH_STAGE_INITIALIZE: u32 = 0x2;

pub const MAX_GRAPH_ARGUMENT_NAME: usize = 256;
pub const MAX_GRAPH_ARGUMENT_DIMENSIONS: usize = 5;
pub const MAX_GRAPH_TENSOR_NAMES: usize = 32;

/// ze_graph_compiler_version_info_t.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CompilerVersion {
    pub major: u16,
    pub minor: u16,
}

/// ze_device_graph_properties_t. A C compile of ze_graph_ext.h puts
/// graphExtensionVersion at 16 and compilerVersion at 20; the whole
/// struct is 32 bytes.
#[repr(C)]
pub struct DeviceGraphProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub graph_extension_version: u32,
    pub compiler_version: CompilerVersion,
    pub graph_formats_supported: u32,
    pub max_ov_opset_version_supported: u32,
}

/// ze_graph_version_info_t, the elf and runtime versions on the 1.6
/// device graph properties.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GraphVersionInfo {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

/// ze_device_graph_properties_2_t. Same prefix as
/// ze_device_graph_properties_t, then elfVersion at 32 and
/// runtimeVersion at 44. A C compile of the header sizes it at 56.
#[repr(C)]
pub struct DeviceGraphProperties2 {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub graph_extension_version: u32,
    pub compiler_version: CompilerVersion,
    pub graph_formats_supported: u32,
    pub max_ov_opset_version_supported: u32,
    pub elf_version: GraphVersionInfo,
    pub runtime_version: GraphVersionInfo,
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
zeroed_default!(DeviceGraphProperties, DeviceGraphProperties2, GraphProperties2, GraphArgumentProperties3);

// ---- The graph DDI table (ze_graph_dditable_ext_t) ---------------------------------

pub type PfnGraphCreate2 = unsafe extern "C" fn(Handle, Handle, *const GraphDesc2, *mut Handle) -> Status;
pub type PfnGraphCreate3 = unsafe extern "C" fn(Handle, Handle, *const GraphDesc2, *mut Handle, *mut Handle) -> Status;
pub type PfnGraphDestroy = unsafe extern "C" fn(Handle) -> Status;
pub type PfnGraphSetArgumentValue = unsafe extern "C" fn(Handle, u32, *const c_void) -> Status;
pub type PfnAppendGraph = unsafe extern "C" fn(Handle, Handle, Handle, u32, *mut Handle) -> Status;
pub type PfnAppendGraphExecute = unsafe extern "C" fn(Handle, Handle, Handle, Handle, u32, *mut Handle) -> Status;
pub type PfnDeviceGetGraphProperties = unsafe extern "C" fn(Handle, *mut DeviceGraphProperties) -> Status;
pub type PfnDeviceGetGraphProperties2 = unsafe extern "C" fn(Handle, *mut DeviceGraphProperties2) -> Status;
pub type PfnGraphGetProperties2 = unsafe extern "C" fn(Handle, *mut GraphProperties2) -> Status;
pub type PfnGraphGetArgumentProperties3 = unsafe extern "C" fn(Handle, u32, *mut GraphArgumentProperties3) -> Status;
pub type PfnGraphInitialize = unsafe extern "C" fn(Handle) -> Status;
pub type PfnBuildLogGetString2 = unsafe extern "C" fn(Handle, *mut u32, *mut c_char) -> Status;
pub type PfnBuildLogDestroy = unsafe extern "C" fn(Handle) -> Status;
/// ze_pfnGraphGetNativeBinary_ext_t. A null buffer asks for the size.
/// The caller owns the buffer filled on the second call.
pub type PfnGraphGetNativeBinary = unsafe extern "C" fn(Handle, *mut usize, *mut u8) -> Status;
/// ze_pfnGraphGetNativeBinary_ext_2_t, extension 1.7. The driver owns
/// the bytes the pointer names. They die with the graph.
pub type PfnGraphGetNativeBinary2 = unsafe extern "C" fn(Handle, *mut usize, *mut *const u8) -> Status;

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
    pub pfn_get_native_binary: Option<PfnGraphGetNativeBinary>,
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
    pub pfn_device_get_graph_properties2: Option<PfnDeviceGetGraphProperties2>,
    // version 1.7
    pub pfn_get_native_binary2: Option<PfnGraphGetNativeBinary2>,
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
    // The published header's CURRENT is 1.20, and a C compile of it is
    // 33 pointers. Versions 1.17, 1.18 and 1.19 add none, so a copy
    // that stops at 1.18 ends at pfnEvict, index 30 (31 pointers).
    // Version 1.20 appends the two below. This backend does not call
    // them. A driver that advertises 1.17 is not read past the fields
    // that version covers.
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

    pub fn device_get_graph_properties2(&self) -> Option<PfnDeviceGetGraphProperties2> {
        covered!(self, version(1, 6), pfn_device_get_graph_properties2)
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

    pub fn get_native_binary(&self) -> Option<PfnGraphGetNativeBinary> {
        covered!(self, version(1, 0), pfn_get_native_binary)
    }

    /// Extension 1.7. A driver that advertises an older version is not
    /// read here, even if the struct the test built has the field.
    pub fn get_native_binary2(&self) -> Option<PfnGraphGetNativeBinary2> {
        covered!(self, version(1, 7), pfn_get_native_binary2)
    }
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
        // ze_device_graph_properties_t, from a C compile of ze_graph_ext.h
        // (ZE_GRAPH_EXT_VERSION_CURRENT 1.20): size 32, graphExtensionVersion
        // at 16, compilerVersion at 20, graphFormatsSupported at 24,
        // maxOVOpsetVersionSupported at 28.
        assert_eq!(size_of::<DeviceGraphProperties>(), 32);
        assert_eq!(std::mem::offset_of!(DeviceGraphProperties, graph_extension_version), 16);
        assert_eq!(std::mem::offset_of!(DeviceGraphProperties, compiler_version), 20);
        assert_eq!(std::mem::offset_of!(DeviceGraphProperties, graph_formats_supported), 24);
        assert_eq!(std::mem::offset_of!(DeviceGraphProperties, max_ov_opset_version_supported), 28);
        // ze_device_graph_properties_2_t: the same prefix, elfVersion at 32,
        // runtimeVersion at 44, size 56.
        assert_eq!(size_of::<DeviceGraphProperties2>(), 56);
        assert_eq!(std::mem::offset_of!(DeviceGraphProperties2, graph_extension_version), 16);
        assert_eq!(std::mem::offset_of!(DeviceGraphProperties2, compiler_version), 20);
        assert_eq!(std::mem::offset_of!(DeviceGraphProperties2, graph_formats_supported), 24);
        assert_eq!(std::mem::offset_of!(DeviceGraphProperties2, max_ov_opset_version_supported), 28);
        assert_eq!(std::mem::offset_of!(DeviceGraphProperties2, elf_version), 32);
        assert_eq!(std::mem::offset_of!(DeviceGraphProperties2, runtime_version), 44);
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
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_device_get_graph_properties2), 19 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_get_properties2), 21 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_graph_initialize), 22 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_create3), 25 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_build_log_get_string2), 27 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_build_log_destroy), 28 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_evict), 30 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_get_argument_properties4), 31 * p);
        assert_eq!(std::mem::offset_of!(GraphDdi, pfn_get_argument_names), 32 * p);
        // ze_graph_dditable_ext_t in the published header
        // (ZE_GRAPH_EXT_VERSION_CURRENT 1.20) is 33 pointers. Through
        // 1.18 the table ends at pfnEvict, 31 pointers.
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
        // 1.5's table ends before pfnDeviceGetGraphProperties2. The slot
        // is not read.
        let bytes = vec![0xABu8; std::mem::offset_of!(GraphDdi, pfn_device_get_graph_properties2)];
        let ext = unsafe { GraphExt::new(bytes.as_ptr() as *const GraphDdi, version(1, 5)) };
        assert!(ext.device_get_graph_properties2().is_none());
    }

    /// A device properties version of 0 is the field left unwritten.
    /// The version the driver advertised for ZE_extension_graph stands,
    /// which on the Arrow Lake machine is 1.17.
    #[test]
    fn a_zero_device_graph_version_keeps_the_advertised_one() {
        assert_eq!(extension_version(version(1, 17), 0), version(1, 17));
        assert_eq!(extension_version(version(1, 17), version(1, 8)), version(1, 8));
        assert_eq!(extension_version(version(1, 17), version(1, 20)), version(1, 17));
        assert_eq!(version(1, 17), 0x0001_0011);
    }
}
