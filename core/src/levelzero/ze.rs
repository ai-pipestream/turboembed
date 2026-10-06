//! The GPU-only part of the Level Zero C API this backend calls, from
//! level_zero/ze_api.h and zes_api.h: modules, kernels, events and
//! sysman. The loader and the structs both backends share live in
//! crate::ze.

use std::ffi::{c_char, c_void};

pub use crate::ze::{
    Api as CoreApi, COMMAND_QUEUE_FLAG_IN_ORDER, COMMAND_QUEUE_MODE_ASYNCHRONOUS, COMMAND_QUEUE_PRIORITY_NORMAL,
    CommandQueueDesc, ContextDesc, DeviceMemoryProperties, DeviceProperties, DriverProperties, Handle,
    HostMemAllocDesc, InitDriverTypeDesc, RESULT_ERROR_OUT_OF_DEVICE_MEMORY, RESULT_ERROR_OUT_OF_HOST_MEMORY,
    RESULT_ERROR_UNINITIALIZED, STRUCTURE_TYPE_COMMAND_QUEUE_DESC, STRUCTURE_TYPE_CONTEXT_DESC,
    STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES, STRUCTURE_TYPE_DEVICE_PROPERTIES, STRUCTURE_TYPE_DRIVER_PROPERTIES,
    STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC, Status, string,
};

pub const ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 0x1;
pub const ZES_STRUCTURE_TYPE_MEM_PROPERTIES: u32 = 0xb;
pub const ZES_STRUCTURE_TYPE_MEM_STATE: u32 = 0x1e;
pub const STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES: u32 = 0x4;
pub const STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES: u32 = 0x5;
pub const STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT: u32 = 0x1_000f;
pub const STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES: u32 = 0x17;
pub const STRUCTURE_TYPE_KERNEL_PROPERTIES: u32 = 0x1e;
pub const STRUCTURE_TYPE_KERNEL_MAX_GROUP_SIZE_EXT_PROPERTIES: u32 = 0x1_0013;
pub const STRUCTURE_TYPE_EVENT_POOL_DESC: u32 = 0x10;
pub const STRUCTURE_TYPE_EVENT_DESC: u32 = 0x11;
pub const EVENT_POOL_FLAG_HOST_VISIBLE: u32 = 1;
pub const EVENT_POOL_FLAG_KERNEL_TIMESTAMP: u32 = 4;
pub const EVENT_SCOPE_FLAG_HOST: u32 = 4;
pub const DEVICE_MODULE_FLAG_FP64: u32 = 2;
pub const MEMORY_TYPE_UNKNOWN: u32 = 0;
pub const MEMORY_TYPE_HOST: u32 = 1;
pub const MEMORY_TYPE_DEVICE: u32 = 2;
pub const MEMORY_TYPE_SHARED: u32 = 3;
pub const STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC: u32 = 0x15;
pub const STRUCTURE_TYPE_MODULE_DESC: u32 = 0x1b;
pub const STRUCTURE_TYPE_KERNEL_DESC: u32 = 0x1d;
pub const MODULE_FORMAT_IL_SPIRV: u32 = 0;
pub const INIT_DRIVER_TYPE_FLAG_GPU: u32 = 1;
pub const DEVICE_TYPE_GPU: u32 = 1;
pub const DEVICE_PROPERTY_FLAG_INTEGRATED: u32 = 1;
pub const ZES_MEM_LOC_DEVICE: u32 = 1;

#[repr(C)]
pub struct ModuleDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub format: u32,
    pub input_size: usize,
    pub input: *const u8,
    pub build_flags: *const c_char,
    pub constants: *const c_void,
}

#[repr(C)]
pub struct KernelDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub name: *const c_char,
}

#[repr(C)]
pub struct DeviceMemAllocDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub ordinal: u32,
}

#[repr(C)]
pub struct GroupCount {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[repr(C)]
pub struct DeviceComputeProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub max_total_group_size: u32,
    pub max_group_size: [u32; 3],
    pub max_group_count: [u32; 3],
    pub max_shared_local_memory: u32,
    pub num_sub_group_sizes: u32,
    pub sub_group_sizes: [u32; 8],
}

#[repr(C)]
pub struct DeviceModuleProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub spirv_version_supported: u32,
    pub flags: u32,
    pub fp16_flags: u32,
    pub fp32_flags: u32,
    pub fp64_flags: u32,
    pub max_argument_size: u32,
    pub printf_buffer_size: u32,
    pub native_kernel_supported: [u8; 16],
}

/// Chained to DeviceProperties (ZE_extension_device_ip_version): the
/// device's IP version, on Intel's GPUs its architecture in bits 22 and up
/// and its release in bits 14 to 21.
#[repr(C)]
pub struct DeviceIpVersion {
    pub stype: u32,
    pub p_next: *const c_void,
    pub ip_version: u32,
}

#[repr(C)]
pub struct MemoryAllocationProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub kind: u32,
    pub id: u64,
    pub page_size: u64,
}

#[repr(C)]
pub struct EventPoolDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub count: u32,
}

#[repr(C)]
pub struct EventDesc {
    pub stype: u32,
    pub p_next: *const c_void,
    pub index: u32,
    pub signal: u32,
    pub wait: u32,
}

/// Chained to KernelProperties: the largest group the kernel can run with,
/// which depends on the register file its build gave it.
#[repr(C)]
pub struct KernelMaxGroupSizeExt {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub max_group_size: u32,
}

#[repr(C)]
pub struct KernelProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub num_kernel_args: u32,
    pub required_group_size: [u32; 3],
    pub required_num_sub_groups: u32,
    pub required_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub max_num_sub_groups: u32,
    pub local_mem_size: u32,
    pub private_mem_size: u32,
    pub spill_mem_size: u32,
    pub uuid: [u8; 32],
}

#[repr(C)]
pub struct SysmanDeviceProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub core: DeviceProperties,
    pub num_subdevices: u32,
    pub serial_number: [c_char; 64],
    pub board_number: [c_char; 64],
    pub brand_name: [c_char; 64],
    pub model_name: [c_char; 64],
    pub vendor_name: [c_char; 64],
    pub driver_version: [c_char; 64],
}

#[repr(C)]
pub struct MemProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub kind: u32,
    pub on_subdevice: u8,
    pub subdevice_id: u32,
    pub location: u32,
    pub physical_size: u64,
    pub bus_width: i32,
    pub num_channels: i32,
}

#[repr(C)]
pub struct MemState {
    pub stype: u32,
    pub p_next: *const c_void,
    pub health: u32,
    pub free: u64,
    pub size: u64,
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
    KernelProperties,
    DeviceComputeProperties,
    DeviceModuleProperties,
    MemoryAllocationProperties,
    SysmanDeviceProperties,
    MemProperties,
    MemState
);

pub struct Api {
    pub core: CoreApi,
    pub device_get_compute_properties: unsafe extern "C" fn(Handle, *mut DeviceComputeProperties) -> Status,
    pub device_get_module_properties: unsafe extern "C" fn(Handle, *mut DeviceModuleProperties) -> Status,
    pub compute: ComputeApi,
    pub sysman: Option<SysmanApi>,
}

/// Contexts, memory, modules, kernels and immediate command lists.
pub struct ComputeApi {
    pub context_create: unsafe extern "C" fn(Handle, *const ContextDesc, *mut Handle) -> Status,
    pub context_destroy: unsafe extern "C" fn(Handle) -> Status,
    pub mem_alloc_device:
        unsafe extern "C" fn(Handle, *const DeviceMemAllocDesc, usize, usize, Handle, *mut *mut c_void) -> Status,
    pub mem_alloc_host: unsafe extern "C" fn(Handle, *const HostMemAllocDesc, usize, usize, *mut *mut c_void) -> Status,
    pub mem_alloc_shared: unsafe extern "C" fn(
        Handle,
        *const DeviceMemAllocDesc,
        *const HostMemAllocDesc,
        usize,
        usize,
        Handle,
        *mut *mut c_void,
    ) -> Status,
    pub mem_free: unsafe extern "C" fn(Handle, *mut c_void) -> Status,
    pub mem_get_alloc_properties:
        unsafe extern "C" fn(Handle, *const c_void, *mut MemoryAllocationProperties, *mut Handle) -> Status,
    pub module_create: unsafe extern "C" fn(Handle, Handle, *const ModuleDesc, *mut Handle, *mut Handle) -> Status,
    pub module_destroy: unsafe extern "C" fn(Handle) -> Status,
    pub module_build_log_get_string: unsafe extern "C" fn(Handle, *mut usize, *mut c_char) -> Status,
    pub module_build_log_destroy: unsafe extern "C" fn(Handle) -> Status,
    pub kernel_create: unsafe extern "C" fn(Handle, *const KernelDesc, *mut Handle) -> Status,
    pub kernel_destroy: unsafe extern "C" fn(Handle) -> Status,
    pub kernel_get_properties: unsafe extern "C" fn(Handle, *mut KernelProperties) -> Status,
    pub kernel_set_group_size: unsafe extern "C" fn(Handle, u32, u32, u32) -> Status,
    pub kernel_set_argument_value: unsafe extern "C" fn(Handle, u32, usize, *const c_void) -> Status,
    pub command_list_create_immediate:
        unsafe extern "C" fn(Handle, Handle, *const CommandQueueDesc, *mut Handle) -> Status,
    pub command_list_destroy: unsafe extern "C" fn(Handle) -> Status,
    pub command_list_append_launch_kernel:
        unsafe extern "C" fn(Handle, Handle, *const GroupCount, Handle, u32, *mut Handle) -> Status,
    pub command_list_append_memory_copy:
        unsafe extern "C" fn(Handle, *mut c_void, *const c_void, usize, Handle, u32, *mut Handle) -> Status,
    pub command_list_host_synchronize: unsafe extern "C" fn(Handle, u64) -> Status,
    pub event_pool_create: unsafe extern "C" fn(Handle, *const EventPoolDesc, u32, *mut Handle, *mut Handle) -> Status,
    pub event_pool_destroy: unsafe extern "C" fn(Handle) -> Status,
    pub event_create: unsafe extern "C" fn(Handle, *const EventDesc, *mut Handle) -> Status,
    pub event_destroy: unsafe extern "C" fn(Handle) -> Status,
    pub event_host_synchronize: unsafe extern "C" fn(Handle, u64) -> Status,
    pub event_host_reset: unsafe extern "C" fn(Handle) -> Status,
    pub event_query_kernel_timestamp: unsafe extern "C" fn(Handle, *mut [u64; 4]) -> Status,
}

pub struct SysmanApi {
    pub init: unsafe extern "C" fn(u32) -> Status,
    pub driver_get: unsafe extern "C" fn(*mut u32, *mut Handle) -> Status,
    pub device_get: unsafe extern "C" fn(Handle, *mut u32, *mut Handle) -> Status,
    pub device_get_properties: unsafe extern "C" fn(Handle, *mut SysmanDeviceProperties) -> Status,
    pub device_enum_memory_modules: unsafe extern "C" fn(Handle, *mut u32, *mut Handle) -> Status,
    pub memory_get_properties: unsafe extern "C" fn(Handle, *mut MemProperties) -> Status,
    pub memory_get_state: unsafe extern "C" fn(Handle, *mut MemState) -> Status,
}

impl Api {
    /// None when the loader is not installed. A loader without
    /// zeInitDrivers predates Level Zero 1.10 and is an error.
    pub fn load() -> Result<Option<Api>, String> {
        let Some(core) = CoreApi::load("levelzero")? else {
            return Ok(None);
        };
        // A closure is one type. Each symbol is a different function
        // pointer, so the lookup is expanded at the field.
        macro_rules! sym {
            ($name:literal) => {
                core.symbol("levelzero", $name)?
            };
        }
        macro_rules! opt {
            ($name:literal) => {
                core.optional_symbol($name)
            };
        }
        let sysman = (|| {
            Some(SysmanApi {
                init: opt!("zesInit")?,
                driver_get: opt!("zesDriverGet")?,
                device_get: opt!("zesDeviceGet")?,
                device_get_properties: opt!("zesDeviceGetProperties")?,
                device_enum_memory_modules: opt!("zesDeviceEnumMemoryModules")?,
                memory_get_properties: opt!("zesMemoryGetProperties")?,
                memory_get_state: opt!("zesMemoryGetState")?,
            })
        })();
        Ok(Some(Api {
            device_get_compute_properties: sym!("zeDeviceGetComputeProperties"),
            device_get_module_properties: sym!("zeDeviceGetModuleProperties"),
            compute: ComputeApi {
                context_create: core.context_create,
                context_destroy: core.context_destroy,
                mem_alloc_device: sym!("zeMemAllocDevice"),
                mem_alloc_host: core.mem_alloc_host,
                mem_alloc_shared: sym!("zeMemAllocShared"),
                mem_free: core.mem_free,
                mem_get_alloc_properties: sym!("zeMemGetAllocProperties"),
                module_create: sym!("zeModuleCreate"),
                module_destroy: sym!("zeModuleDestroy"),
                module_build_log_get_string: sym!("zeModuleBuildLogGetString"),
                module_build_log_destroy: sym!("zeModuleBuildLogDestroy"),
                kernel_create: sym!("zeKernelCreate"),
                kernel_destroy: sym!("zeKernelDestroy"),
                kernel_get_properties: sym!("zeKernelGetProperties"),
                kernel_set_group_size: sym!("zeKernelSetGroupSize"),
                kernel_set_argument_value: sym!("zeKernelSetArgumentValue"),
                command_list_create_immediate: core.command_list_create_immediate,
                command_list_destroy: core.command_list_destroy,
                command_list_append_launch_kernel: sym!("zeCommandListAppendLaunchKernel"),
                command_list_append_memory_copy: sym!("zeCommandListAppendMemoryCopy"),
                command_list_host_synchronize: core.command_list_host_synchronize,
                event_pool_create: sym!("zeEventPoolCreate"),
                event_pool_destroy: sym!("zeEventPoolDestroy"),
                event_create: sym!("zeEventCreate"),
                event_destroy: sym!("zeEventDestroy"),
                event_host_synchronize: sym!("zeEventHostSynchronize"),
                event_host_reset: sym!("zeEventHostReset"),
                event_query_kernel_timestamp: sym!("zeEventQueryKernelTimestamp"),
            },
            sysman,
            core,
        }))
    }

    /// The loader's own version, "" when it does not say.
    pub fn loader_version(&self) -> String {
        self.core.loader_version()
    }
}

pub fn check(what: &str, rc: Status) -> Result<(), String> {
    crate::ze::check("levelzero", what, rc)
}

/// The two-call pattern: ask for the count, then fill that many.
pub fn list(what: &str, f: impl FnMut(*mut u32, *mut Handle) -> Status) -> Result<Vec<Handle>, String> {
    crate::ze::list("levelzero", what, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The sizes and offsets the C compiler gives these structs on
    // x86_64 and aarch64 with the 1.28 headers.
    #[test]
    fn layouts_match_the_c_headers() {
        assert_eq!(size_of::<InitDriverTypeDesc>(), 24);
        assert_eq!(size_of::<DriverProperties>(), 40);
        assert_eq!(size_of::<DeviceProperties>(), 368);
        assert_eq!(std::mem::offset_of!(DeviceProperties, uuid), 96);
        assert_eq!(std::mem::offset_of!(DeviceProperties, name), 112);
        assert_eq!(size_of::<DeviceMemoryProperties>(), 296);
        assert_eq!(size_of::<SysmanDeviceProperties>(), 776);
        assert_eq!(size_of::<MemProperties>(), 48);
        assert_eq!(size_of::<MemState>(), 40);
        assert_eq!(size_of::<ContextDesc>(), 24);
        assert_eq!(size_of::<ModuleDesc>(), 56);
        assert_eq!(std::mem::offset_of!(ModuleDesc, input_size), 24);
        assert_eq!(size_of::<KernelDesc>(), 32);
        assert_eq!(size_of::<DeviceMemAllocDesc>(), 24);
        assert_eq!(size_of::<HostMemAllocDesc>(), 24);
        assert_eq!(size_of::<CommandQueueDesc>(), 40);
        assert_eq!(std::mem::offset_of!(CommandQueueDesc, mode), 28);
        assert_eq!(size_of::<GroupCount>(), 12);
        assert_eq!(size_of::<DeviceComputeProperties>(), 88);
        assert_eq!(std::mem::offset_of!(DeviceComputeProperties, max_shared_local_memory), 44);
        assert_eq!(size_of::<DeviceModuleProperties>(), 64);
        assert_eq!(std::mem::offset_of!(DeviceModuleProperties, flags), 20);
        assert_eq!(size_of::<MemoryAllocationProperties>(), 40);
        assert_eq!(size_of::<KernelProperties>(), 96);
        assert_eq!(size_of::<EventPoolDesc>(), 24);
        assert_eq!(size_of::<EventDesc>(), 32);
        assert_eq!(std::mem::offset_of!(KernelProperties, local_mem_size), 48);
    }
}
