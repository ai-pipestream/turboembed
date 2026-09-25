//! The part of the Level Zero C API this backend calls, from
//! level_zero/ze_api.h, zes_api.h and loader/ze_loader.h. zeInitDrivers
//! needs a 1.10 or later loader.

use std::ffi::{c_char, c_void};

pub type Handle = *mut c_void;
pub type Status = i32;

pub const RESULT_ERROR_UNINITIALIZED: Status = 0x7800_0001;
pub const RESULT_ERROR_OUT_OF_HOST_MEMORY: Status = 0x7000_0002;
pub const RESULT_ERROR_OUT_OF_DEVICE_MEMORY: Status = 0x7000_0003;

pub const STRUCTURE_TYPE_DRIVER_PROPERTIES: u32 = 0x1;
pub const STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 0x3;
pub const STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES: u32 = 0x7;
pub const STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC: u32 = 0x0002_0021;
pub const ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES: u32 = 0x1;
pub const ZES_STRUCTURE_TYPE_MEM_PROPERTIES: u32 = 0xb;
pub const ZES_STRUCTURE_TYPE_MEM_STATE: u32 = 0x1e;
pub const STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES: u32 = 0x4;
pub const STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES: u32 = 0x5;
pub const STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES: u32 = 0x17;
pub const STRUCTURE_TYPE_KERNEL_PROPERTIES: u32 = 0x1e;
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
pub const STRUCTURE_TYPE_COMMAND_QUEUE_DESC: u32 = 0xe;
pub const STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC: u32 = 0x15;
pub const STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC: u32 = 0x16;
pub const STRUCTURE_TYPE_MODULE_DESC: u32 = 0x1b;
pub const STRUCTURE_TYPE_KERNEL_DESC: u32 = 0x1d;
pub const STRUCTURE_TYPE_CONTEXT_DESC: u32 = 0xd;
pub const MODULE_FORMAT_IL_SPIRV: u32 = 0;
pub const COMMAND_QUEUE_FLAG_IN_ORDER: u32 = 2;
pub const COMMAND_QUEUE_MODE_ASYNCHRONOUS: u32 = 2;
pub const COMMAND_QUEUE_PRIORITY_NORMAL: u32 = 0;
pub const INIT_DRIVER_TYPE_FLAG_GPU: u32 = 1;
pub const DEVICE_TYPE_GPU: u32 = 1;
pub const DEVICE_PROPERTY_FLAG_INTEGRATED: u32 = 1;
pub const ZES_MEM_LOC_DEVICE: u32 = 1;

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
pub struct DriverProperties {
    pub stype: u32,
    pub p_next: *mut c_void,
    pub uuid: [u8; 16],
    pub driver_version: u32,
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

#[repr(C)]
pub struct ComponentVersion {
    pub component_name: [c_char; 64],
    pub spec_version: u32,
    pub major: i32,
    pub minor: i32,
    pub patch: i32,
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
    DriverProperties,
    DeviceProperties,
    DeviceMemoryProperties,
    SysmanDeviceProperties,
    MemProperties,
    MemState,
    ComponentVersion
);

pub struct Api {
    // Keeps the function pointers below valid.
    _lib: libloading::Library,
    pub init_drivers: unsafe extern "C" fn(*mut u32, *mut Handle, *mut InitDriverTypeDesc) -> Status,
    pub driver_get_properties: unsafe extern "C" fn(Handle, *mut DriverProperties) -> Status,
    pub device_get: unsafe extern "C" fn(Handle, *mut u32, *mut Handle) -> Status,
    pub device_get_properties: unsafe extern "C" fn(Handle, *mut DeviceProperties) -> Status,
    pub device_get_memory_properties: unsafe extern "C" fn(Handle, *mut u32, *mut DeviceMemoryProperties) -> Status,
    pub device_get_compute_properties: unsafe extern "C" fn(Handle, *mut DeviceComputeProperties) -> Status,
    pub device_get_module_properties: unsafe extern "C" fn(Handle, *mut DeviceModuleProperties) -> Status,
    pub compute: ComputeApi,
    loader_get_versions: Option<unsafe extern "C" fn(*mut usize, *mut ComponentVersion) -> Status>,
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
        let need = |name: &str| format!("levelzero: {LOADER} has no {name}; Level Zero 1.10 or later is needed");
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
            init_drivers: sym!("zeInitDrivers"),
            driver_get_properties: sym!("zeDriverGetProperties"),
            device_get: sym!("zeDeviceGet"),
            device_get_properties: sym!("zeDeviceGetProperties"),
            device_get_memory_properties: sym!("zeDeviceGetMemoryProperties"),
            device_get_compute_properties: sym!("zeDeviceGetComputeProperties"),
            device_get_module_properties: sym!("zeDeviceGetModuleProperties"),
            compute: ComputeApi {
                context_create: sym!("zeContextCreate"),
                context_destroy: sym!("zeContextDestroy"),
                mem_alloc_device: sym!("zeMemAllocDevice"),
                mem_alloc_host: sym!("zeMemAllocHost"),
                mem_alloc_shared: sym!("zeMemAllocShared"),
                mem_free: sym!("zeMemFree"),
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
                command_list_create_immediate: sym!("zeCommandListCreateImmediate"),
                command_list_destroy: sym!("zeCommandListDestroy"),
                command_list_append_launch_kernel: sym!("zeCommandListAppendLaunchKernel"),
                command_list_append_memory_copy: sym!("zeCommandListAppendMemoryCopy"),
                command_list_host_synchronize: sym!("zeCommandListHostSynchronize"),
                event_pool_create: sym!("zeEventPoolCreate"),
                event_pool_destroy: sym!("zeEventPoolDestroy"),
                event_create: sym!("zeEventCreate"),
                event_destroy: sym!("zeEventDestroy"),
                event_host_synchronize: sym!("zeEventHostSynchronize"),
                event_host_reset: sym!("zeEventHostReset"),
                event_query_kernel_timestamp: sym!("zeEventQueryKernelTimestamp"),
            },
            loader_get_versions: opt!("zelLoaderGetVersions"),
            sysman,
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
    if rc != 0 { Err(format!("levelzero: {what} failed with 0x{rc:08x}")) } else { Ok(()) }
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
