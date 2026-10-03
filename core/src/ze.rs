//! The part of the Level Zero C API that both the Intel GPU backend and
//! the NPU backend call, from level_zero/ze_api.h and loader/ze_loader.h.
//! zeInitDrivers needs a 1.10 or later loader. The NPU graph extension is
//! not here: it stays in the npu backend.

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

pub const COMMAND_QUEUE_FLAG_IN_ORDER: u32 = 2;
pub const COMMAND_QUEUE_MODE_ASYNCHRONOUS: u32 = 2;
pub const COMMAND_QUEUE_PRIORITY_NORMAL: u32 = 0;

pub const MAX_EXTENSION_NAME: usize = 256;

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
    ComponentVersion
);

/// The loader and the entry points both backends call. A backend that
/// needs more symbols loads them from the same library through `symbol`.
pub struct Api {
    lib: libloading::Library,
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
    /// zeInitDrivers predates Level Zero 1.10 and is an error. `who` is
    /// the backend name on that error.
    pub fn load(who: &str) -> Result<Option<Api>, String> {
        let Ok(lib) = (unsafe { libloading::Library::new(LOADER) }) else {
            return Ok(None);
        };
        // A closure is one type. Each symbol is a different function
        // pointer, so the lookup is expanded at the field.
        macro_rules! sym {
            ($name:literal) => {
                symbol(&lib, who, $name)?
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
            loader_get_versions: optional(&lib, "zelLoaderGetVersions"),
            lib,
        }))
    }

    /// A further symbol from the same loader. Missing is an error naming
    /// `who` and the symbol.
    pub fn symbol<T: Copy>(&self, who: &str, name: &str) -> Result<T, String> {
        symbol(&self.lib, who, name)
    }

    /// A symbol the loader may omit. Sysman is optional this way.
    pub fn optional_symbol<T: Copy>(&self, name: &str) -> Option<T> {
        optional(&self.lib, name)
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

fn symbol<T: Copy>(lib: &libloading::Library, who: &str, name: &str) -> Result<T, String> {
    let bytes = format!("{name}\0");
    let s = unsafe { lib.get::<T>(bytes.as_bytes()) }
        .map_err(|_| format!("{who}: {LOADER} has no {name}; Level Zero 1.10 or later is needed"))?;
    Ok(*s)
}

fn optional<T: Copy>(lib: &libloading::Library, name: &str) -> Option<T> {
    let bytes = format!("{name}\0");
    unsafe { lib.get::<T>(bytes.as_bytes()) }.ok().map(|s| *s)
}

pub fn check(who: &str, what: &str, rc: Status) -> Result<(), String> {
    if rc != 0 { Err(format!("{who}: {what} failed with 0x{rc:08x}")) } else { Ok(()) }
}

/// The two-call pattern: ask for the count, then fill that many.
pub fn list(who: &str, what: &str, mut f: impl FnMut(*mut u32, *mut Handle) -> Status) -> Result<Vec<Handle>, String> {
    let mut n = 0u32;
    check(who, what, f(&mut n, std::ptr::null_mut()))?;
    let mut out = vec![std::ptr::null_mut(); n as usize];
    check(who, what, f(&mut n, out.as_mut_ptr()))?;
    out.truncate(n as usize);
    Ok(out)
}

pub fn string(b: &[c_char]) -> String {
    crate::backend::cstr(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sizes and offsets a C compiler gives these structs on x86_64
    /// and aarch64 with the 1.28 ze_api.h.
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
        assert_eq!(std::mem::offset_of!(DeviceProperties, uuid), 96);
        assert_eq!(std::mem::offset_of!(DeviceProperties, name), 112);
        assert_eq!(size_of::<DeviceMemoryProperties>(), 296);
    }
}
