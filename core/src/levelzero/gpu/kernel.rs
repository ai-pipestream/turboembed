//! Kernel objects and their launches.

use std::ffi::c_void;

use super::*;
use crate::levelzero::ze::{self, ComputeApi, Handle};

/// A kernel object: its arguments are its own state, so each session
/// creates its own. Arguments are captured when a launch is appended.
pub(crate) struct Kernel {
    pub(super) api: &'static ComputeApi,
    pub handle: Handle,
}

unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { (self.api.kernel_destroy)(self.handle) };
    }
}

/// A kernel argument.
#[derive(Clone, Copy)]
pub(crate) enum Arg {
    /// A device address.
    Ptr(u64),
    I32(i32),
    F32(f32),
    U64(u64),
    /// Local memory of this many bytes.
    Local(usize),
}

impl Kernel {
    /// The largest work-group the kernel can run with, which depends on the
    /// register file the driver's compiler gave it; None from a driver that
    /// does not say.
    pub fn max_group_size(&self) -> Res<Option<u32>> {
        let mut ext = ze::KernelMaxGroupSizeExt {
            stype: ze::STRUCTURE_TYPE_KERNEL_MAX_GROUP_SIZE_EXT_PROPERTIES,
            p_next: std::ptr::null_mut(),
            max_group_size: 0,
        };
        let mut p = ze::KernelProperties {
            stype: ze::STRUCTURE_TYPE_KERNEL_PROPERTIES,
            p_next: &mut ext as *mut ze::KernelMaxGroupSizeExt as *mut c_void,
            ..Default::default()
        };
        ze("zeKernelGetProperties", unsafe { (self.api.kernel_get_properties)(self.handle, &mut p) })?;
        Ok((ext.max_group_size > 0).then_some(ext.max_group_size))
    }

    /// The local memory the driver gives the kernel itself, beyond what its
    /// arguments ask for.
    pub fn local_bytes(&self) -> Res<u32> {
        let mut p = ze::KernelProperties { stype: ze::STRUCTURE_TYPE_KERNEL_PROPERTIES, ..Default::default() };
        ze("zeKernelGetProperties", unsafe { (self.api.kernel_get_properties)(self.handle, &mut p) })?;
        Ok(p.local_mem_size)
    }

    /// Sets the arguments and appends a launch of `groups` work-groups to
    /// the queue, which the caller holds. Allocates nothing.
    pub fn launch(&self, c: &Context, q: &mut Queue, name: &str, args: &[Arg], groups: [u32; 3]) -> Res<()> {
        for (i, a) in args.iter().enumerate() {
            let (n, p) = match a {
                Arg::Ptr(v) | Arg::U64(v) => (8, v as *const u64 as *const c_void),
                Arg::I32(v) => (4, v as *const i32 as *const c_void),
                Arg::F32(v) => (4, v as *const f32 as *const c_void),
                Arg::Local(n) => (*n, std::ptr::null()),
            };
            let rc = unsafe { (self.api.kernel_set_argument_value)(self.handle, i as u32, n, p) };
            if rc != 0 {
                return ze(name, rc);
            }
        }
        let g = ze::GroupCount { x: groups[0], y: groups[1], z: groups[2] };
        let e = c.next_event(q)?;
        let (waits, mut wait) = (u32::from(!q.pending_wait.is_null()), q.pending_wait);
        let rc = unsafe { (self.api.command_list_append_launch_kernel)(q.list, self.handle, &g, e, waits, &mut wait) };
        if rc == 0 {
            q.pending_wait = std::ptr::null_mut();
        }
        c.appended(q, name, rc)
    }
}
