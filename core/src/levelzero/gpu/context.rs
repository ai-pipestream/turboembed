//! A context's module, kernels, memory and in-order queue.

use std::ffi::{CString, c_void};
use std::sync::atomic::Ordering;

use super::*;
use crate::levelzero::ze::{self, Handle};
use crate::status::RUNTIME;
use crate::turbo_text;

impl Context {
    /// oneDNN for the linear layers, opened on this context the first
    /// time; None where it cannot open, which the log says once.
    #[cfg(feature = "levelzero-onednn")]
    pub fn dnnl(&self) -> Option<&crate::levelzero::onednn::Dnnl> {
        self.dnnl
            .get_or_init(|| match crate::levelzero::onednn::Dnnl::open(self.device, self.handle) {
                Ok(d) => {
                    let v = crate::levelzero::onednn::Dnnl::version();
                    self.say(LOG_DEBUG, &format!("levelzero device {}: {v} can run the linear layers", self.ordinal));
                    Some(d)
                }
                Err(e) => {
                    self.say(
                        LOG_WARNING,
                        &format!(
                            "levelzero device {}: oneDNN not used, the backend's own kernels run the linear layers: {}",
                            self.ordinal, e.message
                        ),
                    );
                    None
                }
            })
            .as_ref()
    }

    pub fn say(&self, level: u32, message: &str) {
        if let Some(f) = self.log {
            let t = turbo_text { ptr: message.as_ptr() as *const _, len: message.len() as u64 };
            unsafe { f(self.log_user_data, level, t) };
        }
    }

    /// The encoder's module, built for this device the first time.
    pub fn module(&self) -> Res<Handle> {
        let mut m = self.module.lock().unwrap_or_else(|p| p.into_inner());
        let built = m.get_or_insert_with(|| self.build().map(Shared));
        built.as_ref().map(|s| s.0).map_err(|e| fail(RUNTIME, e.clone()))
    }

    /// The driver's compiler picks each kernel's register file
    /// (-ze-intel-enable-auto-large-GRF-mode): the XMX kernels with 32-token
    /// tiles and the attention need the large one and spill without it. A
    /// driver that refuses the flag builds without it.
    pub(super) fn build(&self) -> Result<Handle, String> {
        match self.build_with(c"-ze-intel-enable-auto-large-GRF-mode") {
            Ok(h) => Ok(h),
            Err(e) => {
                self.say(LOG_WARNING, &format!("{e}; building without the flag"));
                self.build_with(c"")
            }
        }
    }

    pub(super) fn build_with(&self, flags: &std::ffi::CStr) -> Result<Handle, String> {
        let desc = ze::ModuleDesc {
            stype: ze::STRUCTURE_TYPE_MODULE_DESC,
            p_next: std::ptr::null(),
            format: ze::MODULE_FORMAT_IL_SPIRV,
            input_size: SPIRV.len(),
            input: SPIRV.as_ptr(),
            build_flags: flags.as_ptr(),
            constants: std::ptr::null(),
        };
        let (mut handle, mut log) = (std::ptr::null_mut(), std::ptr::null_mut());
        let rc = unsafe { (self.api.module_create)(self.handle, self.device, &desc, &mut handle, &mut log) };
        let text = self.build_log(log);
        if rc != 0 {
            return Err(format!("levelzero: building the encoder's kernels failed with 0x{rc:08x}: {text}"));
        }
        self.say(LOG_DEBUG, &format!("levelzero device {}: the encoder's kernels are built", self.ordinal));
        Ok(handle)
    }

    /// The build log's text, released after reading. Empty when there is
    /// none.
    pub(super) fn build_log(&self, log: Handle) -> String {
        if log.is_null() {
            return String::new();
        }
        let mut n = 0usize;
        let mut text = String::new();
        if unsafe { (self.api.module_build_log_get_string)(log, &mut n, std::ptr::null_mut()) } == 0 && n > 0 {
            let mut buf = vec![0 as std::ffi::c_char; n];
            if unsafe { (self.api.module_build_log_get_string)(log, &mut n, buf.as_mut_ptr()) } == 0 {
                text = crate::backend::cstr(&buf);
            }
        }
        unsafe { (self.api.module_build_log_destroy)(log) };
        text
    }

    /// A kernel from the encoder's module, with its work-group size set.
    pub fn kernel(&self, name: &str, group: [u32; 3]) -> Res<Kernel> {
        let module = self.module()?;
        let cname = CString::new(name).map_err(|_| fail(RUNTIME, "a kernel name holds a NUL"))?;
        let desc = ze::KernelDesc {
            stype: ze::STRUCTURE_TYPE_KERNEL_DESC,
            p_next: std::ptr::null(),
            flags: 0,
            name: cname.as_ptr(),
        };
        let mut handle = std::ptr::null_mut();
        ze(&format!("zeKernelCreate({name})"), unsafe { (self.api.kernel_create)(module, &desc, &mut handle) })?;
        let k = Kernel { api: self.api, handle };
        ze(&format!("zeKernelSetGroupSize({name})"), unsafe {
            (self.api.kernel_set_group_size)(handle, group[0], group[1], group[2])
        })?;
        if profiling() {
            let mut p = ze::KernelProperties { stype: ze::STRUCTURE_TYPE_KERNEL_PROPERTIES, ..Default::default() };
            if unsafe { (self.api.kernel_get_properties)(handle, &mut p) } == 0 {
                self.say(
                    LOG_DEBUG,
                    &format!(
                        "levelzero kernel {name}: group {}x{}x{}, spill {} B, private {} B, local {} B, group at most {:?}",
                        group[0],
                        group[1],
                        group[2],
                        p.spill_mem_size,
                        p.private_mem_size,
                        p.local_mem_size,
                        k.max_group_size().ok().flatten()
                    ),
                );
            }
        }
        Ok(k)
    }

    /// `bytes` of device memory, counted.
    pub fn alloc_device(&self, bytes: usize) -> Res<*mut c_void> {
        let desc = ze::DeviceMemAllocDesc {
            stype: ze::STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            p_next: std::ptr::null(),
            flags: 0,
            ordinal: 0,
        };
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { (self.api.mem_alloc_device)(self.handle, &desc, bytes.max(1), 256, self.device, &mut ptr) };
        ze(&format!("{bytes} bytes of DEVICE memory"), rc)?;
        counted_device();
        Ok(ptr)
    }

    /// `bytes` of host memory the device reads directly, counted.
    pub fn alloc_pinned(&self, bytes: usize) -> Res<*mut c_void> {
        let desc =
            ze::HostMemAllocDesc { stype: ze::STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, p_next: std::ptr::null(), flags: 0 };
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { (self.api.mem_alloc_host)(self.handle, &desc, bytes.max(1), 64, &mut ptr) };
        ze(&format!("{bytes} bytes of PINNED memory"), rc)?;
        counted_device();
        Ok(ptr)
    }

    pub(super) fn alloc_shared(&self, bytes: usize) -> Res<*mut c_void> {
        let d = ze::DeviceMemAllocDesc {
            stype: ze::STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            p_next: std::ptr::null(),
            flags: 0,
            ordinal: 0,
        };
        let h =
            ze::HostMemAllocDesc { stype: ze::STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, p_next: std::ptr::null(), flags: 0 };
        let mut ptr = std::ptr::null_mut();
        let rc = unsafe { (self.api.mem_alloc_shared)(self.handle, &d, &h, bytes.max(1), 64, self.device, &mut ptr) };
        ze(&format!("{bytes} bytes of SHARED memory"), rc)?;
        counted_device();
        Ok(ptr)
    }

    /// Frees memory from one of the alloc functions, warning through the
    /// log when the driver refuses.
    pub fn free(&self, ptr: *mut c_void, what: &str) {
        let rc = unsafe { (self.api.mem_free)(self.handle, ptr) };
        if rc == 0 {
            DEVICE_LIVE.fetch_sub(1, Ordering::Relaxed);
        } else {
            self.say(LOG_WARNING, &format!("levelzero backend: freeing {what}: zeMemFree failed with 0x{rc:08x}"));
        }
    }

    /// What the driver says an address is in this context, and the device
    /// it is on: MEMORY_TYPE_UNKNOWN for memory it did not allocate here.
    pub fn memory_type(&self, ptr: *const c_void) -> (u32, Handle) {
        let mut p = ze::MemoryAllocationProperties {
            stype: ze::STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
            ..Default::default()
        };
        let mut dev = std::ptr::null_mut();
        match unsafe { (self.api.mem_get_alloc_properties)(self.handle, ptr, &mut p, &mut dev) } {
            0 => (p.kind, dev),
            _ => (ze::MEMORY_TYPE_UNKNOWN, std::ptr::null_mut()),
        }
    }

    /// Appends a copy to the queue, which the caller holds.
    ///
    /// # Safety
    /// Both ranges stay valid until the queue has synchronized.
    pub unsafe fn copy(&self, q: &mut Queue, dst: *mut c_void, src: *const c_void, bytes: usize) -> Res<()> {
        let e = self.next_event(q)?;
        let no_waits = std::ptr::null_mut();
        let rc = unsafe { (self.api.command_list_append_memory_copy)(q.list, dst, src, bytes, e, 0, no_waits) };
        self.appended(q, "zeCommandListAppendMemoryCopy", rc)
    }

    /// The event the next append signals. When every event is taken, the
    /// queue is first waited for, and they are all free again.
    #[inline]
    pub(super) fn next_event(&self, q: &mut Queue) -> Res<Handle> {
        if q.used == q.events.len() {
            self.sync(q)?;
        }
        Ok(q.events[q.used])
    }

    /// An append's status. A failed append leaves the driver's immediate
    /// list unable to synchronize or be destroyed, which sync handles.
    #[inline]
    pub(super) fn appended(&self, q: &mut Queue, what: &str, rc: ze::Status) -> Res<()> {
        match rc {
            0 => {
                q.used += 1;
                if profiling() {
                    q.names.push(what.to_owned());
                }
            }
            _ => q.wedged = true,
        }
        ze(what, rc)
    }

    /// Waits for everything appended to the queue, which the caller holds.
    /// After a failed append the list never finishes, so the event of the
    /// last append that succeeded is waited for instead: the list runs in
    /// order, so everything before it is done too. Then that list is set
    /// aside for a new one.
    pub fn sync(&self, q: &mut Queue) -> Res<()> {
        let a = self.api;
        if !q.wedged {
            ze("zeCommandListHostSynchronize", unsafe { (a.command_list_host_synchronize)(q.list, u64::MAX) })?;
        } else {
            if q.used > 0 {
                let last = q.events[q.used - 1];
                ze("zeEventHostSynchronize", unsafe { (a.event_host_synchronize)(last, u64::MAX) })?;
            }
            self.say(
                LOG_WARNING,
                &format!("levelzero device {}: an append failed; the queue is set aside for a new one", self.ordinal),
            );
            // Destroying the old list would wait for it forever; it is left.
            q.list = std::ptr::null_mut();
            q.wedged = false;
            let mut list = std::ptr::null_mut();
            ze("zeCommandListCreateImmediate", unsafe {
                (a.command_list_create_immediate)(self.handle, self.device, &queue_desc(), &mut list)
            })?;
            q.list = list;
        }
        if profiling() {
            let mut p = self.profile.lock().unwrap_or_else(|p| p.into_inner());
            for (name, &e) in q.names.iter().zip(&q.events[..q.used]) {
                // Global start and end, then this context's, in ticks.
                let mut ts = [0u64; 4];
                if unsafe { (a.event_query_kernel_timestamp)(e, &mut ts) } == 0 {
                    let entry = p.entry(name.clone()).or_default();
                    entry.0 += 1;
                    entry.1 += ts[3].wrapping_sub(ts[2]) * self.timer_ns;
                }
            }
            q.names.clear();
        }
        for &e in &q.events[..q.used] {
            ze("zeEventHostReset", unsafe { (a.event_host_reset)(e) })?;
        }
        q.used = 0;
        Ok(())
    }

    /// The queue, which has no list only when replacing it failed.
    pub fn lock_queue(&self) -> Res<std::sync::MutexGuard<'_, Queue>> {
        let q = self.queue.lock().unwrap_or_else(|p| p.into_inner());
        if q.list.is_null() {
            return Err(fail(RUNTIME, format!("levelzero device {}: the context's queue is gone", self.ordinal)));
        }
        Ok(q)
    }
}

/// Events each queue has for its appends to signal, reused once it has
/// synchronized.
pub(super) const EVENTS: u32 = 256;

/// The context's in-order immediate list and the events its appends
/// signal, used under the context's lock.
pub(crate) struct Queue {
    pub list: Handle,
    pub(super) pool: Handle,
    pub(super) events: Vec<Handle>,
    /// Events signalled, or to be, by appends since the last sync.
    pub(super) used: usize,
    /// An append failed since the last sync.
    pub(super) wedged: bool,
    /// With TURBO_LEVELZERO_PROFILE set, what each signalled event's
    /// append was.
    pub(super) names: Vec<String>,
    /// An event of another queue the next launch waits for, or null.
    pub pending_wait: Handle,
}

impl Queue {
    /// The event the last append signals, or null after a sync.
    #[cfg_attr(not(feature = "levelzero-onednn"), allow(dead_code))]
    pub fn last_event(&self) -> Handle {
        if self.used > 0 { self.events[self.used - 1] } else { std::ptr::null_mut() }
    }
}

/// TURBO_LEVELZERO_PROFILE set in the environment: every context times
/// each kernel and copy on the device, and logs the totals at debug level
/// when it is released.
pub(super) fn profiling() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("TURBO_LEVELZERO_PROFILE").is_some())
}

unsafe impl Send for Queue {}
