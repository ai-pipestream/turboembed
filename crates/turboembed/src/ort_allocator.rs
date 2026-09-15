//! ORT's external CUDA callbacks have no session context. Their pool therefore
//! owns its arena independently of engine I/O arenas, until both the final
//! session lease and final outstanding allocation have been released.
//! This path is qualified for ORT 1.28's device-0/default-stream external
//! allocator mode; external queues and other CUDA devices need separate pools
//! and explicit stream-aware reuse. See docs/cuda-allocator-isolation.md.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::Mutex;

use crate::buffer_ffi::{self, turbo_buffer_arena, turbo_buffer_view};

static ALLOCATOR: Mutex<AllocatorState> = Mutex::new(AllocatorState::new());

struct Pool {
    arena: usize,
    placement: i32,
    allocations: HashMap<usize, turbo_buffer_view>,
}

impl Drop for Pool {
    fn drop(&mut self) {
        // State only removes a pool once its allocation registry is empty.
        unsafe { buffer_ffi::turbo_buffer_arena_destroy(self.arena as *mut turbo_buffer_arena) };
    }
}

struct AllocatorState {
    sessions: usize,
    pool: Option<Pool>,
}

impl AllocatorState {
    const fn new() -> Self {
        Self {
            sessions: 0,
            pool: None,
        }
    }

    fn acquire(&mut self, device: i32, placement: i32) -> Result<(), String> {
        let sessions = self
            .sessions
            .checked_add(1)
            .ok_or("too many CUDA sessions")?;
        if self.pool.is_none() {
            let mut arena = ptr::null_mut();
            let status = unsafe { buffer_ffi::turbo_buffer_arena_create(device, &mut arena) };
            if status != buffer_ffi::TURBO_BUFFER_OK || arena.is_null() {
                return Err(format!(
                    "external allocator arena: {}",
                    buffer_ffi::last_error(ptr::null())
                ));
            }
            self.pool = Some(Pool {
                arena: arena as usize,
                placement,
                allocations: HashMap::new(),
            });
        }
        self.sessions = sessions;
        Ok(())
    }

    fn retire_unused(&mut self) {
        if self.sessions == 0 && self.pool.as_ref().is_some_and(|p| p.allocations.is_empty()) {
            self.pool = None;
        }
    }

    fn release_session(&mut self) {
        self.sessions -= 1;
        self.retire_unused();
    }

    fn alloc(&mut self, bytes: usize) -> *mut c_void {
        let Some(cols) = allocation_cols(bytes) else {
            return ptr::null_mut();
        };
        // A retired session cannot start new allocations, but late frees still
        // find the old pool until its last checked-out view has been returned.
        if self.sessions == 0 {
            return ptr::null_mut();
        }
        let Some(pool) = self.pool.as_mut() else {
            return ptr::null_mut();
        };
        let view = match buffer_ffi::rent(
            pool.arena as *mut turbo_buffer_arena,
            buffer_ffi::TURBO_BUFFER_DTYPE_F32,
            pool.placement,
            1,
            cols,
            cols,
        ) {
            Ok(view) => view,
            Err(_) => return ptr::null_mut(),
        };
        let address = view.ptr;
        pool.allocations.insert(address as usize, view);
        address
    }

    fn free(&mut self, ptr: *mut c_void) {
        if let Some(pool) = self.pool.as_mut() {
            if let Some(mut view) = pool.allocations.remove(&(ptr as usize)) {
                buffer_ffi::return_view(pool.arena as *mut turbo_buffer_arena, &mut view);
            }
        }
        self.retire_unused();
    }
}

fn allocation_cols(bytes: usize) -> Option<u32> {
    u32::try_from((bytes.checked_add(3)? / 4).max(1)).ok()
}

/// Keep after Session and its Allocator fields so ORT releases them first.
pub(crate) struct SessionLease;

impl SessionLease {
    pub(crate) fn acquire() -> Result<Self, String> {
        // The existing CUDA provider and TurboBuffer both select device 0.
        ALLOCATOR
            .lock()
            .map_err(|_| "external allocator lock poisoned")?
            .acquire(2, buffer_ffi::TURBO_BUFFER_PLACE_DEVICE)?;
        Ok(Self)
    }
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        ALLOCATOR
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .release_session();
    }
}

pub(crate) fn alloc(bytes: usize) -> *mut c_void {
    ALLOCATOR
        .lock()
        .map(|mut state| state.alloc(bytes))
        .unwrap_or(ptr::null_mut())
}

pub(crate) fn free(ptr: *mut c_void) {
    if let Ok(mut state) = ALLOCATOR.lock() {
        state.free(ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_retains_storage_through_other_session_shutdown_and_late_free() {
        // CPU storage exercises the same owner/registry logic without claiming
        // to execute CUDA. Each test owns independent state.
        let mut state = AllocatorState::new();
        state
            .acquire(1, buffer_ffi::TURBO_BUFFER_PLACE_HOST)
            .unwrap();
        let first = state.alloc(64).cast::<u8>();
        assert!(!first.is_null());
        unsafe { first.write_bytes(0x5a, 64) };
        state
            .acquire(1, buffer_ffi::TURBO_BUFFER_PLACE_HOST)
            .unwrap();
        let second = state.alloc(64);
        assert!(!second.is_null());
        assert_ne!(first.cast::<c_void>(), second);
        state.release_session();
        state.free(second);
        assert_eq!(
            unsafe { std::slice::from_raw_parts(first, 64) },
            &[0x5a; 64]
        );
        state.release_session();
        assert!(state.alloc(64).is_null());
        assert!(state.pool.is_some());
        state.free(first.cast());
        assert!(state.pool.is_none());
        state
            .acquire(1, buffer_ffi::TURBO_BUFFER_PLACE_HOST)
            .unwrap();
        let next = state.alloc(64);
        assert!(!next.is_null());
        state.free(next);
        state.release_session();
        assert!(state.pool.is_none());
    }

    #[test]
    fn failed_load_without_allocations_releases_pool() {
        let mut state = AllocatorState::new();
        state
            .acquire(1, buffer_ffi::TURBO_BUFFER_PLACE_HOST)
            .unwrap();
        state.release_session();
        assert!(state.pool.is_none());
    }

    #[test]
    fn byte_rounding_never_wraps_or_truncates() {
        assert_eq!(allocation_cols(0), Some(1));
        assert_eq!(allocation_cols(1), Some(1));
        assert_eq!(allocation_cols(5), Some(2));
        assert_eq!(allocation_cols(usize::MAX), None);
        if usize::BITS > 32 {
            let max = u32::MAX as u64 * 4;
            assert_eq!(allocation_cols(max as usize), Some(u32::MAX));
            assert_eq!(allocation_cols((max + 1) as usize), None);
        }
    }

    #[test]
    #[ignore = "requires CUDA runtime, device, and catalog MiniLM weights"]
    fn cuda_engines_release_external_pool() {
        use crate::{Device, EmbedOptions, Engine};
        let start = std::time::Instant::now();
        eprintln!("ORT build: {}", ort::info());
        let opts = EmbedOptions::default();
        let first = Engine::create(Device::Cuda).unwrap();
        first.load_model("minilm").unwrap();
        let second = Engine::create(Device::Cuda).unwrap();
        second.load_model("minilm").unwrap();
        let texts = ["The capital of France is Paris.", "A different sentence."];
        let expected = first
            .embed("minilm", &texts, &opts)
            .unwrap()
            .values()
            .to_vec();
        let expected_worker = expected.clone();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let worker_barrier = barrier.clone();
        let worker = std::thread::spawn(move || {
            worker_barrier.wait();
            for _ in 0..8 {
                let result = first.embed("minilm", &texts, &opts).unwrap();
                for (&actual, &expected) in result.values().iter().zip(&expected_worker) {
                    assert!((actual - expected).abs() <= 1e-5);
                }
            }
            drop(first);
        });
        barrier.wait();
        for _ in 0..8 {
            let result = second.embed("minilm", &texts, &opts).unwrap();
            assert_eq!(result.values().len(), expected.len());
            for (&actual, &expected) in result.values().iter().zip(&expected) {
                assert!((actual - expected).abs() <= 1e-5);
            }
        }
        worker.join().unwrap();
        assert_eq!(ALLOCATOR.lock().unwrap().sessions, 1);
        let survivor = second.embed("minilm", &texts, &opts).unwrap();
        drop(second);
        assert_eq!(ALLOCATOR.lock().unwrap().sessions, 1);
        assert_eq!(survivor.values().len(), expected.len());
        drop(survivor);
        let state = ALLOCATOR.lock().unwrap();
        assert_eq!(state.sessions, 0);
        assert!(
            state.pool.is_none(),
            "ORT left external allocations outstanding after shutdown"
        );
        eprintln!(
            "two-engine CUDA isolation completed in {:?}",
            start.elapsed()
        );
    }
}
