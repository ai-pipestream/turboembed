//! Safe ownership for the additive Intel native SDK.
//!
//! Enable `prepared` and set `TURBOEMBED_PREPARED_SDK` when building. The runtime
//! loader must find the installed SDK library (for example through the consuming
//! application's RPATH or `LD_LIBRARY_PATH`). No models or runtimes are downloaded.
//!
//! Contexts and models may be shared. Slots move between threads but require
//! exclusive access for writes and execution. A result borrows its slot until
//! release; repeated prepared execution allocates no Rust result container.
//!
//! ```compile_fail
//! use turboembed::prepared::Slot;
//! fn reuse(slot: &mut Slot) {
//!     let result = slot.execute().unwrap();
//!     slot.execute().unwrap(); // The first output is still leased.
//!     drop(result);
//! }
//! ```
//! ```compile_fail
//! use turboembed::prepared::Slot;
//! fn needs_sync<T: Sync>() {}
//! needs_sync::<Slot>();
//! ```

use crate::prepared_ffi as ffi;
use std::{cell::Cell, ffi::c_char, marker::PhantomData, mem::size_of, ptr::NonNull};

/// A native failure with caller-owned error text.
#[derive(Debug, thiserror::Error)]
#[error("TurboEmbed prepared error {code}: {message}")]
pub struct Error {
    pub code: u32,
    pub message: String,
}
type Outcome<T> = std::result::Result<T, Error>;
fn error_buffer() -> ffi::te_error {
    ffi::te_error {
        code: 0,
        message: [0; 508],
    }
}
fn text(bytes: &[c_char]) -> String {
    let bytes: Vec<u8> = bytes
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
fn checked(code: u32, error: &ffi::te_error) -> Outcome<()> {
    if code == ffi::TE_OK {
        Ok(())
    } else {
        Err(Error {
            code,
            message: text(&error.message),
        })
    }
}
fn nonnull<T>(ptr: *mut T) -> Outcome<NonNull<T>> {
    NonNull::new(ptr).ok_or_else(|| Error {
        code: ffi::TE_INTERNAL,
        message: "native success returned a null handle".into(),
    })
}
macro_rules! descriptor {
    ($ty:ty) => {{
        // These C descriptors contain only integers and byte arrays; all-zero
        // bits are valid. Their required size/version are set before FFI access.
        let mut value: $ty = unsafe { std::mem::zeroed() };
        value.struct_size = size_of::<$ty>() as u32;
        value.version = ffi::TE_PREPARED_VERSION;
        value
    }};
}

/// AUTO selects an Intel GPU. CPU is an explicit choice.
#[derive(Clone, Copy, Debug)]
pub enum Device {
    Auto,
    Gpu { ordinal: u32 },
    Cpu,
}

/// Resolved provider identity, independent of the context's lifetime.
#[derive(Debug)]
pub struct ContextInfo {
    pub device: u32,
    pub ordinal: u32,
    pub capabilities: u64,
    pub device_name: String,
    pub runtime_version: String,
    pub driver_version: String,
}
/// Immutable verified model metadata.
#[derive(Debug)]
pub struct ModelInfo {
    pub dimension: u32,
    pub vocab_size: u32,
    pub max_sequence_length: u32,
    pub max_batch_size: u32,
    pub normalized: bool,
    pub model_id: String,
    pub revision: String,
    pub tokenizer_sha256: String,
    pub pooling: String,
}
/// Explicit adapter transfers and bound tensor sizes; excludes vendor internals
/// and the GPU slot's additional host input staging.
#[derive(Debug)]
pub struct SlotStats {
    pub executions: u64,
    pub input_write_bytes: u64,
    pub output_read_bytes: u64,
    pub owned_input_bytes: u64,
    pub owned_output_bytes: u64,
}

pub struct Context {
    handle: NonNull<ffi::te_context>,
}
// Native context compilation is mutex-protected; metadata is immutable. Rust
// borrowing prevents release while any method is using this raw handle.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}
impl Context {
    pub fn new(device: Device) -> Outcome<Self> {
        let (device, ordinal) = match device {
            Device::Auto => (ffi::TE_DEVICE_AUTO, 0),
            Device::Gpu { ordinal } => (ffi::TE_DEVICE_OPENVINO_GPU, ordinal),
            Device::Cpu => (ffi::TE_DEVICE_OPENVINO_CPU, 0),
        };
        let options = ffi::te_context_options {
            struct_size: size_of::<ffi::te_context_options>() as u32,
            version: ffi::TE_PREPARED_VERSION,
            device,
            ordinal,
        };
        let mut handle = std::ptr::null_mut();
        let mut error = error_buffer();
        unsafe {
            checked(
                ffi::turboembed_prepared_v1_context_create(&options, &mut handle, &mut error),
                &error,
            )?;
        }
        Ok(Self {
            handle: nonnull(handle)?,
        })
    }
    pub fn info(&self) -> Outcome<ContextInfo> {
        let mut out = descriptor!(ffi::te_context_info);
        let mut error = error_buffer();
        unsafe {
            checked(
                ffi::turboembed_prepared_v1_context_info(
                    self.handle.as_ptr(),
                    &mut out,
                    &mut error,
                ),
                &error,
            )?;
        }
        Ok(ContextInfo {
            device: out.device,
            ordinal: out.ordinal,
            capabilities: out.capabilities,
            device_name: text(&out.device_name),
            runtime_version: text(&out.runtime_version),
            driver_version: text(&out.driver_version),
        })
    }
    /// Loads an explicit UTF-8 bundle path. Native models independently retain
    /// their context, so the returned model can outlive this Rust context.
    pub fn load_model(&self, path: &str) -> Outcome<Model> {
        let mut handle = std::ptr::null_mut();
        let mut error = error_buffer();
        unsafe {
            checked(
                ffi::turboembed_prepared_v1_model_load(
                    self.handle.as_ptr(),
                    path.as_ptr().cast(),
                    path.len() as u64,
                    &mut handle,
                    &mut error,
                ),
                &error,
            )?;
        }
        Ok(Model {
            handle: nonnull(handle)?,
        })
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            ffi::turboembed_prepared_v1_context_release(self.handle.as_ptr());
        }
    }
}

pub struct Model {
    handle: NonNull<ffi::te_model>,
}
// The graph/metadata are immutable. Native slot creation serializes cloning and
// compilation through the retained context; releasing this handle needs &mut self.
unsafe impl Send for Model {}
unsafe impl Sync for Model {}
impl Model {
    pub fn info(&self) -> Outcome<ModelInfo> {
        let mut out = descriptor!(ffi::te_model_info);
        let mut error = error_buffer();
        unsafe {
            checked(
                ffi::turboembed_prepared_v1_model_info(self.handle.as_ptr(), &mut out, &mut error),
                &error,
            )?;
        }
        Ok(ModelInfo {
            dimension: out.dimension,
            vocab_size: out.vocab_size,
            max_sequence_length: out.max_sequence_length,
            max_batch_size: out.max_batch_size,
            normalized: out.normalized != 0,
            model_id: text(&out.model_id),
            revision: text(&out.revision),
            tokenizer_sha256: text(&out.tokenizer_sha256),
            pooling: text(&out.pooling),
        })
    }
    /// Compiles one fixed shape. The slot can outlive its Rust model handle.
    pub fn slot(&self, batch: u32, sequence: u32) -> Outcome<Slot> {
        let info = self.info()?;
        let options = ffi::te_slot_options {
            struct_size: size_of::<ffi::te_slot_options>() as u32,
            version: ffi::TE_PREPARED_VERSION,
            batch,
            sequence_length: sequence,
        };
        let mut handle = std::ptr::null_mut();
        let mut error = error_buffer();
        unsafe {
            checked(
                ffi::turboembed_prepared_v1_slot_create(
                    self.handle.as_ptr(),
                    &options,
                    &mut handle,
                    &mut error,
                ),
                &error,
            )?;
        }
        Ok(Slot {
            handle: nonnull(handle)?,
            batch: batch as usize,
            sequence: sequence as usize,
            dimension: info.dimension as usize,
            text_views: Vec::with_capacity(batch as usize),
            not_sync: PhantomData,
        })
    }
}
impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            ffi::turboembed_prepared_v1_model_release(self.handle.as_ptr());
        }
    }
}

pub struct Slot {
    handle: NonNull<ffi::te_slot>,
    batch: usize,
    sequence: usize,
    dimension: usize,
    text_views: Vec<ffi::te_text>,
    not_sync: PhantomData<Cell<()>>,
}
// Slots have independent requests and OpenCL queues, with no creating-thread
// affinity. Only an exclusive owner can write, execute, or drop the handle.
unsafe impl Send for Slot {}
impl Slot {
    fn invalidate(&mut self) -> Outcome<()> {
        let mut error = error_buffer();
        unsafe {
            checked(
                ffi::turboembed_prepared_v1_slot_write_tokens(
                    self.handle.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    &mut error,
                ),
                &error,
            )
        }
    }
    pub fn write_tokens(
        &mut self,
        ids: &[i32],
        mask: &[i32],
        types: Option<&[i32]>,
    ) -> Outcome<()> {
        let count = self.batch * self.sequence;
        if ids.len() != count || mask.len() != count || types.is_some_and(|t| t.len() != count) {
            return self.invalidate();
        }
        let mut error = error_buffer();
        unsafe {
            checked(
                ffi::turboembed_prepared_v1_slot_write_tokens(
                    self.handle.as_ptr(),
                    ids.as_ptr(),
                    mask.as_ptr(),
                    types.map_or(std::ptr::null(), |t| t.as_ptr()),
                    count as u64,
                    &mut error,
                ),
                &error,
            )
        }
    }
    /// Text is UTF-8 and may contain NUL. The descriptor array is reused; its
    /// borrowed string pointers are cleared as soon as the copying call returns.
    pub fn write_text(&mut self, texts: &[&str]) -> Outcome<()> {
        if texts.len() != self.batch {
            return self.invalidate();
        }
        self.text_views.clear();
        self.text_views.extend(texts.iter().map(|s| ffi::te_text {
            ptr: s.as_ptr().cast(),
            byte_length: s.len() as u64,
        }));
        let mut error = error_buffer();
        let code = unsafe {
            ffi::turboembed_prepared_v1_slot_write_text(
                self.handle.as_ptr(),
                self.text_views.as_ptr(),
                self.batch as u64,
                &mut error,
            )
        };
        self.text_views.clear();
        checked(code, &error)
    }
    pub fn execute(&mut self) -> Outcome<EmbeddingResult<'_>> {
        let mut handle = std::ptr::null_mut();
        let mut error = error_buffer();
        unsafe {
            checked(
                ffi::turboembed_prepared_v1_slot_execute(
                    self.handle.as_ptr(),
                    &mut handle,
                    &mut error,
                ),
                &error,
            )?;
        }
        Ok(EmbeddingResult {
            handle: Some(nonnull(handle)?),
            batch: self.batch,
            dimension: self.dimension,
            slot: PhantomData,
        })
    }
    pub fn stats(&self) -> Outcome<SlotStats> {
        let mut out = descriptor!(ffi::te_slot_stats);
        let mut error = error_buffer();
        unsafe {
            checked(
                ffi::turboembed_prepared_v1_slot_stats(self.handle.as_ptr(), &mut out, &mut error),
                &error,
            )?;
        }
        Ok(SlotStats {
            executions: out.executions,
            input_write_bytes: out.input_write_bytes,
            output_read_bytes: out.output_read_bytes,
            owned_input_bytes: out.owned_input_bytes,
            owned_output_bytes: out.owned_output_bytes,
        })
    }
}
impl Drop for Slot {
    fn drop(&mut self) {
        unsafe {
            ffi::turboembed_prepared_v1_slot_release(self.handle.as_ptr());
        }
    }
}

pub struct EmbeddingResult<'slot> {
    handle: Option<NonNull<ffi::te_result>>,
    batch: usize,
    dimension: usize,
    slot: PhantomData<&'slot mut Slot>,
}
// The exclusive slot borrow follows this result when moved; it is not Sync.
unsafe impl Send for EmbeddingResult<'_> {}
impl EmbeddingResult<'_> {
    pub fn batch(&self) -> usize {
        self.batch
    }
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn read_into(&self, output: &mut [f32]) -> Outcome<()> {
        let mut error = error_buffer();
        unsafe {
            checked(
                ffi::turboembed_prepared_v1_result_read(
                    self.handle.unwrap().as_ptr(),
                    output.as_mut_ptr(),
                    output.len() as u64,
                    &mut error,
                ),
                &error,
            )
        }
    }
    pub fn to_vec(&self) -> Outcome<Vec<f32>> {
        let mut output = vec![0.0; self.batch * self.dimension];
        self.read_into(&mut output)?;
        Ok(output)
    }
    pub fn opencl(&mut self) -> Outcome<OpenClView<'_>> {
        let mut out = descriptor!(ffi::te_opencl_view);
        let mut error = error_buffer();
        unsafe {
            checked(
                ffi::turboembed_prepared_v1_result_opencl(
                    self.handle.unwrap().as_ptr(),
                    &mut out,
                    &mut error,
                ),
                &error,
            )?;
        }
        Ok(OpenClView {
            raw: out,
            result: PhantomData,
        })
    }
    /// Releases the lease and reports a GPU completion failure. Drop also
    /// releases, but cannot report such a failure to the caller.
    pub fn close(mut self) -> Outcome<()> {
        self.release()
    }
    fn release(&mut self) -> Outcome<()> {
        if let Some(handle) = self.handle.take() {
            let mut error = error_buffer();
            unsafe {
                checked(
                    ffi::turboembed_prepared_v1_result_release(handle.as_ptr(), &mut error),
                    &error,
                )
            }
        } else {
            Ok(())
        }
    }
}
impl Drop for EmbeddingResult<'_> {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

/// Borrowed GPU resources. The view keeps the result borrowed until it is dropped.
pub struct OpenClView<'result> {
    raw: ffi::te_opencl_view,
    result: PhantomData<&'result mut ()>,
}
#[derive(Clone, Copy, Debug)]
pub struct OpenClHandles {
    pub context: usize,
    pub queue: usize,
    pub buffer: usize,
    pub byte_size: u64,
}
impl OpenClView<'_> {
    /// Returns OpenCL resource identifiers, not host memory addresses.
    ///
    /// # Safety
    /// Use these only while this result lease is alive. Do not release borrowed
    /// references or modify the output buffer. Submit consumers on the supplied
    /// in-order queue; foreign queues/imports are unsupported. Native result
    /// release waits for this queue before allowing output reuse.
    pub unsafe fn raw_handles(&self) -> OpenClHandles {
        OpenClHandles {
            context: self.raw.context,
            queue: self.raw.queue,
            buffer: self.raw.buffer,
            byte_size: self.raw.byte_size,
        }
    }
}
