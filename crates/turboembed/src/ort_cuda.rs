//! NVIDIA provider: ONNX Runtime CUDA EP + IoBinding, or an explicit CPU EP.
//!
//! I/O tensors rent from the engine `turbo_buffer` arena:
//! - CUDA / TensorRT: PINNED mapped token rows + DEVICE hidden states
//! - CPU: HOST token rows + HOST hidden states
//!
//! After load warmup (max batch × max seq), steady-state embed must not
//! increment `turbo_buffer_alloc_counter` or the ORT `gpu_external_alloc`
//! hook. CUDA mean+L2 (mask-weighted) runs on DEVICE into a mapped
//! PINNED result row. Activation D2H (`d2h_hidden_*`) must stay 0.
//! The caller reads the final 384-d row from mapped PINNED
//! (`result_host_bytes`, much smaller than the hidden volume).

use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use inferstream_backend_ort::pool::l2_normalize;
use inferstream_backend_ort::Pooling;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::{IoBinding, Session};
use ort::value::{PrimitiveTensorElementType, Shape, Tensor, ValueType};
use ort::{ortsys, AsPointer};
use tokenizers::Tokenizer;

use crate::buffer_ffi::{
    self, mapped_device_ptr, rent, return_view, turbo_buffer_arena, turbo_buffer_view,
    TURBO_BUFFER_DTYPE_F32, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_DEVICE,
    TURBO_BUFFER_PLACE_HOST, TURBO_BUFFER_PLACE_PINNED,
};
use crate::catalog::CatalogModelSpec;
use crate::wordpiece_ffi::{self, WordPiece};

type Error = String;

const DEFAULT_MAX_SEQ_LEN: usize = 256;
const DEFAULT_MAX_BATCH: usize = 8;

static ORT_EXT_ALLOCS: AtomicU64 = AtomicU64::new(0);
static ORT_EXT_LAST_BYTES: AtomicU64 = AtomicU64::new(0);
static ORT_D2H_BYTES: AtomicU64 = AtomicU64::new(0);
static ORT_D2H_CALLS: AtomicU64 = AtomicU64::new(0);
static ORT_D2H_RESULT_BYTES: AtomicU64 = AtomicU64::new(0);
static ORT_RESULT_HOST_BYTES: AtomicU64 = AtomicU64::new(0);

static EXT_ARENA: Mutex<Option<usize>> = Mutex::new(None);
static EXT_SLABS: Mutex<Option<HashMap<usize, turbo_buffer_view>>> = Mutex::new(None);

#[cfg(turboembed_cuda)]
unsafe extern "C" {
    fn turboembed_cuda_pool_mean_l2(
        hidden_dev: *const f32,
        mask_dev: *const i64,
        out_dev: *mut f32,
        batch: i32,
        seq: i32,
        dim: i32,
        normalize: i32,
    ) -> i32;
    fn turboembed_cuda_pool_cls_l2(
        hidden_dev: *const f32,
        out_dev: *mut f32,
        batch: i32,
        seq: i32,
        dim: i32,
        normalize: i32,
    ) -> i32;
}

/// Where this session is allowed to run. CUDA / TensorRT never become CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrtPlace {
    Cuda,
    Cpu,
    TensorRt,
}

impl OrtPlace {
    /// ABI `turboembed_device`: AUTO(0) and CUDA(2) → CUDA; CPU(1) → CPU;
    /// TENSORRT(3) → TensorRT EP (same ONNX, fail loud if TRT 10 is missing).
    pub fn from_abi(device: i32) -> Result<Self, Error> {
        match device {
            1 => Ok(Self::Cpu),
            0 | 2 => Ok(Self::Cuda),
            3 => Ok(Self::TensorRt),
            other => Err(format!(
                "ORT path does not handle ABI device {other}; \
                 use TURBOEMBED_DEVICE_CUDA / AUTO, CPU, or TENSORRT"
            )),
        }
    }

    fn uses_cuda_buffers(self) -> bool {
        matches!(self, Self::Cuda | Self::TensorRt)
    }
}

struct OrtWork {
    arena: *mut turbo_buffer_arena,
    max_batch: usize,
    max_seq: usize,
    hidden_dim: usize,
    input_ids: turbo_buffer_view,
    attention_mask: turbo_buffer_view,
    token_type_ids: turbo_buffer_view,
    hidden: turbo_buffer_view,
    /// PINNED mapped [max_batch, hidden_dim] used when the C++ caller
    /// did not pass a result row (load warmup / Rust-only embed).
    pool_out: turbo_buffer_view,
}

unsafe impl Send for OrtWork {}

impl OrtWork {
    fn return_all(&mut self) {
        return_view(self.arena, &mut self.input_ids);
        return_view(self.arena, &mut self.attention_mask);
        return_view(self.arena, &mut self.token_type_ids);
        return_view(self.arena, &mut self.hidden);
        return_view(self.arena, &mut self.pool_out);
    }
}

impl Drop for OrtWork {
    fn drop(&mut self) {
        self.return_all();
    }
}

enum TokenFront {
    WordPiece(WordPiece),
    Hf(Tokenizer),
}

pub struct OrtCudaSession {
    session: Mutex<Session>,
    tokens: TokenFront,
    pooling: Pooling,
    normalize: bool,
    input_names: Vec<String>,
    output_name: String,
    place: OrtPlace,
    #[allow(dead_code)]
    cuda_allocator: Option<Allocator>,
    embedding_dim: usize,
    work: OrtWork,
}

fn fail_load(what: &str, detail: impl std::fmt::Display) -> Error {
    format!("{what}: {detail}")
}

fn ort_status(status: ort::sys::OrtStatusPtr) -> Result<(), Error> {
    if status.0.is_null() {
        return Ok(());
    }
    let api = ort::api();
    let msg = unsafe {
        let p = (api.GetErrorMessage)(status.0);
        let s = if p.is_null() {
            "ORT status without message".to_string()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        };
        (api.ReleaseStatus)(status.0);
        s
    };
    Err(msg)
}

extern "C" fn gpu_external_alloc(bytes: usize) -> *mut c_void {
    ORT_EXT_ALLOCS.fetch_add(1, Ordering::Relaxed);
    ORT_EXT_LAST_BYTES.store(bytes as u64, Ordering::Relaxed);
    let arena = match EXT_ARENA.lock() {
        Ok(g) => g.and_then(|p| {
            if p == 0 {
                None
            } else {
                Some(p as *mut turbo_buffer_arena)
            }
        }),
        Err(_) => None,
    };
    let Some(arena) = arena else {
        return ptr::null_mut();
    };
    let cols = ((bytes + 3) / 4).max(1) as u32;
    match rent(
        arena,
        TURBO_BUFFER_DTYPE_F32,
        TURBO_BUFFER_PLACE_DEVICE,
        1,
        cols,
        cols,
    ) {
        Ok(view) => {
            let key = view.ptr as usize;
            if let Ok(mut map) = EXT_SLABS.lock() {
                map.get_or_insert_with(HashMap::new).insert(key, view);
            }
            view.ptr
        }
        Err(_) => ptr::null_mut(),
    }
}

extern "C" fn gpu_external_free(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    let arena = match EXT_ARENA.lock() {
        Ok(g) => g.and_then(|v| {
            if v == 0 {
                None
            } else {
                Some(v as *mut turbo_buffer_arena)
            }
        }),
        Err(_) => None,
    };
    let Some(arena) = arena else {
        return;
    };
    if let Ok(mut map) = EXT_SLABS.lock() {
        if let Some(map) = map.as_mut() {
            if let Some(mut view) = map.remove(&(p as usize)) {
                return_view(arena, &mut view);
            }
        }
    }
}

extern "C" fn gpu_external_empty_cache() {}

fn set_ext_arena(arena: *mut turbo_buffer_arena) {
    if let Ok(mut g) = EXT_ARENA.lock() {
        *g = if arena.is_null() {
            None
        } else {
            Some(arena as usize)
        };
    }
}

fn clear_ext_arena_if(arena: *mut turbo_buffer_arena) {
    if let Ok(mut g) = EXT_ARENA.lock() {
        if *g == Some(arena as usize) {
            *g = None;
        }
    }
}

fn register_cuda_ep_with_arena(builder: &mut SessionBuilder) -> Result<(), Error> {
    let api = ort::api();
    let mut cuda_options: *mut ort::sys::OrtCUDAProviderOptionsV2 = ptr::null_mut();
    ort_status(unsafe { (api.CreateCUDAProviderOptions)(&mut cuda_options) })?;
    if cuda_options.is_null() {
        return Err("CreateCUDAProviderOptions returned null".into());
    }

    let keys = [
        CString::new("device_id").unwrap(),
        CString::new("arena_extend_strategy").unwrap(),
        CString::new("cudnn_conv_algo_search").unwrap(),
    ];
    let vals = [
        CString::new("0").unwrap(),
        CString::new("kSameAsRequested").unwrap(),
        CString::new("HEURISTIC").unwrap(),
    ];
    let key_ptrs: Vec<*const i8> = keys.iter().map(|k| k.as_ptr()).collect();
    let val_ptrs: Vec<*const i8> = vals.iter().map(|v| v.as_ptr()).collect();
    ort_status(unsafe {
        (api.UpdateCUDAProviderOptions)(
            cuda_options,
            key_ptrs.as_ptr(),
            val_ptrs.as_ptr(),
            keys.len(),
        )
    })?;

    let k_alloc = CString::new("gpu_external_alloc").unwrap();
    let k_free = CString::new("gpu_external_free").unwrap();
    let k_empty = CString::new("gpu_external_empty_cache").unwrap();
    ort_status(unsafe {
        (api.UpdateCUDAProviderOptionsWithValue)(
            cuda_options,
            k_alloc.as_ptr(),
            gpu_external_alloc as *mut c_void,
        )
    })?;
    ort_status(unsafe {
        (api.UpdateCUDAProviderOptionsWithValue)(
            cuda_options,
            k_free.as_ptr(),
            gpu_external_free as *mut c_void,
        )
    })?;
    ort_status(unsafe {
        (api.UpdateCUDAProviderOptionsWithValue)(
            cuda_options,
            k_empty.as_ptr(),
            gpu_external_empty_cache as *mut c_void,
        )
    })?;

    let rc = ort_status(unsafe {
        (api.SessionOptionsAppendExecutionProvider_CUDA_V2)(builder.ptr_mut(), cuda_options)
    });
    unsafe { (api.ReleaseCUDAProviderOptions)(cuda_options) };
    rc.map_err(|e| {
        format!("CUDA execution provider unavailable (no silent CPU fallback): {e}")
    })
}

fn find_tokenizer(
    model_path: &str,
    tokenizer_dir: Option<&str>,
) -> Result<std::path::PathBuf, Error> {
    use std::path::{Path, PathBuf};
    if let Some(path) = tokenizer_dir {
        let path = PathBuf::from(path);
        let file = if path.is_dir() {
            path.join("tokenizer.json")
        } else {
            path
        };
        if file.is_file() {
            return Ok(file);
        }
        return Err(fail_load(
            "tokenizer not found",
            format!("{} does not exist", file.display()),
        ));
    }
    let model = Path::new(model_path);
    let mut candidates = Vec::new();
    if let Some(dir) = model.parent() {
        candidates.push(dir.join("tokenizer.json"));
        if let Some(up) = dir.parent() {
            candidates.push(up.join("tokenizer.json"));
        }
    }
    candidates
        .into_iter()
        .find(|c| c.is_file())
        .ok_or_else(|| {
            fail_load(
                "tokenizer not found",
                format!("no tokenizer.json next to {model_path}"),
            )
        })
}

fn require_cuda_device(info: &MemoryInfo<'_>, what: &str) -> Result<(), Error> {
    let device = info.allocation_device();
    if device != AllocationDevice::CUDA {
        return Err(format!(
            "{what} is on allocation device {:?} (cpu_accessible={}); \
             expected CUDA. This is a CPU fallback, not a real NVIDIA embed. \
             Check LD_LIBRARY_PATH (.libs/nvidia/lib) and --features ort-cuda",
            device.as_str(),
            info.is_cpu_accessible()
        ));
    }
    if info.is_cpu_accessible() {
        return Err(format!(
            "{what} is CUDA-named but CPU-accessible; a pinned/host \
             buffer is not a device buffer"
        ));
    }
    Ok(())
}

fn resolve_path(workspace_root: &std::path::Path, raw: &str) -> std::path::PathBuf {
    let path = std::path::PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        workspace_root.join(path)
    }
}

fn i64_slot_mut(view: &turbo_buffer_view, n: usize) -> Result<&mut [i64], Error> {
    if view.ptr.is_null() {
        return Err("token view is null".into());
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(view.ptr.cast::<i64>(), n) })
}

fn f32_view<'a>(view: &turbo_buffer_view, n: usize) -> Result<&'a [f32], Error> {
    if view.ptr.is_null() {
        return Err("f32 view is null".into());
    }
    Ok(unsafe { std::slice::from_raw_parts(view.ptr.cast::<f32>(), n) })
}

fn i64_view<'a>(view: &turbo_buffer_view, n: usize) -> Result<&'a [i64], Error> {
    if view.ptr.is_null() {
        return Err("i64 view is null".into());
    }
    Ok(unsafe { std::slice::from_raw_parts(view.ptr.cast::<i64>(), n) })
}

fn ort_ok<T>(r: ort::Result<T>) -> Result<T, Error> {
    r.map_err(|e| e.to_string())
}

/// `TensorRefMut::from_raw` calls `MemoryInfo::to_owned`, which always
/// builds a CPU MemoryInfo in ort 2.0.0-rc.13. Create the OrtValue
/// ourselves so CUDA / PINNED tags survive.
fn tensor_from_data<T: PrimitiveTensorElementType + std::fmt::Debug>(
    info: &MemoryInfo<'_>,
    data: *mut c_void,
    shape: Shape,
) -> Result<Tensor<T>, Error> {
    let mut value_ptr: *mut ort::sys::OrtValue = ptr::null_mut();
    let nbytes = shape.num_elements() * std::mem::size_of::<T>();
    ort_ok((|| {
        ortsys![
            unsafe CreateTensorWithDataAsOrtValue(
                info.ptr(),
                data,
                nbytes,
                shape.as_ptr(),
                shape.len(),
                T::into_tensor_element_type().into(),
                &mut value_ptr
            )?;
            nonNull(value_ptr)
        ];
        Ok(())
    })())
    .map_err(|e| format!("CreateTensorWithDataAsOrtValue: {e}"))?;
    let nn = NonNull::new(value_ptr).ok_or_else(|| "CreateTensorWithDataAsOrtValue returned null".to_string())?;
    Ok(unsafe { Tensor::<T>::from_ptr(nn, None) })
}

fn mean_pool_into(
    hidden: &[f32],
    mask: &[i64],
    batch: usize,
    seq: usize,
    dim: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(hidden.len(), batch * seq * dim);
    debug_assert_eq!(mask.len(), batch * seq);
    debug_assert_eq!(out.len(), batch * dim);
    out.fill(0.0);
    for b in 0..batch {
        let mut count = 0f32;
        for s in 0..seq {
            if mask[b * seq + s] == 0 {
                continue;
            }
            count += 1.0;
            let row = &hidden[(b * seq + s) * dim..(b * seq + s + 1) * dim];
            let acc = &mut out[b * dim..(b + 1) * dim];
            for (a, v) in acc.iter_mut().zip(row) {
                *a += v;
            }
        }
        if count > 0.0 {
            for a in &mut out[b * dim..(b + 1) * dim] {
                *a /= count;
            }
        }
    }
}

fn cls_pool_into(hidden: &[f32], batch: usize, seq: usize, dim: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), batch * dim);
    for b in 0..batch {
        let src = &hidden[b * seq * dim..b * seq * dim + dim];
        out[b * dim..(b + 1) * dim].copy_from_slice(src);
    }
}

fn note_result_host_read(bytes: usize) {
    ORT_RESULT_HOST_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

#[cfg(turboembed_cuda)]
fn device_pool(
    pooling: Pooling,
    hidden_dev: *const f32,
    mask_dev: *const i64,
    out_dev: *mut f32,
    batch: usize,
    seq: usize,
    dim: usize,
    normalize: bool,
) -> Result<(), Error> {
    let rc = match pooling {
        Pooling::Mean => unsafe {
            turboembed_cuda_pool_mean_l2(
                hidden_dev,
                mask_dev,
                out_dev,
                batch as i32,
                seq as i32,
                dim as i32,
                i32::from(normalize),
            )
        },
        Pooling::Cls => unsafe {
            turboembed_cuda_pool_cls_l2(
                hidden_dev,
                out_dev,
                batch as i32,
                seq as i32,
                dim as i32,
                i32::from(normalize),
            )
        },
    };
    if rc != 0 {
        return Err(format!(
            "CUDA device pool failed (cudaError={rc}); refusing a host hidden D2H stand-in"
        ));
    }
    Ok(())
}

#[cfg(not(turboembed_cuda))]
fn device_pool(
    _pooling: Pooling,
    _hidden_dev: *const f32,
    _mask_dev: *const i64,
    _out_dev: *mut f32,
    _batch: usize,
    _seq: usize,
    _dim: usize,
    _normalize: bool,
) -> Result<(), Error> {
    Err("CUDA device pool requires a TURBO_BUFFER_CUDA build".into())
}

impl OrtCudaSession {
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn pooling(&self) -> Pooling {
        self.pooling
    }

    pub fn normalize(&self) -> bool {
        self.normalize
    }

    pub fn place(&self) -> OrtPlace {
        self.place
    }

    pub fn load(
        spec: &CatalogModelSpec,
        workspace_root: &std::path::Path,
        place: OrtPlace,
        arena: *mut turbo_buffer_arena,
    ) -> Result<Self, Error> {
        if arena.is_null() {
            return Err("ORT embed requires a turbo_buffer arena from the engine".into());
        }
        if !spec.backend.eq_ignore_ascii_case("ort") {
            return Err(format!(
                "nvidia turboembed requires catalog backend=\"ort\", got {:?}",
                spec.backend
            ));
        }
        let catalog_device = spec.device.as_deref().unwrap_or("");
        if place == OrtPlace::Cuda && !catalog_device.eq_ignore_ascii_case("cuda") {
            return Err(format!(
                "CUDA was requested but catalog device is {catalog_device:?}; \
                 refusing to treat a non-cuda catalog entry as CUDA"
            ));
        }
        if place == OrtPlace::TensorRt
            && !catalog_device.eq_ignore_ascii_case("cuda")
            && !catalog_device.eq_ignore_ascii_case("tensorrt")
        {
            return Err(format!(
                "TensorRT was requested but catalog device is {catalog_device:?}; \
                 MiniLM ORT-TRT uses the same ONNX as CUDA (device=cuda|tensorrt)"
            ));
        }
        let model_path = spec.path.as_deref().ok_or_else(|| {
            "catalog nvidia entry is missing path to the .onnx file".to_string()
        })?;
        let model_path = resolve_path(workspace_root, model_path);
        if !model_path.is_file() {
            return Err(fail_load(
                "onnx model not found",
                format!("{} is not a file", model_path.display()),
            ));
        }
        let model_path = model_path
            .to_str()
            .ok_or_else(|| "onnx model path is not UTF-8".to_string())?;

        let tokenizer_hint = spec
            .tokenizer_dir
            .as_deref()
            .map(|p| resolve_path(workspace_root, p).to_string_lossy().into_owned());
        let max_seq = spec
            .max_seq_len
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_MAX_SEQ_LEN);
        let tokens = if let Some(wp) =
            wordpiece_ffi::load_beside_model(model_path, tokenizer_hint.as_deref())
        {
            TokenFront::WordPiece(wp)
        } else {
            let tokenizer_file = find_tokenizer(model_path, tokenizer_hint.as_deref())?;
            let mut tokenizer = Tokenizer::from_file(&tokenizer_file)
                .map_err(|e| fail_load("failed to load tokenizer", e))?;
            tokenizer
                .with_truncation(Some(tokenizers::TruncationParams {
                    max_length: max_seq,
                    ..Default::default()
                }))
                .map_err(|e| fail_load("failed to configure truncation", e))?;
            tokenizer.with_padding(Some(tokenizers::PaddingParams {
                strategy: tokenizers::PaddingStrategy::Fixed(max_seq),
                ..Default::default()
            }));
            TokenFront::Hf(tokenizer)
        };

        let pooling = spec
            .pooling
            .as_deref()
            .map(Pooling::from_config)
            .transpose()
            .map_err(|e| e.to_string())?
            .unwrap_or(Pooling::Mean);

        if place == OrtPlace::Cuda {
            set_ext_arena(arena);
        }

        let mut builder = Session::builder()
            .map_err(|e| fail_load("failed to create ort session builder", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| fail_load("failed to configure ort session", e))?
            .with_memory_pattern(true)
            .map_err(|e| fail_load("failed to enable ORT memory pattern", e))?;

        if place == OrtPlace::Cuda {
            register_cuda_ep_with_arena(&mut builder)?;
        }
        if place == OrtPlace::TensorRt {
            builder = builder
                .with_execution_providers([ort::ep::TensorRT::default()
                    .with_device_id(0)
                    .build()
                    .error_on_failure()])
                .map_err(|e| {
                    fail_load(
                        "TensorRT execution provider unavailable \
                         (libnvinfer.so.10 / libnvonnxparser.so.10); \
                         CUDA/CPU is not a fallback",
                        e,
                    )
                })?;
        }

        let session = builder.commit_from_file(model_path).map_err(|e| {
            if place == OrtPlace::Cuda {
                clear_ext_arena_if(arena);
            }
            fail_load(
                match place {
                    OrtPlace::Cuda => "failed to load onnx model on CUDA EP",
                    OrtPlace::Cpu => "failed to load onnx model on CPU EP",
                    OrtPlace::TensorRt => {
                        "failed to load onnx model on TensorRT EP \
                         (not a silent CUDA or CPU session)"
                    }
                },
                e,
            )
        })?;

        let cuda_allocator = match place {
            OrtPlace::Cuda | OrtPlace::TensorRt => {
                let cuda_mem = MemoryInfo::new(
                    AllocationDevice::CUDA,
                    0,
                    AllocatorType::Device,
                    MemoryType::Default,
                )
                .map_err(|e| fail_load("CUDA MemoryInfo", e))?;
                let alloc = Allocator::new(&session, cuda_mem).map_err(|e| {
                    fail_load(
                        "CUDA device allocator unavailable — CUDA EP is not live; \
                         CUDA was requested so CPU is not accepted",
                        e,
                    )
                })?;
                let gpu_id = ort::ep::get_gpu_device().map_err(|e| {
                    fail_load(
                        "ort::ep::get_gpu_device failed; CUDA EP did not attach a GPU",
                        e,
                    )
                })?;
                if gpu_id < 0 {
                    return Err(format!(
                        "ort GPU device id {gpu_id} is not a real CUDA device; \
                         CUDA was requested so CPU is not accepted"
                    ));
                }
                Some(alloc)
            }
            OrtPlace::Cpu => None,
        };

        const KNOWN: [&str; 3] = ["input_ids", "attention_mask", "token_type_ids"];
        let mut input_names = Vec::new();
        for input in session.inputs() {
            if KNOWN.contains(&input.name()) {
                input_names.push(input.name().to_string());
            } else {
                return Err(fail_load(
                    "unsupported model input",
                    format!("graph input {:?} is not one of {KNOWN:?}", input.name()),
                ));
            }
        }
        if !input_names.iter().any(|n| n == "input_ids") {
            return Err(fail_load(
                "unsupported model",
                "graph has no `input_ids` input",
            ));
        }
        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .ok_or_else(|| fail_load("unsupported model", "graph has no outputs"))?;

        let hidden_dim = match session.outputs().first().map(|o| o.dtype()) {
            Some(ValueType::Tensor { shape, .. }) if !shape.is_empty() => {
                let last = shape[shape.len() - 1];
                if last > 0 {
                    last as usize
                } else {
                    384
                }
            }
            _ => 384,
        };

        let max_batch = spec
            .max_batch_size
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_MAX_BATCH)
            .max(1);
        let token_place = if place.uses_cuda_buffers() {
            TURBO_BUFFER_PLACE_PINNED
        } else {
            TURBO_BUFFER_PLACE_HOST
        };
        let hidden_place = if place.uses_cuda_buffers() {
            TURBO_BUFFER_PLACE_DEVICE
        } else {
            TURBO_BUFFER_PLACE_HOST
        };
        let token_cols = (max_seq * 2) as u32;
        let hidden_cols = (max_seq * hidden_dim) as u32;

        let mut work = OrtWork {
            arena,
            max_batch,
            max_seq,
            hidden_dim,
            input_ids: rent(
                arena,
                TURBO_BUFFER_DTYPE_I32,
                token_place,
                max_batch as u32,
                token_cols,
                token_cols,
            )?,
            attention_mask: rent(
                arena,
                TURBO_BUFFER_DTYPE_I32,
                token_place,
                max_batch as u32,
                token_cols,
                token_cols,
            )?,
            token_type_ids: rent(
                arena,
                TURBO_BUFFER_DTYPE_I32,
                token_place,
                max_batch as u32,
                token_cols,
                token_cols,
            )?,
            hidden: rent(
                arena,
                TURBO_BUFFER_DTYPE_F32,
                hidden_place,
                max_batch as u32,
                hidden_cols,
                hidden_cols,
            )?,
            pool_out: turbo_buffer_view::empty(),
        };
        if place.uses_cuda_buffers() {
            work.pool_out = rent(
                arena,
                TURBO_BUFFER_DTYPE_F32,
                TURBO_BUFFER_PLACE_PINNED,
                max_batch as u32,
                hidden_dim as u32,
                hidden_dim as u32,
            )?;
        }

        let mut loaded = Self {
            session: Mutex::new(session),
            tokens,
            pooling,
            normalize: spec.normalize.unwrap_or(true),
            input_names,
            output_name,
            place,
            cuda_allocator,
            embedding_dim: 0,
            work,
        };
        // Warm the ORT BFC / graph at max batch so later batch=1 does not grow it.
        let warm: Vec<String> = (0..max_batch).map(|_| String::from("x")).collect();
        let (dim, _) = loaded.embed_batch(&warm, None)?;
        if dim == 0 {
            return Err(format!("{:?} warmup produced embedding dim 0", loaded.place));
        }
        loaded.embedding_dim = dim;
        // Second pass at batch=1 matches the receipt hot path.
        let _ = loaded.embed_batch(&[String::from("x")], None)?;
        // cuDNN / ORT CUDA EP can lazily rent extra DEVICE workspace on the
        // first non-trivial mask. Touch a few lengths so the first real
        // sentence does not increment gpu_external_alloc.
        for text in [
            "hello world".to_string(),
            "The capital of France is Paris.".to_string(),
            "a ".repeat(64),
            "inferstream turboembed onnxruntime cuda iobinding".to_string(),
        ] {
            let _ = loaded.embed_batch(&[text], None)?;
        }
        Ok(loaded)
    }

    /// Embed `texts` and write the pooled rows into `out` when provided
    /// (C++ already rented that row). Otherwise allocate a Vec (tests).
    pub(crate) fn embed_batch(
        &self,
        texts: &[String],
        mut out: Option<&mut [f32]>,
    ) -> Result<(usize, Vec<f32>), Error> {
        let batch = texts.len();
        if batch == 0 {
            return Err("embed requires at least one text".into());
        }
        if batch > self.work.max_batch {
            return Err(format!(
                "embed batch {batch} exceeds turbo_buffer warmed max_batch {}",
                self.work.max_batch
            ));
        }

        let seq = self.work.max_seq;
        let token_n = self.work.max_batch * self.work.max_seq;
        {
            let ids = i64_slot_mut(&self.work.input_ids, token_n)?;
            let mask = i64_slot_mut(&self.work.attention_mask, token_n)?;
            let types = i64_slot_mut(&self.work.token_type_ids, token_n)?;
            match &self.tokens {
                TokenFront::WordPiece(wp) => {
                    wordpiece_ffi::wordpiece_hot_alloc_counter_reset();
                    for (b, text) in texts.iter().enumerate() {
                        let row = b * seq;
                        wp.encode_sentence(
                            text,
                            ids[row..].as_mut_ptr() as *mut std::ffi::c_void,
                            mask[row..].as_mut_ptr() as *mut std::ffi::c_void,
                            types[row..].as_mut_ptr() as *mut std::ffi::c_void,
                            seq as u32,
                            seq as u32,
                            8,
                        )?;
                    }
                    if wordpiece_ffi::wordpiece_hot_alloc_counter() != 0 {
                        return Err(
                            "WordPiece hot-path heap token staging reintroduced".into()
                        );
                    }
                }
                TokenFront::Hf(tokenizer) => {
                    let encodings = tokenizer
                        .encode_batch(texts.to_vec(), true)
                        .map_err(|e| format!("tokenization failed: {e}"))?;
                    if encodings.iter().any(|e| e.len() > seq) {
                        return Err(format!("tokenized seq exceeds warmed max_seq {seq}"));
                    }
                    if encodings.is_empty() {
                        return Err("tokenization produced an empty sequence".into());
                    }
                    ids.fill(0);
                    mask.fill(0);
                    types.fill(0);
                    for (b, encoding) in encodings.iter().enumerate() {
                        let row = b * seq;
                        for (i, v) in encoding.get_ids().iter().enumerate() {
                            ids[row + i] = i64::from(*v);
                        }
                        for (i, v) in encoding.get_attention_mask().iter().enumerate() {
                            mask[row + i] = i64::from(*v);
                        }
                        for (i, v) in encoding.get_type_ids().iter().enumerate() {
                            types[row + i] = i64::from(*v);
                        }
                    }
                }
            }
        }

        let shape = [batch as i64, seq as i64];
        unsafe { buffer_ffi::turbo_buffer_cuda_forward_enter() };
        let result = if self.place.uses_cuda_buffers() {
            self.embed_cuda(batch, seq, shape, out.as_deref_mut())
        } else {
            self.embed_cpu_host(batch, seq, shape, out.as_deref_mut())
        };
        unsafe { buffer_ffi::turbo_buffer_cuda_forward_leave() };
        result
    }

    fn embed_cuda(
        &self,
        batch: usize,
        seq: usize,
        shape: [i64; 2],
        mut out: Option<&mut [f32]>,
    ) -> Result<(usize, Vec<f32>), Error> {
        let dims = self.run_cuda(batch, seq, shape)?;
        let dim = match dims.as_slice() {
            [b, s, d] if *b as usize == batch && *s as usize == seq => *d as usize,
            [b, d] if *b as usize == batch => {
                return Err(format!(
                    "CUDA MiniLM path expects last_hidden [batch, seq, dim], got {dims:?}; \
                     refusing a host D2H of a pre-pooled tensor"
                ));
            }
            _ => {
                return Err(format!(
                    "unexpected CUDA output shape {dims:?} from {:?} (batch={batch}, seq={seq})",
                    self.output_name
                ))
            }
        };
        let need = batch * dim;
        let dest_host = if let Some(dst) = out.as_mut() {
            if dst.len() < need {
                return Err("result buffer is smaller than the pooled batch".into());
            }
            dst.as_mut_ptr()
        } else {
            if self.work.pool_out.ptr.is_null() {
                return Err("PINNED mapped pool_out view is null".into());
            }
            self.work.pool_out.ptr.cast::<f32>()
        };
        let out_dev = mapped_device_ptr(dest_host.cast())?;
        let mask_dev = mapped_device_ptr(self.work.attention_mask.ptr)?;
        device_pool(
            self.pooling,
            self.work.hidden.ptr.cast(),
            mask_dev.cast(),
            out_dev.cast(),
            batch,
            seq,
            dim,
            self.normalize,
        )?;
        note_result_host_read(need * std::mem::size_of::<f32>());
        if out.is_some() {
            Ok((dim, Vec::new()))
        } else {
            let host = f32_view(&self.work.pool_out, need)?;
            Ok((dim, host.to_vec()))
        }
    }

    fn embed_cpu_host(
        &self,
        batch: usize,
        seq: usize,
        shape: [i64; 2],
        mut out: Option<&mut [f32]>,
    ) -> Result<(usize, Vec<f32>), Error> {
        let dims = self.run_cpu(batch, seq, shape)?;
        let hidden_n = batch * seq * self.work.hidden_dim;
        let hidden = f32_view(&self.work.hidden, hidden_n)?;
        let mask = i64_view(&self.work.attention_mask, batch * self.work.max_seq)?;
        let mask = &mask[..batch * seq];

        let dim = match (self.pooling, dims.as_slice()) {
            (Pooling::Mean, [b, s, d]) if *b as usize == batch && *s as usize == seq => *d as usize,
            (Pooling::Cls, [b, s, d]) if *b as usize == batch && *s as usize == seq => *d as usize,
            (_, [b, d]) if *b as usize == batch => *d as usize,
            _ => {
                return Err(format!(
                    "unexpected output shape {dims:?} from {:?} (batch={batch}, seq={seq})",
                    self.output_name
                ))
            }
        };

        let need = batch * dim;
        if let Some(dst) = out.as_mut() {
            if dst.len() < need {
                return Err("result buffer is smaller than the pooled batch".into());
            }
            match (self.pooling, dims.as_slice()) {
                (Pooling::Mean, [_, _, _]) => {
                    mean_pool_into(hidden, mask, batch, seq, dim, &mut dst[..need]);
                }
                (Pooling::Cls, [_, _, _]) => {
                    cls_pool_into(hidden, batch, seq, dim, &mut dst[..need]);
                }
                (_, [_, _]) => dst[..need].copy_from_slice(hidden),
                _ => unreachable!(),
            }
            if self.normalize {
                l2_normalize(&mut dst[..need], dim);
            }
            Ok((dim, Vec::new()))
        } else {
            let mut pooled = vec![0.0f32; need];
            match (self.pooling, dims.as_slice()) {
                (Pooling::Mean, [_, _, _]) => {
                    mean_pool_into(hidden, mask, batch, seq, dim, &mut pooled);
                }
                (Pooling::Cls, [_, _, _]) => {
                    cls_pool_into(hidden, batch, seq, dim, &mut pooled);
                }
                (_, [_, _]) => pooled.copy_from_slice(hidden),
                _ => unreachable!(),
            }
            if self.normalize {
                l2_normalize(&mut pooled, dim);
            }
            Ok((dim, pooled))
        }
    }

    fn bind_token_input(
        &self,
        binding: &mut IoBinding,
        name: &str,
        host: *mut c_void,
        shape: [i64; 2],
    ) -> Result<Tensor<i64>, Error> {
        let (info, data) = if self.place.uses_cuda_buffers() {
            let dev = mapped_device_ptr(host)?;
            let info = MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(|e| format!("CUDA token MemoryInfo: {e}"))?;
            (info, dev)
        } else {
            let info = MemoryInfo::new(
                AllocationDevice::CPU,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(|e| format!("CPU token MemoryInfo: {e}"))?;
            (info, host)
        };
        let tensor = tensor_from_data::<i64>(&info, data.cast(), Shape::new(shape))?;
        if self.place.uses_cuda_buffers() {
            require_cuda_device(tensor.memory_info(), &format!("input {name}"))?;
        }
        binding
            .bind_input(name, &tensor)
            .map_err(|e| format!("IoBinding bind_input {name}: {e}"))?;
        Ok(tensor)
    }

    fn run_cuda(&self, batch: usize, seq: usize, shape: [i64; 2]) -> Result<Vec<i64>, Error> {
        let mut session = self
            .session
            .lock()
            .map_err(|_| "ort session mutex poisoned".to_string())?;
        let mut binding = session
            .create_binding()
            .map_err(|e| format!("IoBinding create failed: {e}"))?;

        let mut held = Vec::new();
        for name in &self.input_names {
            let host = match name.as_str() {
                "input_ids" => self.work.input_ids.ptr,
                "attention_mask" => self.work.attention_mask.ptr,
                "token_type_ids" => self.work.token_type_ids.ptr,
                _ => unreachable!("input names validated at load"),
            };
            held.push(self.bind_token_input(&mut binding, name, host, shape)?);
        }

        let hidden_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| format!("CUDA hidden MemoryInfo: {e}"))?;
        let hidden_shape = Shape::new([batch as i64, seq as i64, self.work.hidden_dim as i64]);
        let hidden_tensor =
            tensor_from_data::<f32>(&hidden_info, self.work.hidden.ptr.cast(), hidden_shape)?;
        require_cuda_device(
            hidden_tensor.memory_info(),
            &format!("output {}", self.output_name),
        )?;
        binding
            .bind_output(self.output_name.as_str(), hidden_tensor)
            .map_err(|e| format!("IoBinding bind_output DEVICE hidden: {e}"))?;

        ort_ok((|| {
            ortsys![unsafe RunWithBinding(session.ptr_mut(), ptr::null(), binding.ptr())?];
            Ok(())
        })())
        .map_err(|e| format!("IoBinding CUDA run failed: {e}"))?;
        binding
            .synchronize_outputs()
            .map_err(|e| format!("IoBinding synchronize_outputs: {e}"))?;

        drop(held);
        Ok(vec![batch as i64, seq as i64, self.work.hidden_dim as i64])
    }

    fn run_cpu(&self, batch: usize, seq: usize, shape: [i64; 2]) -> Result<Vec<i64>, Error> {
        let mut session = self
            .session
            .lock()
            .map_err(|_| "ort session mutex poisoned".to_string())?;
        let mut binding = session
            .create_binding()
            .map_err(|e| format!("IoBinding create failed: {e}"))?;

        let mut held = Vec::new();
        for name in &self.input_names {
            let host = match name.as_str() {
                "input_ids" => self.work.input_ids.ptr,
                "attention_mask" => self.work.attention_mask.ptr,
                "token_type_ids" => self.work.token_type_ids.ptr,
                _ => unreachable!("input names validated at load"),
            };
            held.push(self.bind_token_input(&mut binding, name, host, shape)?);
        }

        let hidden_info = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| format!("CPU hidden MemoryInfo: {e}"))?;
        let hidden_shape = Shape::new([batch as i64, seq as i64, self.work.hidden_dim as i64]);
        let hidden_tensor =
            tensor_from_data::<f32>(&hidden_info, self.work.hidden.ptr.cast(), hidden_shape)?;
        binding
            .bind_output(self.output_name.as_str(), hidden_tensor)
            .map_err(|e| format!("IoBinding bind_output HOST hidden: {e}"))?;

        ort_ok((|| {
            ortsys![unsafe RunWithBinding(session.ptr_mut(), ptr::null(), binding.ptr())?];
            Ok(())
        })())
        .map_err(|e| format!("ORT CPU IoBinding run failed: {e}"))?;

        drop(held);

        Ok(vec![batch as i64, seq as i64, self.work.hidden_dim as i64])
    }
}

impl Drop for OrtCudaSession {
    fn drop(&mut self) {
        clear_ext_arena_if(self.work.arena);
    }
}

pub fn hot_path_reset() {
    ORT_EXT_ALLOCS.store(0, Ordering::Relaxed);
    ORT_D2H_BYTES.store(0, Ordering::Relaxed);
    ORT_D2H_CALLS.store(0, Ordering::Relaxed);
    ORT_D2H_RESULT_BYTES.store(0, Ordering::Relaxed);
    ORT_RESULT_HOST_BYTES.store(0, Ordering::Relaxed);
    unsafe {
        buffer_ffi::turbo_buffer_alloc_counter_reset();
        buffer_ffi::turbo_buffer_cuda_forward_allocs_reset();
        buffer_ffi::turbo_buffer_cuda_forward_h2d_reset();
    }
}

pub fn external_allocs() -> u64 {
    ORT_EXT_ALLOCS.load(Ordering::Relaxed)
}

pub fn external_last_bytes() -> u64 {
    ORT_EXT_LAST_BYTES.load(Ordering::Relaxed)
}

pub fn d2h_bytes() -> u64 {
    ORT_D2H_BYTES.load(Ordering::Relaxed)
}

pub fn d2h_calls() -> u64 {
    ORT_D2H_CALLS.load(Ordering::Relaxed)
}

pub fn d2h_result_bytes() -> u64 {
    ORT_D2H_RESULT_BYTES.load(Ordering::Relaxed)
}

pub fn result_host_bytes() -> u64 {
    ORT_RESULT_HOST_BYTES.load(Ordering::Relaxed)
}

pub fn arena_allocs() -> u64 {
    unsafe { buffer_ffi::turbo_buffer_alloc_counter() }
}

pub fn cuda_forward_allocs() -> u64 {
    unsafe { buffer_ffi::turbo_buffer_cuda_forward_allocs() }
}

pub fn cuda_forward_h2d_bytes() -> u64 {
    unsafe { buffer_ffi::turbo_buffer_cuda_forward_h2d_bytes() }
}

pub fn cuda_forward_h2d_calls() -> u64 {
    unsafe { buffer_ffi::turbo_buffer_cuda_forward_h2d_calls() }
}
