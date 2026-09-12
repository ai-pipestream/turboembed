//! NVIDIA provider: ONNX Runtime CUDA EP + IoBinding device buffers.
//!
//! Fail-loud contract:
//! - CUDA EP is registered with `error_on_failure` (no silent CPU fallback).
//! - A CUDA device allocator is created at load; failure means the EP is not live.
//! - Inputs are copied onto `AllocationDevice::CUDA` before bind.
//! - Outputs are bound to CUDA via `bind_output_to_device`.
//! - After `run_binding`, the output tensor's allocation device must be CUDA.
//!   A CPU-resident output is treated as a fake-CUDA session and errors.
//!
//! Pooling is the sentence-transformers MiniLM recipe: attention-mask-weighted
//! mean over tokens, then L2 normalize. That math runs on the host after the
//! hidden-state tensor is copied back from device memory.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use inferstream_backend_ort::pool::{cls_pool, l2_normalize, mean_pool};
use inferstream_backend_ort::Pooling;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::catalog::CatalogModelSpec;
use crate::error::Error;

const DEFAULT_MAX_SEQ_LEN: usize = 512;

pub struct OrtCudaSession {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    pooling: Pooling,
    normalize: bool,
    input_names: Vec<String>,
    output_name: String,
    /// Proven at load: CUDA device allocator exists for this session.
    #[allow(dead_code)]
    cuda_allocator: Allocator,
}

fn fail_load(what: &str, detail: impl std::fmt::Display) -> Error {
    Error::unavailable(format!("{what}: {detail}"))
}

fn find_tokenizer(model_path: &str, tokenizer_dir: Option<&str>) -> Result<PathBuf, Error> {
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
        return Err(Error::unavailable(format!(
            "{what} landed on allocation device {:?} (cpu_accessible={}); \
             expected CUDA. This is a CPU fallback, not a real NVIDIA embed. \
             Check LD_LIBRARY_PATH (.libs/nvidia/lib) and --features ort-cuda",
            device.as_str(),
            info.is_cpu_accessible()
        )));
    }
    if info.is_cpu_accessible() {
        return Err(Error::unavailable(format!(
            "{what} is CUDA-named but CPU-accessible; refusing a pinned/host \
             buffer pretending to be a device buffer"
        )));
    }
    Ok(())
}

impl OrtCudaSession {
    pub fn load(spec: &CatalogModelSpec) -> Result<Self, Error> {
        if !spec.backend.eq_ignore_ascii_case("ort") {
            return Err(Error::unavailable(format!(
                "nvidia turboembed requires catalog backend=\"ort\", got {:?}",
                spec.backend
            )));
        }
        let device = spec.device.as_deref().unwrap_or("");
        if !device.eq_ignore_ascii_case("cuda") {
            return Err(Error::unavailable(format!(
                "nvidia turboembed requires catalog device=\"cuda\", got {device:?}; \
                 refusing CPU or any other EP"
            )));
        }
        let model_path = spec.path.as_deref().ok_or_else(|| {
            Error::invalid("catalog nvidia entry is missing path to the .onnx file")
        })?;
        if !Path::new(model_path).is_file() {
            return Err(fail_load(
                "onnx model not found",
                format!("{model_path} is not a file"),
            ));
        }

        let tokenizer_file = find_tokenizer(model_path, spec.tokenizer_dir.as_deref())?;
        let mut tokenizer = Tokenizer::from_file(&tokenizer_file)
            .map_err(|e| fail_load("failed to load tokenizer", e))?;
        let max_len = spec
            .max_seq_len
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_MAX_SEQ_LEN);
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: max_len,
                ..Default::default()
            }))
            .map_err(|e| fail_load("failed to configure truncation", e))?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        let pooling = spec
            .pooling
            .as_deref()
            .map(Pooling::from_config)
            .transpose()
            .map_err(|e| Error::invalid(e.to_string()))?
            .unwrap_or(Pooling::Mean);

        // error_on_failure: registration failure is a hard error, never a
        // silent fall-through to CPUExecutionProvider.
        let mut builder = Session::builder()
            .map_err(|e| fail_load("failed to create ort session builder", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| fail_load("failed to configure ort session", e))?
            .with_execution_providers([ort::ep::CUDA::default()
                .with_device_id(0)
                .build()
                .error_on_failure()])
            .map_err(|e| {
                fail_load(
                    "CUDA execution provider unavailable (no silent CPU fallback)",
                    e,
                )
            })?;

        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| fail_load("failed to load onnx model on CUDA EP", e))?;

        // Creating a CUDA device allocator fails if the EP is not actually live.
        let cuda_mem = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| fail_load("CUDA MemoryInfo", e))?;
        let cuda_allocator = Allocator::new(&session, cuda_mem).map_err(|e| {
            fail_load(
                "CUDA device allocator unavailable — CUDA EP is not live",
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
            return Err(Error::unavailable(format!(
                "ort GPU device id {gpu_id} is not a real CUDA device"
            )));
        }

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

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            pooling,
            normalize: spec.normalize.unwrap_or(true),
            input_names,
            output_name,
            cuda_allocator,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>, Error> {
        let (dim, values) = self.embed_batch(&[text.to_string()])?;
        debug_assert_eq!(values.len(), dim);
        Ok(values)
    }

    fn embed_batch(&self, texts: &[String]) -> Result<(usize, Vec<f32>), Error> {
        let batch = texts.len();
        if batch == 0 {
            return Err(Error::invalid("embed requires at least one text"));
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::invalid(format!("tokenization failed: {e}")))?;
        let seq = encodings.first().map(|e| e.len()).unwrap_or(0);
        if seq == 0 {
            return Err(Error::invalid("tokenization produced an empty sequence"));
        }

        let mut input_ids = Vec::with_capacity(batch * seq);
        let mut attention_mask = Vec::with_capacity(batch * seq);
        let mut token_type_ids = Vec::with_capacity(batch * seq);
        for encoding in &encodings {
            input_ids.extend(encoding.get_ids().iter().map(|&v| v as i64));
            attention_mask.extend(encoding.get_attention_mask().iter().map(|&v| v as i64));
            token_type_ids.extend(encoding.get_type_ids().iter().map(|&v| v as i64));
        }

        let shape = [batch as i64, seq as i64];
        let mut cuda_inputs = Vec::new();
        for name in &self.input_names {
            let data = match name.as_str() {
                "input_ids" => input_ids.clone(),
                "attention_mask" => attention_mask.clone(),
                "token_type_ids" => token_type_ids.clone(),
                _ => unreachable!("input names validated at load"),
            };
            let host = Tensor::from_array((shape, data))
                .map_err(|e| Error::internal(format!("host tensor build failed: {e}")))?;
            // Host → CUDA device buffer. `.to` uses ORT IoBinding internally
            // and fails if the CUDA EP cannot receive the copy.
            let device = host.to(AllocationDevice::CUDA, 0).map_err(|e| {
                Error::unavailable(format!(
                    "host→CUDA copy for {name} failed (not a real CUDA session): {e}"
                ))
            })?;
            require_cuda_device(device.memory_info(), &format!("input {name}"))?;
            cuda_inputs.push((name.clone(), device));
        }

        let cuda_out_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| Error::internal(format!("CUDA output MemoryInfo: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| Error::internal("ort session mutex poisoned"))?;
        let mut binding = session
            .create_binding()
            .map_err(|e| Error::internal(format!("IoBinding create failed: {e}")))?;
        for (name, tensor) in &cuda_inputs {
            binding
                .bind_input(name.as_str(), tensor)
                .map_err(|e| Error::internal(format!("IoBinding bind_input {name}: {e}")))?;
        }
        binding
            .bind_output_to_device(&self.output_name, &cuda_out_info)
            .map_err(|e| Error::internal(format!("IoBinding bind_output_to_device CUDA: {e}")))?;

        let mut outputs = session
            .run_binding(&binding)
            .map_err(|e| Error::internal(format!("IoBinding CUDA run failed: {e}")))?;
        binding
            .synchronize_outputs()
            .map_err(|e| Error::internal(format!("IoBinding synchronize_outputs: {e}")))?;

        let hidden_gpu = outputs.remove(self.output_name.as_str()).ok_or_else(|| {
            Error::internal(format!("missing output {:?}", self.output_name))
        })?;
        require_cuda_device(
            hidden_gpu.memory_info(),
            &format!("output {}", self.output_name),
        )?;

        // Device → host for mean/L2. The graph ran on CUDA; pooling is the
        // documented MiniLM host reduction (mask-weighted mean + L2).
        let hidden_cpu = hidden_gpu.to(AllocationDevice::CPU, 0).map_err(|e| {
            Error::internal(format!("CUDA→CPU copy of hidden states failed: {e}"))
        })?;
        let (out_shape, hidden) = hidden_cpu
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::internal(format!("output extraction failed: {e}")))?;
        let dims: Vec<i64> = out_shape.iter().copied().collect();

        let mut pooled = match (self.pooling, dims.as_slice()) {
            (Pooling::Mean, [b, s, d]) if *b as usize == batch && *s as usize == seq => {
                mean_pool(hidden, &attention_mask, batch, seq, *d as usize)
            }
            (Pooling::Cls, [b, s, d]) if *b as usize == batch && *s as usize == seq => {
                cls_pool(hidden, batch, seq, *d as usize)
            }
            (_, [b, _d]) if *b as usize == batch => hidden.to_vec(),
            _ => {
                return Err(Error::internal(format!(
                    "unexpected output shape {dims:?} from {:?} (batch={batch}, seq={seq})",
                    self.output_name
                )))
            }
        };
        let dim = pooled.len() / batch;
        if self.normalize {
            l2_normalize(&mut pooled, dim);
        }
        Ok((dim, pooled))
    }
}
