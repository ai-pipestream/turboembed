//! NVIDIA provider: ONNX Runtime CUDA EP + IoBinding, or an explicit CPU EP.
//!
//! Device policy:
//! - **CUDA** (and AUTO): CUDA EP with `error_on_failure`. No silent CPU
//!   fallback. Device allocator + IoBinding outputs must reside on
//!   `AllocationDevice::CUDA` and must not be CPU-accessible.
//! - **CPU**: only when the ABI device is explicitly CPU. Host tensors +
//!   `Session::run`. Same mean+L2 pooling. This is not a CUDA fallback.
//!
//! Pooling is the sentence-transformers MiniLM recipe: attention-mask-weighted
//! mean over tokens, then L2 normalize. On CUDA that math runs on the host
//! after the hidden-state tensor is copied back from device memory.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use inferstream_backend_ort::pool::{cls_pool, l2_normalize, mean_pool};
use inferstream_backend_ort::Pooling;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::catalog::CatalogModelSpec;

type Error = String;

const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// Where this session is allowed to run. CUDA never becomes CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrtPlace {
    Cuda,
    Cpu,
}

impl OrtPlace {
    /// ABI `turboembed_device`: AUTO(0) and CUDA(2) → CUDA; CPU(1) → CPU.
    pub fn from_abi(device: i32) -> Result<Self, Error> {
        match device {
            1 => Ok(Self::Cpu),
            0 | 2 => Ok(Self::Cuda),
            other => Err(format!(
                "ORT path does not handle ABI device {other}; \
                 use TURBOEMBED_DEVICE_CUDA / AUTO or TURBOEMBED_DEVICE_CPU"
            )),
        }
    }
}

pub struct OrtCudaSession {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    pooling: Pooling,
    normalize: bool,
    input_names: Vec<String>,
    output_name: String,
    place: OrtPlace,
    /// Proven at CUDA load: device allocator exists for this session.
    #[allow(dead_code)]
    cuda_allocator: Option<Allocator>,
    /// Sentence-embedding width after pooling (set by a warmup at load).
    embedding_dim: usize,
}

fn fail_load(what: &str, detail: impl std::fmt::Display) -> Error {
    format!("{what}: {detail}")
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

fn resolve_path(workspace_root: &Path, raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        workspace_root.join(path)
    }
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

    pub fn on_cuda(&self) -> bool {
        self.place == OrtPlace::Cuda
    }

    pub fn load(
        spec: &CatalogModelSpec,
        workspace_root: &Path,
        place: OrtPlace,
    ) -> Result<Self, Error> {
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
        let tokenizer_file = find_tokenizer(model_path, tokenizer_hint.as_deref())?;
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
            .map_err(|e| e.to_string())?
            .unwrap_or(Pooling::Mean);

        let mut builder = Session::builder()
            .map_err(|e| fail_load("failed to create ort session builder", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| fail_load("failed to configure ort session", e))?;

        if place == OrtPlace::Cuda {
            // error_on_failure: registration failure is a hard error,
            // never a silent fall-through to CPUExecutionProvider.
            builder = builder
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
        }

        let session = builder.commit_from_file(model_path).map_err(|e| {
            fail_load(
                match place {
                    OrtPlace::Cuda => "failed to load onnx model on CUDA EP",
                    OrtPlace::Cpu => "failed to load onnx model on CPU EP",
                },
                e,
            )
        })?;

        let cuda_allocator = match place {
            OrtPlace::Cuda => {
                // Creating a CUDA device allocator fails if the EP is not live.
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

        let mut loaded = Self {
            session: Mutex::new(session),
            tokenizer,
            pooling,
            normalize: spec.normalize.unwrap_or(true),
            input_names,
            output_name,
            place,
            cuda_allocator,
            embedding_dim: 0,
        };
        // One inference pass at load: prove the requested EP + set dim.
        let (dim, _) = loaded.embed_batch(&[String::from("x")])?;
        if dim == 0 {
            return Err(format!(
                "{:?} warmup produced embedding dim 0",
                loaded.place
            ));
        }
        loaded.embedding_dim = dim;
        Ok(loaded)
    }

    pub(crate) fn embed_batch(&self, texts: &[String]) -> Result<(usize, Vec<f32>), Error> {
        let batch = texts.len();
        if batch == 0 {
            return Err("embed requires at least one text".into());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| format!("tokenization failed: {e}"))?;
        let seq = encodings.first().map(|e| e.len()).unwrap_or(0);
        if seq == 0 {
            return Err("tokenization produced an empty sequence".into());
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
        let dims = match self.place {
            OrtPlace::Cuda => self.run_cuda(&input_ids, &attention_mask, &token_type_ids, shape)?,
            OrtPlace::Cpu => self.run_cpu(&input_ids, &attention_mask, &token_type_ids, shape)?,
        };
        let (dims, hidden) = dims;

        let mut pooled = match (self.pooling, dims.as_slice()) {
            (Pooling::Mean, [b, s, d]) if *b as usize == batch && *s as usize == seq => {
                mean_pool(hidden, &attention_mask, batch, seq, *d as usize)
            }
            (Pooling::Cls, [b, s, d]) if *b as usize == batch && *s as usize == seq => {
                cls_pool(hidden, batch, seq, *d as usize)
            }
            (_, [b, _d]) if *b as usize == batch => hidden.to_vec(),
            _ => {
                return Err(format!(
                    "unexpected output shape {dims:?} from {:?} (batch={batch}, seq={seq})",
                    self.output_name
                ))
            }
        };
        let dim = pooled.len() / batch;
        if self.normalize {
            l2_normalize(&mut pooled, dim);
        }
        Ok((dim, pooled))
    }

    fn run_cuda(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        token_type_ids: &[i64],
        shape: [i64; 2],
    ) -> Result<(Vec<i64>, Vec<f32>), Error> {
        let mut cuda_inputs = Vec::new();
        for name in &self.input_names {
            let data = match name.as_str() {
                "input_ids" => input_ids.to_vec(),
                "attention_mask" => attention_mask.to_vec(),
                "token_type_ids" => token_type_ids.to_vec(),
                _ => unreachable!("input names validated at load"),
            };
            let host = Tensor::from_array((shape, data))
                .map_err(|e| format!("host tensor build failed: {e}"))?;
            let device = host.to(AllocationDevice::CUDA, 0).map_err(|e| {
                format!(
                    "host→CUDA copy for {name} failed (CUDA was requested; \
                     CPU is not a fallback): {e}"
                )
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
        .map_err(|e| format!("CUDA output MemoryInfo: {e}"))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| "ort session mutex poisoned".to_string())?;
        let mut binding = session
            .create_binding()
            .map_err(|e| format!("IoBinding create failed: {e}"))?;
        for (name, tensor) in &cuda_inputs {
            binding
                .bind_input(name.as_str(), tensor)
                .map_err(|e| format!("IoBinding bind_input {name}: {e}"))?;
        }
        binding
            .bind_output_to_device(&self.output_name, &cuda_out_info)
            .map_err(|e| format!("IoBinding bind_output_to_device CUDA: {e}"))?;

        let mut outputs = session
            .run_binding(&binding)
            .map_err(|e| format!("IoBinding CUDA run failed: {e}"))?;
        binding
            .synchronize_outputs()
            .map_err(|e| format!("IoBinding synchronize_outputs: {e}"))?;

        let hidden_gpu = outputs
            .remove(self.output_name.as_str())
            .ok_or_else(|| format!("missing output {:?}", self.output_name))?;
        require_cuda_device(
            hidden_gpu.memory_info(),
            &format!("output {}", self.output_name),
        )?;

        let hidden_cpu = hidden_gpu
            .to(AllocationDevice::CPU, 0)
            .map_err(|e| format!("CUDA→CPU copy of hidden states failed: {e}"))?;
        let (out_shape, hidden) = hidden_cpu
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("output extraction failed: {e}"))?;
        Ok((out_shape.iter().copied().collect(), hidden.to_vec()))
    }

    fn run_cpu(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        token_type_ids: &[i64],
        shape: [i64; 2],
    ) -> Result<(Vec<i64>, Vec<f32>), Error> {
        let mut feed: Vec<(&str, SessionInputValue<'_>)> = Vec::new();
        for name in &self.input_names {
            let data = match name.as_str() {
                "input_ids" => input_ids.to_vec(),
                "attention_mask" => attention_mask.to_vec(),
                "token_type_ids" => token_type_ids.to_vec(),
                _ => unreachable!("input names validated at load"),
            };
            let tensor = Tensor::from_array((shape, data))
                .map_err(|e| format!("host tensor build failed: {e}"))?;
            feed.push((name.as_str(), tensor.into()));
        }

        let mut session = self
            .session
            .lock()
            .map_err(|_| "ort session mutex poisoned".to_string())?;
        let outputs = session
            .run(feed)
            .map_err(|e| format!("ORT CPU run failed: {e}"))?;
        let (out_shape, hidden) = outputs[self.output_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("CPU output extraction failed: {e}"))?;
        Ok((out_shape.iter().copied().collect(), hidden.to_vec()))
    }
}
