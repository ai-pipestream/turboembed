//! Real GenAI `TextEmbeddingPipeline` (crate feature `genai`).

use std::path::Path;
use std::sync::Mutex;

use inferstream_backend::BackendError;

use crate::ffi::ffi;
use crate::{
    embedding_dim_from_config, missing_ir_files, resolve_device, Embedder, OpenVinoConfig, Pooling,
};

pub(crate) struct GenaiEmbedder {
    pipe: Mutex<cxx::UniquePtr<ffi::Pipeline>>,
    models_path: String,
    device: String,
    pooling: Pooling,
    normalize: bool,
    dim: Option<usize>,
}

fn load_error(what: &str, detail: impl std::fmt::Display) -> BackendError {
    BackendError::Unavailable(format!("{what}: {detail}"))
}

impl GenaiEmbedder {
    pub(crate) fn load(config: OpenVinoConfig) -> Result<Self, BackendError> {
        if config.models_path.is_empty() {
            return Err(BackendError::InvalidRequest(
                "openvino backend requires `path` pointing at an OpenVINO GenAI \
                 model directory (openvino_model.xml + openvino_tokenizer.xml)"
                    .into(),
            ));
        }
        let dir = Path::new(&config.models_path);
        if !dir.is_dir() {
            return Err(load_error(
                "OpenVINO GenAI model directory not found",
                format!("{} is not a directory", config.models_path),
            ));
        }
        let missing = missing_ir_files(dir);
        if !missing.is_empty() {
            return Err(load_error(
                "OpenVINO GenAI model directory is incomplete",
                format!(
                    "{} is missing {missing:?}; need the GenAI layout \
                     (openvino_model.xml/.bin + openvino_tokenizer.xml/.bin). \
                     Fetch with `make fetch-ov-genai` or see docs/intel-genai-embed.md",
                    config.models_path
                ),
            ));
        }

        let available = ffi::available_devices().map_err(|e| {
            load_error(
                "failed to query OpenVINO devices",
                format!("{e} (is OpenVINO on the loader path?)"),
            )
        })?;
        let device = resolve_device(config.device, &available)?;
        let normalize = config.normalize.unwrap_or(true);
        let max_length = config.max_seq_len.unwrap_or(0) as u32;

        tracing::info!(
            models_path = %config.models_path,
            device = %device,
            available = ?available,
            pooling = config.pooling.as_str(),
            normalize,
            "loading OpenVINO GenAI TextEmbeddingPipeline"
        );

        let pipe = ffi::load_pipeline(
            &config.models_path,
            &device,
            config.pooling.as_genai_u8(),
            normalize,
            max_length,
            max_length > 0,
        )
        .map_err(|e| load_error("failed to construct TextEmbeddingPipeline", format!("{e}")))?;

        let dim = embedding_dim_from_config(dir);
        Ok(Self {
            pipe: Mutex::new(pipe),
            models_path: config.models_path,
            device,
            pooling: config.pooling,
            normalize,
            dim,
        })
    }
}

impl Embedder for GenaiEmbedder {
    fn embed(&self, texts: &[String]) -> Result<(usize, Vec<f32>), BackendError> {
        if texts.is_empty() {
            return Err(BackendError::InvalidRequest(
                "input \"text\" contained no elements".into(),
            ));
        }
        let mut guard = self
            .pipe
            .lock()
            .map_err(|_| BackendError::Internal("genai pipeline mutex poisoned".into()))?;
        let pipe = guard
            .as_ref()
            .ok_or_else(|| BackendError::Internal("genai pipeline handle is null".into()))?;
        let owned: Vec<String> = texts.to_vec();
        let flat = pipe
            .embed_documents(&owned)
            .map_err(|e| BackendError::Internal(format!("TextEmbeddingPipeline failed: {e}")))?;
        let dim = if !flat.is_empty() {
            let d = flat.len().checked_div(texts.len()).unwrap_or(0);
            if d == 0 || flat.len() != d * texts.len() {
                return Err(BackendError::Internal(format!(
                    "ragged GenAI embedding (len={}, batch={})",
                    flat.len(),
                    texts.len()
                )));
            }
            d
        } else {
            pipe.embedding_dim()
        };
        if dim == 0 {
            return Err(BackendError::Internal(
                "TextEmbeddingPipeline returned empty embeddings".into(),
            ));
        }
        Ok((dim, flat))
    }

    fn device(&self) -> &str {
        &self.device
    }

    fn pooling(&self) -> Pooling {
        self.pooling
    }

    fn normalize(&self) -> bool {
        self.normalize
    }

    fn models_path(&self) -> &str {
        &self.models_path
    }

    fn embedding_dim(&self) -> Option<usize> {
        self.dim
    }
}
