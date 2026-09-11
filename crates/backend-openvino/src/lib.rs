//! OpenVINO backend stub for inferstream.
//!
//! Targets Intel CPU / integrated & discrete GPU (Arc/Battlemage) / NPU
//! through the OpenVINO runtime (via the `openvino` Rust bindings). This is
//! a *backend* inside the inferstream façade, not a dependency on OpenVINO
//! Model Server — inferstream owns the gRPC surface, auth, and routing;
//! OpenVINO only executes graphs.
//!
//! **Environment:** the OpenVINO runtime resolves its plugins at load time.
//! On oneAPI installs, `source /opt/intel/oneapi/setvars.sh` (or the
//! standalone OpenVINO `setupvars.sh`) must run in the shell/service unit
//! that builds and launches `inferstream-intel`, and GPU/NPU targets need the
//! Level Zero + compute-runtime packages on the host.
//!
//! On `inferstream-intel`, this backend competes with llama.cpp-SYCL in the
//! planned Battlemage bake-off (see README); both stay feature-gated so the
//! bake-off is a config change, not a rebuild of the façade.
//!
//! This crate currently compiles without the OpenVINO runtime and reports
//! `Unavailable` at runtime.

use async_trait::async_trait;
use inferstream_backend::{Backend, BackendError, ModelMetadata};
use inferstream_protocol::inference::{ModelInferRequest, ModelInferResponse};

/// Stub OpenVINO backend.
#[derive(Debug, Default, Clone)]
pub struct OpenVinoBackend {
    /// Filesystem path to the model (IR `.xml` or ONNX) this backend would load.
    pub model_path: Option<String>,
    /// OpenVINO device string, e.g. `"CPU"`, `"GPU"`, `"NPU"`, `"AUTO"`.
    pub device: Option<String>,
}

impl OpenVinoBackend {
    pub fn new(model_path: Option<String>, device: Option<String>) -> Self {
        Self { model_path, device }
    }

    fn unavailable() -> BackendError {
        BackendError::Unavailable(
            "OpenVINO backend is a stub in v0.1; no compiled model is loaded".into(),
        )
    }
}

#[async_trait]
impl Backend for OpenVinoBackend {
    fn id(&self) -> &str {
        "openvino"
    }

    async fn model_ready(&self, _model_name: &str, _model_version: &str) -> bool {
        false
    }

    async fn model_metadata(
        &self,
        _model_name: &str,
        _model_version: &str,
    ) -> Result<ModelMetadata, BackendError> {
        Err(Self::unavailable())
    }

    async fn infer(&self, _request: ModelInferRequest) -> Result<ModelInferResponse, BackendError> {
        Err(Self::unavailable())
    }
}
