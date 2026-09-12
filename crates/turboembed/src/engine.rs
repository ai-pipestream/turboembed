use std::path::Path;

use crate::catalog::{Arch, Catalog, CatalogModelSpec};
use crate::error::Error;

#[cfg(feature = "ort-cuda")]
use std::collections::HashMap;
#[cfg(feature = "ort-cuda")]
use std::sync::Mutex;

#[cfg(feature = "ort-cuda")]
use crate::ort_cuda::OrtCudaSession;

/// Live device advertised by a successfully created engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cuda,
}

impl Device {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
        }
    }
}

/// In-process embedder bound to one catalog + arch.
///
/// Catalog aliases never resolve to a mock. The NVIDIA path requires
/// `--features ort-cuda` and a live CUDA EP.
pub struct Engine {
    arch: Arch,
    catalog: Catalog,
    #[cfg(feature = "ort-cuda")]
    sessions: Mutex<HashMap<String, OrtCudaSession>>,
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("arch", &self.arch)
            .field("device", &self.device())
            .finish()
    }
}

impl Engine {
    pub fn open(arch: Arch) -> Result<Self, Error> {
        Self::open_catalog(arch, Catalog::builtin())
    }

    pub fn open_file(arch: Arch, catalog_path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::open_catalog(arch, Catalog::from_file(catalog_path)?)
    }

    fn open_catalog(arch: Arch, catalog: Catalog) -> Result<Self, Error> {
        match arch {
            Arch::Nvidia => {
                #[cfg(not(feature = "ort-cuda"))]
                {
                    let _ = catalog;
                    return Err(Error::unavailable(
                        "catalog alias embeds on nvidia require a real ORT CUDA \
                         provider; rebuild with --features ort-cuda. There is no \
                         mock path for catalog names",
                    ));
                }
                #[cfg(feature = "ort-cuda")]
                {
                    Ok(Self {
                        arch,
                        catalog,
                        sessions: Mutex::new(HashMap::new()),
                    })
                }
            }
            Arch::Intel => Err(Error::unavailable(
                "intel turboembed provider is not in this binary; rebuild with \
                 the OpenVINO GenAI feature on inferstream-intel. Catalog \
                 aliases will not fall back to mock",
            )),
            Arch::Apple => Err(Error::unavailable(
                "apple turboembed provider is not in this binary; rebuild with \
                 the MLX feature on inferstream-apple. Catalog aliases will \
                 not fall back to mock",
            )),
        }
    }

    pub fn arch(&self) -> Arch {
        self.arch
    }

    pub fn device(&self) -> Device {
        Device::Cuda
    }

    /// `embed("minilm", text)` — catalog alias + UTF-8 text → one FP32 vector.
    pub fn embed(&self, alias: &str, text: &str) -> Result<Vec<f32>, Error> {
        let spec = self.catalog.resolve_embed(alias, self.arch)?;
        self.embed_resolved(alias, spec, text)
    }

    #[cfg(not(feature = "ort-cuda"))]
    fn embed_resolved(
        &self,
        _alias: &str,
        _spec: &CatalogModelSpec,
        _text: &str,
    ) -> Result<Vec<f32>, Error> {
        Err(Error::unavailable(
            "catalog alias embeds require --features ort-cuda; no mock",
        ))
    }

    #[cfg(feature = "ort-cuda")]
    fn embed_resolved(
        &self,
        alias: &str,
        spec: &CatalogModelSpec,
        text: &str,
    ) -> Result<Vec<f32>, Error> {
        {
            let sessions = self
                .sessions
                .lock()
                .map_err(|_| Error::internal("session cache mutex poisoned"))?;
            if let Some(session) = sessions.get(alias) {
                return session.embed(text);
            }
        }
        let loaded = OrtCudaSession::load(spec)?;
        let vector = loaded.embed(text)?;
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|_| Error::internal("session cache mutex poisoned"))?;
        sessions.entry(alias.to_string()).or_insert(loaded);
        Ok(vector)
    }
}
