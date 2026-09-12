//! turboembed — shared C ABI (`include/turboembed.h`) for catalog embeds.
//!
//! NVIDIA: ORT CUDA EP + IoBinding device buffers behind `--features ort-cuda`.
//! Catalog aliases never resolve to a mock. Missing the real feature is an
//! error. Zero Python.

pub mod catalog;
pub mod engine;
pub mod error;
pub mod ffi;

#[cfg(feature = "ort-cuda")]
mod ort_cuda;

pub use catalog::{Arch, Catalog, CatalogError, CatalogModelSpec};
pub use engine::{Device, Engine};
pub use error::Error;

/// Cosine similarity of two equal-length vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_minilm_is_not_mock() {
        let catalog = Catalog::builtin();
        let spec = catalog.resolve_embed("minilm", Arch::Nvidia).unwrap();
        assert_ne!(spec.backend, "mock");
        assert_eq!(spec.device.as_deref(), Some("cuda"));
    }

    #[cfg(not(feature = "ort-cuda"))]
    #[test]
    fn nvidia_engine_errors_without_ort_cuda_feature() {
        let err = Engine::open(Arch::Nvidia).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("ort-cuda"),
            "expected feature name in error, got {msg}"
        );
        assert!(
            !msg.to_ascii_lowercase().contains("mock path for catalog")
                || msg.contains("no mock"),
            "{msg}"
        );
    }

    #[cfg(not(feature = "ort-cuda"))]
    #[test]
    fn intel_and_apple_do_not_pretend() {
        for arch in [Arch::Intel, Arch::Apple] {
            let msg = Engine::open(arch).unwrap_err().to_string();
            assert!(
                msg.contains("will not fall back to mock"),
                "{arch:?}: {msg}"
            );
        }
    }

    #[test]
    fn cosine_identical_is_one() {
        let v = [0.3_f32, 0.4, 0.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }
}
