//! Optional per-arch golden embeddings. Missing files are not an error.

use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Schema matches `testdata/reference_embeddings/` so existing goldens can
/// be copied under `testdata/e2e/goldens/<arch>/<alias>.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct GoldenEmbedding {
    #[serde(default)]
    pub model: Option<String>,
    pub text: String,
    pub dim: u32,
    #[serde(default)]
    pub vector: Option<Vec<f32>>,
    #[serde(default)]
    pub vector_head: Option<Vec<f32>>,
    #[serde(default)]
    pub vector_tail: Option<Vec<f32>>,
}

pub fn load_golden(path: impl AsRef<Path>) -> Result<GoldenEmbedding, String> {
    let text = std::fs::read_to_string(path.as_ref()).map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

/// `{goldens_dir}/{target}/{alias}.json` when that file exists.
pub fn lookup_golden(goldens_dir: &Path, target: &str, alias: &str) -> Option<PathBuf> {
    let path = goldens_dir.join(target).join(format!("{alias}.json"));
    path.is_file().then_some(path)
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Compare `got` to a golden. Full `vector` when present; otherwise
/// head/tail slices (same convention as the reference-embedding README).
pub fn golden_cosine(got: &[f32], golden: &GoldenEmbedding) -> Result<f32, String> {
    if got.len() != golden.dim as usize {
        return Err(format!(
            "dim mismatch: got {} expected {}",
            got.len(),
            golden.dim
        ));
    }
    if let Some(ref vector) = golden.vector {
        if vector.len() != got.len() {
            return Err(format!(
                "golden vector length {} != dim {}",
                vector.len(),
                golden.dim
            ));
        }
        return Ok(cosine(got, vector));
    }
    let head = golden
        .vector_head
        .as_deref()
        .ok_or_else(|| "golden has neither vector nor vector_head".to_string())?;
    let tail = golden
        .vector_tail
        .as_deref()
        .ok_or_else(|| "golden has vector_head but no vector_tail".to_string())?;
    if head.len() > got.len() || tail.len() > got.len() {
        return Err("golden head/tail longer than embedding".into());
    }
    let head_c = cosine(&got[..head.len()], head);
    let tail_c = cosine(&got[got.len() - tail.len()..], tail);
    Ok(head_c.min(tail_c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one() {
        let v = [0.2f32, 0.0, 0.4];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn golden_full_vector() {
        let g = GoldenEmbedding {
            model: Some("minilm".into()),
            text: "hello".into(),
            dim: 2,
            vector: Some(vec![1.0, 0.0]),
            vector_head: None,
            vector_tail: None,
        };
        let score = golden_cosine(&[1.0, 0.0], &g).unwrap();
        assert!((score - 1.0).abs() < 1e-6);
    }
}
