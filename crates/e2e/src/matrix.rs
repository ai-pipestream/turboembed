//! Which aliases each arch must support, plus expected embedding dims.
//!
//! Override with `--matrix path.json` / `INFERSTREAM_E2E_MATRIX` when a host
//! serves a subset (or extra) aliases. The built-in matrix matches
//! `config/catalog.toml` + the README alias table.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::target::Target;

const BUILTIN_JSON: &str = include_str!("../../../testdata/e2e/matrix.json");

/// One embedding alias in the matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedAlias {
    pub alias: String,
    pub dim: u32,
    #[serde(default)]
    pub required: bool,
    pub arches: Vec<String>,
}

/// One generative LLM alias in the matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmAlias {
    pub alias: String,
    /// Hard-fail if this alias is missing from ListModels. Default false:
    /// skip when the host did not put it on `serve`.
    #[serde(default)]
    pub required: bool,
    pub arches: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Matrix {
    pub embeds: Vec<EmbedAlias>,
    pub llms: Vec<LlmAlias>,
}

impl Matrix {
    pub fn builtin() -> Self {
        serde_json::from_str(BUILTIN_JSON)
            .expect("testdata/e2e/matrix.json must parse; covered by unit test")
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| e.to_string())?;
        serde_json::from_str(&text).map_err(|e| e.to_string())
    }

    pub fn embed(&self, alias: &str) -> Option<&EmbedAlias> {
        self.embeds.iter().find(|e| e.alias == alias)
    }

    pub fn llm(&self, alias: &str) -> Option<&LlmAlias> {
        self.llms.iter().find(|e| e.alias == alias)
    }

    pub fn embed_on_target(&self, target: Target) -> impl Iterator<Item = &EmbedAlias> {
        self.embeds
            .iter()
            .filter(move |e| target.is_mock() || e.arches.iter().any(|a| a == target.as_str()))
    }

    pub fn llm_on_target(&self, target: Target) -> impl Iterator<Item = &LlmAlias> {
        self.llms
            .iter()
            .filter(move |e| target.is_mock() || e.arches.iter().any(|a| a == target.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_minilm_required_everywhere() {
        let m = Matrix::builtin();
        let minilm = m.embed("minilm").expect("minilm in matrix");
        assert!(minilm.required);
        assert_eq!(minilm.dim, 384);
        for arch in ["nvidia", "intel", "apple"] {
            assert!(minilm.arches.iter().any(|a| a == arch), "{arch}");
        }
    }

    #[test]
    fn builtin_mpnet_not_on_apple() {
        let m = Matrix::builtin();
        let mpnet = m.embed("mpnet").expect("mpnet");
        assert!(!mpnet.required);
        assert_eq!(mpnet.dim, 768);
        assert!(!mpnet.arches.iter().any(|a| a == "apple"));
        assert!(m.embed_on_target(Target::Apple).all(|e| e.alias != "mpnet"));
        assert!(m
            .embed_on_target(Target::Nvidia)
            .any(|e| e.alias == "mpnet"));
    }

    #[test]
    fn builtin_qwen_05b_on_intel() {
        let m = Matrix::builtin();
        let q = m.llm("qwen-0.5b").expect("qwen-0.5b");
        assert!(q.arches.iter().any(|a| a == "intel"));
        assert!(!q.required, "skip if this host did not serve it");
    }
}
