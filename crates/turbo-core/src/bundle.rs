//! Model bundles: a directory with a `bundle.json` manifest (version 2).
//!
//! The manifest is the single source of per-model truth: identity, tokenizer
//! files, the frozen contract (pooling, normalization, sequence limit,
//! prefixes, labels), and one artifact per provider format, each with a
//! SHA-256. `Bundle::open` verifies every listed file before anything is
//! loaded; a mismatch is `TURBO_E_BUNDLE_INTEGRITY` and the bundle is not
//! usable. Providers read the contract; they never infer it from a name.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::types::{Aggregation, DType, Modality, ModelKind, Normalize, Pooling, Task};

/// Manifest version this library reads.
pub const BUNDLE_VERSION: u32 = 2;

/// File name of the manifest inside a bundle directory.
pub const MANIFEST_NAME: &str = "bundle.json";

/// A file entry with its recorded hash.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct FileEntry {
    /// Path relative to the bundle directory. No `..`, no absolute paths.
    pub path: String,
    /// Lower-case hex SHA-256 of the file contents.
    pub sha256: String,
    /// Optional free-text provenance (exporter, conversion command).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<String>,
}

/// Tokenizer section.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct TokenizerSpec {
    /// `wordpiece`, `bpe`, `unigram`, `sentencepiece`, `gguf`, `mock`.
    pub kind: String,
    /// Files by role, for example `"tokenizer.json"`.
    #[serde(default)]
    pub files: BTreeMap<String, FileEntry>,
    /// Chat template (Jinja) for generative models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
}

/// Prompt prefixes.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Prompts {
    /// Prefix for query-role inputs.
    #[serde(default)]
    pub query: String,
    /// Prefix for document-role inputs.
    #[serde(default)]
    pub document: String,
}

/// The frozen model contract.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
pub struct Contract {
    /// `mean`, `cls`, `last`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pooling: Option<String>,
    /// `l2` or `none`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalize: Option<String>,
    /// Maximum sequence length in tokens.
    #[serde(default)]
    pub max_seq: u32,
    /// Embedding dimension.
    #[serde(default)]
    pub dim: u32,
    /// Matryoshka dimensions the model was trained to truncate to.
    #[serde(default)]
    pub truncate_dims: Vec<u32>,
    /// Prompt prefixes.
    #[serde(default)]
    pub prompts: Prompts,
    /// `cosine`, `dot`, `euclidean`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub similarity_fn: Option<String>,
    /// Model compute dtype name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    /// Vocabulary size.
    #[serde(default)]
    pub vocab_size: u32,
    /// Classifier labels in index order.
    #[serde(default)]
    pub labels: Vec<String>,
    /// `softmax`, `sigmoid`, `none`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation: Option<String>,
    /// Token-classification aggregation: `none`, `simple`, `first`, `max`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregation: Option<String>,
    /// Tagging scheme: `BIO`, `BILOU`, `IOB1`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tagging: Option<String>,
}

/// Declared limits.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Limits {
    /// Maximum batch a session may declare; 0 = provider default.
    #[serde(default)]
    pub max_batch: u32,
    /// True when the artifact is compiled for one fixed shape (NPUs).
    #[serde(default)]
    pub fixed_shape: bool,
}

/// The manifest as stored on disk.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Manifest {
    /// Must equal [`BUNDLE_VERSION`].
    pub bundle_version: u32,
    /// Model identifier (for example a Hub id).
    pub model_id: String,
    /// Revision or commit.
    #[serde(default)]
    pub revision: String,
    /// SPDX license of the weights.
    #[serde(default)]
    pub license: String,
    /// Primary task name: `embed`, `rerank`, `classify`, `token_classify`, `generate`, `run`.
    pub task: String,
    /// Model kind name: `embedding`, `reranker`, `classifier`, `token_classifier`, `generative`, `generic`.
    pub kind: String,
    /// `text`, `audio`, `image`, `video`.
    #[serde(default = "default_modality")]
    pub modality: String,
    /// Architecture family, informational.
    #[serde(default)]
    pub family: String,
    /// Tokenizer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<TokenizerSpec>,
    /// Frozen contract.
    #[serde(default)]
    pub contract: Contract,
    /// Artifacts by format: `onnx`, `openvino_ir`, `gguf`, `hef`, `mlx_safetensors`, `static`, `mock`.
    #[serde(default)]
    pub artifacts: BTreeMap<String, FileEntry>,
    /// Limits.
    #[serde(default)]
    pub limits: Limits,
}

fn default_modality() -> String {
    "text".to_string()
}

/// A verified bundle.
#[derive(Clone, Debug)]
pub struct Bundle {
    dir: PathBuf,
    manifest: Manifest,
    task: Task,
    kind: ModelKind,
    modality: Modality,
}

impl Bundle {
    /// Open and verify a bundle directory. Every file the manifest lists is
    /// hashed and compared; the first mismatch fails the open.
    pub fn open(dir: &Path) -> Result<Self> {
        if !dir.is_dir() {
            return Err(Error::bundle_not_found(format!(
                "bundle directory `{}` does not exist or is not a directory",
                dir.display()
            )));
        }
        let manifest_path = dir.join(MANIFEST_NAME);
        let text = std::fs::read_to_string(&manifest_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::bundle_not_found(format!("`{}` is missing", manifest_path.display()))
            } else {
                Error::from(e)
            }
        })?;
        let manifest: Manifest = serde_json::from_str(&text)?;
        Self::from_manifest(dir.to_path_buf(), manifest)
    }

    /// Build from an already-parsed manifest; still verifies files.
    pub fn from_manifest(dir: PathBuf, manifest: Manifest) -> Result<Self> {
        if manifest.bundle_version != BUNDLE_VERSION {
            return Err(Error::bundle_invalid(format!(
                "bundle_version {} is not supported; this library reads {}",
                manifest.bundle_version, BUNDLE_VERSION
            )));
        }
        if manifest.model_id.trim().is_empty() {
            return Err(Error::bundle_invalid("model_id is empty"));
        }
        let task = task_from_name(&manifest.task)?;
        let kind = kind_from_name(&manifest.kind)?;
        let modality = modality_from_name(&manifest.modality)?;
        if manifest.artifacts.is_empty() && manifest.tokenizer.as_ref().is_none_or(|t| t.files.is_empty()) {
            return Err(Error::bundle_invalid("manifest lists no artifacts and no tokenizer files"));
        }
        validate_contract(&manifest, kind)?;

        let bundle = Self { dir, manifest, task, kind, modality };
        bundle.verify_files()?;
        Ok(bundle)
    }

    fn verify_files(&self) -> Result<()> {
        let mut entries: Vec<(&str, &FileEntry)> = Vec::new();
        if let Some(tok) = &self.manifest.tokenizer {
            for (role, entry) in &tok.files {
                entries.push((role, entry));
            }
        }
        for (format, entry) in &self.manifest.artifacts {
            entries.push((format, entry));
        }
        for (label, entry) in entries {
            let path = self.resolve(&entry.path)?;
            let actual = sha256_file(&path)?;
            if !actual.eq_ignore_ascii_case(&entry.sha256) {
                return Err(Error::bundle_integrity(format!(
                    "`{}` ({label}) hashes to {actual} but the manifest records {}",
                    entry.path, entry.sha256
                )));
            }
        }
        Ok(())
    }

    /// Resolve a manifest-relative path, rejecting escapes.
    pub fn resolve(&self, relative: &str) -> Result<PathBuf> {
        let rel = Path::new(relative);
        if rel.is_absolute()
            || rel.components().any(|c| {
                matches!(
                    c,
                    std::path::Component::ParentDir | std::path::Component::RootDir | std::path::Component::Prefix(_)
                )
            })
        {
            return Err(Error::bundle_invalid(format!(
                "manifest path `{relative}` must be relative and inside the bundle"
            )));
        }
        let full = self.dir.join(rel);
        if !full.is_file() {
            return Err(Error::bundle_not_found(format!(
                "manifest lists `{relative}` but `{}` is not a file",
                full.display()
            )));
        }
        Ok(full)
    }

    /// Bundle directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Parsed manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Primary task.
    pub fn task(&self) -> Task {
        self.task
    }

    /// Model kind.
    pub fn kind(&self) -> ModelKind {
        self.kind
    }

    /// Modality.
    pub fn modality(&self) -> Modality {
        self.modality
    }

    /// Frozen contract.
    pub fn contract(&self) -> &Contract {
        &self.manifest.contract
    }

    /// Artifact entry for a format, if the bundle has one.
    pub fn artifact(&self, format: &str) -> Option<&FileEntry> {
        self.manifest.artifacts.get(format)
    }

    /// Absolute path of an artifact, verifying it exists.
    pub fn artifact_path(&self, format: &str) -> Result<PathBuf> {
        let entry = self.artifact(format).ok_or_else(|| {
            Error::bundle_no_artifact(format!(
                "bundle `{}` has no `{format}` artifact; it has: {}",
                self.manifest.model_id,
                self.manifest.artifacts.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })?;
        self.resolve(&entry.path)
    }

    /// Pooling from the contract, if declared.
    pub fn pooling(&self) -> Result<Option<Pooling>> {
        self.manifest.contract.pooling.as_deref().map(Pooling::from_name).transpose()
    }

    /// Normalization from the contract, if declared.
    pub fn normalize(&self) -> Result<Option<Normalize>> {
        match self.manifest.contract.normalize.as_deref() {
            None => Ok(None),
            Some("l2") => Ok(Some(Normalize::L2)),
            Some("none") => Ok(Some(Normalize::None)),
            Some(other) => Err(Error::bundle_invalid(format!("unknown normalize `{other}`"))),
        }
    }

    /// Aggregation from the contract, if declared.
    pub fn aggregation(&self) -> Result<Option<Aggregation>> {
        self.manifest.contract.aggregation.as_deref().map(Aggregation::from_name).transpose()
    }

    /// Compute dtype from the contract, if declared.
    pub fn dtype(&self) -> Result<Option<DType>> {
        self.manifest.contract.dtype.as_deref().map(DType::from_name).transpose()
    }

    /// Hex SHA-256 of the primary tokenizer file (`tokenizer.json` or the first listed), or empty.
    pub fn tokenizer_sha256(&self) -> &str {
        match &self.manifest.tokenizer {
            Some(t) => t
                .files
                .get("tokenizer.json")
                .or_else(|| t.files.values().next())
                .map(|e| e.sha256.as_str())
                .unwrap_or(""),
            None => "",
        }
    }
}

fn validate_contract(m: &Manifest, kind: ModelKind) -> Result<()> {
    let c = &m.contract;
    match kind {
        ModelKind::Embedding => {
            if c.dim == 0 {
                return Err(Error::bundle_invalid("embedding bundle must declare contract.dim"));
            }
            if c.pooling.is_none() {
                return Err(Error::bundle_invalid("embedding bundle must declare contract.pooling"));
            }
            if c.normalize.is_none() {
                return Err(Error::bundle_invalid("embedding bundle must declare contract.normalize"));
            }
            for &d in &c.truncate_dims {
                if d == 0 || d > c.dim {
                    return Err(Error::bundle_invalid(format!(
                        "contract.truncate_dims entry {d} is outside 1..={}",
                        c.dim
                    )));
                }
            }
        }
        ModelKind::Classifier | ModelKind::TokenClassifier => {
            if c.labels.is_empty() {
                return Err(Error::bundle_invalid("classifier bundle must declare contract.labels"));
            }
        }
        ModelKind::Reranker | ModelKind::Generative | ModelKind::Generic => {}
    }
    if c.max_seq == 0 && !matches!(kind, ModelKind::Generic) {
        return Err(Error::bundle_invalid("contract.max_seq must be declared and non-zero"));
    }
    Ok(())
}

/// Parse a task name as used in manifests.
pub fn task_from_name(name: &str) -> Result<Task> {
    Ok(match name {
        "embed" => Task::Embed,
        "rerank" => Task::Rerank,
        "classify" => Task::Classify,
        "token_classify" => Task::TokenClassify,
        "generate" => Task::Generate,
        "tokenize" => Task::Tokenize,
        "run" => Task::Run,
        "chunk" => Task::Chunk,
        other => return Err(Error::bundle_invalid(format!("unknown task `{other}`"))),
    })
}

/// Parse a model kind name as used in manifests.
pub fn kind_from_name(name: &str) -> Result<ModelKind> {
    Ok(match name {
        "embedding" => ModelKind::Embedding,
        "reranker" => ModelKind::Reranker,
        "classifier" => ModelKind::Classifier,
        "token_classifier" => ModelKind::TokenClassifier,
        "generative" => ModelKind::Generative,
        "generic" => ModelKind::Generic,
        other => return Err(Error::bundle_invalid(format!("unknown model kind `{other}`"))),
    })
}

/// Parse a modality name as used in manifests.
pub fn modality_from_name(name: &str) -> Result<Modality> {
    Ok(match name {
        "text" => Modality::Text,
        "audio" => Modality::Audio,
        "image" => Modality::Image,
        "video" => Modality::Video,
        other => return Err(Error::bundle_invalid(format!("unknown modality `{other}`"))),
    })
}

/// Hex SHA-256 of a file, streamed.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// Hex SHA-256 of bytes.
pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(TABLE[(b >> 4) as usize] as char);
        out.push(TABLE[(b & 0xf) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_abi as abi;

    fn write_bundle(dir: &Path, artifact_body: &[u8], recorded_sha: Option<&str>) {
        std::fs::write(dir.join("mock.json"), artifact_body).unwrap();
        let sha = recorded_sha.map(str::to_string).unwrap_or_else(|| sha256_bytes(artifact_body));
        let manifest = serde_json::json!({
            "bundle_version": 2,
            "model_id": "turbo/mock-8d",
            "revision": "test",
            "task": "embed",
            "kind": "embedding",
            "contract": {"pooling": "mean", "normalize": "l2", "max_seq": 16, "dim": 8},
            "artifacts": {"mock": {"path": "mock.json", "sha256": sha}}
        });
        std::fs::write(dir.join(MANIFEST_NAME), manifest.to_string()).unwrap();
    }

    #[test]
    fn opens_and_verifies() {
        let tmp = tempfile::tempdir().unwrap();
        write_bundle(tmp.path(), b"{}", None);
        let b = Bundle::open(tmp.path()).unwrap();
        assert_eq!(b.task(), Task::Embed);
        assert_eq!(b.kind(), ModelKind::Embedding);
        assert_eq!(b.pooling().unwrap(), Some(Pooling::Mean));
        assert!(b.artifact_path("mock").unwrap().ends_with("mock.json"));
        assert_eq!(b.artifact_path("onnx").unwrap_err().code(), abi::TURBO_E_BUNDLE_NO_ARTIFACT);
    }

    #[test]
    fn hash_mismatch_is_integrity_error() {
        let tmp = tempfile::tempdir().unwrap();
        write_bundle(tmp.path(), b"{}", Some(&"0".repeat(64)));
        let err = Bundle::open(tmp.path()).unwrap_err();
        assert_eq!(err.code(), abi::TURBO_E_BUNDLE_INTEGRITY);
    }

    #[test]
    fn missing_dir_and_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(Bundle::open(&tmp.path().join("nope")).unwrap_err().code(), abi::TURBO_E_BUNDLE_NOT_FOUND);
        assert_eq!(Bundle::open(tmp.path()).unwrap_err().code(), abi::TURBO_E_BUNDLE_NOT_FOUND);
    }

    #[test]
    fn rejects_path_escape() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest = serde_json::json!({
            "bundle_version": 2, "model_id": "x", "task": "embed", "kind": "embedding",
            "contract": {"pooling": "mean", "normalize": "l2", "max_seq": 1, "dim": 1},
            "artifacts": {"mock": {"path": "../etc/passwd", "sha256": "00"}}
        });
        std::fs::write(tmp.path().join(MANIFEST_NAME), manifest.to_string()).unwrap();
        assert_eq!(Bundle::open(tmp.path()).unwrap_err().code(), abi::TURBO_E_BUNDLE_INVALID);
    }

    #[test]
    fn wrong_version_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("mock.json"), b"{}").unwrap();
        let manifest = serde_json::json!({
            "bundle_version": 1, "model_id": "x", "task": "embed", "kind": "embedding",
            "contract": {"pooling": "mean", "normalize": "l2", "max_seq": 1, "dim": 1},
            "artifacts": {"mock": {"path": "mock.json", "sha256": sha256_bytes(b"{}")}}
        });
        std::fs::write(tmp.path().join(MANIFEST_NAME), manifest.to_string()).unwrap();
        assert_eq!(Bundle::open(tmp.path()).unwrap_err().code(), abi::TURBO_E_BUNDLE_INVALID);
    }
}
