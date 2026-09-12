//! Maintainer source-of-truth for `--update-manifest`.
//!
//! Mirrors the nvidia ORT / llama-cpp and apple MLX resolutions in
//! `config/catalog.toml`. Hashes themselves live in the committed JSON
//! manifests; this module only names the Hugging Face repos and the files
//! we pin.

use std::collections::BTreeMap;

pub const HF_BASE: &str = "https://huggingface.co";
pub const USER_AGENT: &str = "inferstream-xtask/1.0";

pub const ONNX_FILES: &[&str] = &["onnx/model.onnx", "tokenizer.json", "config.json"];
pub const ONNX_SIDECAR_PREFIX: &str = "onnx/model.onnx";

pub fn onnx_repos() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        ("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
        ("minilm-l12", "sentence-transformers/all-MiniLM-L12-v2"),
        ("mpnet", "sentence-transformers/all-mpnet-base-v2"),
        ("bge-small", "Xenova/bge-small-en-v1.5"),
        ("bge-base", "Xenova/bge-base-en-v1.5"),
        ("bge-large", "Xenova/bge-large-en-v1.5"),
        ("bge-m3", "Xenova/bge-m3"),
        ("e5-small", "Xenova/multilingual-e5-small"),
        ("e5-base", "Xenova/multilingual-e5-base"),
        ("e5-large", "Xenova/multilingual-e5-large"),
        ("gte-small", "Xenova/gte-small"),
        ("gte-base", "Xenova/gte-base"),
        ("nomic-embed-text", "nomic-ai/nomic-embed-text-v1.5"),
    ])
}

/// Apple/MLX runtime repos (weights fetched into `models/mlx/<alias>/`).
pub fn mlx_embed_repos() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        ("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
        ("minilm-l12", "sentence-transformers/all-MiniLM-L12-v2"),
        ("bge-small", "BAAI/bge-small-en-v1.5"),
        ("bge-base", "BAAI/bge-base-en-v1.5"),
        ("bge-large", "BAAI/bge-large-en-v1.5"),
        ("bge-m3", "BAAI/bge-m3"),
        ("e5-small", "intfloat/multilingual-e5-small"),
        ("e5-base", "intfloat/multilingual-e5-base"),
        ("e5-large", "intfloat/multilingual-e5-large"),
        ("gte-small", "thenlper/gte-small"),
        ("gte-base", "thenlper/gte-base"),
    ])
}

pub struct LlmSource {
    pub repo: &'static str,
    pub files: &'static [&'static str],
    pub dest: &'static str,
    pub tokenizer_repo: &'static str,
    pub tokenizer_files: &'static [&'static str],
}

pub fn llm_sources() -> BTreeMap<&'static str, LlmSource> {
    BTreeMap::from([
        (
            "qwen-0.5b",
            LlmSource {
                repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                files: &["qwen2.5-0.5b-instruct-q8_0.gguf"],
                dest: "models/gguf/qwen-0.5b",
                tokenizer_repo: "Qwen/Qwen2.5-0.5B-Instruct",
                tokenizer_files: &["tokenizer.json"],
            },
        ),
        (
            "qwen-7b",
            LlmSource {
                repo: "Qwen/Qwen2.5-7B-Instruct-GGUF",
                files: &[
                    "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
                    "qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf",
                ],
                dest: "models/gguf/qwen-7b",
                tokenizer_repo: "Qwen/Qwen2.5-7B-Instruct",
                tokenizer_files: &["tokenizer.json"],
            },
        ),
    ])
}

pub fn llm_aliases() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([("default-llm", "qwen-0.5b")])
}

pub fn mlx_llm_repos() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        ("default-llm", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
        ("qwen-0.5b", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
        ("qwen-7b", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
    ])
}

pub fn llm_known_aliases() -> BTreeMap<String, String> {
    let mut known = BTreeMap::new();
    for (alias, spec) in llm_sources() {
        known.insert(alias.to_string(), spec.repo.to_string());
    }
    for (alias, target) in llm_aliases() {
        if let Some(spec) = llm_sources().get(target) {
            known.insert(alias.to_string(), spec.repo.to_string());
        }
    }
    known
}

/// Files we keep when pinning an MLX / safetensors repo.
pub fn keep_mlx_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".md")
        || lower.ends_with(".png")
        || lower.ends_with(".jpg")
        || lower.ends_with(".jpeg")
        || lower.ends_with(".gif")
        || lower == ".gitattributes"
        || lower.starts_with("onnx/")
        || lower.contains("openvino")
        || lower.starts_with("flax_model")
        || lower.starts_with("tf_model")
        || lower.starts_with("rust_model")
        || lower.contains("pytorch_model")
    {
        return false;
    }
    lower.ends_with(".safetensors")
        || lower.ends_with(".safetensors.index.json")
        || lower == "config.json"
        || lower == "tokenizer.json"
        || lower == "tokenizer_config.json"
        || lower == "special_tokens_map.json"
        || lower == "generation_config.json"
        || lower == "vocab.json"
        || lower == "merges.txt"
        || lower == "modules.json"
        || lower == "preprocessor_config.json"
        || lower == "sentence_bert_config.json"
        || lower == "1_pooling/config.json"
        || lower.ends_with(".json") && !lower.contains("onnx")
}

pub fn onnx_file_list(repo_files: &[String]) -> Result<Vec<String>, String> {
    let sidecars: Vec<String> = repo_files
        .iter()
        .filter(|f| {
            f.starts_with(ONNX_SIDECAR_PREFIX)
                && *f != "onnx/model.onnx"
                && f[ONNX_SIDECAR_PREFIX.len()..].contains("data")
        })
        .cloned()
        .collect();
    let missing: Vec<&str> = ONNX_FILES
        .iter()
        .copied()
        .filter(|f| !repo_files.iter().any(|r| r == f))
        .collect();
    if !missing.is_empty() {
        return Err(format!("required file(s) not in repo: {missing:?}"));
    }
    let mut files = vec![ONNX_FILES[0].to_string()];
    files.extend(sidecars);
    files.extend(ONNX_FILES[1..].iter().map(|s| s.to_string()));
    Ok(files)
}
