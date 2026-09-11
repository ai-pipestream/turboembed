//! Hash-verified fetch of inferstream model artifacts.
//!
//! Downloads revision-pinned ONNX / GGUF / tokenizer files from Hugging Face
//! and checks every byte against the committed JSON manifests in
//! `models/manifests/`. No Python, no `huggingface_hub`.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const HF_BASE: &str = "https://huggingface.co";
pub const USER_AGENT: &str = "inferstream-fetch/1.0";
const CHUNK: usize = 1 << 20;

/// Alias → ONNX source repo. Mirrors nvidia `backend = "ort"` resolutions
/// in `config/catalog.toml`. Used by `--update-manifest`.
pub const ONNX_REPOS: &[(&str, &str)] = &[
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
];

pub const ONNX_FILES: &[&str] = &["onnx/model.onnx", "tokenizer.json", "config.json"];
pub const ONNX_SIDECAR_PREFIX: &str = "onnx/model.onnx";

/// apple/MLX runtime repos recorded in the embedding manifest (`mlx_repos`).
pub const MLX_REPOS: &[(&str, &str)] = &[
    ("minilm", "mlx-community/all-MiniLM-L6-v2-4bit"),
    ("minilm-l12", "sentence-transformers/all-MiniLM-L12-v2"),
    ("bge-small", "mlx-community/bge-small-en-v1.5-4bit"),
    ("bge-base", "BAAI/bge-base-en-v1.5"),
    ("bge-large", "BAAI/bge-large-en-v1.5"),
    ("bge-m3", "BAAI/bge-m3"),
    ("e5-small", "intfloat/multilingual-e5-small"),
    ("e5-base", "intfloat/multilingual-e5-base"),
    ("e5-large", "intfloat/multilingual-e5-large"),
    ("gte-small", "thenlper/gte-small"),
    ("gte-base", "thenlper/gte-base"),
];

/// Concrete LLM families hashed into `models/manifests/llms.json`.
pub const LLM_SOURCES: &[LlmSource] = &[
    LlmSource {
        alias: "qwen-0.5b",
        repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        files: &["qwen2.5-0.5b-instruct-q8_0.gguf"],
        dest: "models/gguf/qwen-0.5b",
        tokenizer_repo: "Qwen/Qwen2.5-0.5B-Instruct",
        tokenizer_files: &["tokenizer.json"],
    },
    LlmSource {
        alias: "qwen-7b",
        repo: "Qwen/Qwen2.5-7B-Instruct-GGUF",
        files: &[
            "qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
            "qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf",
        ],
        dest: "models/gguf/qwen-7b",
        tokenizer_repo: "Qwen/Qwen2.5-7B-Instruct",
        tokenizer_files: &["tokenizer.json"],
    },
];

/// Logical catalog aliases that share a fetched artifact family.
pub const LLM_ALIASES: &[(&str, &str)] = &[("default-llm", "qwen-0.5b")];

pub const LLM_MLX_REPOS: &[(&str, &str)] = &[
    ("default-llm", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
    ("qwen-0.5b", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
    ("qwen-7b", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
];

#[derive(Debug, Clone, Copy)]
pub struct LlmSource {
    pub alias: &'static str,
    pub repo: &'static str,
    pub files: &'static [&'static str],
    pub dest: &'static str,
    pub tokenizer_repo: &'static str,
    pub tokenizer_files: &'static [&'static str],
}

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("{0}")]
    Msg(String),
    #[error("http: {0}")]
    Http(String),
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

impl FetchError {
    pub fn msg(s: impl Into<String>) -> Self {
        Self::Msg(s.into())
    }
}

pub type Result<T> = std::result::Result<T, FetchError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none", rename = "_comment")]
    pub comment: Option<String>,
    #[serde(default)]
    pub models: BTreeMap<String, ModelEntry>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mlx_repos: BTreeMap<String, MlxRepo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alias_of: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dest: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<FileEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<TokenizerEntry>,
    /// OVMS export entries record the pipeline suffix used to split dest roots.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerEntry {
    pub repo: String,
    pub revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dest: Option<String>,
    pub files: Vec<FileEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxRepo {
    pub repo: String,
    pub revision: String,
}

/// One downloadable artifact: (repo, revision, dest, file).
#[derive(Debug, Clone)]
pub struct ArtifactSpec {
    pub repo: String,
    pub revision: String,
    pub dest: String,
    pub file: FileEntry,
}

pub fn onnx_repo(alias: &str) -> Option<&'static str> {
    ONNX_REPOS
        .iter()
        .find(|(a, _)| *a == alias)
        .map(|(_, r)| *r)
}

pub fn mlx_repo(alias: &str) -> Option<&'static str> {
    MLX_REPOS.iter().find(|(a, _)| *a == alias).map(|(_, r)| *r)
}

pub fn llm_source(alias: &str) -> Option<&'static LlmSource> {
    LLM_SOURCES.iter().find(|s| s.alias == alias)
}

pub fn llm_alias_of(alias: &str) -> Option<&'static str> {
    LLM_ALIASES
        .iter()
        .find(|(a, _)| *a == alias)
        .map(|(_, t)| *t)
}

pub fn llm_known_aliases() -> BTreeMap<String, String> {
    let mut known = BTreeMap::new();
    for spec in LLM_SOURCES {
        known.insert(spec.alias.to_string(), spec.repo.to_string());
    }
    for (alias, target) in LLM_ALIASES {
        if let Some(spec) = llm_source(target) {
            known.insert((*alias).to_string(), spec.repo.to_string());
        }
    }
    known
}

pub fn embedding_known_aliases() -> BTreeMap<String, String> {
    ONNX_REPOS
        .iter()
        .map(|(a, r)| ((*a).to_string(), (*r).to_string()))
        .collect()
}

pub fn load_manifest(path: &Path) -> Result<Manifest> {
    let text = fs::read_to_string(path)?;
    let manifest: Manifest = serde_json::from_str(&text)?;
    if manifest.schema_version != 1 {
        return Err(FetchError::msg(format!(
            "{}: unsupported schema_version {:?}",
            path.display(),
            manifest.schema_version
        )));
    }
    Ok(manifest)
}

pub fn write_manifest(path: &Path, manifest: &Manifest) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut body = serde_json::to_string_pretty(manifest)?;
    body.push('\n');
    fs::write(path, body)?;
    Ok(())
}

/// Return (canonical_alias, entry), following a single `alias_of` hop.
pub fn resolve_model_entry<'a>(
    manifest: &'a Manifest,
    alias: &str,
) -> Result<(&'a str, &'a ModelEntry)> {
    let (key, entry) = manifest
        .models
        .get_key_value(alias)
        .ok_or_else(|| FetchError::msg(format!("alias {alias:?} is not in the manifest")))?;
    if let Some(target) = entry.alias_of.as_deref() {
        let (target_key, target_entry) =
            manifest.models.get_key_value(target).ok_or_else(|| {
                FetchError::msg(format!(
                    "{alias}: alias_of {target:?} is not in the manifest"
                ))
            })?;
        if target_entry.alias_of.is_some() {
            return Err(FetchError::msg(format!(
                "{alias}: nested alias_of is not supported"
            )));
        }
        return Ok((target_key.as_str(), target_entry));
    }
    Ok((key.as_str(), entry))
}

pub fn artifact_specs(entry: &ModelEntry) -> Result<Vec<ArtifactSpec>> {
    let dest = entry
        .dest
        .as_deref()
        .ok_or_else(|| FetchError::msg("manifest entry is missing dest"))?;
    let repo = entry
        .repo
        .as_deref()
        .ok_or_else(|| FetchError::msg("manifest entry is missing repo"))?;
    let revision = entry
        .revision
        .as_deref()
        .ok_or_else(|| FetchError::msg("manifest entry is missing revision"))?;
    let mut out = Vec::new();
    for f in &entry.files {
        out.push(ArtifactSpec {
            repo: repo.to_string(),
            revision: revision.to_string(),
            dest: dest.to_string(),
            file: f.clone(),
        });
    }
    if let Some(tok) = &entry.tokenizer {
        let tok_dest = tok.dest.as_deref().unwrap_or(dest);
        for f in &tok.files {
            out.push(ArtifactSpec {
                repo: tok.repo.clone(),
                revision: tok.revision.clone(),
                dest: tok_dest.to_string(),
                file: f.clone(),
            });
        }
    }
    Ok(out)
}

pub fn resolve_url(repo: &str, revision: &str, path: &str) -> String {
    format!("{HF_BASE}/{repo}/resolve/{revision}/{path}")
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn human(size: u64) -> String {
    if size >= 1 << 30 {
        format!("{:.2} GiB", size as f64 / ((1u64 << 30) as f64))
    } else if size >= 1 << 20 {
        format!("{:.1} MiB", size as f64 / ((1u64 << 20) as f64))
    } else {
        format!("{size} B")
    }
}

fn http_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| FetchError::Http(e.to_string()))
}

/// Download `url`, returning (sha256, size). If `dest` is given, write the
/// file atomically (tmp file in the same directory, then rename).
///
/// `file://` URLs are supported so tests can exercise the write path offline.
pub fn stream_download(url: &str, dest: Option<&Path>) -> Result<(String, u64)> {
    if let Some(rest) = url.strip_prefix("file://") {
        let src = Path::new(rest);
        let mut file = File::open(src)?;
        return stream_from_reader(&mut file, dest);
    }
    let client = http_client()?;
    let mut resp = client
        .get(url)
        .send()
        .map_err(|e| FetchError::Http(format!("{url}: {e}")))?;
    if !resp.status().is_success() {
        return Err(FetchError::Http(format!(
            "{url}: HTTP {} {}",
            resp.status().as_u16(),
            resp.status().canonical_reason().unwrap_or("")
        )));
    }
    stream_from_reader(&mut resp, dest)
}

fn stream_from_reader<R: Read>(reader: &mut R, dest: Option<&Path>) -> Result<(String, u64)> {
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut tmp_path: Option<PathBuf> = None;
    let mut out: Option<File> = None;

    if let Some(dest) = dest {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        // Unique-ish part file so two concurrent fetches of the same dest
        // don't clobber each other (pid is enough for this CLI).
        let tmp = {
            let mut n = 0u32;
            loop {
                let candidate = dest.with_file_name(format!(
                    "{}.{}.part",
                    dest.file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("download"),
                    std::process::id() + n
                ));
                if !candidate.exists() {
                    break candidate;
                }
                n += 1;
            }
        };
        out = Some(File::create(&tmp)?);
        tmp_path = Some(tmp);
    }

    let mut buf = vec![0u8; CHUNK];
    let result = (|| -> Result<(String, u64)> {
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            size += n as u64;
            if let Some(f) = out.as_mut() {
                f.write_all(&buf[..n])?;
            }
        }
        if let Some(f) = out.as_mut() {
            f.flush()?;
        }
        if let (Some(tmp), Some(dest)) = (tmp_path.as_ref(), dest) {
            fs::rename(tmp, dest)?;
            tmp_path = None;
        }
        Ok((format!("{:x}", hasher.finalize()), size))
    })();

    if let Some(tmp) = tmp_path {
        let _ = fs::remove_file(tmp);
    }
    result
}

#[derive(Deserialize)]
struct HfModelInfo {
    sha: Option<String>,
    #[serde(default)]
    siblings: Vec<HfSibling>,
}

#[derive(Deserialize)]
struct HfSibling {
    rfilename: String,
}

/// `(main-branch commit hash, list of files)` via the HF API.
pub fn repo_info(repo: &str) -> Result<(String, Vec<String>)> {
    let url = format!("{HF_BASE}/api/models/{repo}");
    let client = http_client()?;
    let resp = client
        .get(&url)
        .send()
        .map_err(|e| FetchError::Http(format!("{url}: {e}")))?;
    if !resp.status().is_success() {
        return Err(FetchError::Http(format!(
            "{url}: HTTP {}",
            resp.status().as_u16()
        )));
    }
    let info: HfModelInfo = resp
        .json()
        .map_err(|e| FetchError::Http(format!("{url}: {e}")))?;
    let sha = info
        .sha
        .ok_or_else(|| FetchError::msg(format!("HF API returned no commit sha for {repo}")))?;
    let files = info.siblings.into_iter().map(|s| s.rfilename).collect();
    Ok((sha, files))
}

pub fn onnx_file_list(repo: &str, repo_files: &[String]) -> Result<Vec<String>> {
    let sidecars: Vec<String> = {
        let mut s: Vec<String> = repo_files
            .iter()
            .filter(|f| {
                f.starts_with(ONNX_SIDECAR_PREFIX)
                    && f.as_str() != "onnx/model.onnx"
                    && f[ONNX_SIDECAR_PREFIX.len()..].contains("data")
            })
            .cloned()
            .collect();
        s.sort();
        s
    };
    let missing: Vec<&str> = ONNX_FILES
        .iter()
        .copied()
        .filter(|f| !repo_files.iter().any(|r| r == f))
        .collect();
    if !missing.is_empty() {
        return Err(FetchError::msg(format!(
            "{repo}: required file(s) not in repo: {missing:?}"
        )));
    }
    let mut out = vec![ONNX_FILES[0].to_string()];
    out.extend(sidecars);
    out.extend(ONNX_FILES[1..].iter().map(|s| (*s).to_string()));
    Ok(out)
}

pub fn select_aliases(
    all: bool,
    aliases: &[String],
    known: &BTreeMap<String, String>,
) -> Result<Vec<String>> {
    if all {
        return Ok(known.keys().cloned().collect());
    }
    if aliases.is_empty() {
        return Err(FetchError::msg(
            "error: no aliases given (use --all or --list)",
        ));
    }
    let mut bad = Vec::new();
    for a in aliases {
        if !known.contains_key(a) {
            bad.push(a.clone());
        }
    }
    if !bad.is_empty() {
        let known_list: Vec<&str> = known.keys().map(|s| s.as_str()).collect();
        return Err(FetchError::msg(format!(
            "error: unknown alias(es): {}\nknown: {}",
            bad.join(", "),
            known_list.join(", ")
        )));
    }
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for a in aliases {
        if seen.insert(a.clone()) {
            out.push(a.clone());
        }
    }
    Ok(out)
}

pub fn cmd_list(
    manifest_path: &Path,
    fallback_repos: &BTreeMap<String, String>,
    out: &mut impl Write,
) -> Result<i32> {
    if manifest_path.exists() {
        let manifest = load_manifest(manifest_path)?;
        writeln!(out, "{:<18} {:<42} revision", "alias", "repo")?;
        for (alias, entry) in &manifest.models {
            if let Some(target) = &entry.alias_of {
                writeln!(out, "{alias:<18} alias_of {target}")?;
                continue;
            }
            let repo = entry.repo.as_deref().unwrap_or("-");
            let rev = entry.revision.as_deref().unwrap_or("-");
            let short = if rev.len() >= 12 { &rev[..12] } else { rev };
            writeln!(out, "{alias:<18} {repo:<42} {short}")?;
        }
    } else {
        writeln!(out, "{:<18} repo (manifest not generated yet)", "alias")?;
        for (alias, repo) in fallback_repos {
            writeln!(out, "{alias:<18} {repo}")?;
        }
    }
    Ok(0)
}

/// Offline SHA-256 check. `dest_for` maps a relative dest (or ovms file path)
/// onto an absolute directory that already contains the file.
pub fn cmd_verify(
    aliases: &[String],
    manifest: &Manifest,
    root: &Path,
    out: &mut impl Write,
    err: &mut impl Write,
) -> Result<i32> {
    let mut failures = Vec::new();
    let mut seen = HashSet::new();
    for alias in aliases {
        let (key, entry) = resolve_model_entry(manifest, alias)?;
        if !seen.insert(key.to_string()) {
            writeln!(
                out,
                "  skip      {alias} (alias of {key}, already verified)"
            )?;
            continue;
        }
        if alias != key {
            writeln!(out, "--- {alias}  ->  {key} ---")?;
        }
        for spec in artifact_specs(entry)? {
            let path = root.join(&spec.dest).join(&spec.file.path);
            let label = format!(
                "{key}: {}",
                path.strip_prefix(root).unwrap_or(&path).display()
            );
            if !path.exists() {
                failures.push(format!("{label} — MISSING"));
                writeln!(out, "  missing   {label}")?;
                continue;
            }
            let actual = sha256_file(&path)?;
            if actual != spec.file.sha256 {
                failures.push(format!(
                    "{label} — sha256 mismatch (expected {}, got {actual})",
                    spec.file.sha256
                ));
                writeln!(out, "  MISMATCH  {label}")?;
            } else {
                writeln!(out, "  ok        {label}")?;
            }
        }
    }
    if !failures.is_empty() {
        writeln!(err, "\nverify FAILED ({} problem(s)):", failures.len())?;
        for msg in &failures {
            writeln!(err, "  {msg}")?;
        }
        return Ok(1);
    }
    writeln!(
        out,
        "\nverify OK — all files present with matching SHA-256."
    )?;
    Ok(0)
}

/// Verify an OVMS-layout manifest: `hf_tokenizer_*` files live under `hf_out`,
/// everything else under `ovms_out`. Paths in the manifest are already the
/// relative keys (`tokenizer_<name>/1/...`, `hf_tokenizer_<name>/tokenizer.json`).
pub fn cmd_verify_ovms(
    aliases: &[String],
    manifest: &Manifest,
    ovms_out: &Path,
    hf_out: &Path,
    out: &mut impl Write,
    err: &mut impl Write,
) -> Result<i32> {
    let mut failures = Vec::new();
    for alias in aliases {
        let entry = manifest
            .models
            .get(alias)
            .ok_or_else(|| FetchError::msg(format!("alias {alias:?} is not in the manifest")))?;
        if entry.alias_of.is_some() {
            continue;
        }
        let name = entry
            .name
            .as_deref()
            .ok_or_else(|| FetchError::msg(format!("{alias}: ovms entry is missing name")))?;
        for f in &entry.files {
            let path = ovms_artifact_path(name, ovms_out, hf_out, &f.path);
            if !path.exists() {
                failures.push(format!("{alias}: {path} — MISSING", path = path.display()));
                writeln!(out, "  missing   {alias}: {}", f.path)?;
                continue;
            }
            let actual = sha256_file(&path)?;
            if actual != f.sha256 {
                failures.push(format!(
                    "{alias}: {} — sha256 mismatch (expected {}, got {actual})",
                    path.display(),
                    f.sha256
                ));
                writeln!(out, "  MISMATCH  {alias}: {}", f.path)?;
            } else {
                writeln!(out, "  ok        {alias}: {}", f.path)?;
            }
        }
    }
    if !failures.is_empty() {
        writeln!(err, "\nverify FAILED ({} problem(s)):", failures.len())?;
        for msg in &failures {
            writeln!(err, "  {msg}")?;
        }
        return Ok(1);
    }
    writeln!(
        out,
        "\nverify OK — all OVMS artifacts present with matching SHA-256."
    )?;
    Ok(0)
}

pub fn ovms_artifact_path(name: &str, ovms_out: &Path, hf_out: &Path, rel: &str) -> PathBuf {
    if rel == format!("hf_tokenizer_{name}/tokenizer.json")
        || rel.starts_with(&format!("hf_tokenizer_{name}/"))
    {
        hf_out.join(rel)
    } else {
        ovms_out.join(rel)
    }
}

pub fn cmd_fetch(
    aliases: &[String],
    manifest: &Manifest,
    root: &Path,
    smoke_hint: Option<&str>,
    out: &mut impl Write,
    err: &mut impl Write,
) -> Result<i32> {
    let mut downloaded = 0u32;
    let mut skipped = 0u32;
    let mut seen = HashSet::new();
    for alias in aliases {
        let (key, entry) = resolve_model_entry(manifest, alias)?;
        if !seen.insert(key.to_string()) {
            writeln!(out, "--- {alias}  ->  {key} (already fetched) ---")?;
            continue;
        }
        let dest_root = root.join(entry.dest.as_deref().unwrap_or("."));
        let label = if alias == key {
            alias.to_string()
        } else {
            format!("{alias} -> {key}")
        };
        let repo = entry.repo.as_deref().unwrap_or("?");
        let rev = entry.revision.as_deref().unwrap_or("?");
        let short = if rev.len() >= 12 { &rev[..12] } else { rev };
        let dest_disp = dest_root.strip_prefix(root).unwrap_or(&dest_root);
        writeln!(
            out,
            "--- {label}  <-  {repo}@{short}  ->  {} ---",
            dest_disp.display()
        )?;
        for spec in artifact_specs(entry)? {
            let path = root.join(&spec.dest).join(&spec.file.path);
            if path.exists() {
                if sha256_file(&path)? == spec.file.sha256 {
                    writeln!(
                        out,
                        "  ok (cached)  {}  [{}]",
                        spec.file.path,
                        human(spec.file.size)
                    )?;
                    skipped += 1;
                    continue;
                }
                writeln!(out, "  stale hash, re-downloading  {}", spec.file.path)?;
            }
            let url = resolve_url(&spec.repo, &spec.revision, &spec.file.path);
            writeln!(
                out,
                "  downloading  {}  [{}] ...",
                spec.file.path,
                human(spec.file.size)
            )?;
            let _ = out.flush();
            let (actual, size) = match stream_download(&url, Some(&path)) {
                Ok(v) => v,
                Err(e) => {
                    writeln!(err, "error: {url}: {e}")?;
                    return Ok(1);
                }
            };
            if actual != spec.file.sha256 {
                let _ = fs::remove_file(&path);
                writeln!(
                    err,
                    "error: SHA-256 mismatch for {key}/{}\n  expected {}\n  got      {actual}\n  url      {url}\nThe file was deleted. If upstream legitimately changed, re-pin with --update-manifest and review the diff.",
                    spec.file.path, spec.file.sha256
                )?;
                return Ok(1);
            }
            if size != spec.file.size {
                writeln!(
                    err,
                    "error: size mismatch for {key}/{} (expected {}, got {size})",
                    spec.file.path, spec.file.size
                )?;
                return Ok(1);
            }
            let short_hash = if actual.len() >= 16 {
                &actual[..16]
            } else {
                &actual
            };
            writeln!(
                out,
                "  verified     {}  sha256={short_hash}…",
                spec.file.path
            )?;
            downloaded += 1;
        }
    }
    writeln!(
        out,
        "\nDone: {downloaded} downloaded, {skipped} already present and verified."
    )?;
    if let Some(hint) = smoke_hint {
        write!(out, "{hint}")?;
        if !hint.ends_with('\n') {
            writeln!(out)?;
        }
    } else {
        writeln!(
            out,
            "Add the aliases to `serve` in config/nvidia.toml and restart;"
        )?;
        writeln!(
            out,
            "verify with: scripts/smoke-embeddings.sh <host:port> <bearer-token>"
        )?;
    }
    Ok(0)
}

const EMBED_COMMENT: &str = "SHA-256 manifest for inferstream model artifacts. Generated by cargo run -p inferstream-fetch -- --update-manifest; do not edit hashes by hand. Revisions are exact HF commit hashes (never floating branches). mlx_repos records the repos+revisions the apple/MLX runtime is expected to use (informational).";

const LLM_COMMENT: &str = "SHA-256 manifest for inferstream LLM artifacts. Generated by cargo run -p inferstream-fetch -- --llms --update-manifest; do not edit hashes by hand. Revisions are exact HF commit hashes (never floating branches). mlx_repos records the repos+revisions the apple/MLX runtime is expected to use (informational). default-llm is alias_of qwen-0.5b.";

pub fn cmd_update_manifest(
    aliases: &[String],
    manifest_path: &Path,
    root: &Path,
    store: bool,
    out: &mut impl Write,
    err: &mut impl Write,
) -> Result<i32> {
    let mut manifest = if manifest_path.exists() {
        load_manifest(manifest_path)?
    } else {
        Manifest {
            schema_version: 1,
            comment: Some(EMBED_COMMENT.to_string()),
            models: BTreeMap::new(),
            mlx_repos: BTreeMap::new(),
        }
    };

    for alias in aliases {
        let repo = onnx_repo(alias)
            .ok_or_else(|| FetchError::msg(format!("{alias}: not in ONNX_REPOS")))?;
        writeln!(out, "--- pinning {alias}  <-  {repo} ---")?;
        let (revision, repo_files) = repo_info(repo)?;
        writeln!(out, "  revision {revision}")?;
        let dest = format!("models/onnx/{alias}");
        let mut files = Vec::new();
        for rel in onnx_file_list(repo, &repo_files)? {
            let url = resolve_url(repo, &revision, &rel);
            let target = if store {
                Some(root.join(&dest).join(&rel))
            } else {
                None
            };
            writeln!(out, "  hashing {rel} ...")?;
            let _ = out.flush();
            let (digest, size) = match stream_download(&url, target.as_deref()) {
                Ok(v) => v,
                Err(e) => {
                    writeln!(err, "error: {url}: {e}")?;
                    return Ok(1);
                }
            };
            writeln!(out, "    sha256={digest}  size={}", human(size))?;
            files.push(FileEntry {
                path: rel,
                sha256: digest,
                size,
            });
        }
        manifest.models.insert(
            alias.clone(),
            ModelEntry {
                alias_of: None,
                repo: Some(repo.to_string()),
                revision: Some(revision),
                dest: Some(dest),
                files,
                tokenizer: None,
                name: None,
            },
        );
        if let Some(mlx) = mlx_repo(alias) {
            let (mlx_rev, _) = repo_info(mlx)?;
            manifest.mlx_repos.insert(
                alias.clone(),
                MlxRepo {
                    repo: mlx.to_string(),
                    revision: mlx_rev.clone(),
                },
            );
            let short = if mlx_rev.len() >= 12 {
                &mlx_rev[..12]
            } else {
                &mlx_rev
            };
            writeln!(out, "  mlx runtime repo {mlx}@{short}")?;
        }
    }

    write_manifest(manifest_path, &manifest)?;
    writeln!(
        out,
        "\nManifest written: {} — review and commit it.",
        manifest_path
            .strip_prefix(root)
            .unwrap_or(manifest_path)
            .display()
    )?;
    Ok(0)
}

pub fn cmd_update_llm_manifest(
    aliases: &[String],
    manifest_path: &Path,
    root: &Path,
    store: bool,
    out: &mut impl Write,
    err: &mut impl Write,
) -> Result<i32> {
    let mut manifest = if manifest_path.exists() {
        load_manifest(manifest_path)?
    } else {
        Manifest {
            schema_version: 1,
            comment: Some(LLM_COMMENT.to_string()),
            models: BTreeMap::new(),
            mlx_repos: BTreeMap::new(),
        }
    };

    let mut families = Vec::new();
    for alias in aliases {
        families.push(llm_alias_of(alias).unwrap_or(alias.as_str()).to_string());
    }
    families.sort();
    families.dedup();

    for alias in &families {
        let spec = llm_source(alias)
            .ok_or_else(|| FetchError::msg(format!("{alias}: not in LLM_SOURCES")))?;
        writeln!(out, "--- pinning {alias}  <-  {} ---", spec.repo)?;
        let (revision, repo_files) = repo_info(spec.repo)?;
        writeln!(out, "  revision {revision}")?;
        let missing: Vec<&&str> = spec
            .files
            .iter()
            .filter(|f| !repo_files.iter().any(|r| r == *f))
            .collect();
        if !missing.is_empty() {
            writeln!(
                err,
                "error: {}: required file(s) not in repo: {missing:?}",
                spec.repo
            )?;
            return Ok(1);
        }
        let dest = spec.dest.to_string();
        let mut files = Vec::new();
        for rel in spec.files {
            let url = resolve_url(spec.repo, &revision, rel);
            let target = if store {
                Some(root.join(&dest).join(rel))
            } else {
                None
            };
            writeln!(out, "  hashing {rel} ...")?;
            let _ = out.flush();
            let (digest, size) = match stream_download(&url, target.as_deref()) {
                Ok(v) => v,
                Err(e) => {
                    writeln!(err, "error: {url}: {e}")?;
                    return Ok(1);
                }
            };
            writeln!(out, "    sha256={digest}  size={}", human(size))?;
            files.push(FileEntry {
                path: (*rel).to_string(),
                sha256: digest,
                size,
            });
        }
        writeln!(out, "  pinning tokenizer  <-  {}", spec.tokenizer_repo)?;
        let (tok_rev, tok_files) = repo_info(spec.tokenizer_repo)?;
        writeln!(out, "    revision {tok_rev}")?;
        let tok_missing: Vec<&&str> = spec
            .tokenizer_files
            .iter()
            .filter(|f| !tok_files.iter().any(|r| r == *f))
            .collect();
        if !tok_missing.is_empty() {
            writeln!(
                err,
                "error: {}: required file(s) not in repo: {tok_missing:?}",
                spec.tokenizer_repo
            )?;
            return Ok(1);
        }
        let mut tok_hashed = Vec::new();
        for rel in spec.tokenizer_files {
            let url = resolve_url(spec.tokenizer_repo, &tok_rev, rel);
            let target = if store {
                Some(root.join(&dest).join(rel))
            } else {
                None
            };
            writeln!(out, "  hashing {rel} ...")?;
            let _ = out.flush();
            let (digest, size) = match stream_download(&url, target.as_deref()) {
                Ok(v) => v,
                Err(e) => {
                    writeln!(err, "error: {url}: {e}")?;
                    return Ok(1);
                }
            };
            writeln!(out, "    sha256={digest}  size={}", human(size))?;
            tok_hashed.push(FileEntry {
                path: (*rel).to_string(),
                sha256: digest,
                size,
            });
        }
        manifest.models.insert(
            alias.clone(),
            ModelEntry {
                alias_of: None,
                repo: Some(spec.repo.to_string()),
                revision: Some(revision),
                dest: Some(dest),
                files,
                tokenizer: Some(TokenizerEntry {
                    repo: spec.tokenizer_repo.to_string(),
                    revision: tok_rev,
                    dest: None,
                    files: tok_hashed,
                }),
                name: None,
            },
        );
    }

    for (alias, target) in LLM_ALIASES {
        if aliases.iter().any(|a| a == alias) || families.iter().any(|f| f == target) {
            manifest.models.insert(
                (*alias).to_string(),
                ModelEntry {
                    alias_of: Some((*target).to_string()),
                    repo: None,
                    revision: None,
                    dest: None,
                    files: Vec::new(),
                    tokenizer: None,
                    name: None,
                },
            );
        }
    }

    for (alias, mlx_repo_name) in LLM_MLX_REPOS {
        let family = llm_alias_of(alias).unwrap_or(alias);
        if !aliases.iter().any(|a| a == alias)
            && !families.iter().any(|f| f == family)
            && !families.iter().any(|f| f == alias)
        {
            continue;
        }
        let (mlx_rev, _) = repo_info(mlx_repo_name)?;
        manifest.mlx_repos.insert(
            (*alias).to_string(),
            MlxRepo {
                repo: (*mlx_repo_name).to_string(),
                revision: mlx_rev.clone(),
            },
        );
        let short = if mlx_rev.len() >= 12 {
            &mlx_rev[..12]
        } else {
            &mlx_rev
        };
        writeln!(out, "  mlx runtime repo {alias}: {mlx_repo_name}@{short}")?;
    }

    write_manifest(manifest_path, &manifest)?;
    writeln!(
        out,
        "\nManifest written: {} — review and commit it.",
        manifest_path
            .strip_prefix(root)
            .unwrap_or(manifest_path)
            .display()
    )?;
    Ok(0)
}

/// Loose catalog shape used only to check nvidia fetch paths against the
/// committed manifests. Extra fields are ignored.
#[derive(Debug, Deserialize)]
struct CatalogFile {
    #[serde(default)]
    models: BTreeMap<String, CatalogAlias>,
}

#[derive(Debug, Deserialize)]
struct CatalogAlias {
    #[serde(default)]
    nvidia: Option<CatalogArch>,
    #[serde(default)]
    apple: Option<CatalogArch>,
}

#[derive(Debug, Deserialize)]
struct CatalogArch {
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    tokenizer_dir: Option<String>,
}

pub fn catalog_nvidia_ort_fetch_paths(catalog_toml: &str) -> Result<Vec<(String, String, String)>> {
    let catalog: CatalogFile =
        toml::from_str(catalog_toml).map_err(|e| FetchError::msg(e.to_string()))?;
    let mut out = Vec::new();
    for (alias, tables) in catalog.models {
        let Some(nvidia) = tables.nvidia else {
            continue;
        };
        if nvidia.backend.as_deref() != Some("ort") {
            continue;
        }
        let path = nvidia.path.unwrap_or_default();
        if !path.starts_with("models/onnx/") {
            continue;
        }
        out.push((alias, path, nvidia.tokenizer_dir.unwrap_or_default()));
    }
    Ok(out)
}

pub fn catalog_nvidia_llamacpp_fetch_paths(catalog_toml: &str) -> Result<Vec<(String, String)>> {
    let catalog: CatalogFile =
        toml::from_str(catalog_toml).map_err(|e| FetchError::msg(e.to_string()))?;
    let mut out = Vec::new();
    for (alias, tables) in catalog.models {
        let Some(nvidia) = tables.nvidia else {
            continue;
        };
        if nvidia.backend.as_deref() != Some("llama-cpp") {
            continue;
        }
        let path = nvidia.path.unwrap_or_default();
        if !path.starts_with("models/gguf/") {
            continue;
        }
        out.push((alias, path));
    }
    Ok(out)
}

pub fn catalog_apple_llm_tokenizer_dirs(catalog_toml: &str) -> Result<Vec<(String, String)>> {
    let catalog: CatalogFile =
        toml::from_str(catalog_toml).map_err(|e| FetchError::msg(e.to_string()))?;
    let mut out = Vec::new();
    for (alias, tables) in catalog.models {
        let Some(apple) = tables.apple else {
            continue;
        };
        if apple.backend.as_deref() != Some("mlx") {
            continue;
        }
        let tok = apple.tokenizer_dir.unwrap_or_default();
        if !tok.starts_with("models/gguf/") {
            continue;
        }
        out.push((alias, tok));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn embeddings_manifest() -> PathBuf {
        repo_root().join("models/manifests/embeddings.json")
    }

    fn llms_manifest() -> PathBuf {
        repo_root().join("models/manifests/llms.json")
    }

    fn catalog_text() -> String {
        fs::read_to_string(repo_root().join("config/catalog.toml")).unwrap()
    }

    fn hex40(s: &str) -> bool {
        s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit())
    }

    fn hex64(s: &str) -> bool {
        s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
    }

    #[test]
    fn embeddings_schema_and_pins() {
        let m = load_manifest(&embeddings_manifest()).unwrap();
        assert_eq!(m.schema_version, 1);
        let known: HashSet<&str> = ONNX_REPOS.iter().map(|(a, _)| *a).collect();
        let have: HashSet<&str> = m.models.keys().map(|s| s.as_str()).collect();
        assert_eq!(have, known, "manifest aliases must match ONNX_REPOS");
        for (alias, entry) in &m.models {
            assert_eq!(entry.repo.as_deref(), onnx_repo(alias));
            assert!(
                hex40(entry.revision.as_deref().unwrap()),
                "{alias} revision"
            );
            let expected_dest = format!("models/onnx/{alias}");
            assert_eq!(entry.dest.as_deref(), Some(expected_dest.as_str()));
            let paths: Vec<&str> = entry.files.iter().map(|f| f.path.as_str()).collect();
            for required in ONNX_FILES {
                assert!(paths.contains(required), "{alias} missing {required}");
            }
            for f in &entry.files {
                assert!(hex64(&f.sha256), "{alias} {}", f.path);
                assert!(f.size > 0);
            }
        }
        for alias in ["bge-m3", "e5-large"] {
            let paths: Vec<&str> = m.models[alias]
                .files
                .iter()
                .map(|f| f.path.as_str())
                .collect();
            assert!(
                paths.contains(&"onnx/model.onnx_data"),
                "{alias} needs external-data sidecar"
            );
        }
        for (alias, entry) in &m.mlx_repos {
            assert_eq!(entry.repo, mlx_repo(alias).unwrap());
            assert!(hex40(&entry.revision), "{alias} mlx revision");
        }
    }

    #[test]
    fn embeddings_match_catalog() {
        let m = load_manifest(&embeddings_manifest()).unwrap();
        let catalog = catalog_text();
        for (alias, path, tok_dir) in catalog_nvidia_ort_fetch_paths(&catalog).unwrap() {
            let entry = m
                .models
                .get(&alias)
                .unwrap_or_else(|| panic!("catalog alias {alias} missing from embedding manifest"));
            let dest = entry.dest.as_deref().unwrap();
            let fetched: HashSet<String> = entry
                .files
                .iter()
                .map(|f| format!("{dest}/{}", f.path))
                .collect();
            assert!(
                fetched.contains(&path),
                "catalog model path {path} not fetched for {alias}"
            );
            assert!(
                fetched.contains(&format!("{tok_dir}/tokenizer.json")),
                "catalog tokenizer_dir {tok_dir} has no fetched tokenizer.json"
            );
        }
    }

    #[test]
    fn llms_schema_and_pins() {
        let m = load_manifest(&llms_manifest()).unwrap();
        assert_eq!(m.schema_version, 1);
        let mut known: HashSet<&str> = LLM_SOURCES.iter().map(|s| s.alias).collect();
        known.extend(LLM_ALIASES.iter().map(|(a, _)| *a));
        let have: HashSet<&str> = m.models.keys().map(|s| s.as_str()).collect();
        assert_eq!(have, known);
        assert_eq!(
            m.models["default-llm"].alias_of.as_deref(),
            Some("qwen-0.5b")
        );
        for spec in LLM_SOURCES {
            let entry = &m.models[spec.alias];
            assert_eq!(entry.repo.as_deref(), Some(spec.repo));
            assert!(hex40(entry.revision.as_deref().unwrap()));
            assert_eq!(entry.dest.as_deref(), Some(spec.dest));
            let paths: Vec<&str> = entry.files.iter().map(|f| f.path.as_str()).collect();
            for required in spec.files {
                assert!(
                    paths.contains(required),
                    "{} missing {required}",
                    spec.alias
                );
            }
            for f in &entry.files {
                assert!(hex64(&f.sha256));
                assert!(f.size > 0);
            }
            let tok = entry.tokenizer.as_ref().expect("tokenizer pin");
            assert_eq!(tok.repo, spec.tokenizer_repo);
            assert!(hex40(&tok.revision));
            let tok_paths: Vec<&str> = tok.files.iter().map(|f| f.path.as_str()).collect();
            assert!(tok_paths.contains(&"tokenizer.json"));
            for f in &tok.files {
                assert!(hex64(&f.sha256));
                assert!(f.size > 0);
            }
        }
        let paths: Vec<&str> = m.models["qwen-7b"]
            .files
            .iter()
            .map(|f| f.path.as_str())
            .collect();
        assert!(paths.contains(&"qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf"));
        assert!(paths.contains(&"qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf"));
        for (alias, repo) in LLM_MLX_REPOS {
            let entry = &m.mlx_repos[*alias];
            assert_eq!(entry.repo, *repo);
            assert!(hex40(&entry.revision));
        }
        let (key, entry) = resolve_model_entry(&m, "default-llm").unwrap();
        assert_eq!(key, "qwen-0.5b");
        assert_eq!(entry.dest.as_deref(), Some("models/gguf/qwen-0.5b"));
        let specs = artifact_specs(entry).unwrap();
        let repos: HashSet<&str> = specs.iter().map(|s| s.repo.as_str()).collect();
        let paths: HashSet<&str> = specs.iter().map(|s| s.file.path.as_str()).collect();
        assert!(repos.contains("Qwen/Qwen2.5-0.5B-Instruct-GGUF"));
        assert!(repos.contains("Qwen/Qwen2.5-0.5B-Instruct"));
        assert!(paths.contains("qwen2.5-0.5b-instruct-q8_0.gguf"));
        assert!(paths.contains("tokenizer.json"));
    }

    #[test]
    fn llms_match_catalog() {
        let m = load_manifest(&llms_manifest()).unwrap();
        let catalog = catalog_text();
        for (alias, path) in catalog_nvidia_llamacpp_fetch_paths(&catalog).unwrap() {
            let (_key, entry) = resolve_model_entry(&m, &alias).unwrap();
            let fetched: HashSet<String> = artifact_specs(entry)
                .unwrap()
                .into_iter()
                .map(|s| format!("{}/{}", s.dest, s.file.path))
                .collect();
            assert!(
                fetched.contains(&path),
                "catalog GGUF path {path} not fetched for {alias}"
            );
        }
        for (alias, tok_dir) in catalog_apple_llm_tokenizer_dirs(&catalog).unwrap() {
            let (_key, entry) = resolve_model_entry(&m, &alias).unwrap();
            let fetched: HashSet<String> = artifact_specs(entry)
                .unwrap()
                .into_iter()
                .map(|s| format!("{}/{}", s.dest, s.file.path))
                .collect();
            assert!(
                fetched.contains(&format!("{tok_dir}/tokenizer.json")),
                "apple tokenizer_dir {tok_dir} has no fetched tokenizer.json"
            );
        }
    }

    fn make_fixture() -> (tempfile::TempDir, Manifest) {
        let tmp = tempfile::tempdir().unwrap();
        let payload = b"tiny onnx stand-in\n";
        let dest = tmp.path().join("models/onnx/tiny");
        fs::create_dir_all(&dest).unwrap();
        fs::write(dest.join("model.bin"), payload).unwrap();
        let digest = {
            let mut h = Sha256::new();
            h.update(payload);
            format!("{:x}", h.finalize())
        };
        let manifest = Manifest {
            schema_version: 1,
            comment: None,
            models: BTreeMap::from([(
                "tiny".into(),
                ModelEntry {
                    alias_of: None,
                    repo: Some("example/tiny".into()),
                    revision: Some("0".repeat(40)),
                    dest: Some("models/onnx/tiny".into()),
                    files: vec![FileEntry {
                        path: "model.bin".into(),
                        sha256: digest,
                        size: payload.len() as u64,
                    }],
                    tokenizer: None,
                    name: None,
                },
            )]),
            mlx_repos: BTreeMap::new(),
        };
        (tmp, manifest)
    }

    #[test]
    fn verify_ok() {
        let (tmp, manifest) = make_fixture();
        let mut out = Cursor::new(Vec::new());
        let mut err = Cursor::new(Vec::new());
        let rc = cmd_verify(&["tiny".into()], &manifest, tmp.path(), &mut out, &mut err).unwrap();
        assert_eq!(rc, 0);
    }

    #[test]
    fn verify_detects_corruption() {
        let (tmp, manifest) = make_fixture();
        fs::write(tmp.path().join("models/onnx/tiny/model.bin"), b"tampered").unwrap();
        let mut out = Cursor::new(Vec::new());
        let mut err = Cursor::new(Vec::new());
        let rc = cmd_verify(&["tiny".into()], &manifest, tmp.path(), &mut out, &mut err).unwrap();
        assert_eq!(rc, 1);
    }

    #[test]
    fn verify_detects_missing() {
        let (tmp, manifest) = make_fixture();
        fs::remove_file(tmp.path().join("models/onnx/tiny/model.bin")).unwrap();
        let mut out = Cursor::new(Vec::new());
        let mut err = Cursor::new(Vec::new());
        let rc = cmd_verify(&["tiny".into()], &manifest, tmp.path(), &mut out, &mut err).unwrap();
        assert_eq!(rc, 1);
    }

    #[test]
    fn fetch_is_idempotent_offline() {
        let (tmp, manifest) = make_fixture();
        let mut out = Cursor::new(Vec::new());
        let mut err = Cursor::new(Vec::new());
        let rc = cmd_fetch(
            &["tiny".into()],
            &manifest,
            tmp.path(),
            None,
            &mut out,
            &mut err,
        )
        .unwrap();
        assert_eq!(rc, 0);
    }

    #[test]
    fn stream_download_atomic_write_and_hash() {
        let tmp = tempfile::tempdir().unwrap();
        let payload = b"streamed bytes";
        let src = tmp.path().join("source.bin");
        fs::write(&src, payload).unwrap();
        let dest = tmp.path().join("out/copy.bin");
        let url = format!("file://{}", src.display());
        let (digest, size) = stream_download(&url, Some(&dest)).unwrap();
        let mut h = Sha256::new();
        h.update(payload);
        assert_eq!(digest, format!("{:x}", h.finalize()));
        assert_eq!(size, payload.len() as u64);
        assert_eq!(fs::read(&dest).unwrap(), payload);
        let leftovers: Vec<_> = fs::read_dir(dest.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.ends_with(".part"))
            })
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn manifest_rejects_unknown_schema() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.json");
        fs::write(&path, r#"{"schema_version":99,"models":{}}"#).unwrap();
        let err = load_manifest(&path).unwrap_err();
        assert!(err.to_string().contains("unsupported schema_version"));
    }

    #[test]
    fn ovms_dest_split() {
        let ovms = Path::new("/work/models/ovms-embedder");
        let hf = Path::new("/home/me/ovms-models");
        assert_eq!(
            ovms_artifact_path(
                "bge_base",
                ovms,
                hf,
                "tokenizer_bge_base/1/openvino_tokenizer.xml"
            ),
            ovms.join("tokenizer_bge_base/1/openvino_tokenizer.xml")
        );
        assert_eq!(
            ovms_artifact_path("bge_base", ovms, hf, "hf_tokenizer_bge_base/tokenizer.json"),
            hf.join("hf_tokenizer_bge_base/tokenizer.json")
        );
    }
}
