use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::sources::{HF_BASE, USER_AGENT};

pub const CHUNK: usize = 1 << 20;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub _comment: Option<String>,
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
pub struct FileEntry {
    pub path: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlxRepo {
    pub repo: String,
    pub revision: String,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Msg(String),
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("http {status}: {url}")]
    Http { status: u16, url: String },
}

impl From<String> for Error {
    fn from(value: String) -> Self {
        Self::Msg(value)
    }
}

impl From<&str> for Error {
    fn from(value: &str) -> Self {
        Self::Msg(value.to_string())
    }
}

pub fn load_manifest(path: &Path) -> Result<Manifest, Error> {
    let text = fs::read_to_string(path)?;
    let manifest: Manifest = serde_json::from_str(&text)?;
    if manifest.schema_version != 1 {
        return Err(Error::Msg(format!(
            "{}: unsupported schema_version {:?}",
            path.display(),
            manifest.schema_version
        )));
    }
    Ok(manifest)
}

pub fn write_manifest(path: &Path, manifest: &Manifest) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut body = serde_json::to_string_pretty(manifest)?;
    body.push('\n');
    fs::write(path, body)?;
    Ok(())
}

pub fn resolve_model_entry<'a>(
    manifest: &'a Manifest,
    alias: &str,
) -> Result<(String, &'a ModelEntry), Error> {
    let entry = manifest
        .models
        .get(alias)
        .ok_or_else(|| Error::Msg(format!("unknown alias {alias:?}")))?;
    if let Some(target) = &entry.alias_of {
        let target_entry = manifest.models.get(target).ok_or_else(|| {
            Error::Msg(format!("{alias}: alias_of {target:?} is not in the manifest"))
        })?;
        if target_entry.alias_of.is_some() {
            return Err(Error::Msg(format!(
                "{alias}: nested alias_of is not supported"
            )));
        }
        return Ok((target.clone(), target_entry));
    }
    Ok((alias.to_string(), entry))
}

pub struct ArtifactSpec {
    pub repo: String,
    pub revision: String,
    pub dest: String,
    pub file: FileEntry,
}

pub fn artifact_specs(entry: &ModelEntry) -> Result<Vec<ArtifactSpec>, Error> {
    let dest = entry
        .dest
        .clone()
        .ok_or_else(|| Error::Msg("manifest entry missing dest".into()))?;
    let repo = entry
        .repo
        .clone()
        .ok_or_else(|| Error::Msg("manifest entry missing repo".into()))?;
    let revision = entry
        .revision
        .clone()
        .ok_or_else(|| Error::Msg("manifest entry missing revision".into()))?;
    let mut out = Vec::new();
    for f in &entry.files {
        out.push(ArtifactSpec {
            repo: repo.clone(),
            revision: revision.clone(),
            dest: dest.clone(),
            file: f.clone(),
        });
    }
    if let Some(tok) = &entry.tokenizer {
        let tok_dest = tok.dest.clone().unwrap_or_else(|| dest.clone());
        for f in &tok.files {
            out.push(ArtifactSpec {
                repo: tok.repo.clone(),
                revision: tok.revision.clone(),
                dest: tok_dest.clone(),
                file: f.clone(),
            });
        }
    }
    Ok(out)
}

pub fn resolve_url(repo: &str, revision: &str, path: &str) -> String {
    format!("{HF_BASE}/{repo}/resolve/{revision}/{path}")
}

pub fn sha256_file(path: &Path) -> Result<String, Error> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_encode(&hasher.finalize()))
}

pub fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
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

fn http_get(url: &str) -> Result<ureq::Response, Error> {
    match ureq::get(url)
        .set("User-Agent", USER_AGENT)
        .timeout(std::time::Duration::from_secs(120))
        .call()
    {
        Ok(resp) => Ok(resp),
        Err(ureq::Error::Status(code, _)) => Err(Error::Http {
            status: code,
            url: url.to_string(),
        }),
        Err(e) => Err(Error::Msg(format!("{url}: {e}"))),
    }
}

pub fn repo_info(repo: &str) -> Result<(String, Vec<String>), Error> {
    let url = format!("{HF_BASE}/api/models/{repo}");
    let resp = http_get(&url)?;
    let body = resp.into_string().map_err(|e| Error::Msg(e.to_string()))?;
    let info: serde_json::Value = serde_json::from_str(&body)?;
    let sha = info
        .get("sha")
        .and_then(|v| v.as_str())
        .ok_or_else(|| Error::Msg(format!("HF API returned no commit sha for {repo}")))?
        .to_string();
    let files = info
        .get("siblings")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| s.get("rfilename").and_then(|v| v.as_str()).map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    Ok((sha, files))
}

/// Download `url`, returning (sha256, size). When `dest` is set, write
/// atomically (temp file in the same directory, then rename).
pub fn stream_download(url: &str, dest: Option<&Path>) -> Result<(String, u64), Error> {
    let resp = http_get(url)?;
    let mut reader = resp.into_reader();
    let mut hasher = Sha256::new();
    let mut size = 0u64;
    let mut tmp_path: Option<PathBuf> = None;
    let mut out: Option<fs::File> = None;
    if let Some(dest) = dest {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = dest.parent().unwrap().join(format!(
            ".{}.{}.part",
            dest.file_name().unwrap().to_string_lossy(),
            std::process::id()
        ));
        out = Some(fs::File::create(&tmp)?);
        tmp_path = Some(tmp);
        let _ = dest; // dest used after hash
    }
    let mut buf = vec![0u8; CHUNK];
    let result = (|| -> Result<(), Error> {
        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            size += n as u64;
            if let Some(file) = out.as_mut() {
                file.write_all(&buf[..n])?;
            }
        }
        if let Some(file) = out.as_mut() {
            file.sync_all()?;
        }
        Ok(())
    })();
    if result.is_err() {
        if let Some(tmp) = &tmp_path {
            let _ = fs::remove_file(tmp);
        }
        result?;
    }
    if let (Some(tmp), Some(dest)) = (tmp_path, dest) {
        fs::rename(&tmp, dest)?;
    }
    Ok((hex_encode(&hasher.finalize()), size))
}

pub fn cmd_list(manifest_path: &Path, fallback: &BTreeMap<String, String>) -> Result<i32, Error> {
    if manifest_path.exists() {
        let manifest = load_manifest(manifest_path)?;
        println!("{:<18} {:<42} revision", "alias", "repo");
        for (alias, entry) in &manifest.models {
            if let Some(target) = &entry.alias_of {
                println!("{alias:<18} alias_of {target}");
                continue;
            }
            let repo = entry.repo.as_deref().unwrap_or("-");
            let rev = entry.revision.as_deref().unwrap_or("-");
            let short = if rev.len() >= 12 { &rev[..12] } else { rev };
            println!("{alias:<18} {repo:<42} {short}");
        }
    } else {
        println!("{:<18} repo (manifest not generated yet)", "alias");
        for (alias, repo) in fallback {
            println!("{alias:<18} {repo}");
        }
    }
    Ok(0)
}

pub fn cmd_verify(aliases: &[String], manifest: &Manifest, root: &Path) -> Result<i32, Error> {
    let mut failures = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for alias in aliases {
        let (key, entry) = resolve_model_entry(manifest, alias)?;
        if !seen.insert(key.clone()) {
            println!("  skip      {alias} (alias of {key}, already verified)");
            continue;
        }
        if alias != &key {
            println!("--- {alias}  ->  {key} ---");
        }
        for spec in artifact_specs(entry)? {
            let path = root.join(&spec.dest).join(&spec.file.path);
            let label = format!("{key}: {}", path.strip_prefix(root).unwrap_or(&path).display());
            if !path.exists() {
                failures.push(format!("{label} — MISSING"));
                println!("  missing   {label}");
                continue;
            }
            let actual = sha256_file(&path)?;
            if actual != spec.file.sha256 {
                failures.push(format!(
                    "{label} — sha256 mismatch (expected {}, got {actual})",
                    spec.file.sha256
                ));
                println!("  MISMATCH  {label}");
            } else {
                println!("  ok        {label}");
            }
        }
    }
    if !failures.is_empty() {
        eprintln!("\nverify FAILED ({} problem(s)):", failures.len());
        for msg in &failures {
            eprintln!("  {msg}");
        }
        return Ok(1);
    }
    println!("\nverify OK — all files present with matching SHA-256.");
    Ok(0)
}

pub fn cmd_fetch(
    aliases: &[String],
    manifest: &Manifest,
    root: &Path,
    smoke_hint: Option<&str>,
) -> Result<i32, Error> {
    let mut downloaded = 0usize;
    let mut skipped = 0usize;
    let mut seen = std::collections::BTreeSet::new();
    for alias in aliases {
        let (key, entry) = resolve_model_entry(manifest, alias)?;
        if !seen.insert(key.clone()) {
            println!("--- {alias}  ->  {key} (already fetched) ---");
            continue;
        }
        let dest_root = root.join(entry.dest.as_deref().unwrap_or("."));
        let label = if alias == &key {
            alias.clone()
        } else {
            format!("{alias} -> {key}")
        };
        let repo = entry.repo.as_deref().unwrap_or("?");
        let rev = entry.revision.as_deref().unwrap_or("?");
        let short = if rev.len() >= 12 { &rev[..12] } else { rev };
        println!(
            "--- {label}  <-  {repo}@{short}  ->  {} ---",
            dest_root.strip_prefix(root).unwrap_or(&dest_root).display()
        );
        for spec in artifact_specs(entry)? {
            let path = root.join(&spec.dest).join(&spec.file.path);
            if path.exists() && sha256_file(&path)? == spec.file.sha256 {
                println!(
                    "  ok (cached)  {}  [{}]",
                    spec.file.path,
                    human(spec.file.size)
                );
                skipped += 1;
                continue;
            }
            if path.exists() {
                println!("  stale hash, re-downloading  {}", spec.file.path);
            }
            let url = resolve_url(&spec.repo, &spec.revision, &spec.file.path);
            println!(
                "  downloading  {}  [{}] ...",
                spec.file.path,
                human(spec.file.size)
            );
            let (actual, size) = match stream_download(&url, Some(&path)) {
                Ok(v) => v,
                Err(Error::Http { status, url }) => {
                    eprintln!("error: {url}: HTTP {status}");
                    return Ok(1);
                }
                Err(e) => return Err(e),
            };
            if actual != spec.file.sha256 {
                let _ = fs::remove_file(&path);
                eprintln!(
                    "error: SHA-256 mismatch for {key}/{}\n  expected {}\n  got      {actual}\n  url      {url}\nThe file was deleted. If upstream legitimately changed, re-pin with --update-manifest and review the diff.",
                    spec.file.path, spec.file.sha256
                );
                return Ok(1);
            }
            if size != spec.file.size {
                eprintln!(
                    "error: size mismatch for {key}/{} (expected {}, got {size})",
                    spec.file.path, spec.file.size
                );
                return Ok(1);
            }
            println!(
                "  verified     {}  sha256={}…",
                spec.file.path,
                &actual[..16.min(actual.len())]
            );
            downloaded += 1;
        }
    }
    println!("\nDone: {downloaded} downloaded, {skipped} already present and verified.");
    if let Some(hint) = smoke_hint {
        println!("{hint}");
    }
    Ok(0)
}

pub fn empty_manifest(comment: &str) -> Manifest {
    Manifest {
        schema_version: 1,
        _comment: Some(comment.to_string()),
        models: BTreeMap::new(),
        mlx_repos: BTreeMap::new(),
    }
}

pub fn hash_remote(
    repo: &str,
    revision: &str,
    rel: &str,
    dest: Option<&Path>,
) -> Result<(String, u64), Error> {
    let url = resolve_url(repo, revision, rel);
    stream_download(&url, dest)
}

pub fn pin_mlx_repo(alias: &str, repo: &str) -> Result<MlxRepo, Error> {
    let (revision, _) = repo_info(repo)?;
    println!("  mlx runtime repo {alias}: {repo}@{}", &revision[..12.min(revision.len())]);
    Ok(MlxRepo {
        repo: repo.to_string(),
        revision,
    })
}

pub fn select_aliases(
    requested: &[String],
    all: bool,
    known: &BTreeMap<String, String>,
) -> Result<Vec<String>, Error> {
    if all {
        return Ok(known.keys().cloned().collect());
    }
    if requested.is_empty() {
        return Err(Error::Msg(
            "error: no aliases given (use --all or --list)".into(),
        ));
    }
    let bad: Vec<&str> = requested
        .iter()
        .filter(|a| !known.contains_key(a.as_str()))
        .map(String::as_str)
        .collect();
    if !bad.is_empty() {
        return Err(Error::Msg(format!(
            "error: unknown alias(es): {}\nknown: {}",
            bad.join(", "),
            known.keys().cloned().collect::<Vec<_>>().join(", ")
        )));
    }
    let mut out = Vec::new();
    for a in requested {
        if !out.contains(a) {
            out.push(a.clone());
        }
    }
    Ok(out)
}

pub fn known_from_manifest(manifest: &Manifest) -> BTreeMap<String, String> {
    manifest
        .models
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.repo
                    .clone()
                    .or_else(|| v.alias_of.clone())
                    .unwrap_or_default(),
            )
        })
        .collect()
}
