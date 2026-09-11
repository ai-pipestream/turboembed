//! Hash-verified fetch of inferstream model artifacts.
//!
//! Replaces `scripts/fetch_models.py`. No Python. Manifests stay JSON with
//! pinned HF revisions and SHA-256 per file.
//!
//!     cargo xtask fetch --embeddings --all
//!     cargo xtask fetch --llms qwen-0.5b
//!     cargo xtask fetch --mlx minilm default-llm
//!     cargo xtask verify --embeddings --all
//!     cargo xtask list --llms
//!     cargo xtask update-manifest --mlx --all

mod manifest;
mod sources;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use manifest::{
    cmd_fetch, cmd_list, cmd_verify, empty_manifest, hash_remote, known_from_manifest,
    load_manifest, pin_mlx_repo, repo_info, select_aliases, write_manifest, Error, FileEntry,
    ModelEntry, TokenizerEntry,
};

#[cfg(test)]
use manifest::{artifact_specs, resolve_model_entry};
use sources::{
    keep_mlx_file, llm_aliases, llm_known_aliases, llm_sources, mlx_embed_repos, mlx_llm_repos,
    onnx_file_list, onnx_repos,
};

#[derive(Parser)]
#[command(
    name = "inferstream-xtask",
    about = "Hash-verified model fetch / verify"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Fetch {
        aliases: Vec<String>,
        #[arg(long)]
        all: bool,
        #[arg(long)]
        embeddings: bool,
        #[arg(long)]
        llms: bool,
        #[arg(long)]
        mlx: bool,
        #[arg(long)]
        ovms: bool,
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Verify {
        aliases: Vec<String>,
        #[arg(long)]
        all: bool,
        #[arg(long)]
        embeddings: bool,
        #[arg(long)]
        llms: bool,
        #[arg(long)]
        mlx: bool,
        #[arg(long)]
        ovms: bool,
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    List {
        #[arg(long)]
        embeddings: bool,
        #[arg(long)]
        llms: bool,
        #[arg(long)]
        mlx: bool,
        #[arg(long)]
        ovms: bool,
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    UpdateManifest {
        aliases: Vec<String>,
        #[arg(long)]
        all: bool,
        #[arg(long)]
        embeddings: bool,
        #[arg(long)]
        llms: bool,
        #[arg(long)]
        mlx: bool,
        #[arg(long)]
        no_store: bool,
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long)]
        root: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Embeddings,
    Llms,
    Mlx,
    Ovms,
}

impl Kind {
    fn detect(embeddings: bool, llms: bool, mlx: bool, ovms: bool) -> Result<Self, Error> {
        let flags = [embeddings, llms, mlx, ovms]
            .into_iter()
            .filter(|b| *b)
            .count();
        if flags > 1 {
            return Err(Error::Msg(
                "specify only one of --embeddings / --llms / --mlx / --ovms".into(),
            ));
        }
        Ok(if llms {
            Self::Llms
        } else if mlx {
            Self::Mlx
        } else if ovms {
            Self::Ovms
        } else {
            Self::Embeddings
        })
    }

    fn default_manifest(self) -> &'static str {
        match self {
            Self::Embeddings => "models/manifests/embeddings.json",
            Self::Llms => "models/manifests/llms.json",
            Self::Mlx => "models/manifests/mlx.json",
            Self::Ovms => "models/manifests/ovms-embeddings.json",
        }
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code as u8),
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<i32, Error> {
    let cli = Cli::parse();
    match cli.command {
        Command::List {
            embeddings,
            llms,
            mlx,
            ovms,
            manifest,
        } => {
            let kind = Kind::detect(embeddings, llms, mlx, ovms)?;
            let root = repo_root();
            let path = manifest.unwrap_or_else(|| root.join(kind.default_manifest()));
            let fallback = fallback_repos(kind);
            cmd_list(&path, &fallback)
        }
        Command::Verify {
            aliases,
            all,
            embeddings,
            llms,
            mlx,
            ovms,
            manifest,
            root,
            out,
        } => operate(
            Kind::detect(embeddings, llms, mlx, ovms)?,
            aliases,
            all,
            manifest,
            root,
            out,
            Mode::Verify,
        ),
        Command::Fetch {
            aliases,
            all,
            embeddings,
            llms,
            mlx,
            ovms,
            manifest,
            root,
            out,
        } => operate(
            Kind::detect(embeddings, llms, mlx, ovms)?,
            aliases,
            all,
            manifest,
            root,
            out,
            Mode::Fetch,
        ),
        Command::UpdateManifest {
            aliases,
            all,
            embeddings,
            llms,
            mlx,
            no_store,
            manifest,
            root,
        } => {
            let kind = Kind::detect(embeddings, llms, mlx, false)?;
            if kind == Kind::Ovms {
                return Err(Error::Msg(
                    "Intel OVMS IR export is not invoked from xtask (OpenVINO's own \
                     conversion toolchain). Re-hash existing IR with verify --ovms."
                        .into(),
                ));
            }
            let root = root.unwrap_or_else(repo_root);
            let path = manifest.unwrap_or_else(|| root.join(kind.default_manifest()));
            match kind {
                Kind::Embeddings => update_embeddings(&aliases, all, &path, &root, !no_store),
                Kind::Llms => update_llms(&aliases, all, &path, &root, !no_store),
                Kind::Mlx => update_mlx(&aliases, all, &path, &root, !no_store),
                Kind::Ovms => unreachable!(),
            }
        }
    }
}

enum Mode {
    Fetch,
    Verify,
}

fn fallback_repos(kind: Kind) -> BTreeMap<String, String> {
    match kind {
        Kind::Embeddings => onnx_repos()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        Kind::Llms => llm_known_aliases(),
        Kind::Mlx => {
            let mut m = BTreeMap::new();
            for (k, v) in mlx_embed_repos() {
                m.insert(k.to_string(), v.to_string());
            }
            for (k, v) in mlx_llm_repos() {
                m.insert(k.to_string(), v.to_string());
            }
            m
        }
        Kind::Ovms => BTreeMap::new(),
    }
}

fn operate(
    kind: Kind,
    aliases: Vec<String>,
    all: bool,
    manifest: Option<PathBuf>,
    root: Option<PathBuf>,
    out: Option<PathBuf>,
    mode: Mode,
) -> Result<i32, Error> {
    let root = root.unwrap_or_else(repo_root);
    let path = manifest.unwrap_or_else(|| root.join(kind.default_manifest()));
    if !path.exists() {
        return Err(Error::Msg(format!(
            "error: manifest not found: {}\nGenerate it with update-manifest (maintainers) or fetch it from git.",
            path.display()
        )));
    }
    let manifest = load_manifest(&path)?;
    let known = known_from_manifest(&manifest);
    let aliases = select_aliases(&aliases, all, &known)?;
    let verify_root = if kind == Kind::Ovms {
        out.unwrap_or_else(|| PathBuf::from("/work/models/ovms-embedder"))
    } else {
        root.clone()
    };
    match mode {
        Mode::Verify => cmd_verify(&aliases, &manifest, &verify_root),
        Mode::Fetch => {
            if kind == Kind::Ovms {
                eprintln!(
                    "Intel OVMS artifacts are pre-exported IR (not downloaded here).\n\
                     Place them under --out (default /work/models/ovms-embedder) and run:\n  \
                     cargo xtask verify --ovms --all"
                );
                return cmd_verify(&aliases, &manifest, &verify_root);
            }
            let hint = match kind {
                Kind::Llms => Some(
                    "Add the aliases to `serve` in the arch config and restart;\n\
                     verify with: scripts/smoke-llms.sh <host:port> <bearer-token>",
                ),
                Kind::Mlx => Some(
                    "Apple loads these directories through native MLX (Swift FFI).\n\
                     verify with: scripts/smoke-apple.sh",
                ),
                Kind::Embeddings => Some(
                    "Add the aliases to `serve` in config/nvidia.toml and restart;\n\
                     verify with: scripts/smoke-embeddings.sh <host:port> <bearer-token>",
                ),
                Kind::Ovms => None,
            };
            cmd_fetch(&aliases, &manifest, &root, hint)
        }
    }
}

fn update_embeddings(
    aliases: &[String],
    all: bool,
    manifest_path: &Path,
    root: &Path,
    store: bool,
) -> Result<i32, Error> {
    let known: BTreeMap<String, String> = onnx_repos()
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    let aliases = select_aliases(aliases, all, &known)?;
    let mut manifest = if manifest_path.exists() {
        load_manifest(manifest_path)?
    } else {
        empty_manifest(
            "SHA-256 manifest for inferstream ONNX embedding artifacts. Generated by \
             cargo xtask update-manifest --embeddings; do not edit hashes by hand.",
        )
    };
    for alias in &aliases {
        let repo = *onnx_repos()
            .get(alias.as_str())
            .ok_or_else(|| Error::Msg(format!("unknown embedding alias {alias}")))?;
        println!("--- pinning {alias}  <-  {repo} ---");
        let (revision, repo_files) = repo_info(repo)?;
        println!("  revision {revision}");
        let dest = format!("models/onnx/{alias}");
        let rels = onnx_file_list(&repo_files).map_err(|e| Error::Msg(format!("{repo}: {e}")))?;
        let mut files = Vec::new();
        for rel in rels {
            let target = store.then(|| root.join(&dest).join(&rel));
            println!("  hashing {rel} ...");
            let (digest, size) = hash_remote(repo, &revision, &rel, target.as_deref())?;
            println!("    sha256={digest}  size={}", manifest::human(size));
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
            },
        );
        if let Some(mlx_repo) = mlx_embed_repos().get(alias.as_str()) {
            manifest
                .mlx_repos
                .insert(alias.clone(), pin_mlx_repo(alias, mlx_repo)?);
        }
    }
    write_manifest(manifest_path, &manifest)?;
    println!(
        "\nManifest written: {} — review and commit it.",
        manifest_path
            .strip_prefix(root)
            .unwrap_or(manifest_path)
            .display()
    );
    Ok(0)
}

fn update_llms(
    aliases: &[String],
    all: bool,
    manifest_path: &Path,
    root: &Path,
    store: bool,
) -> Result<i32, Error> {
    let aliases = select_aliases(aliases, all, &llm_known_aliases())?;
    let mut manifest = if manifest_path.exists() {
        load_manifest(manifest_path)?
    } else {
        empty_manifest(
            "SHA-256 manifest for inferstream LLM artifacts. Generated by \
             cargo xtask update-manifest --llms; do not edit hashes by hand. \
             default-llm is alias_of qwen-0.5b.",
        )
    };
    let mut families = Vec::new();
    for alias in &aliases {
        families.push(
            llm_aliases()
                .get(alias.as_str())
                .copied()
                .unwrap_or(alias.as_str())
                .to_string(),
        );
    }
    families.sort();
    families.dedup();

    let sources = llm_sources();
    for alias in &families {
        let spec = sources
            .get(alias.as_str())
            .ok_or_else(|| Error::Msg(format!("unknown LLM family {alias}")))?;
        println!("--- pinning {alias}  <-  {} ---", spec.repo);
        let (revision, repo_files) = repo_info(spec.repo)?;
        println!("  revision {revision}");
        let missing: Vec<_> = spec
            .files
            .iter()
            .copied()
            .filter(|f| !repo_files.iter().any(|r| r == f))
            .collect();
        if !missing.is_empty() {
            eprintln!(
                "error: {}: required file(s) not in repo: {missing:?}",
                spec.repo
            );
            return Ok(1);
        }
        let dest = spec.dest.to_string();
        let mut files = Vec::new();
        for rel in spec.files {
            let target = store.then(|| root.join(&dest).join(rel));
            println!("  hashing {rel} ...");
            let (digest, size) = hash_remote(spec.repo, &revision, rel, target.as_deref())?;
            println!("    sha256={digest}  size={}", manifest::human(size));
            files.push(FileEntry {
                path: rel.to_string(),
                sha256: digest,
                size,
            });
        }
        println!("  pinning tokenizer  <-  {}", spec.tokenizer_repo);
        let (tok_rev, tok_files) = repo_info(spec.tokenizer_repo)?;
        println!("    revision {tok_rev}");
        let tok_missing: Vec<_> = spec
            .tokenizer_files
            .iter()
            .copied()
            .filter(|f| !tok_files.iter().any(|r| r == f))
            .collect();
        if !tok_missing.is_empty() {
            eprintln!(
                "error: {}: required file(s) not in repo: {tok_missing:?}",
                spec.tokenizer_repo
            );
            return Ok(1);
        }
        let mut tok_hashed = Vec::new();
        for rel in spec.tokenizer_files {
            let target = store.then(|| root.join(&dest).join(rel));
            println!("  hashing {rel} ...");
            let (digest, size) =
                hash_remote(spec.tokenizer_repo, &tok_rev, rel, target.as_deref())?;
            println!("    sha256={digest}  size={}", manifest::human(size));
            tok_hashed.push(FileEntry {
                path: rel.to_string(),
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
            },
        );
    }
    for (alias, target) in llm_aliases() {
        if aliases.iter().any(|a| a == alias || a == target) || families.iter().any(|f| f == target)
        {
            manifest.models.insert(
                alias.to_string(),
                ModelEntry {
                    alias_of: Some(target.to_string()),
                    repo: None,
                    revision: None,
                    dest: None,
                    files: Vec::new(),
                    tokenizer: None,
                },
            );
        }
    }
    for (alias, mlx_repo) in mlx_llm_repos() {
        let in_scope = aliases.iter().any(|a| a == alias)
            || llm_aliases()
                .get(alias)
                .is_some_and(|t| families.iter().any(|f| f == t))
            || families.iter().any(|f| f == alias);
        if in_scope {
            manifest
                .mlx_repos
                .insert(alias.to_string(), pin_mlx_repo(alias, mlx_repo)?);
        }
    }
    write_manifest(manifest_path, &manifest)?;
    println!(
        "\nManifest written: {} — review and commit it.",
        manifest_path
            .strip_prefix(root)
            .unwrap_or(manifest_path)
            .display()
    );
    Ok(0)
}

fn update_mlx(
    aliases: &[String],
    all: bool,
    manifest_path: &Path,
    root: &Path,
    store: bool,
) -> Result<i32, Error> {
    let known = fallback_repos(Kind::Mlx);
    let aliases = select_aliases(aliases, all, &known)?;
    let mut manifest = if manifest_path.exists() {
        load_manifest(manifest_path)?
    } else {
        empty_manifest(
            "SHA-256 manifest for inferstream Apple MLX weights (safetensors + config + tokenizer). \
             Generated by cargo xtask update-manifest --mlx; do not edit hashes by hand. \
             Runtime loads these directories via native MLX (Swift FFI) — no Python.",
        )
    };
    let mut repos = BTreeMap::new();
    repos.extend(mlx_embed_repos());
    repos.extend(mlx_llm_repos());

    // default-llm shares qwen-0.5b weights.
    let mut families = Vec::new();
    for alias in &aliases {
        if alias == "default-llm" {
            families.push("qwen-0.5b".to_string());
        } else {
            families.push(alias.clone());
        }
    }
    families.sort();
    families.dedup();

    for alias in &families {
        let repo = *repos
            .get(alias.as_str())
            .ok_or_else(|| Error::Msg(format!("unknown MLX alias {alias}")))?;
        println!("--- pinning {alias}  <-  {repo} ---");
        let (revision, repo_files) = repo_info(repo)?;
        println!("  revision {revision}");
        let dest = format!("models/mlx/{alias}");
        let rels: Vec<String> = {
            let mut v: Vec<String> = repo_files
                .into_iter()
                .filter(|f| keep_mlx_file(f))
                .collect();
            v.sort();
            v
        };
        if rels.is_empty() {
            return Err(Error::Msg(format!("{repo}: no MLX weight files to pin")));
        }
        let mut files = Vec::new();
        for rel in rels {
            let target = store.then(|| root.join(&dest).join(&rel));
            println!("  hashing {rel} ...");
            let (digest, size) = hash_remote(repo, &revision, &rel, target.as_deref())?;
            println!("    sha256={digest}  size={}", manifest::human(size));
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
                revision: Some(revision.clone()),
                dest: Some(dest),
                files,
                tokenizer: None,
            },
        );
        manifest.mlx_repos.insert(
            alias.clone(),
            manifest::MlxRepo {
                repo: repo.to_string(),
                revision,
            },
        );
    }
    if aliases.iter().any(|a| a == "default-llm") || families.iter().any(|f| f == "qwen-0.5b") {
        manifest.models.insert(
            "default-llm".into(),
            ModelEntry {
                alias_of: Some("qwen-0.5b".into()),
                repo: None,
                revision: None,
                dest: None,
                files: Vec::new(),
                tokenizer: None,
            },
        );
        if let Some(q) = manifest.mlx_repos.get("qwen-0.5b").cloned() {
            manifest.mlx_repos.insert("default-llm".into(), q);
        }
    }
    write_manifest(manifest_path, &manifest)?;
    println!(
        "\nManifest written: {} — review and commit it.",
        manifest_path
            .strip_prefix(root)
            .unwrap_or(manifest_path)
            .display()
    );
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn hex40(s: &str) -> bool {
        s.len() == 40 && s.chars().all(|c| c.is_ascii_hexdigit())
    }
    fn hex64(s: &str) -> bool {
        s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())
    }

    fn root() -> PathBuf {
        repo_root()
    }

    #[test]
    fn embeddings_manifest_covers_onnx_repos() {
        let m = load_manifest(&root().join("models/manifests/embeddings.json")).unwrap();
        assert_eq!(m.schema_version, 1);
        let expected: std::collections::BTreeSet<_> =
            onnx_repos().keys().copied().map(str::to_string).collect();
        let actual: std::collections::BTreeSet<_> = m.models.keys().cloned().collect();
        assert_eq!(actual, expected);
        for (alias, entry) in &m.models {
            assert_eq!(
                entry.repo.as_deref(),
                Some(*onnx_repos().get(alias.as_str()).unwrap())
            );
            assert!(hex40(entry.revision.as_deref().unwrap()));
            assert_eq!(
                entry.dest.as_deref(),
                Some(format!("models/onnx/{alias}").as_str())
            );
            let paths: Vec<_> = entry.files.iter().map(|f| f.path.as_str()).collect();
            for req in sources::ONNX_FILES {
                assert!(paths.contains(req), "{alias} missing {req}");
            }
            for f in &entry.files {
                assert!(hex64(&f.sha256));
                assert!(f.size > 0);
            }
        }
        for alias in ["bge-m3", "e5-large"] {
            let paths: Vec<_> = m.models[alias]
                .files
                .iter()
                .map(|f| f.path.as_str())
                .collect();
            assert!(paths.contains(&"onnx/model.onnx_data"), "{alias}");
        }
    }

    #[test]
    fn llm_manifest_structure() {
        let m = load_manifest(&root().join("models/manifests/llms.json")).unwrap();
        assert_eq!(m.schema_version, 1);
        assert_eq!(
            m.models["default-llm"].alias_of.as_deref(),
            Some("qwen-0.5b")
        );
        let (key, entry) = resolve_model_entry(&m, "default-llm").unwrap();
        assert_eq!(key, "qwen-0.5b");
        let specs = artifact_specs(entry).unwrap();
        let paths: Vec<_> = specs.iter().map(|s| s.file.path.as_str()).collect();
        assert!(paths.contains(&"qwen2.5-0.5b-instruct-q8_0.gguf"));
        assert!(paths.contains(&"tokenizer.json"));
        let q7 = &m.models["qwen-7b"];
        let q7_paths: Vec<_> = q7.files.iter().map(|f| f.path.as_str()).collect();
        assert!(q7_paths.iter().any(|p| p.contains("00001-of-00002")));
        assert!(q7_paths.iter().any(|p| p.contains("00002-of-00002")));
    }

    #[test]
    fn catalog_nvidia_ort_paths_are_fetched() {
        let m = load_manifest(&root().join("models/manifests/embeddings.json")).unwrap();
        let catalog: toml::Value =
            toml::from_str(&fs::read_to_string(root().join("config/catalog.toml")).unwrap())
                .unwrap();
        for (alias, tables) in catalog["models"].as_table().unwrap() {
            let Some(nvidia) = tables.get("nvidia") else {
                continue;
            };
            if nvidia.get("backend").and_then(|v| v.as_str()) != Some("ort") {
                continue;
            }
            let Some(path) = nvidia.get("path").and_then(|v| v.as_str()) else {
                continue;
            };
            if !path.starts_with("models/onnx/") {
                continue;
            }
            let entry = m.models.get(alias).expect(alias);
            let fetched: Vec<_> = entry
                .files
                .iter()
                .map(|f| format!("{}/{}", entry.dest.as_deref().unwrap(), f.path))
                .collect();
            assert!(fetched.iter().any(|f| f == path), "{alias} path {path}");
        }
    }

    #[test]
    fn catalog_apple_llm_tokenizer_is_in_llm_manifest() {
        let m = load_manifest(&root().join("models/manifests/llms.json")).unwrap();
        let catalog: toml::Value =
            toml::from_str(&fs::read_to_string(root().join("config/catalog.toml")).unwrap())
                .unwrap();
        for (alias, tables) in catalog["models"].as_table().unwrap() {
            let Some(apple) = tables.get("apple") else {
                continue;
            };
            if apple.get("backend").and_then(|v| v.as_str()) != Some("mlx") {
                continue;
            }
            let Some(tok_dir) = apple.get("tokenizer_dir").and_then(|v| v.as_str()) else {
                continue;
            };
            if !tok_dir.starts_with("models/gguf/") && !tok_dir.starts_with("models/mlx/") {
                continue;
            }
            if !m.models.contains_key(alias) {
                continue;
            }
            let (_key, entry) = resolve_model_entry(&m, alias).unwrap();
            let fetched: Vec<_> = artifact_specs(entry)
                .unwrap()
                .into_iter()
                .map(|s| format!("{}/{}", s.dest, s.file.path))
                .collect();
            assert!(
                fetched
                    .iter()
                    .any(|f| f == &format!("{tok_dir}/tokenizer.json")),
                "{alias} tokenizer_dir {tok_dir} not fetched: {fetched:?}"
            );
        }
    }

    #[test]
    fn verify_fixture_ok_and_corruption() {
        let tmp = std::env::temp_dir().join(format!(
            "inferstream-xtask-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("models/onnx/tiny")).unwrap();
        let payload = b"tiny onnx stand-in\n";
        fs::write(tmp.join("models/onnx/tiny/model.bin"), payload).unwrap();
        let digest = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(payload);
            manifest::hex_encode(&h.finalize())
        };
        let mut models = BTreeMap::new();
        models.insert(
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
            },
        );
        let manifest = manifest::Manifest {
            schema_version: 1,
            _comment: None,
            models,
            mlx_repos: BTreeMap::new(),
        };
        assert_eq!(cmd_verify(&["tiny".into()], &manifest, &tmp).unwrap(), 0);
        fs::write(tmp.join("models/onnx/tiny/model.bin"), b"tampered").unwrap();
        assert_eq!(cmd_verify(&["tiny".into()], &manifest, &tmp).unwrap(), 1);
        fs::remove_file(tmp.join("models/onnx/tiny/model.bin")).unwrap();
        assert_eq!(cmd_verify(&["tiny".into()], &manifest, &tmp).unwrap(), 1);
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn unknown_schema_rejected() {
        let tmp =
            std::env::temp_dir().join(format!("inferstream-xtask-schema-{}", std::process::id()));
        fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("bad.json");
        fs::write(&path, r#"{"schema_version":99,"models":{}}"#).unwrap();
        assert!(load_manifest(&path).is_err());
        let _ = fs::remove_dir_all(&tmp);
    }
}
