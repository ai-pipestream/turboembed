//! inferstream-fetch — hash-verified model artifact fetch / verify / re-pin.
//!
//! Replaces the former Python fetcher. Reads committed JSON manifests in
//! `models/manifests/` and talks to Hugging Face over HTTPS.

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use inferstream_fetch::{
    cmd_fetch, cmd_list, cmd_update_llm_manifest, cmd_update_manifest,
    cmd_update_ov_genai_manifest, cmd_verify, cmd_verify_ovms, embedding_known_aliases,
    llm_known_aliases, load_manifest, ov_genai_known_aliases, select_aliases, FetchError,
};

#[derive(Parser, Debug)]
#[command(
    name = "inferstream-fetch",
    about = "Hash-verified fetch of inferstream model artifacts (embeddings by default; LLMs with --llms; OpenVINO GenAI with --ov-genai; OVMS verify with --ovms)."
)]
struct Args {
    /// Catalog aliases to fetch (or use --all).
    aliases: Vec<String>,
    /// Operate on every alias in the selected manifest / source table.
    #[arg(long)]
    all: bool,
    /// List aliases and pinned sources.
    #[arg(long)]
    list: bool,
    /// Operate on generative LLM artifacts (`models/manifests/llms.json`).
    #[arg(long)]
    llms: bool,
    /// Operate on Intel in-process OpenVINO GenAI dirs (`models/manifests/ov-genai-embeddings.json`).
    #[arg(long)]
    ov_genai: bool,
    /// Operate on Intel OVMS embedding artifacts (`models/manifests/ovms-embeddings.json`).
    /// Verify / list only — IR export is a one-off outside this tool.
    #[arg(long)]
    ovms: bool,
    /// Verify existing files against the manifest; no downloads.
    #[arg(long)]
    verify_only: bool,
    /// Maintainer mode: re-pin revisions, download, hash, rewrite manifest.
    #[arg(long)]
    update_manifest: bool,
    /// With --update-manifest: hash from the stream without writing files.
    #[arg(long)]
    no_store: bool,
    /// Manifest path (default depends on --llms / --ov-genai / --ovms).
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Repo root that dest paths are relative to.
    #[arg(long)]
    root: Option<PathBuf>,
    /// OVMS model dir (`--ovms` verify). Default: `/work/models/ovms-embedder`.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Dir for `hf_tokenizer_<name>/` (`--ovms` verify). Default: `$HOME/ovms-models`.
    #[arg(long)]
    hf_out: Option<PathBuf>,
}

fn default_root() -> PathBuf {
    // Walk up from CWD (and from the executable) looking for the workspace
    // Cargo.toml + models/manifests. Falls back to CWD.
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.to_path_buf());
        }
    }
    // cargo run puts the binary in target/debug — walk up from there too.
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let p = PathBuf::from(manifest_dir);
        if let Some(ws) = p.parent().and_then(|p| p.parent()) {
            candidates.push(ws.to_path_buf());
        }
    }
    for start in candidates {
        let mut cur = start;
        loop {
            if cur.join("models/manifests").is_dir() && cur.join("Cargo.toml").is_file() {
                return cur;
            }
            if !cur.pop() {
                break;
            }
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn default_manifest(root: &std::path::Path, llms: bool, ov_genai: bool, ovms: bool) -> PathBuf {
    let name = if ovms {
        "ovms-embeddings.json"
    } else if ov_genai {
        "ov-genai-embeddings.json"
    } else if llms {
        "llms.json"
    } else {
        "embeddings.json"
    };
    root.join("models/manifests").join(name)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            let _ = writeln!(io::stderr(), "{e}");
            // usage-style errors from select_aliases already include "error:"
            match e {
                FetchError::Msg(s) if s.starts_with("error: no aliases") => ExitCode::from(2),
                FetchError::Msg(s) if s.starts_with("error: unknown alias") => ExitCode::from(2),
                _ => ExitCode::from(1),
            }
        }
    }
}

fn run() -> inferstream_fetch::Result<i32> {
    let args = Args::parse();
    let mode_flags = [args.llms, args.ov_genai, args.ovms]
        .into_iter()
        .filter(|v| *v)
        .count();
    if mode_flags > 1 {
        return Err(FetchError::msg(
            "error: --llms, --ov-genai, and --ovms are mutually exclusive",
        ));
    }

    let root = args.root.clone().unwrap_or_else(default_root);
    let manifest_path = args
        .manifest
        .clone()
        .unwrap_or_else(|| default_manifest(&root, args.llms, args.ov_genai, args.ovms));

    let stdout = io::stdout();
    let stderr = io::stderr();
    let mut out = stdout.lock();
    let mut err = stderr.lock();

    if args.list {
        let fallback = if args.llms {
            llm_known_aliases()
        } else if args.ov_genai {
            ov_genai_known_aliases()
        } else if args.ovms {
            // Prefer the committed manifest; fall back to empty (list still
            // works once the file is present).
            Default::default()
        } else {
            embedding_known_aliases()
        };
        return cmd_list(&manifest_path, &fallback, &mut out);
    }

    if args.update_manifest {
        if args.ovms {
            writeln!(
                err,
                "error: --update-manifest --ovms is a one-off IR export, not part of fetch.\n\
                 See contrib/offline-once/README.md (OpenVINO + torch; not invoked by Make / CI)."
            )?;
            return Ok(2);
        }
        if args.llms {
            let aliases = select_aliases(args.all, &args.aliases, &llm_known_aliases())?;
            return cmd_update_llm_manifest(
                &aliases,
                &manifest_path,
                &root,
                !args.no_store,
                &mut out,
                &mut err,
            );
        }
        if args.ov_genai {
            let aliases = select_aliases(args.all, &args.aliases, &ov_genai_known_aliases())?;
            return cmd_update_ov_genai_manifest(
                &aliases,
                &manifest_path,
                &root,
                !args.no_store,
                &mut out,
                &mut err,
            );
        }
        let aliases = select_aliases(args.all, &args.aliases, &embedding_known_aliases())?;
        return cmd_update_manifest(
            &aliases,
            &manifest_path,
            &root,
            !args.no_store,
            &mut out,
            &mut err,
        );
    }

    if !manifest_path.exists() {
        writeln!(
            err,
            "error: manifest not found: {}\nGenerate it with --update-manifest (maintainers) or fetch it from git.",
            manifest_path.display()
        )?;
        return Ok(1);
    }
    let manifest = load_manifest(&manifest_path)?;
    let known: std::collections::BTreeMap<String, String> = manifest
        .models
        .keys()
        .map(|k| (k.clone(), String::new()))
        .collect();
    let aliases = select_aliases(args.all, &args.aliases, &known)?;

    if args.verify_only {
        if args.ovms {
            let ovms_out = args
                .out
                .unwrap_or_else(|| PathBuf::from("/work/models/ovms-embedder"));
            let hf_out = args.hf_out.unwrap_or_else(|| {
                let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
                PathBuf::from(home).join("ovms-models")
            });
            return cmd_verify_ovms(&aliases, &manifest, &ovms_out, &hf_out, &mut out, &mut err);
        }
        return cmd_verify(&aliases, &manifest, &root, &mut out, &mut err);
    }

    if args.ovms {
        writeln!(
            err,
            "error: fetching OVMS IR is a one-off export (OpenVINO + torch), not a download.\n\
             Verify existing artifacts with --ovms --verify-only.\n\
             See contrib/offline-once/README.md."
        )?;
        return Ok(2);
    }

    let smoke = if args.llms {
        Some(
            "Add the aliases to `serve` in the arch config and restart;\n\
             verify with: scripts/smoke-llms.sh <host:port> <bearer-token>\n",
        )
    } else if args.ov_genai {
        Some(
            "Add the aliases to `serve` in config/intel.toml and rebuild with --features openvino-genai;\n\
             verify with: scripts/smoke-embeddings.sh <host:port> <bearer-token>\n\
             See docs/intel-genai-embed.md (krick-1 Battlemage smoke).\n",
        )
    } else {
        None
    };
    cmd_fetch(&aliases, &manifest, &root, smoke, &mut out, &mut err)
}
