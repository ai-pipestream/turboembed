//! inferstream-fetch — hash-verified model artifact fetch / verify / re-pin.
//!
//! Replaces the former Python fetcher. Reads committed JSON manifests in
//! `models/manifests/` and talks to Hugging Face over HTTPS.

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use inferstream_fetch::{
    cmd_fetch, cmd_list, cmd_update_corpus_manifest, cmd_update_llm_manifest, cmd_update_manifest,
    cmd_update_ov_genai_manifest, cmd_update_rerank_manifest, cmd_verify, corpus_known_aliases,
    embedding_known_aliases, llm_known_aliases, load_manifest, ov_genai_known_aliases,
    rerank_known_aliases, select_aliases, FetchError,
};

#[derive(Parser, Debug)]
#[command(
    name = "inferstream-fetch",
    about = "Hash-verified fetch of inferstream model artifacts (embeddings by default; LLMs with --llms; OpenVINO GenAI with --ov-genai; text corpora with --corpus)."
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
    /// Operate on text corpora (`models/manifests/corpus.json`): Tiny
    /// Shakespeare soak text + STS-style sentence pairs.
    #[arg(long)]
    corpus: bool,
    /// Operate on TurboRerank cross-encoder weights (`models/manifests/rerankers.json`).
    #[arg(long)]
    rerankers: bool,
    /// Verify existing files against the manifest; no downloads.
    #[arg(long)]
    verify_only: bool,
    /// Maintainer mode: re-pin revisions, download, hash, rewrite manifest.
    #[arg(long)]
    update_manifest: bool,
    /// With --update-manifest: hash from the stream without writing files.
    #[arg(long)]
    no_store: bool,
    /// Manifest path (default depends on --llms / --ov-genai / --corpus).
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Repo root that dest paths are relative to.
    #[arg(long)]
    root: Option<PathBuf>,
}

fn default_manifest(
    root: &std::path::Path,
    llms: bool,
    ov_genai: bool,
    corpus: bool,
    rerankers: bool,
) -> PathBuf {
    let name = if ov_genai {
        "ov-genai-embeddings.json"
    } else if llms {
        "llms.json"
    } else if corpus {
        "corpus.json"
    } else if rerankers {
        "rerankers.json"
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
    let mode_flags = [args.llms, args.ov_genai, args.corpus, args.rerankers]
        .into_iter()
        .filter(|v| *v)
        .count();
    if mode_flags > 1 {
        return Err(FetchError::msg(
            "error: --llms, --ov-genai, --corpus, and --rerankers are mutually exclusive",
        ));
    }

    let root = args
        .root
        .clone()
        .unwrap_or_else(inferstream_fetch::workspace_root);
    let manifest_path = args.manifest.clone().unwrap_or_else(|| {
        default_manifest(&root, args.llms, args.ov_genai, args.corpus, args.rerankers)
    });

    let stdout = io::stdout();
    let stderr = io::stderr();
    let mut out = stdout.lock();
    let mut err = stderr.lock();

    if args.list {
        let fallback = if args.llms {
            llm_known_aliases()
        } else if args.ov_genai {
            ov_genai_known_aliases()
        } else if args.corpus {
            corpus_known_aliases()
        } else if args.rerankers {
            rerank_known_aliases()
        } else {
            embedding_known_aliases()
        };
        return cmd_list(&manifest_path, &fallback, &mut out);
    }

    if args.update_manifest {
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
        if args.corpus {
            let aliases = select_aliases(args.all, &args.aliases, &corpus_known_aliases())?;
            return cmd_update_corpus_manifest(
                &aliases,
                &manifest_path,
                &root,
                !args.no_store,
                &mut out,
                &mut err,
            );
        }
        if args.rerankers {
            let aliases = select_aliases(args.all, &args.aliases, &rerank_known_aliases())?;
            return cmd_update_rerank_manifest(
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
        return cmd_verify(&aliases, &manifest, &root, &mut out, &mut err);
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
             See docs/intel-genai-embed.md (Machine B Battlemage smoke).\n",
        )
    } else if args.corpus {
        Some(
            "Corpus ready under testdata/corpus/.\n\
             Chunk + embed: cargo run -p inferstream-e2e -- --parity-goldens\n\
             See testdata/corpus/README.md and docs/e2e-parity.md.\n",
        )
    } else if args.rerankers {
        Some(
            "Cross-encoder weights ready under models/rerank/.\n\
             Prove: make test-turborerank\n\
             See docs/turborerank-architecture.md.\n",
        )
    } else {
        None
    };
    cmd_fetch(&aliases, &manifest, &root, smoke, &mut out, &mut err)
}
