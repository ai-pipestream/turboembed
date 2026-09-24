//! `turbo-bundle`: import, verify, and inspect Turbo model bundles.
//!
//! ```text
//! turbo-bundle import --source <model dir> --output <bundle dir> --license Apache-2.0 \
//!     [--artifact onnx=model.onnx] [--artifact openvino_ir=model.xml] [--artifact gguf=model.gguf] \
//!     [--static-from model.safetensors[:tensor]] [--truncate-dims 256,128] [--max-batch 32]
//! turbo-bundle verify <bundle dir>
//! turbo-bundle inspect <bundle dir>
//! ```

#![deny(missing_docs)]

mod import;
mod safetensors;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use turbo_core::bundle::Bundle;

#[derive(Parser)]
#[command(name = "turbo-bundle", version, about = "Import, verify, and inspect Turbo model bundles")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

// The import subcommand has many flags; boxing them would only obscure the clap derive.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Command {
    /// Import a model directory into a new bundle.
    Import {
        /// Source model directory (sentence-transformers or Hugging Face layout).
        #[arg(long)]
        source: PathBuf,
        /// Output bundle directory; must not exist.
        #[arg(long)]
        output: PathBuf,
        /// SPDX license identifier of the weights (required).
        #[arg(long)]
        license: Option<String>,
        /// Model identifier to record (default: from config.json or the directory name).
        #[arg(long)]
        model_id: Option<String>,
        /// Revision or commit to record.
        #[arg(long)]
        revision: Option<String>,
        /// Task override: embed, rerank, classify, token_classify, generate, run.
        #[arg(long)]
        task: Option<String>,
        /// Kind override: embedding, reranker, classifier, token_classifier, generative, generic.
        #[arg(long)]
        kind: Option<String>,
        /// Artifact as format=path (onnx, openvino_ir, gguf, hef, hailo_tables, safetensors, hf_config). Repeatable.
        #[arg(long = "artifact", value_name = "FORMAT=PATH")]
        artifacts: Vec<String>,
        /// Build a `static` embedding table from a safetensors file: path[:tensor].
        #[arg(long, value_name = "PATH[:TENSOR]")]
        static_from: Option<String>,
        /// Matryoshka dimensions the model supports, comma separated.
        #[arg(long, value_delimiter = ',')]
        truncate_dims: Vec<u32>,
        /// Maximum batch to record (0 = provider default).
        #[arg(long, default_value_t = 0)]
        max_batch: u32,
        /// Mark the artifact as compiled for one fixed shape.
        #[arg(long)]
        fixed_shape: bool,
        /// Override the sequence limit.
        #[arg(long)]
        max_seq: Option<u32>,
    },
    /// Verify a bundle's manifest and file hashes.
    Verify {
        /// Bundle directory.
        bundle: PathBuf,
    },
    /// Print a bundle's manifest as JSON after verifying it.
    Inspect {
        /// Bundle directory.
        bundle: PathBuf,
    },
}

fn run() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.command {
        Command::Import {
            source,
            output,
            license,
            model_id,
            revision,
            task,
            kind,
            artifacts,
            static_from,
            truncate_dims,
            max_batch,
            fixed_shape,
            max_seq,
        } => {
            let mut parsed = Vec::new();
            for a in artifacts {
                let (f, p) = a.split_once('=').ok_or_else(|| format!("--artifact `{a}` must be FORMAT=PATH"))?;
                if f.is_empty() || p.is_empty() {
                    return Err(format!("--artifact `{a}` must be FORMAT=PATH"));
                }
                parsed.push((f.to_string(), PathBuf::from(p)));
            }
            let static_from = static_from.map(|s| match s.rsplit_once(':') {
                Some((p, t)) if !t.contains('/') && !t.contains('\\') && !p.is_empty() => {
                    (PathBuf::from(p), Some(t.to_string()))
                }
                _ => (PathBuf::from(s), None),
            });
            let report = import::import(&import::ImportRequest {
                source,
                output,
                model_id,
                revision,
                license,
                task,
                kind,
                artifacts: parsed,
                static_from,
                truncate_dims,
                max_batch,
                fixed_shape,
                max_seq,
            })?;
            for n in &report.notes {
                eprintln!("note: {n}");
            }
            println!("wrote {}", report.output.display());
            println!("{}", serde_json::to_string_pretty(&report.manifest).map_err(|e| e.to_string())?);
            Ok(())
        }
        Command::Verify { bundle } => {
            let b = Bundle::open(&bundle).map_err(|e| e.to_string())?;
            println!(
                "ok: {} ({}, {}) in {}",
                b.manifest().model_id,
                b.manifest().kind,
                b.manifest().artifacts.keys().cloned().collect::<Vec<_>>().join(", "),
                bundle.display()
            );
            Ok(())
        }
        Command::Inspect { bundle } => {
            let b = Bundle::open(&bundle).map_err(|e| e.to_string())?;
            println!("{}", serde_json::to_string_pretty(b.manifest()).map_err(|e| e.to_string())?);
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
