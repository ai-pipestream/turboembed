//! `inferstream-e2e` — run the shared suite against one live arch server.

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use inferstream_e2e::{
    infer_target_from_addr, run_suite, CatalogIndex, Matrix, SuiteConfig, SuiteFilter, Target,
};

#[derive(Parser, Debug)]
#[command(
    name = "inferstream-e2e",
    about = "Unified inferstream E2E harness: the same gRPC suite against nvidia / intel / apple (or the local mock)."
)]
struct Args {
    /// Architecture of the server under test (`nvidia` | `intel` | `apple` | `mock`).
    #[arg(long, env = "INFERSTREAM_E2E_TARGET")]
    target: Option<Target>,

    /// `host:port` or `http://host:port`. Defaults to the live worker for `--target`.
    #[arg(long, env = "INFERSTREAM_E2E_ADDR")]
    addr: Option<String>,

    /// Bearer token. Empty string disables auth. Default `change-me`.
    #[arg(long, env = "INFERSTREAM_E2E_TOKEN")]
    token: Option<String>,

    /// Optional JSON matrix overriding the built-in alias / dim table.
    #[arg(long, env = "INFERSTREAM_E2E_MATRIX")]
    matrix: Option<PathBuf>,

    /// Directory of optional per-arch goldens (`<dir>/<target>/<alias>.json`).
    #[arg(long, env = "INFERSTREAM_E2E_GOLDENS")]
    goldens: Option<PathBuf>,

    /// Restrict embed/LLM cases to these aliases (comma-separated).
    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,

    /// Slice of the suite to run.
    #[arg(long, value_enum, default_value_t = SuiteFilter::All)]
    suite: SuiteFilter,

    /// `max_tokens` for ModelStreamInfer.
    #[arg(long, default_value_t = 32)]
    max_tokens: i64,

    /// Minimum cosine similarity when a golden file is present.
    #[arg(long, default_value_t = 0.99)]
    cosine_min: f32,
}

fn default_goldens() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("testdata/e2e/goldens"));
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        candidates.push(
            PathBuf::from(manifest)
                .join("../../testdata/e2e/goldens")
                .canonicalize()
                .unwrap_or_else(|_| {
                    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/e2e/goldens")
                }),
        );
    }
    candidates.into_iter().find(|p| p.is_dir())
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();
    let addr = args.addr.clone().unwrap_or_else(|| {
        args.target
            .unwrap_or(Target::Nvidia)
            .default_addr()
            .to_string()
    });
    let target = args.target.unwrap_or_else(|| infer_target_from_addr(&addr));

    let matrix = match args.matrix {
        Some(path) => match Matrix::from_path(&path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("error: matrix {}: {e}", path.display());
                return ExitCode::from(2);
            }
        },
        None => Matrix::builtin(),
    };
    let catalog = match CatalogIndex::builtin() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: catalog: {e}");
            return ExitCode::from(2);
        }
    };

    let token = match args.token {
        Some(t) if t.is_empty() => None,
        Some(t) => Some(t),
        None => Some("change-me".into()),
    };

    let config = SuiteConfig {
        target,
        addr: addr.clone(),
        token,
        matrix,
        catalog,
        goldens_dir: args.goldens.or_else(default_goldens),
        only: args.only.into_iter().collect::<HashSet<_>>(),
        filter: args.suite,
        max_tokens: args.max_tokens,
        cosine_min: args.cosine_min,
    };

    match run_suite(config).await {
        Ok(report) => {
            print!("{}", report.format(target, &addr));
            if report.failed() {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(1)
        }
    }
}
