//! `inferstream-e2e` — run the shared suite against one live arch server.

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use inferstream_e2e::fetch::{default_arch_config, e2e_workspace_root, load_serve_list};
use inferstream_e2e::{
    ensure_plan, infer_target_from_addr, plan_fetches, run_suite, CatalogIndex, FetchScope, Matrix,
    SuiteConfig, SuiteFilter, Target,
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

    /// Fetch missing SHA-256-pinned artifacts for this target, then run the suite.
    /// Idempotent: files already on disk with a matching hash are skipped.
    /// Default set is required embeds (`minilm`) plus `default-llm` / `qwen-0.5b`.
    /// Mock target is a no-op. Env: `INFERSTREAM_E2E_FETCH=true`.
    #[arg(long, env = "INFERSTREAM_E2E_FETCH")]
    fetch: bool,

    /// Like `--fetch`, but exit after ensuring artifacts (no gRPC).
    #[arg(long)]
    fetch_only: bool,

    /// With `--fetch`: every matrix alias on this target (includes `qwen-7b`).
    #[arg(long)]
    fetch_all: bool,

    /// With `--fetch`: aliases from `config/<arch>.toml` `serve`.
    #[arg(long)]
    fetch_serve: bool,
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

    let do_fetch = args.fetch || args.fetch_only || args.fetch_all || args.fetch_serve;
    if args.fetch_all && args.fetch_serve {
        eprintln!("error: --fetch-all and --fetch-serve are mutually exclusive");
        return ExitCode::from(2);
    }
    if do_fetch {
        let scope = if args.fetch_all {
            FetchScope::All
        } else if args.fetch_serve {
            FetchScope::Serve
        } else {
            FetchScope::Default
        };
        let only = args.only.iter().cloned().collect::<HashSet<_>>();
        let root = e2e_workspace_root();
        let serve = if scope == FetchScope::Serve {
            match load_serve_list(&default_arch_config(&root, target)) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("error: serve list: {e}");
                    return ExitCode::from(2);
                }
            }
        } else {
            Vec::new()
        };
        let plan = plan_fetches(target, scope, &only, &serve, &matrix);
        println!("inferstream-e2e  fetch  target={target}  scope={scope}");
        print!("{}", plan.format());
        if let Err(e) = ensure_plan(&plan, &root) {
            eprintln!("error: fetch: {e}");
            return ExitCode::from(1);
        }
        if args.fetch_only {
            println!("fetch-only: artifacts ensured, suite not run");
            return ExitCode::SUCCESS;
        }
    }

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
