//! `inferstream-e2e` — run the shared suite against one live arch server,
//! or cross-arch embedding parity (`--parity-goldens` / `--parity-cross`).

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::process::ExitCode;
use std::str::FromStr;

use clap::Parser;
use inferstream_e2e::fetch::{default_arch_config, e2e_workspace_root, load_serve_list};
use inferstream_e2e::{
    ensure_plan, infer_target_from_addr, plan_corpus, plan_fetches, run_parity, run_suite,
    with_corpus, CatalogIndex, FetchScope, Matrix, ParityConfig, ParityMode, SuiteConfig,
    SuiteFilter, Target, DEFAULT_DRIFT_ALIASES, DEFAULT_PARITY_ALIASES,
};

#[derive(Parser, Debug)]
#[command(
    name = "inferstream-e2e",
    about = "Unified inferstream E2E harness: the same gRPC suite against nvidia / intel / apple (or the local mock). Parity: --parity-goldens / --parity-cross."
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

    /// Restrict embed/LLM/parity cases to these aliases (comma-separated).
    #[arg(long, value_delimiter = ',')]
    only: Vec<String>,

    /// Slice of the suite to run.
    #[arg(long, value_enum, default_value_t = SuiteFilter::All)]
    suite: SuiteFilter,

    /// `max_tokens` for ModelStreamInfer.
    #[arg(long, default_value_t = 32)]
    max_tokens: i64,

    /// Minimum cosine similarity when a golden file is present (regular suite).
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

    /// Also fetch the SHA-pinned text corpus (Tiny Shakespeare + STS pairs).
    /// Off by default so CI stays light. Env: `INFERSTREAM_E2E_FETCH_CORPUS=true`.
    #[arg(long, env = "INFERSTREAM_E2E_FETCH_CORPUS")]
    fetch_corpus: bool,

    /// Compare live Embed vectors to `testdata/e2e/goldens/<arch>/<alias>.json`.
    #[arg(long)]
    parity_goldens: bool,

    /// With `--parity-goldens`: write dumps instead of comparing.
    #[arg(long)]
    parity_write: bool,

    /// Pairwise cosine across `--peer` live addrs and/or `--dump` vector files.
    #[arg(long)]
    parity_cross: bool,

    /// Like `--parity-cross` over the popular-model drift list
    /// (`DEFAULT_DRIFT_ALIASES`: minilm, bge-*, e5-*, gte-*, …). Reuses
    /// the same cosine floors. See docs/turboembed-drift.md.
    #[arg(long)]
    drift: bool,

    /// Live peer for `--parity-cross` / `--parity-goldens`: `arch=host:port`.
    /// Repeatable. Env fallbacks: `INFERSTREAM_E2E_{NVIDIA,INTEL,APPLE}_ADDR`.
    #[arg(long = "peer", value_name = "ARCH=ADDR")]
    peers: Vec<String>,

    /// Saved vector dump for `--parity-cross`: `arch=path` (file or directory).
    #[arg(long = "dump", value_name = "ARCH=PATH")]
    dumps: Vec<String>,

    /// Max extra Shakespeare sentence units when the full soak file is present.
    #[arg(long, default_value_t = 24)]
    soak_limit: usize,
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

fn parse_arch_kv<T, E: std::fmt::Display>(
    raw: &[String],
    parse_val: impl Fn(&str) -> Result<T, E>,
    kind: &str,
) -> Result<BTreeMap<Target, T>, String> {
    let mut out = BTreeMap::new();
    for item in raw {
        let Some((arch, val)) = item.split_once('=') else {
            return Err(format!("error: {kind} must be arch=value (got {item:?})"));
        };
        let target =
            Target::from_str(arch.trim()).map_err(|e| format!("error: {kind} {item:?}: {e}"))?;
        let parsed = parse_val(val.trim()).map_err(|e| format!("error: {kind} {item:?}: {e}"))?;
        out.insert(target, parsed);
    }
    Ok(out)
}

fn env_peer(var: &str, target: Target, peers: &mut BTreeMap<Target, String>) {
    if peers.contains_key(&target) {
        return;
    }
    if let Ok(addr) = std::env::var(var) {
        let addr = addr.trim().to_string();
        if !addr.is_empty() {
            peers.insert(target, addr);
        }
    }
}

fn token_from(args: &Args) -> Option<String> {
    match &args.token {
        Some(t) if t.is_empty() => None,
        Some(t) => Some(t.clone()),
        None => Some("change-me".into()),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();
    if args.parity_goldens && args.parity_cross {
        eprintln!("error: --parity-goldens and --parity-cross are mutually exclusive");
        return ExitCode::from(2);
    }
    if args.drift && (args.parity_goldens || args.parity_cross) {
        eprintln!("error: --drift is mutually exclusive with --parity-goldens / --parity-cross");
        return ExitCode::from(2);
    }
    if args.parity_write && !args.parity_goldens {
        eprintln!("error: --parity-write requires --parity-goldens");
        return ExitCode::from(2);
    }

    let addr = args.addr.clone().unwrap_or_else(|| {
        args.target
            .unwrap_or(Target::Nvidia)
            .default_addr()
            .to_string()
    });
    let target = args.target.unwrap_or_else(|| infer_target_from_addr(&addr));

    let matrix = match &args.matrix {
        Some(path) => match Matrix::from_path(path) {
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

    let do_fetch =
        args.fetch || args.fetch_only || args.fetch_all || args.fetch_serve || args.fetch_corpus;
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
        let mut plan = if args.fetch || args.fetch_only || args.fetch_all || args.fetch_serve {
            plan_fetches(target, scope, &only, &serve, &matrix)
        } else {
            inferstream_e2e::FetchPlan::default()
        };
        if args.fetch_corpus {
            plan = with_corpus(plan);
        }
        if plan.batches.is_empty() && args.fetch_corpus {
            plan = plan_corpus();
        }
        println!("inferstream-e2e  fetch  target={target}  scope={scope}");
        print!("{}", plan.format());
        if let Err(e) = ensure_plan(&plan, &root) {
            eprintln!("error: fetch: {e}");
            return ExitCode::from(1);
        }
        if args.fetch_only && !args.parity_goldens && !args.parity_cross && !args.drift {
            println!("fetch-only: artifacts ensured, suite not run");
            return ExitCode::SUCCESS;
        }
    }

    if args.parity_goldens || args.parity_cross || args.drift {
        return run_parity_cli(&args, target, &addr, matrix, catalog).await;
    }

    let config = SuiteConfig {
        target,
        addr: addr.clone(),
        token: token_from(&args),
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

async fn run_parity_cli(
    args: &Args,
    target: Target,
    addr: &str,
    matrix: Matrix,
    catalog: CatalogIndex,
) -> ExitCode {
    let mut peers = match parse_arch_kv(&args.peers, |s| Ok::<_, String>(s.to_string()), "--peer") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let dumps = match parse_arch_kv(&args.dumps, |s| Ok::<_, String>(PathBuf::from(s)), "--dump") {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    env_peer("INFERSTREAM_E2E_NVIDIA_ADDR", Target::Nvidia, &mut peers);
    env_peer("INFERSTREAM_E2E_INTEL_ADDR", Target::Intel, &mut peers);
    env_peer("INFERSTREAM_E2E_APPLE_ADDR", Target::Apple, &mut peers);
    if args.parity_goldens && peers.is_empty() {
        peers.insert(target, addr.to_string());
    }

    let aliases = if !args.only.is_empty() {
        args.only.clone()
    } else if args.drift {
        DEFAULT_DRIFT_ALIASES
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    } else {
        DEFAULT_PARITY_ALIASES
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    };

    let goldens_dir = args
        .goldens
        .clone()
        .or_else(default_goldens)
        .unwrap_or_else(|| e2e_workspace_root().join("testdata/e2e/goldens"));

    let config = ParityConfig {
        mode: if args.parity_goldens {
            ParityMode::Goldens {
                write: args.parity_write,
            }
        } else {
            ParityMode::Cross
        },
        goldens_dir,
        aliases,
        workspace: e2e_workspace_root(),
        token: token_from(args),
        catalog,
        matrix,
        peers,
        dumps,
        soak_limit: args.soak_limit,
    };

    match run_parity(config).await {
        Ok(report) => {
            let label = if args.parity_goldens {
                "parity-goldens"
            } else if args.drift {
                "drift"
            } else {
                "parity-cross"
            };
            print!("{}", report.format(target, label));
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
