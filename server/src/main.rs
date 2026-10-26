//! turbo-kserve: the Open Inference Protocol over gRPC for turboembed
//! bundles (docs/kserve.md).

use std::path::PathBuf;
use std::process::ExitCode;

use turbo_kserve::catalog::Catalog;
use turbo_kserve::config::{Args, USAGE};
use turbo_kserve::probe::{self, HEALTH_USAGE, Probe};
use turbo_kserve::proto::health::health_check_response::ServingStatus;
use turbo_kserve::{Serve, Server};

/// Resolves to the name of the first stop signal: SIGINT (Ctrl-C) or, on
/// Unix, SIGTERM as well, what an orchestrator sends to stop a service.
async fn stop_signal() -> &'static str {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
                return "SIGINT";
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => "SIGINT",
            _ = term.recv() => "SIGTERM",
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        "SIGINT"
    }
}

/// The catalog in `dir`, fetching over HTTPS.
#[cfg(feature = "catalog-http")]
fn catalog(dir: &str, index: Option<&str>) -> Result<Catalog, String> {
    Catalog::http(std::path::PathBuf::from(dir), index.map(std::path::Path::new))
}

/// A server built without `catalog-http` has no way to fetch a model.
#[cfg(not(feature = "catalog-http"))]
fn catalog(_dir: &str, _index: Option<&str>) -> Result<Catalog, String> {
    Err("this turbo-kserve was built without the catalog-http feature, so it cannot fetch models".into())
}

/// `turbo-kserve health ...`: one health check against a running server.
async fn health(args: impl Iterator<Item = String>) -> ExitCode {
    let p = match Probe::parse(args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("turbo-kserve health: {e}\n{HEALTH_USAGE}");
            return ExitCode::from(2);
        }
    };
    match probe::check(&p).await {
        Ok(ServingStatus::Serving) => {
            println!("SERVING");
            ExitCode::SUCCESS
        }
        Ok(s) => {
            println!("{}", s.as_str_name());
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("turbo-kserve health: {e}");
            ExitCode::FAILURE
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let mut argv = std::env::args().skip(1).peekable();
    if argv.peek().map(String::as_str) == Some("health") {
        argv.next();
        return health(argv).await;
    }
    let args = match Args::parse(argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("turbo-kserve: {e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let catalog = match &args.catalog_dir {
        Some(dir) => match catalog(dir, args.catalog_index.as_deref()) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("turbo-kserve: catalog: {e}");
                return ExitCode::FAILURE;
            }
        },
        None => None,
    };
    let serve = Serve {
        admin: args.admin,
        vocabulary: args.vocabulary_dir.map(PathBuf::from),
        catalog,
        training: args.training_dir.map(PathBuf::from),
        sink: args.sink,
        tls: args.tls,
        metrics_listen: args.metrics_listen,
        ..Serve::default()
    };
    let server = match Server::start(args.listen, args.models, args.max_message_bytes, serve).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("turbo-kserve: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("turbo-kserve: listening on {}", server.local_addr());
    if let Some(m_addr) = server.metrics_addr() {
        eprintln!("turbo-kserve: metrics on {m_addr}");
    }
    // A stop signal during load stops the server once the load returns:
    // the library has no way to stop a load part way.
    let stop = tokio::spawn(stop_signal());
    if let Err(e) = server.load().await {
        eprintln!("turbo-kserve: {e}");
        return ExitCode::FAILURE;
    }
    eprintln!("turbo-kserve: ready");
    let name = stop.await.unwrap_or("SIGINT");
    // The listener closes, requests in flight finish, and new connections
    // are refused; then the process exits.
    eprintln!("turbo-kserve: stopping on {name}");
    server.drain();
    server.stop().await;
    eprintln!("turbo-kserve: stopped");
    ExitCode::SUCCESS
}
