//! turbo-kserve: the Open Inference Protocol over gRPC for turboembed
//! bundles (docs/kserve.md).

use std::process::ExitCode;

use turbo_kserve::Server;
use turbo_kserve::config::{Args, USAGE};

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

#[tokio::main]
async fn main() -> ExitCode {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("turbo-kserve: {e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let server = match Server::start(args.listen, args.models, args.max_message_bytes).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("turbo-kserve: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("turbo-kserve: listening on {}", server.local_addr());
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
    server.stop().await;
    eprintln!("turbo-kserve: stopped");
    ExitCode::SUCCESS
}
