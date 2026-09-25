//! turbo-kserve: the Open Inference Protocol over gRPC for turboembed
//! bundles (docs/kserve.md).

use std::process::ExitCode;

use turbo_kserve::Server;
use turbo_kserve::config::{Args, USAGE};

#[tokio::main]
async fn main() -> ExitCode {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("turbo-kserve: {e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let server = match Server::start(args.listen, args.models).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("turbo-kserve: {e}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!("turbo-kserve: listening on {}", server.local_addr());
    if let Err(e) = server.load().await {
        eprintln!("turbo-kserve: {e}");
        return ExitCode::FAILURE;
    }
    eprintln!("turbo-kserve: ready");
    let _ = tokio::signal::ctrl_c().await;
    server.stop().await;
    ExitCode::SUCCESS
}
