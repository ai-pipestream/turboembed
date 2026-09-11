//! inferstream binary: load config, build the routing table, serve gRPC.

use clap::Parser;
use tracing_subscriber::EnvFilter;

use inferstream_server::config::Config;

/// Multi-backend KServe OIP V2 gRPC streaming inference façade.
#[derive(Debug, Parser)]
#[command(name = "inferstream", version)]
struct Args {
    /// Path to the TOML config file.
    #[arg(
        long,
        short,
        env = "INFERSTREAM_CONFIG",
        default_value = "config/example.toml"
    )]
    config: String,

    /// Override the listen address from the config file.
    #[arg(long, env = "INFERSTREAM_LISTEN")]
    listen: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let mut config = Config::from_file(&args.config)?;
    if let Some(listen) = args.listen {
        config.listen = listen;
    }

    let (bound_tx, _bound_rx) = tokio::sync::oneshot::channel();
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutdown signal received");
    };
    inferstream_server::serve(config, bound_tx, shutdown).await?;
    Ok(())
}
