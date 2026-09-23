//! `inferstream`: the Turbo inference server.
//!
//! Two listeners over one engine: HTTP (the Open Inference Protocol v2
//! REST binding, the OpenAI-shaped `/v1/embeddings`, `/v1/rerank`,
//! `/v1/chat/completions` with streaming, `/v1/classify`, and `/info`) and
//! gRPC (`inference.GRPCInferenceService`). Every model is a bundle on a
//! named provider device with a pool of fixed-shape sessions; see
//! `server/README.md`.
//!
//! ```text
//! inferstream --provider-lib target/debug/libturbo_provider_cuda.so \
//!     --model name=minilm,bundle=~/opt/bundles/minilm-onnx,provider=cuda \
//!     --model bundle=~/opt/bundles/qwen05-gguf,provider=ggml,generations=2 \
//!     [--http 0.0.0.0:8000] [--grpc 0.0.0.0:8001]
//! inferstream --config server.json
//! ```

#![deny(missing_docs)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;

use turbo_inferstream::config::Config;
use turbo_inferstream::engine::Engine;
use turbo_inferstream::{grpc, http};

#[derive(Parser)]
#[command(
    name = "inferstream",
    version,
    about = "Turbo inference server: OIP v2 (gRPC and REST) and OpenAI-shaped routes"
)]
struct Cli {
    /// JSON configuration file (`provider_libs`, `models`); flags add to it.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Provider library to load; repeatable.
    #[arg(long = "provider-lib")]
    provider_libs: Vec<PathBuf>,
    /// Model to serve: `bundle=DIR,provider=ID[,name=N][,ordinal=K][,buckets=1x128;8x256][,sessions=S][,generations=G]`; repeatable.
    #[arg(long = "model")]
    models: Vec<String>,
    /// HTTP listen address (the flag overrides the variable).
    #[arg(long, env = "INFERSTREAM_HTTP", default_value = "127.0.0.1:8000")]
    http: SocketAddr,
    /// gRPC listen address (the flag overrides the variable).
    #[arg(long, env = "INFERSTREAM_GRPC", default_value = "127.0.0.1:8001")]
    grpc: SocketAddr,
    /// Directory of static pages to serve at `/` (for example `demo/search`).
    #[arg(long)]
    pages: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let mut config = match &cli.config {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) => match serde_json::from_str::<Config>(&text) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("error: {}: {e}", path.display());
                    return ExitCode::from(2);
                }
            },
            Err(e) => {
                eprintln!("error: {}: {e}", path.display());
                return ExitCode::from(2);
            }
        },
        None => Config::default(),
    };
    config.provider_libs.extend(cli.provider_libs.iter().cloned());
    for m in &cli.models {
        match Config::parse_model_flag(m) {
            Ok(spec) => config.models.push(spec),
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        }
    }
    let engine = match Engine::load(&config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let http_listener = match tokio::net::TcpListener::bind(cli.http).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("error: cannot listen on {} (http): {e}", cli.http);
            return ExitCode::from(2);
        }
    };
    let pages = cli.pages.clone().or(config.pages.clone());
    let app = match &pages {
        Some(dir) => match http::with_pages(http::router(engine.clone()), dir) {
            Ok(app) => app,
            Err(e) => {
                eprintln!("error: --pages {}: {e}", dir.display());
                return ExitCode::from(2);
            }
        },
        None => http::router(engine.clone()),
    };
    let grpc_svc = grpc::GrpcInferenceServiceServer::new(grpc::Service { engine: engine.clone() });
    let ext_svc = grpc::InferstreamExtensionServer::new(grpc::ExtService { engine: engine.clone() });
    let reflection = match tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(grpc::FILE_DESCRIPTOR_SET)
        .build_v1()
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: gRPC reflection: {e}");
            return ExitCode::from(2);
        }
    };
    eprintln!("inferstream: http on http://{}  grpc on {}  models: {}", cli.http, cli.grpc, engine.names().join(", "));

    let http_task = tokio::spawn(async move {
        axum::serve(http_listener, app).with_graceful_shutdown(shutdown()).await.map_err(|e| format!("http: {e}"))
    });
    let grpc_addr = cli.grpc;
    let grpc_task = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(grpc_svc)
            .add_service(ext_svc)
            .add_service(reflection)
            .serve_with_shutdown(grpc_addr, shutdown())
            .await
            .map_err(|e| format!("grpc on {grpc_addr}: {e}"))
    });
    let (h, g) = tokio::join!(http_task, grpc_task);
    let mut failed = false;
    for r in [h, g] {
        match r {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("error: {e}");
                failed = true;
            }
            Err(e) => {
                eprintln!("error: server task: {e}");
                failed = true;
            }
        }
    }
    let _ = Arc::strong_count(&engine);
    if failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

/// Resolves on SIGINT (and SIGTERM on Unix).
async fn shutdown() {
    #[cfg(unix)]
    {
        let mut term = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: cannot listen for SIGTERM: {e}");
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
