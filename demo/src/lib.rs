//! A web page for trying a running turbo-kserve from a browser
//! (docs/demo.md). Browsers do not speak gRPC, so this serves the page and
//! bridges the Connect protocol to the server's gRPC: the page's protobuf
//! goes to the server, and the server's back, as the same bytes. It holds
//! no model, links no backend and decodes no message.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic::transport::{Channel, Endpoint};

pub mod allow;
mod codec;
pub mod connect;

pub use allow::Allow;

/// The largest message forwarded either way: the server's own default
/// limit.
pub const MAX_MESSAGE_BYTES: usize = 64 << 20;

const INDEX: &str = include_str!("../static/index.html");
const SCRIPT: &str = include_str!("../static/app.js");
const STYLE: &str = include_str!("../static/style.css");

#[derive(Clone)]
struct App {
    channel: Channel,
    allow: Arc<Allow>,
}

/// The demo, answering HTTP until stopped or dropped.
pub struct Demo {
    addr: SocketAddr,
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<std::io::Result<()>>,
}

impl Demo {
    /// Listens on `listen` and forwards the calls `allow` names to the
    /// server at `server` (`http://HOST:PORT`). The server need not be up:
    /// each call connects as needed.
    pub async fn start(listen: SocketAddr, server: &str, allow: Allow) -> Result<Demo, String> {
        let endpoint = Endpoint::from_shared(server.to_string()).map_err(|e| format!("server {server}: {e}"))?;
        if !matches!(endpoint.uri().scheme_str(), Some("http" | "https")) || endpoint.uri().authority().is_none() {
            return Err(format!("server {server}: must be an http:// or https:// URL"));
        }
        let app = App { channel: endpoint.connect_lazy(), allow: Arc::new(allow) };
        let router = Router::new()
            .route("/", get(|| async { asset("text/html; charset=utf-8", INDEX) }))
            .route("/app.js", get(|| async { asset("text/javascript; charset=utf-8", SCRIPT) }))
            .route("/style.css", get(|| async { asset("text/css; charset=utf-8", STYLE) }))
            .route("/{service}/{method}", post(connect::call).fallback(only_post))
            .with_state(app);
        let listener = TcpListener::bind(listen).await.map_err(|e| format!("listen on {listen}: {e}"))?;
        let addr = listener.local_addr().map_err(|e| format!("listen on {listen}: {e}"))?;
        let (stop, stopped) = oneshot::channel::<()>();
        let task = tokio::spawn(async move {
            axum::serve(listener, router)
                .with_graceful_shutdown(async {
                    let _ = stopped.await;
                })
                .await
        });
        Ok(Demo { addr, stop: Some(stop), task })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stops listening and waits for calls in flight.
    pub async fn stop(mut self) {
        if let Some(s) = self.stop.take() {
            let _ = s.send(());
        }
        let _ = (&mut self.task).await;
    }
}

impl Drop for Demo {
    fn drop(&mut self) {
        if let Some(s) = self.stop.take() {
            let _ = s.send(());
        }
    }
}

fn asset(kind: &'static str, body: &'static str) -> Response {
    ([(header::CONTENT_TYPE, kind), (header::CACHE_CONTROL, "no-cache")], body).into_response()
}

/// A call is a POST; GET serves only the page's files.
async fn only_post() -> Response {
    (StatusCode::METHOD_NOT_ALLOWED, [(header::ALLOW, "POST")]).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn start_rejects_server_without_http_or_https_scheme() {
        let listen: SocketAddr = "127.0.0.1:0".parse().unwrap();
        for bad in ["127.0.0.1:8081", "ftp://127.0.0.1:8081", "ws://127.0.0.1:8081", "/ just-a-path"] {
            assert!(Demo::start(listen, bad, Allow::default()).await.is_err(), "{bad}");
        }
    }
}
