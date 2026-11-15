//! A web page for trying a running turbo-kserve from a browser
//! (docs/demo.md). Browsers do not speak gRPC, so this serves the page and
//! bridges the Connect protocol to the server's gRPC: the page's protobuf
//! goes to the server, and the server's back, as the same bytes. It holds
//! no model and links no backend. It decodes no message, but for one
//! field: the user of a catalog call, which it sets to its own identity
//! (user.rs). One call it answers itself: which writes it forwards
//! (`DemoService/Describe`, demo/proto).

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tonic::transport::Channel;

pub mod allow;
mod budget;
mod codec;
pub mod connect;
mod frames;
pub mod gate;
pub mod limits;
mod tls;
pub mod upload;
mod upstream;
pub mod user;

pub use allow::Allow;
pub use tls::Tls;
pub use upstream::{ClientIdentity, UpstreamTls};

const INDEX: &str = include_str!("../static/index.html");
const SCRIPT: &str = include_str!("../static/app.js");
const STYLE: &str = include_str!("../static/style.css");

#[derive(Clone)]
struct App {
    channel: Channel,
    allow: Arc<Allow>,
    gate: Arc<gate::Gate>,
    user: Option<Arc<str>>,
    /// Turns true once the demo is stopping.
    stopping: watch::Receiver<bool>,
}

/// How the demo is reached and what it forwards.
#[derive(Debug, Clone, Default)]
pub struct Options {
    /// The methods forwarded.
    pub allow: Allow,
    /// A bearer token every call must carry. Required when the demo
    /// listens on an address other than loopback with a write allowed.
    pub token: Option<String>,
    /// Other origins whose pages may call, each `http(s)://HOST[:PORT]`.
    pub origins: Vec<String>,
    /// TLS for the listener; plain HTTP without.
    pub tls: Option<Tls>,
    /// TLS for the gRPC server; separate from the page listener.
    pub upstream_tls: Option<UpstreamTls>,
    /// Who the page's catalog calls act for: licence acceptances and
    /// installs are recorded for this user, never for a name the page
    /// sends. Without it those calls are read-only and name no user.
    pub user: Option<String>,
}

/// How long [`Demo::stop`] waits for calls in flight before cutting them.
pub const STOP_GRACE: Duration = Duration::from_secs(5);

/// The demo, answering HTTP until stopped or dropped.
pub struct Demo {
    addr: SocketAddr,
    stop: watch::Sender<bool>,
    task: JoinHandle<std::io::Result<()>>,
}

impl Demo {
    /// Listens on `listen` and forwards the calls `allow` names to the
    /// server at `server` (`http://HOST:PORT`). For HTTPS, use `start_with`
    /// and explicit upstream TLS trust. The server need not be up:
    /// each call connects as needed.
    pub async fn start(listen: SocketAddr, server: &str, allow: Allow) -> Result<Demo, String> {
        Demo::start_with(listen, server, Options { allow, ..Options::default() }).await
    }

    /// As `start`, with a token, other origins or TLS.
    pub async fn start_with(listen: SocketAddr, server: &str, options: Options) -> Result<Demo, String> {
        if !listen.ip().is_loopback() && options.allow.writes() && options.token.is_none() {
            return Err(format!(
                "listen on {listen}: a method that changes the server is allowed, so a token is required off loopback"
            ));
        }
        let gate = gate::Gate::new(options.token, &options.origins, options.tls.is_some())?;
        if let Some(u) = &options.user {
            user::check(u)?;
        }
        let tls = options.tls.as_ref().map(tls::config).transpose()?;
        let endpoint = upstream::endpoint(server, options.upstream_tls.as_ref())?;
        let (stop, stopping) = watch::channel(false);
        let mut stopped = stopping.clone();
        let app = App {
            stopping,
            channel: endpoint.connect_lazy(),
            allow: Arc::new(options.allow),
            gate: Arc::new(gate),
            user: options.user.map(Arc::from),
        };
        let router = Router::new()
            .route("/", get(|| async { asset("text/html; charset=utf-8", INDEX) }))
            .route("/app.js", get(|| async { asset("text/javascript; charset=utf-8", SCRIPT) }))
            .route("/style.css", get(|| async { asset("text/css; charset=utf-8", STYLE) }))
            .route(connect::DESCRIBE, post(connect::describe).options(preflight).fallback(only_post))
            .route("/{service}/{method}", post(connect::call).options(preflight).fallback(only_post))
            .route("/upload/{service}/{method}", put(upload::call).options(preflight).fallback(only_put))
            .with_state(app);
        let listener = TcpListener::bind(listen).await.map_err(|e| format!("listen on {listen}: {e}"))?;
        let addr = listener.local_addr().map_err(|e| format!("listen on {listen}: {e}"))?;
        let stopped = async move {
            let _ = stopped.wait_for(|s| *s).await;
        };
        let task = match tls {
            None => tokio::spawn(async move { axum::serve(listener, router).with_graceful_shutdown(stopped).await }),
            Some(config) => {
                let listener = tls::Listener::new(listener, config).map_err(|e| format!("listen on {listen}: {e}"))?;
                tokio::spawn(async move { axum::serve(listener, router).with_graceful_shutdown(stopped).await })
            }
        };
        Ok(Demo { addr, stop, task })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Stops listening, ends every open stream as `unavailable` and waits
    /// for the calls in flight, up to [`STOP_GRACE`]. See
    /// [`Self::stop_within`].
    pub async fn stop(self) {
        self.stop_within(STOP_GRACE).await;
    }

    /// [`Self::stop`] with the wait given: a call still open after `grace`
    /// (an upload its caller never finishes) is cut. Answers whether every
    /// call ended by itself.
    pub async fn stop_within(mut self, grace: Duration) -> bool {
        self.stop.send_replace(true);
        let clean = tokio::time::timeout(grace, &mut self.task).await.is_ok();
        if !clean {
            self.task.abort();
            let _ = (&mut self.task).await;
        }
        clean
    }
}

impl Drop for Demo {
    fn drop(&mut self) {
        self.stop.send_replace(true);
    }
}

fn asset(kind: &'static str, body: &'static str) -> Response {
    ([(header::CONTENT_TYPE, kind), (header::CACHE_CONTROL, "no-cache")], body).into_response()
}

/// A CORS preflight, granted only to a listed origin.
async fn preflight(State(app): State<App>, headers: HeaderMap) -> Response {
    app.gate.preflight(&headers)
}

/// A call is a POST; GET serves only the page's files.
async fn only_post() -> Response {
    (StatusCode::METHOD_NOT_ALLOWED, [(header::ALLOW, "POST")]).into_response()
}

/// An upload is a PUT.
async fn only_put() -> Response {
    (StatusCode::METHOD_NOT_ALLOWED, [(header::ALLOW, "PUT")]).into_response()
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
