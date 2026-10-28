//! Tower gRPC metrics/deadline middleware and HTTP `/metrics` listener.

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use bytes::Bytes;
use http_body::{Body, Frame, SizeHint};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic::Code;
use tonic::codegen::http::{HeaderMap, Request, Response};

use super::Metrics;
use crate::deadline::{Deadline, deadline_status};
use crate::state::State;

/// Tower layer instrumenting every gRPC request with Prometheus metrics and
/// `grpc-timeout` deadline propagation.
#[derive(Clone)]
pub(crate) struct MetricsLayer {
    metrics: Arc<Metrics>,
}

impl MetricsLayer {
    pub(crate) fn new(metrics: Arc<Metrics>) -> Self {
        Self { metrics }
    }
}

impl<S> tower::Layer<S> for MetricsLayer {
    type Service = MetricsService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        MetricsService { inner, metrics: Arc::clone(&self.metrics) }
    }
}

#[derive(Clone)]
pub(crate) struct MetricsService<S> {
    inner: S,
    metrics: Arc<Metrics>,
}

impl<S, ReqBody, ResBody> tower::Service<Request<ReqBody>> for MetricsService<S>
where
    S: tower::Service<Request<ReqBody>, Response = Response<ResBody>> + Clone + Send + 'static,
    S::Future: Send + 'static,
    ResBody: Body<Data = Bytes> + Default + Send + 'static,
{
    type Response = Response<MetricsBody<ResBody>>;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request<ReqBody>) -> Self::Future {
        let start = Instant::now();
        let (service, method) = split_grpc_path(req.uri().path());
        let metrics = Arc::clone(&self.metrics);

        let deadline = match Deadline::from_headers(req.headers()) {
            Ok(d) => d,
            Err(status) => {
                let res = status_response(status, Arc::clone(&metrics), service, method, start);
                return Box::pin(async move { Ok(res) });
            }
        };
        if deadline.is_expired() {
            let res = status_response(deadline_status(), Arc::clone(&metrics), service, method, start);
            return Box::pin(async move { Ok(res) });
        }
        req.extensions_mut().insert(deadline.clone());
        let hard_at = deadline.hard_at();
        let fut = self.inner.call(req);

        Box::pin(async move {
            let mut cancel_guard = CallCancelGuard {
                metrics: Arc::clone(&metrics),
                service: service.clone(),
                method: method.clone(),
                start,
                disarmed: false,
            };
            let res = match hard_at {
                Some(at) => tokio::select! {
                    biased;
                    r = fut => r,
                    () = tokio::time::sleep_until(at) => {
                        deadline.trip();
                        cancel_guard.disarmed = true;
                        return Ok(status_response(deadline_status(), metrics, service, method, start));
                    }
                },
                None => fut.await,
            };
            let res = match res {
                Ok(r) => {
                    cancel_guard.disarmed = true;
                    r
                }
                Err(e) => {
                    cancel_guard.disarmed = true;
                    metrics.record_rpc(&service, &method, Code::Unknown, start.elapsed());
                    return Err(e);
                }
            };
            let res = crate::too_large(res);
            let initial_code = parse_grpc_status(res.headers());
            Ok(res.map(|body| MetricsBody {
                inner: Box::pin(body),
                metrics,
                service,
                method,
                start,
                code: initial_code,
                recorded: false,
            }))
        })
    }
}

struct CallCancelGuard {
    metrics: Arc<Metrics>,
    service: String,
    method: String,
    start: Instant,
    disarmed: bool,
}

impl Drop for CallCancelGuard {
    fn drop(&mut self) {
        if !self.disarmed {
            self.metrics.record_rpc(&self.service, &self.method, Code::Cancelled, self.start.elapsed());
        }
    }
}

fn status_response<B: Default>(
    status: tonic::Status,
    metrics: Arc<Metrics>,
    service: String,
    method: String,
    start: Instant,
) -> Response<MetricsBody<B>> {
    let code = status.code();
    let http_res: Response<B> = status.into_http();
    metrics.record_rpc(&service, &method, code, start.elapsed());
    http_res.map(|body| MetricsBody {
        inner: Box::pin(body),
        metrics,
        service,
        method,
        start,
        code: Some(code),
        recorded: true,
    })
}

/// Response body wrapper that inspects gRPC trailers and records RPC count and
/// latency when the response finishes or drops.
pub(crate) struct MetricsBody<B> {
    inner: Pin<Box<B>>,
    metrics: Arc<Metrics>,
    service: String,
    method: String,
    start: Instant,
    code: Option<Code>,
    recorded: bool,
}

impl<B> MetricsBody<B> {
    fn finish(&mut self, default_code: Code) {
        if !self.recorded {
            self.recorded = true;
            let code = self.code.unwrap_or(default_code);
            self.metrics.record_rpc(&self.service, &self.method, code, self.start.elapsed());
        }
    }
}

impl<B> Drop for MetricsBody<B> {
    fn drop(&mut self) {
        self.finish(Code::Cancelled);
    }
}

impl<B: Body<Data = Bytes>> Body for MetricsBody<B> {
    type Data = Bytes;
    type Error = B::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        match self.inner.as_mut().poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(trailers) = frame.trailers_ref()
                    && let Some(c) = parse_grpc_status(trailers)
                {
                    self.code = Some(c);
                }
                Poll::Ready(Some(Ok(frame)))
            }
            Poll::Ready(Some(Err(e))) => {
                self.finish(Code::Unknown);
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                self.finish(Code::Cancelled);
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

include!(concat!(env!("OUT_DIR"), "/grpc_methods.rs"));

/// The service and method labels of a gRPC path: those of a method the
/// protos define, else `unknown` for both, so a client cannot add series
/// by sending junk paths.
fn split_grpc_path(path: &str) -> (String, String) {
    let known = GRPC_METHODS.binary_search(&path).is_ok();
    match path.trim_start_matches('/').split_once('/') {
        Some((svc, method)) if known => (svc.to_owned(), method.to_owned()),
        _ => ("unknown".to_owned(), "unknown".to_owned()),
    }
}

fn parse_grpc_status(headers: &HeaderMap) -> Option<Code> {
    let val = headers.get("grpc-status")?.to_str().ok()?;
    let n: i32 = val.parse().ok()?;
    Some(Code::from_i32(n))
}

pub(super) fn code_name(code: Code) -> &'static str {
    match code {
        Code::Ok => "OK",
        Code::Cancelled => "CANCELLED",
        Code::Unknown => "UNKNOWN",
        Code::InvalidArgument => "INVALID_ARGUMENT",
        Code::DeadlineExceeded => "DEADLINE_EXCEEDED",
        Code::NotFound => "NOT_FOUND",
        Code::AlreadyExists => "ALREADY_EXISTS",
        Code::PermissionDenied => "PERMISSION_DENIED",
        Code::ResourceExhausted => "RESOURCE_EXHAUSTED",
        Code::FailedPrecondition => "FAILED_PRECONDITION",
        Code::Aborted => "ABORTED",
        Code::OutOfRange => "OUT_OF_RANGE",
        Code::Unimplemented => "UNIMPLEMENTED",
        Code::Internal => "INTERNAL",
        Code::Unavailable => "UNAVAILABLE",
        Code::DataLoss => "DATA_LOSS",
        Code::Unauthenticated => "UNAUTHENTICATED",
    }
}

/// Starts the HTTP `/metrics` listener on `listen`, returning the bound address,
/// shutdown sender, and server task handle.
pub(crate) async fn start_http_listener(
    listen: SocketAddr,
    state: Arc<State>,
) -> Result<(SocketAddr, oneshot::Sender<()>, JoinHandle<()>), String> {
    let listener = TcpListener::bind(listen).await.map_err(|e| format!("metrics listen on {listen}: {e}"))?;
    let addr = listener.local_addr().map_err(|e| format!("metrics listen on {listen}: {e}"))?;
    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                accepted = listener.accept() => {
                    let Ok((mut stream, _)) = accepted else { continue };
                    let state = Arc::clone(&state);
                    tokio::spawn(async move {
                        let mut buf = [0u8; 1024];
                        let n = stream.read(&mut buf).await.unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        let req = String::from_utf8_lossy(&buf[..n]);
                        let first_line = req.lines().next().unwrap_or("");
                        let mut parts = first_line.split_whitespace();
                        let method = parts.next().unwrap_or("");
                        let path = parts.next().unwrap_or("");
                        let path_only = path.split('?').next().unwrap_or(path);
                        let (status, content_type, body) = if method != "GET" && method != "HEAD" {
                            ("405 Method Not Allowed", "text/plain; charset=utf-8", "method not allowed\n".to_owned())
                        } else if path_only == "/metrics" {
                            ("200 OK", "text/plain; version=0.0.4; charset=utf-8", state.metrics.render(&state))
                        } else {
                            ("404 Not Found", "text/plain; charset=utf-8", "not found\n".to_owned())
                        };
                        let head = format!(
                            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = stream.write_all(head.as_bytes()).await;
                        if method != "HEAD" {
                            let _ = stream.write_all(body.as_bytes()).await;
                        }
                        let _ = stream.shutdown().await;
                    });
                }
            }
        }
    });
    Ok((addr, stop_tx, handle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_defined_methods_get_their_own_labels() {
        assert!(GRPC_METHODS.windows(2).all(|w| w[0] < w[1]), "binary search needs the list sorted");
        let named = |s: &str, m: &str| (s.to_owned(), m.to_owned());
        assert_eq!(
            split_grpc_path("/inference.GRPCInferenceService/ServerReady"),
            named("inference.GRPCInferenceService", "ServerReady")
        );
        assert_eq!(split_grpc_path("/grpc.health.v1.Health/Check"), named("grpc.health.v1.Health", "Check"));
        for junk in [
            "/inference.GRPCInferenceService/Nope",
            "/junk.Svc/x",
            "/",
            "",
            "/a/b/c",
            "inference.GRPCInferenceService/ServerReady",
        ] {
            assert_eq!(split_grpc_path(junk), named("unknown", "unknown"), "{junk:?}");
        }
    }
}
