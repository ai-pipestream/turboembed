//! The ceiling on one call's request body: every frame of a stream
//! counted as it arrives, gRPC's five-byte prefixes included, and the call
//! refused RESOURCE_EXHAUSTED once the total passes the ceiling. Nothing
//! is held: each frame goes on to the service as it came, so memory
//! follows the frame, not the request.

use std::pin::Pin;
use std::task::{Context, Poll};

use http_body::{Body as HttpBody, Frame, SizeHint};
use tonic::Status;
use tonic::body::Body;
use tonic::codegen::{Bytes, http};

/// A request body counted against `ceiling`.
pub(crate) struct Capped {
    inner: Body,
    seen: u64,
    ceiling: u64,
}

/// The request with its body counted against `ceiling`.
pub(crate) fn cap(ceiling: u64) -> impl Fn(http::Request<Body>) -> http::Request<Body> + Clone {
    move |r: http::Request<Body>| r.map(|inner| Body::new(Capped { inner, seen: 0, ceiling }))
}

impl HttpBody for Capped {
    type Data = Bytes;
    type Error = Status;

    fn poll_frame(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Frame<Bytes>, Status>>> {
        let this = &mut *self;
        if this.seen > this.ceiling {
            return Poll::Ready(None);
        }
        match Pin::new(&mut this.inner).poll_frame(cx) {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    this.seen += data.len() as u64;
                    if this.seen > this.ceiling {
                        let msg = format!("the request stream is over max_stream_bytes {}", this.ceiling);
                        return Poll::Ready(Some(Err(Status::resource_exhausted(msg))));
                    }
                }
                Poll::Ready(Some(Ok(frame)))
            }
            other => other,
        }
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn read(body: &mut Capped) -> Vec<Result<usize, Status>> {
        let mut out = Vec::new();
        while let Some(f) = std::future::poll_fn(|cx| Pin::new(&mut *body).poll_frame(cx)).await {
            out.push(f.map(|f| f.into_data().map(|d| d.len()).unwrap_or(0)));
        }
        out
    }

    #[tokio::test]
    async fn a_body_at_the_ceiling_passes_and_one_byte_over_is_refused() {
        let at = Capped { inner: Body::new(http_body_full(10)), seen: 0, ceiling: 10 };
        let mut at = at;
        assert_eq!(read(&mut at).await.into_iter().map(|r| r.unwrap()).collect::<Vec<_>>(), vec![10]);
        let mut over = Capped { inner: Body::new(http_body_full(11)), seen: 0, ceiling: 10 };
        let got = read(&mut over).await;
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].as_ref().unwrap_err().code(), tonic::Code::ResourceExhausted);
    }

    /// A body of `n` bytes in one frame.
    fn http_body_full(n: usize) -> impl HttpBody<Data = Bytes, Error = std::convert::Infallible> + Send + 'static {
        struct One(Option<Bytes>);
        impl HttpBody for One {
            type Data = Bytes;
            type Error = std::convert::Infallible;
            fn poll_frame(
                mut self: Pin<&mut Self>,
                _: &mut Context<'_>,
            ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
                Poll::Ready(self.0.take().map(|b| Ok(Frame::data(b))))
            }
        }
        One(Some(Bytes::from(vec![0u8; n])))
    }
}
