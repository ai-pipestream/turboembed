//! One Connect deadline across upload, channel readiness and response.
use std::future::Future;
use std::time::Duration;

use axum::http::HeaderMap;
use tokio::time::Instant;
use tonic::Status;

#[derive(Clone, Copy)]
pub(crate) struct Budget(Option<Instant>);

impl Budget {
    pub(crate) fn new(headers: &HeaderMap, started: Instant) -> Result<Self, Status> {
        let Some(value) = headers.get("connect-timeout-ms") else { return Ok(Self(None)) };
        let value = value.to_str().map_err(|_| Status::invalid_argument("invalid connect-timeout-ms"))?;
        if value.is_empty() || value.len() > 10 || !value.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Status::invalid_argument("connect-timeout-ms must be 1 to 10 decimal digits"));
        }
        let ms: u64 = value.parse().map_err(|_| Status::invalid_argument("invalid connect-timeout-ms"))?;
        Ok(Self(Some(started + Duration::from_millis(ms))))
    }

    pub(crate) fn remaining(self) -> Result<Option<Duration>, Status> {
        self.0
            .map(|end| end.checked_duration_since(Instant::now()).filter(|d| !d.is_zero()).ok_or_else(expired))
            .transpose()
    }

    pub(crate) async fn wait<T>(self, future: impl Future<Output = T>) -> Result<T, Status> {
        match self.0 {
            None => Ok(future.await),
            Some(end) => {
                tokio::select! {
                    biased;
                    _ = tokio::time::sleep_until(end) => Err(expired()),
                    result = future => Ok(result),
                }
            }
        }
    }
}

fn expired() -> Status {
    Status::deadline_exceeded("Connect deadline exceeded")
}
