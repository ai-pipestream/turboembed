//! gRPC deadline (`grpc-timeout`) parsing and cooperative cancellation
//! across unary and streaming RPCs (issue `#1300`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::time::Instant;
use tonic::Status;
use tonic::codegen::http::HeaderMap;
use tonic::metadata::MetadataMap;

use crate::api::{Failure, Result, TURBO_E_INVALID_STATE};

/// Header name for gRPC per-RPC deadlines under the HTTP/2 transport spec.
pub const GRPC_TIMEOUT_HEADER: &str = "grpc-timeout";

/// Parses a `grpc-timeout` wire value (`1..=8` ASCII digits followed by unit
/// `'H' | 'M' | 'S' | 'm' | 'u' | 'n'`).
pub fn parse_grpc_timeout(raw: &[u8]) -> Option<Duration> {
    if raw.len() < 2 || raw.len() > 9 {
        return None;
    }
    let (digits, unit) = raw.split_at(raw.len() - 1);
    if !digits.iter().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let val_str = std::str::from_utf8(digits).ok()?;
    let val: u64 = val_str.parse().ok()?;
    match unit[0] {
        b'H' => Some(Duration::from_secs(val.saturating_mul(3600))),
        b'M' => Some(Duration::from_secs(val.saturating_mul(60))),
        b'S' => Some(Duration::from_secs(val)),
        b'm' => Some(Duration::from_millis(val)),
        b'u' => Some(Duration::from_micros(val)),
        b'n' => Some(Duration::from_nanos(val)),
        _ => None,
    }
}

/// Computes the cooperative worker stop budget and outer hard cutoff for a
/// requested `grpc-timeout` duration so in-flight steps finish, release their
/// held sessions, and answer `DEADLINE_EXCEEDED` before tonic's transport
/// timeout maps an elapsed timer to `CANCELLED`.
fn budgets(requested: Duration) -> (Duration, Duration) {
    if requested <= Duration::from_millis(5) {
        return (Duration::ZERO, Duration::ZERO);
    }
    let stop_margin = (requested / 4).clamp(Duration::from_millis(2), Duration::from_millis(20));
    let hard_margin = (requested / 8).clamp(Duration::from_millis(1), Duration::from_millis(10));
    (requested.saturating_sub(stop_margin), requested.saturating_sub(hard_margin))
}

/// Cooperative deadline and cancellation token for an in-flight gRPC call.
#[derive(Clone, Debug)]
pub struct Deadline {
    stop_at: Option<Instant>,
    hard_at: Option<Instant>,
    cancel: Arc<AtomicBool>,
    timed_out: Arc<AtomicBool>,
}

impl Default for Deadline {
    fn default() -> Self {
        Self::none()
    }
}

impl Deadline {
    /// Returns an unbounded [`Deadline`] with no timeout.
    pub fn none() -> Self {
        Self {
            stop_at: None,
            hard_at: None,
            cancel: Arc::new(AtomicBool::new(false)),
            timed_out: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Creates a [`Deadline`] expiring after `requested`.
    pub fn with_timeout(requested: Duration) -> Self {
        let now = Instant::now();
        let (stop_dur, hard_dur) = budgets(requested);
        let expired = stop_dur.is_zero();
        let stop_at = now + stop_dur;
        let hard_at = now + hard_dur;
        let cancel = Arc::new(AtomicBool::new(expired));
        let timed_out = Arc::new(AtomicBool::new(expired));
        if !expired && let Ok(handle) = tokio::runtime::Handle::try_current() {
            let cancel_t = Arc::clone(&cancel);
            let timed_out_t = Arc::clone(&timed_out);
            handle.spawn(async move {
                tokio::time::sleep_until(stop_at).await;
                timed_out_t.store(true, Ordering::Release);
                cancel_t.store(true, Ordering::Release);
            });
        }
        Self { stop_at: Some(stop_at), hard_at: Some(hard_at), cancel, timed_out }
    }

    /// Parses `grpc-timeout` from an HTTP header map.
    pub fn from_headers(headers: &HeaderMap) -> std::result::Result<Self, Status> {
        let Some(raw) = headers.get(GRPC_TIMEOUT_HEADER) else {
            return Ok(Self::none());
        };
        let dur = parse_grpc_timeout(raw.as_bytes())
            .ok_or_else(|| Status::invalid_argument("invalid grpc-timeout header"))?;
        Ok(Self::with_timeout(dur))
    }

    /// Extracts the [`Deadline`] attached by the server middleware, or parses
    /// `grpc-timeout` from `req.metadata()`.
    pub fn from_request<T>(req: &tonic::Request<T>) -> std::result::Result<Self, Status> {
        if let Some(d) = req.extensions().get::<Self>() {
            return Ok(d.clone());
        }
        Self::from_metadata(req.metadata())
    }

    /// Parses `grpc-timeout` from gRPC request metadata.
    pub fn from_metadata(metadata: &MetadataMap) -> std::result::Result<Self, Status> {
        let Some(raw) = metadata.get(GRPC_TIMEOUT_HEADER) else {
            return Ok(Self::none());
        };
        let dur = parse_grpc_timeout(raw.as_bytes())
            .ok_or_else(|| Status::invalid_argument("invalid grpc-timeout header"))?;
        Ok(Self::with_timeout(dur))
    }

    /// Returns the cooperative stop instant when a deadline is configured.
    pub fn stop_at(&self) -> Option<Instant> {
        self.stop_at
    }

    /// Returns the outer hard cutoff instant when a deadline is configured.
    pub fn hard_at(&self) -> Option<Instant> {
        self.hard_at
    }

    /// Shared cancellation flag flipped to `true` when the deadline expires.
    pub fn cancel_flag(&self) -> &Arc<AtomicBool> {
        &self.cancel
    }

    /// Trips the deadline immediately, signalling all cooperative workers.
    pub fn trip(&self) {
        self.timed_out.store(true, Ordering::Release);
        self.cancel.store(true, Ordering::Release);
    }

    /// Returns `true` if the deadline has expired.
    pub fn is_expired(&self) -> bool {
        if self.timed_out.load(Ordering::Acquire) {
            return true;
        }
        if let Some(at) = self.stop_at
            && Instant::now() >= at
        {
            self.trip();
            return true;
        }
        false
    }

    /// Returns `Err(Status::deadline_exceeded(...))` if the deadline has expired.
    pub fn check(&self) -> std::result::Result<(), Status> {
        if self.is_expired() {
            return Err(deadline_status());
        }
        Ok(())
    }

    /// Returns `Err(Failure)` if the deadline has expired, for blocking worker
    /// functions returning `crate::api::Result`.
    pub fn check_alive(&self) -> Result<()> {
        if self.is_expired() {
            return Err(Failure::new(TURBO_E_INVALID_STATE, 0, "gRPC deadline exceeded"));
        }
        Ok(())
    }

    /// Binds a stream's [`crate::stream::Dead`] flag so it is tripped as soon
    /// as this deadline expires.
    pub(crate) fn bind_stream_dead(&self, dead: Arc<crate::stream::Dead>) {
        let Some(at) = self.stop_at else {
            return;
        };
        if self.is_expired() {
            dead.set();
            return;
        }
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let timed_out = Arc::clone(&self.timed_out);
            let cancel = Arc::clone(&self.cancel);
            handle.spawn(async move {
                tokio::time::sleep_until(at).await;
                timed_out.store(true, Ordering::Release);
                cancel.store(true, Ordering::Release);
                dead.set();
            });
        }
    }
}

/// Standard `DEADLINE_EXCEEDED` status returned when `grpc-timeout` expires.
pub fn deadline_status() -> Status {
    Status::deadline_exceeded("gRPC deadline exceeded")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_grpc_timeout_units_and_rejects_invalid_headers() {
        assert_eq!(parse_grpc_timeout(b"2H"), Some(Duration::from_secs(7200)));
        assert_eq!(parse_grpc_timeout(b"3M"), Some(Duration::from_secs(180)));
        assert_eq!(parse_grpc_timeout(b"42S"), Some(Duration::from_secs(42)));
        assert_eq!(parse_grpc_timeout(b"15m"), Some(Duration::from_millis(15)));
        assert_eq!(parse_grpc_timeout(b"250u"), Some(Duration::from_micros(250)));
        assert_eq!(parse_grpc_timeout(b"99n"), Some(Duration::from_nanos(99)));
        assert!(parse_grpc_timeout(b"").is_none());
        assert!(parse_grpc_timeout(b"S").is_none());
        assert!(parse_grpc_timeout(b"123456789S").is_none());
        assert!(parse_grpc_timeout(b"10x").is_none());
        assert!(parse_grpc_timeout(b"-5S").is_none());
    }
}
