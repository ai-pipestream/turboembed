//! Bearer-token (API key) authentication as a tonic interceptor.
//!
//! Clients send `authorization: Bearer <token>`; the token must match one of
//! the configured keys. Comparison is constant-time per candidate to avoid
//! trivially timing-leaking key material. TLS/mTLS is layered separately at
//! the transport (see README); this interceptor only handles application
//! auth.

use std::collections::HashSet;
use std::sync::Arc;

use tonic::{Request, Status};

/// Interceptor state. Cheap to clone; the token set is immutable after boot.
#[derive(Clone)]
pub struct BearerAuth {
    tokens: Arc<HashSet<String>>,
}

impl BearerAuth {
    pub fn new(tokens: HashSet<String>) -> Self {
        Self {
            tokens: Arc::new(tokens),
        }
    }

    /// Check a request's `authorization` metadata.
    pub fn check<T>(&self, request: Request<T>) -> Result<Request<T>, Status> {
        let header = request
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| Status::unauthenticated("missing authorization metadata"))?;
        let token = header
            .strip_prefix("Bearer ")
            .or_else(|| header.strip_prefix("bearer "))
            .ok_or_else(|| {
                Status::unauthenticated("authorization metadata must be \"Bearer <token>\"")
            })?;
        if self.tokens.iter().any(|t| constant_time_eq(t, token)) {
            Ok(request)
        } else {
            Err(Status::unauthenticated("invalid bearer token"))
        }
    }
}

/// Constant-time string comparison (length leaks, contents do not).
fn constant_time_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auth() -> BearerAuth {
        BearerAuth::new(HashSet::from(["good-key".to_string()]))
    }

    fn request_with_header(value: Option<&str>) -> Request<()> {
        let mut request = Request::new(());
        if let Some(value) = value {
            request
                .metadata_mut()
                .insert("authorization", value.parse().unwrap());
        }
        request
    }

    #[test]
    fn accepts_valid_token() {
        assert!(auth()
            .check(request_with_header(Some("Bearer good-key")))
            .is_ok());
    }

    #[test]
    fn rejects_missing_header() {
        assert!(auth().check(request_with_header(None)).is_err());
    }

    #[test]
    fn rejects_wrong_token() {
        assert!(auth()
            .check(request_with_header(Some("Bearer bad-key")))
            .is_err());
    }

    #[test]
    fn rejects_non_bearer_scheme() {
        assert!(auth()
            .check(request_with_header(Some("Basic good-key")))
            .is_err());
    }
}
