//! Who may call through the demo, checked before the allowlist: the page's
//! origin, and a bearer token when one is set.
//!
//! The page is served from the demo's own origin, so by default no other
//! origin is let in: a call whose `Origin` is not the demo's own is
//! refused, and no CORS header is ever sent. Origins named with
//! `--cors-origin` are let in and granted CORS; there is no wildcard.

use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use tonic::Code;

/// The request headers a page may send on a call.
const ALLOW_HEADERS: &str = "content-type, connect-protocol-version, connect-timeout-ms, authorization";
/// The response headers a page may read: the server's error metadata.
const EXPOSE_HEADERS: &str = "turbo-code, turbo-field";

#[derive(Debug, Default)]
pub(crate) struct Gate {
    token: Option<String>,
    origins: Vec<String>,
}

impl Gate {
    pub(crate) fn new(token: Option<String>, origins: &[String]) -> Result<Gate, String> {
        if let Some(t) = &token
            && (t.is_empty() || !t.bytes().all(|b| b.is_ascii_graphic()))
        {
            return Err("the token must be one or more visible ASCII characters".into());
        }
        let origins = origins.iter().map(|o| origin(o)).collect::<Result<_, _>>()?;
        Ok(Gate { token, origins })
    }

    /// Why a call is refused, if it is.
    pub(crate) fn check(&self, headers: &HeaderMap) -> Result<(), (Code, String)> {
        if let Some(o) = headers.get(header::ORIGIN) {
            let o = o.to_str().unwrap_or("");
            if !self.listed(o) && !same_origin(o, headers) {
                return Err((Code::PermissionDenied, format!("origin {o} is not allowed")));
            }
        }
        let Some(token) = &self.token else { return Ok(()) };
        let given = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).and_then(bearer);
        match given {
            Some(g) if same(g.as_bytes(), token.as_bytes()) => Ok(()),
            Some(_) => Err((Code::Unauthenticated, "the bearer token is wrong".into())),
            None => Err((Code::Unauthenticated, "a bearer token is required".into())),
        }
    }

    /// The CORS headers for a call from a listed origin.
    pub(crate) fn grant(&self, origin: Option<&HeaderValue>, reply: &mut HeaderMap) {
        if self.origins.is_empty() {
            return;
        }
        reply.append(header::VARY, HeaderValue::from_static("origin"));
        if let Some(o) = origin
            && self.listed(o.to_str().unwrap_or(""))
        {
            reply.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, o.clone());
            reply.insert(header::ACCESS_CONTROL_EXPOSE_HEADERS, HeaderValue::from_static(EXPOSE_HEADERS));
        }
    }

    /// The answer to a CORS preflight: a grant for a POST from a listed
    /// origin, else 403 with no CORS headers.
    pub(crate) fn preflight(&self, headers: &HeaderMap) -> Response {
        let post = headers.get(header::ACCESS_CONTROL_REQUEST_METHOD).is_some_and(|m| m.as_bytes() == b"POST");
        let mut reply = HeaderMap::new();
        self.grant(headers.get(header::ORIGIN), &mut reply);
        if !post || !reply.contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN) {
            return (StatusCode::FORBIDDEN, reply).into_response();
        }
        reply.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("POST"));
        reply.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, HeaderValue::from_static(ALLOW_HEADERS));
        reply.insert(header::ACCESS_CONTROL_MAX_AGE, HeaderValue::from_static("600"));
        (StatusCode::NO_CONTENT, reply).into_response()
    }

    fn listed(&self, origin: &str) -> bool {
        self.origins.iter().any(|o| o.eq_ignore_ascii_case(origin))
    }
}

/// An origin as `--cors-origin` takes it: `http://` or `https://` and a
/// host with an optional port, nothing after.
pub fn origin(s: &str) -> Result<String, String> {
    let s = s.trim();
    let bad = || Err(format!("--cors-origin {s}: must be http://HOST[:PORT] or https://HOST[:PORT]"));
    let lower = s.to_ascii_lowercase();
    let Some((scheme, host)) = lower.split_once("://") else { return bad() };
    if !matches!(scheme, "http" | "https") || host.is_empty() {
        return bad();
    }
    if !host.bytes().all(|b| b.is_ascii_alphanumeric() || b"-.:[]".contains(&b)) {
        return bad();
    }
    Ok(lower)
}

/// Whether `origin` is the demo's own: its host and port are the request's
/// `Host`.
fn same_origin(origin: &str, headers: &HeaderMap) -> bool {
    let Some(host) = headers.get(header::HOST).and_then(|h| h.to_str().ok()) else { return false };
    let Some((_, authority)) = origin.split_once("://") else { return false };
    authority.eq_ignore_ascii_case(host)
}

fn bearer(v: &str) -> Option<&str> {
    let (scheme, token) = v.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
}

/// Equal, in time that does not depend on where they differ.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origins_are_scheme_and_host_only_and_wildcards_are_refused() {
        assert_eq!(origin(" HTTPS://App.Example:8443 ").unwrap(), "https://app.example:8443");
        assert_eq!(origin("http://[::1]:8080").unwrap(), "http://[::1]:8080");
        for bad in
            ["*", "null", "http://*", "https://a.example/", "https://a.example/x", "ftp://a", "http://", "a.example"]
        {
            assert!(origin(bad).is_err(), "{bad}");
        }
        assert!(Gate::new(Some(String::new()), &[]).is_err());
        assert!(Gate::new(Some("two words".into()), &[]).is_err());
        assert_eq!(bearer("Bearer  abc "), Some("abc"));
        assert_eq!(bearer("Basic abc"), None);
        assert!(same(b"abc", b"abc") && !same(b"abc", b"abd") && !same(b"abc", b"ab"));
    }
}
