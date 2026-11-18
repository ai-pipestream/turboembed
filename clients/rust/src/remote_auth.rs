//! Transport security checks for bearer token authentication on remote
//! `turbo-kserve` endpoints.

use std::net::{Ipv4Addr, Ipv6Addr};

use tonic::transport::Uri;

use crate::error::{ClientError, Result};
use crate::remote::RemoteEndpointConfig;

/// Refuses sending a bearer token over a plain `http://` endpoint unless the
/// target host is loopback (`127.0.0.1`, `localhost`, `[::1]`).
pub(crate) fn ensure_secure_bearer_transport(cfg: &RemoteEndpointConfig) -> Result<()> {
    if cfg.bearer_token.is_none() {
        return Ok(());
    }
    let uri: Uri = cfg
        .endpoint
        .parse()
        .map_err(|e| ClientError::InvalidArgument(format!("invalid gRPC endpoint `{}`: {e}", cfg.endpoint)))?;
    let scheme = uri.scheme_str().unwrap_or_default();
    if scheme.eq_ignore_ascii_case("https") {
        return Ok(());
    }
    if scheme.eq_ignore_ascii_case("http") {
        if let Some(host) = uri.host()
            && is_loopback_host(host)
        {
            return Ok(());
        }
        return Err(ClientError::InvalidArgument(format!(
            "refusing to send bearer token over plaintext http:// to non-loopback endpoint `{}`; use https:// or a loopback address",
            cfg.endpoint
        )));
    }
    Err(ClientError::InvalidArgument(format!(
        "refusing to send bearer token over unsupported endpoint scheme `{}` in `{}`",
        scheme, cfg.endpoint
    )))
}

fn is_loopback_host(host: &str) -> bool {
    let bare = host.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(host);
    if bare.eq_ignore_ascii_case("localhost") || bare.eq_ignore_ascii_case("localhost.") {
        return true;
    }
    if let Ok(v4) = bare.parse::<Ipv4Addr>() {
        return v4.is_loopback();
    }
    if let Ok(v6) = bare.parse::<Ipv6Addr>() {
        return v6.is_loopback();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_plain_http_without_bearer_token() {
        let cfg = RemoteEndpointConfig::new("http://kserve.internal:8001");
        assert!(ensure_secure_bearer_transport(&cfg).is_ok());
    }

    #[test]
    fn allows_bearer_token_over_https_and_loopback_http() {
        for ep in [
            "https://kserve.example.com:8443",
            "HTTPS://kserve.example.com",
            "http://127.0.0.1:8001",
            "http://localhost:8001",
            "http://LOCALHOST:8001",
            "http://[::1]:8001",
        ] {
            let cfg = RemoteEndpointConfig::new(ep).with_bearer_token("secret");
            assert!(ensure_secure_bearer_transport(&cfg).is_ok(), "expected {ep} to be allowed");
        }
    }

    #[test]
    fn refuses_bearer_token_over_non_loopback_http() {
        for ep in [
            "http://kserve.example.com:8001",
            "http://10.0.0.5:8001",
            "http://192.168.1.10:8001",
            "http://[2001:db8::1]:8001",
            "http://0.0.0.0:8001",
        ] {
            let cfg = RemoteEndpointConfig::new(ep).with_bearer_token("secret");
            let err = ensure_secure_bearer_transport(&cfg).unwrap_err();
            assert!(matches!(err, ClientError::InvalidArgument(_)), "expected InvalidArgument for {ep}, got {err:?}");
        }
    }
}
