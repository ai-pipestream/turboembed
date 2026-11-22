//! Endpoint and redirect policy for the Drive API and its media service.
use super::super::transport::{is_loopback_authority, origin};
use ureq::http::uri::Authority;

/// Accept an HTTPS origin, or explicit loopback HTTP when enabled by the caller.
pub(super) fn valid(origin: &str, loopback_http: bool) -> bool {
    let Some(text) = origin.strip_prefix("https://") else {
        return loopback_http && origin.strip_prefix("http://").is_some_and(is_loopback_authority);
    };
    let Ok(authority) = text.parse::<Authority>() else { return false };
    if text.contains('@') {
        return false;
    }
    let host = authority.host();
    if host.is_empty() || host.starts_with('[') && host[1..host.len() - 1].parse::<std::net::Ipv6Addr>().is_err() {
        return false;
    }
    let suffix = &text[host.len()..];
    suffix.is_empty()
        || suffix
            .strip_prefix(':')
            .is_some_and(|p| p.bytes().all(|b| b.is_ascii_digit()) && p.parse::<u16>().is_ok_and(|port| port > 0))
}

pub(super) fn allows_redirect(endpoint: &str, target: &str, media: bool) -> bool {
    let (destination, _) = origin(target);
    valid(destination, endpoint.starts_with("http://")) && (media || destination.eq_ignore_ascii_case(endpoint))
}

#[cfg(test)]
mod tests {
    use super::allows_redirect;
    use crate::fetch::drive::resolve;

    #[test]
    fn media_targets_obey_the_transport_policy_on_every_request() {
        let local = "http://127.0.0.1:80";
        let api = "https://api.example";
        for location in [
            "http://other.example/x",
            "//other.example/x",
            "http://127.0.0.1:80@other.example/x",
            "http://127.0.0.1.other.example:80/x",
            "http://[::ffff:192.0.2.1]:80/x",
            "http://localhost/x",
            "https://:443/x",
            "https://other.example:invalid/x",
            "https://user@other.example/x",
        ] {
            assert!(!allows_redirect(local, &resolve(local, "/files", location), true), "{location}");
        }
        assert!(!allows_redirect(api, "http://127.0.0.1:80/x", true));
        for endpoint in [local, api] {
            assert!(allows_redirect(endpoint, "https://content.example/x", true));
            assert!(!allows_redirect(endpoint, "https://content.example/x", false));
        }
    }
}
