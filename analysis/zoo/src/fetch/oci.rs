//! An OCI registry as a place files are read from: the distribution API
//! over the zoo's own HTTPS stack. A manifest and a blob are each read by
//! digest, so an address names its own bytes. A registry that asks for a
//! token (most do, for public repositories too) is given an anonymous one
//! from the service its challenge names.

use std::collections::HashMap;
use std::sync::Mutex;

use super::drive::resolve;
use super::transport::Transport;
use crate::digest::Sha256;
use crate::index::Url;

/// The manifests the registry is asked for: an OCI image manifest, or a
/// Docker one of the same shape.
const MANIFESTS: &str =
    "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json";

/// The most redirects one request is followed through.
const REDIRECTS: usize = 10;

/// `url` as its `scheme://host[:port]` and the path and query after it.
fn origin(url: &str) -> (&str, &str) {
    let after = url.find("://").map_or(0, |at| at + 3);
    url.split_at(url[after..].find('/').map_or(url.len(), |at| after + at))
}

/// The most a manifest or a token reply may be.
const SMALL: u64 = 4 << 20;

/// Whether `authority` is an explicit loopback host with a non-zero port
/// (`127.0.0.1:<port>`, `[::1]:<port>`, or `localhost:<port>`) and no
/// userinfo (`@`).
fn is_loopback_authority(authority: &str) -> bool {
    if authority.contains('@') {
        return false;
    }
    let port = authority
        .strip_prefix("127.0.0.1:")
        .or_else(|| authority.strip_prefix("[::1]:"))
        .or_else(|| authority.strip_prefix("localhost:"));
    port.is_some_and(|p| {
        !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) && p.parse::<u16>().is_ok_and(|n| n > 0)
    })
}

/// OCI registries, read without credentials. As a [`Transport`] it serves
/// a file whose address is a blob of a registry
/// (`https://<registry>/v2/<name>/blobs/sha256:<hex>`). A token goes only
/// to the registry that asked for it: a redirect to the registry's
/// storage is followed without it, over HTTPS only (or loopback HTTP
/// when constructed via [`OciRegistry::with`] with a loopback base).
pub struct OciRegistry {
    /// Sends the request and its token and follows no redirect.
    signed: ureq::Agent,
    /// Follows a token request (redirects are followed manually and
    /// validated per hop) with no token of its own.
    plain: ureq::Agent,
    /// A registry host and the address asked in its place.
    stand_in: Option<(String, String)>,
    /// Whether plaintext `http://` is permitted, strictly bounded to
    /// loopback authorities (`127.0.0.1:<port>`, `[::1]:<port>`,
    /// `localhost:<port>`). Always `false` for [`OciRegistry::new`] and
    /// `https://` bases.
    allow_loopback_http: bool,
    /// The token held for each repository, by `scheme://host/v2/<name>`.
    tokens: Mutex<HashMap<String, String>>,
}

impl OciRegistry {
    /// Registries at their own HTTPS addresses.
    pub fn new() -> OciRegistry {
        OciRegistry::build(None)
    }

    /// As [`OciRegistry::new`], with every request for the registry
    /// `host` sent to `base` in its place: `https://host[:port]`, or
    /// `http://` on this machine alone (`127.0.0.1:<port>`,
    /// `[::1]:<port>`, or `localhost:<port>`), where a token crosses no
    /// network. Even with a loopback `http://` base, every token
    /// endpoint and every redirect hop is validated before a request is
    /// sent and may only target `https://` or an explicit loopback
    /// `http://` authority. Any other base is refused.
    pub fn with(host: &str, base: &str) -> Result<OciRegistry, String> {
        if host.is_empty() || host.contains(['/', '?', '#', '@']) || !host.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(format!("{host:?}: a registry host must be host[:port]"));
        }
        let base = base.trim_end_matches('/');
        let local = base.strip_prefix("http://").is_some_and(is_loopback_authority);
        let named = base.strip_prefix("https://").is_some_and(|h| {
            !h.is_empty() && !h.contains(['/', '?', '#', '@']) && h.bytes().all(|b| b.is_ascii_graphic())
        });
        if !(local || named) {
            return Err(format!("{base}: a registry's address is https://host[:port], or http:// on this machine"));
        }
        Ok(OciRegistry::build(Some((format!("https://{host}"), base.to_owned()))))
    }

    fn build(stand_in: Option<(String, String)>) -> OciRegistry {
        let allow_loopback_http = stand_in.as_ref().is_some_and(|(_, base)| base.starts_with("http://"));
        let agent = || {
            ureq::Agent::config_builder()
                .https_only(!allow_loopback_http)
                .max_redirects(0)
                .redirect_auth_headers(ureq::config::RedirectAuthHeaders::Never)
                .accept_encoding("identity")
                .http_status_as_error(false)
                .build()
                .into()
        };
        OciRegistry {
            signed: agent(),
            plain: agent(),
            stand_in,
            allow_loopback_http,
            tokens: Mutex::new(HashMap::new()),
        }
    }

    /// Validates `url` against the client's endpoint policy before any
    /// request is sent.
    fn check_url(&self, url: &str, what: &str) -> Result<(), String> {
        if let Some(rest) = url.strip_prefix("https://") {
            let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
            if !authority.is_empty() && !authority.contains('@') && url.bytes().all(|b| b.is_ascii_graphic()) {
                return Ok(());
            }
            return Err(format!("{what} {url:?} is not a valid https:// address"));
        }
        if self.allow_loopback_http {
            if let Some(rest) = url.strip_prefix("http://") {
                let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
                if is_loopback_authority(authority) && url.bytes().all(|b| b.is_ascii_graphic()) {
                    return Ok(());
                }
            }
            return Err(format!("{what} {url:?} is not an https:// or loopback http:// address"));
        }
        Err(format!("{what} {url:?} is not an https:// address"))
    }

    /// The manifest of `<registry>/<name>` whose digest is `digest`, held
    /// to that digest.
    pub fn manifest(&self, registry: &str, name: &str, digest: &Sha256) -> Result<Vec<u8>, String> {
        let url = format!("https://{registry}/v2/{name}/manifests/sha256:{digest}");
        let bytes = self.send(&url, Some(MANIFESTS), SMALL)?;
        let held = Sha256::of(&bytes);
        if held != *digest {
            return Err(format!("{url}: the registry served a manifest whose SHA-256 is {held}"));
        }
        Ok(bytes)
    }

    /// One GET of `url`, at most `limit` bytes and one more. A 401 that
    /// names a token service is answered once with an anonymous token.
    fn send(&self, url: &str, accept: Option<&str>, limit: u64) -> Result<Vec<u8>, String> {
        let target = match &self.stand_in {
            Some((host, base)) if url.strip_prefix(host.as_str()).is_some_and(|rest| rest.starts_with('/')) => {
                format!("{base}{}", &url[host.len()..])
            }
            _ => url.to_owned(),
        };
        // The API resource is the segment before the last: a repository's
        // own name may hold `manifests` or `blobs` as a segment.
        let (repository, resource) = target
            .rsplit_once('/')
            .and_then(|(head, _)| head.rsplit_once('/'))
            .filter(|(repository, resource)| repository.contains("/v2/") && matches!(*resource, "manifests" | "blobs"))
            .ok_or_else(|| format!("{url}: not a manifest or a blob of a registry"))?;
        let (repository, is_manifest) = (repository.to_owned(), resource == "manifests");
        self.check_url(&target, "the registry target").map_err(|e| format!("{url}: {e}"))?;
        let held = self.tokens.lock().unwrap().get(&repository).cloned();
        let mut resp = self.ask(&target, accept, held.as_deref()).map_err(|e| format!("{url}: {e}"))?;
        if resp.status().as_u16() == 401 {
            let challenge = resp.headers().get("www-authenticate").and_then(|v| v.to_str().ok()).unwrap_or_default();
            let token = self.token(challenge).map_err(|e| format!("{url}: {e}"))?;
            resp = self.ask(&target, accept, Some(&token)).map_err(|e| format!("{url}: {e}"))?;
            self.tokens.lock().unwrap().insert(repository, token);
        }
        // Each redirect is followed here, one hop at a time and without
        // the token. A blob may be served from the registry's storage. A
        // manifest is the registry's own answer: a hop that leaves the
        // registry is reported as that, not read and found to have
        // another digest. Every hop is validated before a request is sent.
        let registry = origin(&target).0.to_owned();
        let mut at = target;
        let mut hops = 0;
        while resp.status().is_redirection() {
            let to = resp.headers().get("location").and_then(|v| v.to_str().ok());
            let to = to.ok_or_else(|| format!("{url}: a redirect with no Location"))?;
            let (base, path) = origin(&at);
            let to = resolve(base, path, to);
            if is_manifest && origin(&to).0 != registry {
                return Err(format!(
                    "{url}: redirected to {to}, which is not the registry; name the registry's own host"
                ));
            }
            self.check_url(&to, "the redirect target").map_err(|e| format!("{url}: {e}"))?;
            hops += 1;
            if hops > REDIRECTS {
                return Err(format!("{url}: more than {REDIRECTS} redirects"));
            }
            resp = self.ask(&to, None, None).map_err(|e| format!("{url}: following its redirect: {e}"))?;
            at = to;
        }
        if !resp.status().is_success() {
            return Err(format!("{url}: the registry answered {}", resp.status()));
        }
        resp.body_mut().with_config().limit(limit.saturating_add(1)).read_to_vec().map_err(|e| format!("{url}: {e}"))
    }

    fn ask(
        &self,
        target: &str,
        accept: Option<&str>,
        token: Option<&str>,
    ) -> Result<ureq::http::Response<ureq::Body>, ureq::Error> {
        let mut req = self.signed.get(target);
        if let Some(accept) = accept {
            req = req.header("Accept", accept);
        }
        if let Some(token) = token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
        req.call()
    }

    /// An anonymous token from the service a `WWW-Authenticate: Bearer`
    /// challenge names.
    fn token(&self, challenge: &str) -> Result<String, String> {
        let (realm, query) = token_request(challenge)?;
        self.check_url(&realm, "the token service")?;
        let mut at = if query.is_empty() {
            realm.clone()
        } else {
            format!("{realm}{}{query}", if realm.contains('?') { '&' } else { '?' })
        };
        let mut resp = self.plain.get(&at).call().map_err(|e| format!("the token service {realm}: {e}"))?;
        let mut hops = 0;
        while resp.status().is_redirection() {
            let to = resp
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| format!("the token service {realm}: a redirect with no Location"))?;
            let (base, path) = origin(&at);
            let to = resolve(base, path, to);
            self.check_url(&to, "the token redirect")?;
            hops += 1;
            if hops > REDIRECTS {
                return Err(format!("the token service {realm}: more than {REDIRECTS} redirects"));
            }
            resp = self
                .plain
                .get(&to)
                .call()
                .map_err(|e| format!("the token service {realm}: following its redirect: {e}"))?;
            at = to;
        }
        if !resp.status().is_success() {
            return Err(format!("the token service {realm}: answered {}", resp.status()));
        }
        let body = resp.body_mut().with_config().limit(SMALL).read_to_vec().map_err(|e| e.to_string())?;
        let v: serde_json::Value =
            serde_json::from_slice(&body).map_err(|e| format!("the token service {realm}: its reply: {e}"))?;
        let token = ["token", "access_token"].iter().find_map(|k| v.get(*k).and_then(|t| t.as_str()));
        match token {
            Some(t) if !t.is_empty() && t.bytes().all(|b| b.is_ascii_graphic()) => Ok(t.to_owned()),
            _ => Err(format!("the token service {realm} gave no token")),
        }
    }
}

impl Default for OciRegistry {
    fn default() -> OciRegistry {
        OciRegistry::new()
    }
}

impl Transport for OciRegistry {
    fn get(&self, url: &Url, size: u64) -> Result<Vec<u8>, String> {
        self.send(url.as_str(), None, size)
    }
}

/// The token service's address and the query that asks it for the token
/// a `Bearer realm="...",service="...",scope="..."` challenge describes.
fn token_request(challenge: &str) -> Result<(String, String), String> {
    let rest = challenge
        .strip_prefix("Bearer ")
        .or_else(|| challenge.strip_prefix("bearer "))
        .ok_or_else(|| format!("the registry needs a sign-in the zoo does not make ({challenge:?})"))?;
    let (mut realm, mut query) = (None, Vec::new());
    // Each parameter is `name="value"`, where a scope may hold commas, or
    // `name=value` up to the next comma.
    let mut rest = rest.trim_start_matches([',', ' ']);
    while let Some((name, after)) = rest.split_once('=') {
        let (value, tail) = match after.strip_prefix('"') {
            Some(quoted) => quoted.split_once('"').ok_or_else(|| format!("a malformed challenge ({challenge:?})"))?,
            None => after.split_once(',').unwrap_or((after, "")),
        };
        match name.trim() {
            "realm" => realm = Some(value.to_owned()),
            name @ ("service" | "scope") => query.push(format!("{name}={}", encode(value))),
            _ => {}
        }
        rest = tail.trim_start_matches([',', ' ']);
    }
    let realm = realm.ok_or_else(|| format!("a challenge that names no token service ({challenge:?})"))?;
    Ok((realm, query.join("&")))
}

/// `s` percent-encoded for a query value.
fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~' | b':' | b'/') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{OciRegistry, token_request};

    #[test]
    fn a_registry_is_asked_in_clear_text_on_this_machine_alone() {
        for base in [
            "https://mirror.example.org",
            "https://mirror.example.org:5000/",
            "http://127.0.0.1:5000",
            "http://[::1]:5000",
            "http://localhost:5000",
        ] {
            assert!(OciRegistry::with("registry.example.org", base).is_ok(), "{base}");
        }
        for base in [
            "http://mirror.example.org",
            "http://mirror.example.org:5000",
            "http://127.0.0.1.example.org:5000",
            "http://127.0.0.1:5000@mirror.example.org",
            "http://[::1]:5000@mirror.example.org",
            "http://[::1].example.org:5000",
            "http://[::1]",
            "http://[::ffff:127.0.0.1]:5000",
            "http://127.0.0.1",
            "http://127.0.0.1:0",
            "http://127.0.0.1:99999",
            "http://127.0.0.1:5000/path",
            "http://localhost",
            "https://mirror.example.org/path",
            "ftp://127.0.0.1:21",
            "",
        ] {
            let e = OciRegistry::with("registry.example.org", base).err().expect(base);
            assert!(e.contains("or http:// on this machine"), "{base}: {e}");
        }
        for host in ["", "registry.example.org/path", "user@registry.example.org", "registry.example.org?x"] {
            assert!(OciRegistry::with(host, "https://mirror.example.org").is_err(), "{host}");
        }
    }

    #[test]
    fn token_and_redirect_urls_never_escape_https_or_loopback_policy() {
        let prod = OciRegistry::new();
        assert!(prod.check_url("https://auth.docker.io/token", "token").is_ok());
        for bad in [
            "http://127.0.0.1:5000/token",
            "http://[::1]:5000/token",
            "http://localhost:5000/token",
            "http://mirror.example.org/token",
            "ftp://example.org/token",
        ] {
            assert!(prod.check_url(bad, "token").is_err(), "prod should refuse {bad}");
        }

        let https_mirror = OciRegistry::with("registry.example.org", "https://mirror.example.org").unwrap();
        assert!(https_mirror.check_url("http://127.0.0.1:5000/token", "token").is_err());

        for loopback in ["http://127.0.0.1:5000", "http://[::1]:5000", "http://localhost:5000"] {
            let local = OciRegistry::with("registry.example.org", loopback).unwrap();
            assert!(local.check_url("https://cdn.example.org/blob", "redirect").is_ok());
            assert!(local.check_url(&format!("{loopback}/storage/abc"), "redirect").is_ok());
            for escape in [
                "http://mirror.example.org/token",
                "http://127.0.0.1.example.org:5000/token",
                "http://127.0.0.1:5000@mirror.example.org/token",
                "http://[::1]:5000@mirror.example.org/token",
                "http://[::1].example.org:5000/token",
                "http://localhost/token",
                "http://127.0.0.1/token",
                "http://[::1]/token",
                "http://[::ffff:127.0.0.1]:5000/token",
                "ftp://127.0.0.1:21/file",
            ] {
                assert!(local.check_url(escape, "redirect").is_err(), "{loopback} should refuse {escape}");
            }
        }
    }

    #[test]
    fn a_bearer_challenge_names_the_token_request() {
        // Docker Hub's own challenge, as recorded from an anonymous
        // manifest request for a public repository.
        let docker = r#"Bearer realm="https://auth.docker.io/token",service="registry.docker.io",scope="repository:pipestreamai/turbo-en-masc-pos-modernbert-base-2:pull""#;
        assert_eq!(
            token_request(docker).unwrap(),
            (
                "https://auth.docker.io/token".to_owned(),
                "service=registry.docker.io&scope=repository:pipestreamai/turbo-en-masc-pos-modernbert-base-2:pull"
                    .to_owned()
            )
        );
        let spaced = r#"Bearer realm="https://r.example.org/token", scope="repository:a/b:pull,push", error="x y""#;
        assert_eq!(
            token_request(spaced).unwrap(),
            ("https://r.example.org/token".to_owned(), "scope=repository:a/b:pull%2Cpush".to_owned())
        );
        let unquoted = r#"Bearer error=invalid_token, realm="https://r.example.org/token",service=r.example.org"#;
        assert_eq!(
            token_request(unquoted).unwrap(),
            ("https://r.example.org/token".to_owned(), "service=r.example.org".to_owned())
        );
        assert!(token_request(r#"Basic realm="registry""#).unwrap_err().contains("does not make"));
        assert!(token_request(r#"Bearer service="x""#).unwrap_err().contains("names no token service"));
        assert!(token_request(r#"Bearer realm="https://r.example.org"#).unwrap_err().contains("malformed"));
        assert!(token_request("").is_err());
    }
}
