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

/// The most a manifest or a token reply may be.
const SMALL: u64 = 4 << 20;

/// OCI registries, read without credentials. As a [`Transport`] it serves
/// a file whose address is a blob of a registry
/// (`https://<registry>/v2/<name>/blobs/sha256:<hex>`). A token goes only
/// to the registry that asked for it: a redirect to the registry's
/// storage is followed without it, over HTTPS only.
pub struct OciRegistry {
    /// Sends the request and its token and follows no redirect.
    signed: ureq::Agent,
    /// Follows a redirect and fetches a token, with no token of its own.
    plain: ureq::Agent,
    /// A registry host and the address asked in its place.
    stand_in: Option<(String, String)>,
    /// The token held for each repository, by `scheme://host/v2/<name>`.
    tokens: Mutex<HashMap<String, String>>,
}

impl OciRegistry {
    /// Registries at their own HTTPS addresses.
    pub fn new() -> OciRegistry {
        OciRegistry::build(None)
    }

    /// As [`OciRegistry::new`], with every request for the registry
    /// `host` sent to `base` (`scheme://host[:port]`) in its place. A
    /// plain `http://` base sends tokens in clear text and is only for a
    /// stand-in on loopback in tests.
    pub fn with(host: &str, base: &str) -> OciRegistry {
        OciRegistry::build(Some((format!("https://{host}"), base.trim_end_matches('/').to_owned())))
    }

    fn build(stand_in: Option<(String, String)>) -> OciRegistry {
        let https = !stand_in.as_ref().is_some_and(|(_, base)| base.starts_with("http://"));
        let agent = |redirects| {
            ureq::Agent::config_builder()
                .https_only(https)
                .max_redirects(redirects)
                .redirect_auth_headers(ureq::config::RedirectAuthHeaders::Never)
                .accept_encoding("identity")
        };
        OciRegistry {
            signed: agent(0).http_status_as_error(false).build().into(),
            plain: agent(10).build().into(),
            stand_in,
            tokens: Mutex::new(HashMap::new()),
        }
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
        let repository = match target.find("/manifests/").or_else(|| target.find("/blobs/")) {
            Some(at) => target[..at].to_owned(),
            None => return Err(format!("{url}: not a manifest or a blob of a registry")),
        };
        let held = self.tokens.lock().unwrap().get(&repository).cloned();
        let mut resp = self.ask(&target, accept, held.as_deref()).map_err(|e| format!("{url}: {e}"))?;
        if resp.status().as_u16() == 401 {
            let challenge = resp.headers().get("www-authenticate").and_then(|v| v.to_str().ok()).unwrap_or_default();
            let token = self.token(challenge).map_err(|e| format!("{url}: {e}"))?;
            resp = self.ask(&target, accept, Some(&token)).map_err(|e| format!("{url}: {e}"))?;
            self.tokens.lock().unwrap().insert(repository, token);
        }
        if resp.status().is_redirection() {
            let to = resp.headers().get("location").and_then(|v| v.to_str().ok());
            let to = to.ok_or_else(|| format!("{url}: a redirect with no Location"))?;
            let origin = target.find("/v2/").unwrap_or(target.len());
            let to = resolve(&target[..origin], &target[origin..], to);
            // A blob may be served from the registry's storage. A manifest
            // is the registry's own answer: one sent elsewhere is reported
            // as that, not read and found to have another digest.
            if target.contains("/manifests/") && !to.starts_with(&format!("{}/", &target[..origin])) {
                return Err(format!(
                    "{url}: redirected to {to}, which is not the registry; name the registry's own host"
                ));
            }
            resp = self.plain.get(&to).call().map_err(|e| format!("{url}: following its redirect: {e}"))?;
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
        let https = !self.stand_in.as_ref().is_some_and(|(_, base)| base.starts_with("http://"));
        if https && !realm.starts_with("https://") {
            return Err(format!("the token service {realm:?} is not an https:// address"));
        }
        let url = format!("{realm}{}{query}", if realm.contains('?') { '&' } else { '?' });
        let mut resp = self.plain.get(&url).call().map_err(|e| format!("the token service {realm}: {e}"))?;
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
    use super::token_request;

    #[test]
    fn a_bearer_challenge_names_the_token_request() {
        let docker = r#"Bearer realm="https://auth.docker.io/token",service="registry.docker.io",scope="repository:library/x:pull""#;
        assert_eq!(
            token_request(docker).unwrap(),
            (
                "https://auth.docker.io/token".to_owned(),
                "service=registry.docker.io&scope=repository:library/x:pull".to_owned()
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
