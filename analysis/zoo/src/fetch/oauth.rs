//! Google sign-in for a registered Drive (#1744): OAuth 2.0 authorization
//! code with PKCE (RFC 7636, S256) over the zoo's own HTTPS stack, no SDK.
//! An application that owns a redirect (a web page) calls
//! [`GoogleOAuth::authorize_url`] and [`GoogleOAuth::exchange`]; a command
//! line or desktop calls [`GoogleOAuth::sign_in_loopback`]. The refresh
//! token of the grant is what a holder keeps, and [`Renewing`] turns it
//! into access tokens as Drive needs them.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Deserialize;
use sha2::Digest;

/// Google's consent page.
pub const GOOGLE_AUTH_ENDPOINT: &str = "https://accounts.google.com/o/oauth2/v2/auth";
/// Google's token endpoint.
pub const GOOGLE_TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";
/// Read access to the user's Drive files, all a zoo source needs.
pub const DRIVE_READONLY: &str = "https://www.googleapis.com/auth/drive.readonly";

/// An OAuth client registered with Google: its id, and its secret for a
/// client type that has one (a desktop client's is not confidential).
#[derive(Clone)]
pub struct GoogleOAuth {
    client_id: String,
    client_secret: Option<String>,
    auth_endpoint: String,
    token_endpoint: String,
    agent: ureq::Agent,
}

/// What the token endpoint granted. A refresh token comes with the first
/// grant of an offline sign-in, and usually not with a refresh.
#[derive(Clone, PartialEq, Eq)]
pub struct Grant {
    pub access_token: String,
    /// Seconds the access token lasts, from when it was granted.
    pub expires_in: u64,
    pub refresh_token: Option<String>,
}

impl std::fmt::Debug for Grant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let refresh = if self.refresh_token.is_some() { "Some(..)" } else { "None" };
        write!(f, "Grant {{ access_token: .., expires_in: {}, refresh_token: {refresh} }}", self.expires_in)
    }
}

/// One sign-in's secrets: the PKCE code verifier, and the `state` the
/// redirect must bring back. Each is 43 characters of base64url from 32
/// random bytes.
#[derive(Clone)]
pub struct Pkce {
    pub verifier: String,
    pub state: String,
}

impl Pkce {
    pub fn new() -> Pkce {
        Pkce { verifier: base64url(&random_bytes()), state: base64url(&random_bytes()) }
    }

    /// The S256 challenge: base64url of the verifier's SHA-256.
    pub fn challenge(&self) -> String {
        base64url(&sha2::Sha256::digest(self.verifier.as_bytes()))
    }
}

impl Default for Pkce {
    fn default() -> Pkce {
        Pkce::new()
    }
}

impl GoogleOAuth {
    /// The client `client_id` against Google's own endpoints.
    pub fn new(client_id: &str, client_secret: Option<&str>) -> Result<GoogleOAuth, String> {
        GoogleOAuth::with_endpoints(client_id, client_secret, GOOGLE_AUTH_ENDPOINT, GOOGLE_TOKEN_ENDPOINT)
    }

    /// The client against other endpoints: `https://`, or `http://` for a
    /// server on this machine.
    pub fn with_endpoints(
        client_id: &str,
        client_secret: Option<&str>,
        auth_endpoint: &str,
        token_endpoint: &str,
    ) -> Result<GoogleOAuth, String> {
        if !plain(client_id) || client_secret.is_some_and(|s| !plain(s)) {
            return Err("an OAuth client id or secret is empty or holds spaces or control characters".into());
        }
        for e in [auth_endpoint, token_endpoint] {
            let local = e.starts_with("http://127.0.0.1:") || e.starts_with("http://localhost:");
            if !(e.starts_with("https://") || local) || !plain(e) {
                return Err(format!("{e}: an OAuth endpoint is https://, or http:// on this machine"));
            }
        }
        // Errors come with a JSON body that names them; read it, not the
        // status alone.
        let agent = ureq::Agent::config_builder()
            .https_only(token_endpoint.starts_with("https://"))
            .http_status_as_error(false)
            .max_redirects(0)
            .timeout_global(Some(Duration::from_secs(30)))
            .build()
            .into();
        Ok(GoogleOAuth {
            client_id: client_id.to_owned(),
            client_secret: client_secret.map(str::to_owned),
            auth_endpoint: auth_endpoint.to_owned(),
            token_endpoint: token_endpoint.to_owned(),
            agent,
        })
    }

    pub fn client_id(&self) -> &str {
        &self.client_id
    }

    pub fn client_secret(&self) -> Option<&str> {
        self.client_secret.as_deref()
    }

    /// The consent page to send the user to: read-only Drive access,
    /// offline (so the grant holds a refresh token), consent asked again so
    /// a second sign-in still gets one.
    pub fn authorize_url(&self, redirect_uri: &str, pkce: &Pkce) -> String {
        let query = [
            ("client_id", self.client_id.as_str()),
            ("redirect_uri", redirect_uri),
            ("response_type", "code"),
            ("scope", DRIVE_READONLY),
            ("state", pkce.state.as_str()),
            ("code_challenge", &pkce.challenge()),
            ("code_challenge_method", "S256"),
            ("access_type", "offline"),
            ("prompt", "consent"),
        ];
        let query: Vec<String> = query.iter().map(|(k, v)| format!("{k}={}", super::drive::encode(v))).collect();
        format!("{}?{}", self.auth_endpoint, query.join("&"))
    }

    /// The grant for the `code` the redirect to `redirect_uri` brought.
    pub fn exchange(&self, code: &str, redirect_uri: &str, pkce: &Pkce) -> Result<Grant, String> {
        self.token(&[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("code_verifier", &pkce.verifier),
        ])
    }

    /// A new access token from a stored refresh token.
    pub fn refresh(&self, refresh_token: &str) -> Result<Grant, String> {
        self.token(&[("grant_type", "refresh_token"), ("refresh_token", refresh_token)])
    }

    /// Signs in through the browser and this machine's loopback: a
    /// listener on `127.0.0.1` on a free port, `open` called with the
    /// consent page's address (to open a browser, or print it), then the
    /// one redirect that brings a code or an error back, its `state`
    /// checked. Gives up after `wait`.
    pub fn sign_in_loopback(&self, wait: Duration, open: impl FnOnce(&str)) -> Result<Grant, String> {
        let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| format!("a loopback listener: {e}"))?;
        let port = listener.local_addr().map_err(|e| e.to_string())?.port();
        let redirect = format!("http://127.0.0.1:{port}");
        let pkce = Pkce::new();
        open(&self.authorize_url(&redirect, &pkce));
        listener.set_nonblocking(true).map_err(|e| e.to_string())?;
        let until = Instant::now() + wait;
        loop {
            let conn = match listener.accept() {
                Ok((conn, _)) => conn,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < until => {
                    std::thread::sleep(Duration::from_millis(50));
                    continue;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    return Err(format!("no sign-in came back within {}s", wait.as_secs()));
                }
                Err(e) => return Err(format!("the loopback listener: {e}")),
            };
            // Anything but the redirect (a browser's favicon request) is
            // answered and passed over.
            if let Some(answer) = redirected(conn)? {
                let (code, state) = answer?;
                if state != pkce.state {
                    return Err("the sign-in came back with another state; try again".into());
                }
                return self.exchange(&code, &redirect, &pkce);
            }
        }
    }

    fn token(&self, form: &[(&str, &str)]) -> Result<Grant, String> {
        let mut form = form.to_vec();
        form.push(("client_id", &self.client_id));
        if let Some(s) = &self.client_secret {
            form.push(("client_secret", s));
        }
        let fail = |e: ureq::Error| format!("{}: {e}", self.token_endpoint);
        let mut resp = self.agent.post(&self.token_endpoint).send_form(form).map_err(fail)?;
        let status = resp.status().as_u16();
        let body = resp.body_mut().with_config().limit(1 << 16).read_to_vec().map_err(fail)?;
        let reply: TokenReply = serde_json::from_slice(&body)
            .map_err(|e| format!("{}: status {status}, a token reply: {e}", self.token_endpoint))?;
        match (status, reply.access_token) {
            (200, Some(access_token)) if plain(&access_token) => Ok(Grant {
                access_token,
                expires_in: reply.expires_in.unwrap_or(3600),
                refresh_token: reply.refresh_token.filter(|t| plain(t)),
            }),
            _ => Err(format!(
                "{}: status {status}: {} {}",
                self.token_endpoint,
                reply.error.as_deref().unwrap_or("no access token"),
                reply.error_description.as_deref().unwrap_or(""),
            )
            .trim_end()
            .to_owned()),
        }
    }
}

/// Access tokens from one stored refresh token: fetched on first use and
/// again a minute before each runs out, or when Drive turns one down.
pub struct Renewing {
    oauth: GoogleOAuth,
    refresh_token: String,
    held: Mutex<Option<(String, Instant)>>,
}

impl Renewing {
    pub fn new(oauth: GoogleOAuth, refresh_token: &str) -> Result<Renewing, String> {
        if !plain(refresh_token) {
            return Err("a refresh token is empty or holds spaces or control characters".into());
        }
        Ok(Renewing { oauth, refresh_token: refresh_token.to_owned(), held: Mutex::new(None) })
    }

    /// A current access token; with `renew`, a new one whatever is held.
    pub fn bearer(&self, renew: bool) -> Result<String, String> {
        let mut held = self.held.lock().map_err(|e| e.to_string())?;
        match held.as_ref() {
            Some((token, until)) if !renew && Instant::now() < *until => Ok(token.clone()),
            _ => {
                let grant = self.oauth.refresh(&self.refresh_token)?;
                let lasts = Duration::from_secs(grant.expires_in.saturating_sub(60));
                *held = Some((grant.access_token.clone(), Instant::now() + lasts));
                Ok(grant.access_token)
            }
        }
    }
}

#[derive(Deserialize)]
struct TokenReply {
    access_token: Option<String>,
    expires_in: Option<u64>,
    refresh_token: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

/// The code and state of a redirect, its error, or `None` for any other
/// request; each is answered with a short page.
type Redirect = Option<Result<(String, String), String>>;

fn redirected(mut conn: std::net::TcpStream) -> Result<Redirect, String> {
    conn.set_nonblocking(false).map_err(|e| e.to_string())?;
    conn.set_read_timeout(Some(Duration::from_secs(10))).map_err(|e| e.to_string())?;
    let mut line = String::new();
    BufReader::new((&conn).take(8192)).read_line(&mut line).map_err(|e| format!("the redirect: {e}"))?;
    let target = line.split(' ').nth(1).unwrap_or_default();
    let query = target.split_once('?').map_or("", |(_, q)| q);
    let param = |k: &str| query.split('&').filter_map(|kv| kv.split_once('=')).find(|(n, _)| *n == k).map(|(_, v)| v);
    let found = match (param("code"), param("state"), param("error")) {
        (_, _, Some(e)) => Some(Err(format!("the sign-in was not completed: {}", decode(e)))),
        (Some(code), Some(state), None) => Some(Ok((decode(code), decode(state)))),
        _ => None,
    };
    let page = if found.is_some() { "Signed in. This window can be closed." } else { "Not found." };
    let status = if found.is_some() { "200 OK" } else { "404 Not Found" };
    let reply = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{page}",
        page.len()
    );
    let _ = conn.write_all(reply.as_bytes());
    Ok(found)
}

/// Undoes percent-encoding, and `+` as a space; a bad escape is kept.
fn decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
        match b[i] {
            b'%' if i + 2 < b.len() => match (hex(b[i + 1]), hex(b[i + 2])) {
                (Some(h), Some(l)) => {
                    out.push(h << 4 | l);
                    i += 3;
                    continue;
                }
                _ => out.push(b'%'),
            },
            b'+' => out.push(b' '),
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Printable ASCII and nothing else: what a token, id or endpoint is.
fn plain(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_graphic())
}

/// Base64url without padding (RFC 4648 section 5).
fn base64url(bytes: &[u8]) -> String {
    const ABC: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
        for i in 0..=chunk.len() {
            out.push(ABC[(n >> (18 - 6 * i) & 63) as usize] as char);
        }
    }
    out
}

/// 32 bytes from the system's random source: `/dev/urandom` on Unix, and
/// elsewhere std's hasher keys, which std draws from that source, run
/// through SHA-256.
fn random_bytes() -> [u8; 32] {
    let mut out = [0u8; 32];
    #[cfg(unix)]
    if std::fs::File::open("/dev/urandom").and_then(|mut f| f.read_exact(&mut out)).is_ok() {
        return out;
    }
    use std::hash::{BuildHasher, Hasher};
    let mut seed = sha2::Sha256::new();
    for i in 0..8u64 {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(i);
        seed.update(h.finish().to_le_bytes());
    }
    out.copy_from_slice(&seed.finalize());
    out
}

#[cfg(test)]
mod tests {
    use super::{Pkce, base64url, decode};

    #[test]
    fn the_challenge_is_rfc_7636_appendix_b() {
        let pkce = Pkce { verifier: "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk".into(), state: String::new() };
        assert_eq!(pkce.challenge(), "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
    }

    #[test]
    fn base64url_has_no_padding() {
        assert_eq!(base64url(b""), "");
        assert_eq!(base64url(b"f"), "Zg");
        assert_eq!(base64url(b"fo"), "Zm8");
        assert_eq!(base64url(b"foo"), "Zm9v");
        assert_eq!(base64url(&[0xfb, 0xff]), "-_8");
        let a = Pkce::new();
        assert_eq!(a.verifier.len(), 43);
        assert_ne!(a.verifier, Pkce::new().verifier);
    }

    #[test]
    fn a_redirect_query_is_decoded() {
        assert_eq!(decode("4%2F0Ab+c%C3%A9"), "4/0Ab cé");
        assert_eq!(decode("50%"), "50%");
        assert_eq!(decode("%zz"), "%zz");
    }
}
