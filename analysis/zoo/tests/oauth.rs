//! Google sign-in for a registered Drive (#1744), against a small server
//! started in-process that answers both the OAuth token endpoint and the
//! Drive API: the code exchange, refresh, the loopback sign-in, and a
//! Drive folder read with a stored grant whose access token runs out or
//! is turned down.
#![cfg(feature = "http")]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use turbo_zoo::{
    Acceptances, DRIVE_READONLY, DriveAuth, GoogleDrive, GoogleOAuth, Index, Pkce, Policy, Renewing, Sha256, Sha512,
    UserId, Zoo,
};

const TEXT: &[u8] = b"hello hello hello\n";

/// What the server saw: each request's method and target, its headers
/// (lower-cased) and its body.
#[derive(Debug, Clone)]
struct Request {
    line: String,
    headers: Vec<String>,
    body: String,
}

/// How the server answers.
#[derive(Default)]
struct Script {
    /// The access token each refresh grants, in turn: `tok-1`, `tok-2`...
    granted: usize,
    /// How long each granted token lasts.
    expires_in: u64,
    /// Access tokens Drive turns down with a 401.
    revoked: Vec<String>,
}

type Seen = Arc<Mutex<Vec<Request>>>;

fn server(script: Script) -> (String, Seen) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let seen: Seen = Arc::default();
    let log = seen.clone();
    let script = Arc::new(Mutex::new(script));
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let mut conn = conn.unwrap();
            let mut reader = BufReader::new(conn.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let mut headers = Vec::new();
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).unwrap();
                if h.trim_end().is_empty() {
                    break;
                }
                headers.push(h.trim_end().to_ascii_lowercase());
            }
            let length = headers
                .iter()
                .find_map(|h| h.strip_prefix("content-length: "))
                .map_or(0, |n| n.parse::<usize>().unwrap());
            let mut body = vec![0; length];
            reader.read_exact(&mut body).unwrap();
            let req = Request { line: line.trim_end().to_owned(), headers, body: String::from_utf8(body).unwrap() };
            log.lock().unwrap().push(req.clone());
            let (status, body) = answer(&mut script.lock().unwrap(), &req);
            let head = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            conn.write_all(head.as_bytes()).unwrap();
            conn.write_all(&body).unwrap();
        }
    });
    (endpoint, seen)
}

fn answer(script: &mut Script, req: &Request) -> (&'static str, Vec<u8>) {
    let target = req.line.split(' ').nth(1).unwrap_or_default();
    if req.line.starts_with("POST /token ") {
        let form = |k: &str| req.body.split('&').find_map(|kv| kv.strip_prefix(&format!("{k}="))).map(str::to_owned);
        if form("client_id").as_deref() != Some("client-1") {
            return ("401 Unauthorized", br#"{"error": "invalid_client"}"#.to_vec());
        }
        let grant = match form("grant_type").as_deref() {
            Some("authorization_code") if form("code").as_deref() == Some("code-1") => true,
            Some("refresh_token") if form("refresh_token").as_deref() == Some("refresh-1") => false,
            _ => return ("400 Bad Request", br#"{"error": "invalid_grant", "error_description": "Bad"}"#.to_vec()),
        };
        script.granted += 1;
        let refresh = if grant { r#", "refresh_token": "refresh-1""# } else { "" };
        let body =
            format!(r#"{{"access_token": "tok-{}", "expires_in": {}{refresh}}}"#, script.granted, script.expires_in);
        return ("200 OK", body.into_bytes());
    }
    let bearer = req.headers.iter().find_map(|h| h.strip_prefix("authorization: bearer ")).unwrap_or_default();
    if script.revoked.iter().any(|t| t == bearer) {
        return ("401 Unauthorized", Vec::new());
    }
    // A Drive that has every folder and file asked for: one id per name.
    if target.starts_with("/drive/v3/files?") {
        return ("200 OK", format!(r#"{{"files": [{{"id": "f", "size": "{}"}}]}}"#, TEXT.len()).into_bytes());
    }
    if target.starts_with("/drive/v3/files/f?alt=media") {
        return ("200 OK", TEXT.to_vec());
    }
    ("404 Not Found", Vec::new())
}

fn oauth(endpoint: &str) -> GoogleOAuth {
    GoogleOAuth::with_endpoints("client-1", Some("secret-1"), &format!("{endpoint}/auth"), &format!("{endpoint}/token"))
        .unwrap()
}

fn index() -> Index {
    let json = format!(
        r#"{{"zoo_version": 1, "artifacts": [{{"id": "hello", "kind": "KIND_LEXICON", "format": "FORMAT_TEXT",
          "version": "1", "license": "Apache-2.0", "license_terms": "https://example.org/LICENSE", "trained_on": [],
          "class": "CLASS_PERMISSIVE", "attribution": "Example.",
          "files": [{{"path": "hello.txt", "url": "https://example.org/hello.txt", "size": {}, "sha256": "{}",
            "sha512": "{}"}}]}}]}}"#,
        TEXT.len(),
        Sha256::of(TEXT),
        Sha512::of(TEXT),
    );
    Index::parse(json.as_bytes()).unwrap()
}

/// Fetches `hello` through `drive` into a fresh cache.
fn fetch(drive: &GoogleDrive, name: &str) -> Result<Vec<u8>, String> {
    let dir = std::env::temp_dir().join(format!("turbo-zoo-oauth-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let zoo = Zoo::new(index(), dir.clone(), Policy::default(), drive);
    let alice: UserId = "alice".parse().unwrap();
    let got = zoo.fetch("hello", &alice, &Acceptances::default()).map(|f| std::fs::read(&f.files[0]).unwrap());
    let _ = std::fs::remove_dir_all(&dir);
    got.map_err(|e| e.to_string())
}

#[test]
fn the_consent_page_asks_for_read_only_drive_offline_with_a_challenge() {
    let o = GoogleOAuth::new("client-1", None).unwrap();
    let pkce = Pkce::new();
    let url = o.authorize_url("http://127.0.0.1:5000", &pkce);
    assert!(url.starts_with("https://accounts.google.com/o/oauth2/v2/auth?client_id=client-1&"), "{url}");
    for part in [
        "redirect_uri=http%3A%2F%2F127.0.0.1%3A5000",
        "response_type=code",
        "code_challenge_method=S256",
        "access_type=offline",
        "prompt=consent",
    ] {
        assert!(url.contains(part), "{part}: {url}");
    }
    assert!(url.contains(&format!("scope={}", DRIVE_READONLY.replace(':', "%3A").replace('/', "%2F"))), "{url}");
    assert!(url.contains(&format!("state={}", pkce.state)), "{url}");
    assert!(url.contains(&format!("code_challenge={}", pkce.challenge())), "{url}");
    assert!(!url.contains(&pkce.verifier), "the verifier never leaves");
}

#[test]
fn a_code_is_exchanged_and_a_refresh_token_refreshed() {
    let (endpoint, seen) = server(Script { expires_in: 3599, ..Script::default() });
    let o = oauth(&endpoint);
    let pkce = Pkce::new();
    let grant = o.exchange("code-1", "http://127.0.0.1:5000", &pkce).unwrap();
    assert_eq!((grant.access_token.as_str(), grant.expires_in), ("tok-1", 3599));
    assert_eq!(grant.refresh_token.as_deref(), Some("refresh-1"));
    assert!(!format!("{grant:?}").contains("tok-1"), "Debug shows no token");
    let again = o.refresh("refresh-1").unwrap();
    assert_eq!((again.access_token.as_str(), again.refresh_token), ("tok-2", None));
    let body = seen.lock().unwrap()[0].body.clone();
    for part in ["grant_type=authorization_code", "code=code-1", "client_secret=secret-1"] {
        assert!(body.contains(part), "{part}: {body}");
    }
    assert!(body.contains(&format!("code_verifier={}", pkce.verifier)), "{body}");
    let e = o.refresh("refresh-other").unwrap_err();
    assert!(e.contains("status 400: invalid_grant Bad"), "{e}");
}

#[test]
fn the_loopback_sign_in_reads_the_code_back_and_checks_the_state() {
    let (endpoint, _) = server(Script { expires_in: 3600, ..Script::default() });
    let o = oauth(&endpoint);
    // The browser: follows the consent page's redirect back with a code,
    // the state it was given, or another.
    let browse = |state_of: fn(&str) -> String| {
        move |url: &str| {
            let param = |k: &str| {
                let v = url.split(['?', '&']).find_map(|kv| kv.strip_prefix(&format!("{k}="))).unwrap();
                v.replace("%3A", ":").replace("%2F", "/")
            };
            let back = format!("{}/?code=code-1&state={}", param("redirect_uri"), state_of(&param("state")));
            std::thread::spawn(move || {
                let host = back.strip_prefix("http://").unwrap().split('/').next().unwrap().to_owned();
                let mut s = std::net::TcpStream::connect(&host).unwrap();
                // A favicon request first, which is passed over.
                s.write_all(format!("GET /favicon.ico HTTP/1.1\r\nHost: {host}\r\n\r\n").as_bytes()).unwrap();
                let mut s = std::net::TcpStream::connect(&host).unwrap();
                let target = back.split_once(&host).unwrap().1;
                s.write_all(format!("GET {target} HTTP/1.1\r\nHost: {host}\r\n\r\n").as_bytes()).unwrap();
                let mut page = String::new();
                let _ = s.read_to_string(&mut page);
            });
        }
    };
    let grant = o.sign_in_loopback(Duration::from_secs(10), browse(|s| s.to_owned())).unwrap();
    assert_eq!(grant.refresh_token.as_deref(), Some("refresh-1"));
    let e = o.sign_in_loopback(Duration::from_secs(10), browse(|_| "forged".into())).unwrap_err();
    assert!(e.contains("another state"), "{e}");
    let e = o.sign_in_loopback(Duration::from_millis(200), |_| {}).unwrap_err();
    assert!(e.contains("no sign-in came back"), "{e}");
}

#[test]
fn a_stored_grant_reads_drive_and_renews_a_token_drive_turns_down() {
    let (endpoint, seen) = server(Script { expires_in: 3600, revoked: vec!["tok-1".into()], ..Script::default() });
    let renewing = Renewing::new(oauth(&endpoint), "refresh-1").unwrap();
    let drive = GoogleDrive::with("gdrive://top", Some(&endpoint), DriveAuth::Refresh(Arc::new(renewing))).unwrap();
    assert_eq!(fetch(&drive, "revoked").unwrap(), TEXT);
    let seen = seen.lock().unwrap();
    let tokens = seen.iter().filter(|r| r.line.starts_with("POST /token ")).count();
    assert_eq!(tokens, 2, "one grant, and one more after the 401: {seen:?}");
    let last = seen.last().unwrap();
    assert!(last.headers.contains(&"authorization: bearer tok-2".to_string()), "{last:?}");
}

#[test]
fn a_token_about_to_run_out_is_renewed_and_a_live_one_reused() {
    // Lasting under a minute, each token is renewed before its next use.
    let (endpoint, seen) = server(Script { expires_in: 30, ..Script::default() });
    let renewing = Arc::new(Renewing::new(oauth(&endpoint), "refresh-1").unwrap());
    assert_eq!(renewing.bearer(false).unwrap(), "tok-1");
    assert_eq!(renewing.bearer(false).unwrap(), "tok-2");
    let (endpoint, _) = server(Script { expires_in: 3600, ..Script::default() });
    let lasting = Renewing::new(oauth(&endpoint), "refresh-1").unwrap();
    assert_eq!(lasting.bearer(false).unwrap(), "tok-1");
    assert_eq!(lasting.bearer(false).unwrap(), "tok-1");
    assert_eq!(lasting.bearer(true).unwrap(), "tok-2");
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[test]
fn clients_endpoints_and_tokens_are_checked_before_any_request() {
    assert!(GoogleOAuth::new("", None).is_err());
    assert!(GoogleOAuth::new("client 1", None).is_err());
    assert!(GoogleOAuth::new("client-1", Some("")).is_err());
    let e = GoogleOAuth::with_endpoints("c", None, "http://example.org/auth", "https://example.org/token");
    assert!(e.err().unwrap().contains("http:// on this machine"));
    let o = GoogleOAuth::new("client-1", None).unwrap();
    assert!(Renewing::new(o, "refresh\n1").is_err());
}
