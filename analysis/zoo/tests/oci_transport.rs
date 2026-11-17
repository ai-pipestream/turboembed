//! Socket-level protocol and endpoint-policy tests for [`OciRegistry`]
//! (#2305, #2306). Each test binds a real loopback TCP listener (IPv4
//! or IPv6) and verifies that plaintext HTTP is strictly bounded to
//! explicit loopback authorities across the initial base, token
//! challenges, token redirects, and manifest/blob redirects.
#![cfg(feature = "http")]

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use turbo_zoo::{OciRegistry, Sha256, Transport, Url};

const HOST: &str = "registry.example.org";
const NAME: &str = "models/tiny";
const BODY: &[u8] = b"hello oci blob\n";

type Seen = Arc<Mutex<Vec<(String, bool)>>>;
type Responder = Box<dyn Fn(&str, &str, Option<&str>) -> (&'static str, String, Vec<u8>) + Send + Sync>;

fn spawn_loopback(bind_addr: &str, responder: Responder) -> (String, Seen) {
    let listener = TcpListener::bind(bind_addr).unwrap();
    let addr = listener.local_addr().unwrap();
    let base = match addr {
        std::net::SocketAddr::V4(v4) => format!("http://{v4}"),
        std::net::SocketAddr::V6(v6) => format!("http://[{}]:{}", v6.ip(), v6.port()),
    };
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    let at = base.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let mut conn = conn.unwrap();
            let mut reader = BufReader::new(conn.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let target = line.split(' ').nth(1).unwrap_or_default().to_owned();
            let mut token = None;
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).unwrap();
                if h.trim_end().is_empty() {
                    break;
                }
                if let Some(t) = h.trim_end().to_ascii_lowercase().strip_prefix("authorization: bearer ") {
                    token = Some(t.to_owned());
                }
            }
            log.lock().unwrap().push((target.clone(), token.is_some()));
            let (status, extra, body) = responder(&at, &target, token.as_deref());
            let head =
                format!("HTTP/1.1 {status}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            conn.write_all(head.as_bytes()).unwrap();
            conn.write_all(&body).unwrap();
        }
    });
    (base, seen)
}

fn ok_blob(bytes: &[u8]) -> (&'static str, String, Vec<u8>) {
    ("200 OK", String::new(), bytes.to_vec())
}

fn challenge(realm: &str) -> (&'static str, String, Vec<u8>) {
    (
        "401 Unauthorized",
        format!("WWW-Authenticate: Bearer realm=\"{realm}\",service=\"stand-in\",scope=\"repository:{NAME}:pull\"\r\n"),
        b"{}".to_vec(),
    )
}

mod socket_protocol {
    use super::*;

    #[test]
    fn ipv4_and_ipv6_loopback_serve_blobs_with_token_and_storage_redirect() {
        for bind in ["127.0.0.1:0", "[::1]:0"] {
            let (base, seen) = spawn_loopback(
                bind,
                Box::new(|base, target, token| {
                    if target.starts_with("/token?") {
                        return ("200 OK", String::new(), br#"{"token": "tok-1"}"#.to_vec());
                    }
                    if target == "/storage/blob" {
                        return match token {
                            Some(_) => ("400 Bad Request", String::new(), b"token leaked".to_vec()),
                            None => ok_blob(BODY),
                        };
                    }
                    if token != Some("tok-1") {
                        return challenge(&format!("{base}/token"));
                    }
                    ("307 Temporary Redirect", "Location: /storage/blob\r\n".to_owned(), Vec::new())
                }),
            );
            let client = OciRegistry::with(HOST, &base).unwrap();
            let digest = Sha256::of(BODY);
            let url: Url = format!("https://{HOST}/v2/{NAME}/blobs/sha256:{digest}").parse().unwrap();
            assert_eq!(client.get(&url, BODY.len() as u64).unwrap(), BODY);
            let seen = seen.lock().unwrap();
            assert!(seen.iter().any(|(t, has_tok)| t == "/storage/blob" && !has_tok), "{seen:?}");
        }
    }

    #[test]
    fn non_loopback_http_token_realm_is_refused_before_request() {
        for bad_realm in [
            "http://auth.example.org/token",
            "http://127.0.0.1.example.org:5000/token",
            "http://127.0.0.1:5000@auth.example.org/token",
            "http://localhost/token",
        ] {
            let realm = bad_realm.to_owned();
            let (base, seen) = spawn_loopback("127.0.0.1:0", Box::new(move |_, _, _| challenge(&realm)));
            let client = OciRegistry::with(HOST, &base).unwrap();
            let url: Url = format!("https://{HOST}/v2/{NAME}/blobs/sha256:{}", Sha256::of(BODY)).parse().unwrap();
            let err = client.get(&url, BODY.len() as u64).unwrap_err();
            assert!(err.contains("not an https:// or loopback http:// address"), "{bad_realm}: {err}");
            assert_eq!(seen.lock().unwrap().len(), 1);
        }
    }

    #[test]
    fn token_endpoint_redirect_to_non_loopback_http_is_refused_while_loopback_redirect_works() {
        let (base, seen) = spawn_loopback(
            "127.0.0.1:0",
            Box::new(|base, target, token| {
                if target.starts_with("/token?") {
                    return ("302 Found", "Location: /token2\r\n".to_owned(), Vec::new());
                }
                if target == "/token2" {
                    return ("200 OK", String::new(), br#"{"token": "tok-2"}"#.to_vec());
                }
                if token != Some("tok-2") {
                    return challenge(&format!("{base}/token"));
                }
                ok_blob(BODY)
            }),
        );
        let client = OciRegistry::with(HOST, &base).unwrap();
        let url: Url = format!("https://{HOST}/v2/{NAME}/blobs/sha256:{}", Sha256::of(BODY)).parse().unwrap();
        assert_eq!(client.get(&url, BODY.len() as u64).unwrap(), BODY);
        assert!(seen.lock().unwrap().iter().any(|(t, _)| t == "/token2"));

        let (base, _) = spawn_loopback(
            "127.0.0.1:0",
            Box::new(|base, target, _| {
                if target.starts_with("/token?") {
                    return ("302 Found", "Location: http://evil.example.org/token\r\n".to_owned(), Vec::new());
                }
                challenge(&format!("{base}/token"))
            }),
        );
        let client = OciRegistry::with(HOST, &base).unwrap();
        let err = client.get(&url, BODY.len() as u64).unwrap_err();
        assert!(err.contains("not an https:// or loopback http:// address"), "{err}");
    }

    #[test]
    fn blob_redirect_to_non_loopback_http_or_userinfo_escape_is_refused_before_request() {
        for bad_location in [
            "http://storage.example.org/blob",
            "//storage.example.org/blob",
            "http://127.0.0.1:5000@storage.example.org/blob",
            "http://[::1].example.org:5000/blob",
        ] {
            let loc = bad_location.to_owned();
            let (base, seen) = spawn_loopback(
                "127.0.0.1:0",
                Box::new(move |base, target, token| {
                    if target.starts_with("/token?") {
                        return ("200 OK", String::new(), br#"{"token": "tok-3"}"#.to_vec());
                    }
                    if token != Some("tok-3") {
                        return challenge(&format!("{base}/token"));
                    }
                    ("307 Temporary Redirect", format!("Location: {loc}\r\n"), Vec::new())
                }),
            );
            let client = OciRegistry::with(HOST, &base).unwrap();
            let url: Url = format!("https://{HOST}/v2/{NAME}/blobs/sha256:{}", Sha256::of(BODY)).parse().unwrap();
            let err = client.get(&url, BODY.len() as u64).unwrap_err();
            assert!(err.contains("not an https:// or loopback http:// address"), "{bad_location}: {err}");
            assert_eq!(seen.lock().unwrap().len(), 3);
        }
    }
}
