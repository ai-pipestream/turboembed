//! The demo over TLS: a call made through a TLS client the test CA
//! trusts, a plain-HTTP client that gets no answer, and a key that does
//! not load, and an upload PUT over TLS. The certificates in tests/tls are for these tests only.

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use common::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;
use tokio_rustls::rustls::crypto::ring;
use tokio_rustls::rustls::pki_types::pem::PemObject;
use tokio_rustls::rustls::pki_types::{CertificateDer, ServerName};
use tokio_rustls::rustls::{ClientConfig, RootCertStore};
use turbo_kserve::Serve;
use turbo_kserve_demo::{Allow, Demo, Options, Tls};

fn file(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/tls").join(name)
}

fn tls() -> Options {
    Options { tls: Some(Tls { cert: file("cert.pem"), key: file("key.pem") }), ..Options::default() }
}

fn client() -> TlsConnector {
    let mut roots = RootCertStore::empty();
    roots.add(CertificateDer::from_pem_file(file("ca.pem")).unwrap()).unwrap();
    let config = ClientConfig::builder_with_provider(Arc::new(ring::default_provider()))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_root_certificates(roots)
        .with_no_client_auth();
    TlsConnector::from(Arc::new(config))
}

#[tokio::test]
async fn a_call_over_tls_is_answered_and_plain_http_is_not() {
    let s = server(&["tiny-bert-bundle"], Serve::default()).await;
    let d = demo_with(s.local_addr(), tls()).await;
    let tcp = TcpStream::connect(d.local_addr()).await.unwrap();
    let mut conn = client().connect(ServerName::try_from("localhost").unwrap(), tcp).await.unwrap();
    let req = "POST /inference.GRPCInferenceService/ServerLive HTTP/1.1\r\nHost: localhost\r\n\
               Content-Type: application/proto\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    conn.write_all(req.as_bytes()).await.unwrap();
    let mut reply = Vec::new();
    let _ = conn.read_to_end(&mut reply).await;
    let reply = String::from_utf8_lossy(&reply);
    assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
    assert!(reply.contains("content-type: application/proto"), "{reply}");

    let mut plain = TcpStream::connect(d.local_addr()).await.unwrap();
    plain.write_all(req.as_bytes()).await.unwrap();
    let mut reply = Vec::new();
    let _ = plain.read_to_end(&mut reply).await;
    assert!(!reply.starts_with(b"HTTP/1.1 200"));
}

#[tokio::test]
async fn a_certificate_or_key_that_does_not_load_fails_start() {
    let to = "http://127.0.0.1:1";
    let listen = "127.0.0.1:0".parse().unwrap();
    for (cert, key) in
        [("ca.pem", "missing.pem"), ("missing.pem", "key.pem"), ("key.pem", "key.pem"), ("cert.pem", "cert.pem")]
    {
        let o = Options { tls: Some(Tls { cert: file(cert), key: file(key) }), ..Options::default() };
        assert!(Demo::start_with(listen, to, o).await.is_err(), "{cert} {key}");
    }
    // A key that does not match the certificate.
    let o = Options { tls: Some(Tls { cert: file("ca.pem"), key: file("key.pem") }), ..Options::default() };
    assert!(Demo::start_with(listen, to, o).await.is_err());
}

#[tokio::test]
async fn tls_scheme_is_part_of_origin_and_idle_handshake_expires() {
    use std::time::Duration;
    let s = server(&["tiny-bert-bundle"], Serve::default()).await;
    let d = demo_with(s.local_addr(), tls()).await;
    let mut idle = TcpStream::connect(d.local_addr()).await.unwrap();
    for (origin, status) in [("https://localhost", 200), ("http://localhost", 403)] {
        let tcp = TcpStream::connect(d.local_addr()).await.unwrap();
        let mut conn = client().connect(ServerName::try_from("localhost").unwrap(), tcp).await.unwrap();
        let req = format!(
            "POST /inference.GRPCInferenceService/ServerLive HTTP/1.1\r\nHost: localhost\r\nOrigin: {origin}\r\nContent-Type: application/proto\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        conn.write_all(req.as_bytes()).await.unwrap();
        let mut reply = Vec::new();
        let _ = tokio::time::timeout(Duration::from_secs(2), conn.read_to_end(&mut reply)).await.unwrap();
        assert!(reply.starts_with(format!("HTTP/1.1 {status}").as_bytes()), "{}", String::from_utf8_lossy(&reply));
    }
    let mut one = [0];
    let read = tokio::time::timeout(Duration::from_secs(12), idle.read(&mut one)).await.expect("handshake deadline");
    assert!(matches!(read, Ok(0) | Err(_)), "idle TLS handshake stayed open");
}

/// A PUT upload over TLS: refused by default, forwarded under --allow (to
/// an index the server does not have, so the answer is the server's).
#[tokio::test]
async fn an_upload_over_tls_is_gated_and_forwarded() {
    let s = server(&["tiny-bert-bundle"], Serve { admin: true, ..Serve::default() }).await;
    let path = "/upload/turbo.search.v1.TurboSearchService/IndexDocuments";
    let item = [0x0a, 4, b'n', b'o', b'n', b'e'];
    let mut body = vec![0, 0, 0, 0, item.len() as u8];
    body.extend_from_slice(&item);
    let allowed = Allow::with(&["/turbo.search.v1.TurboSearchService/IndexDocuments".to_string()]).unwrap();
    for (allow, status) in [(Allow::default(), "403"), (allowed, "404")] {
        let d = demo_with(s.local_addr(), Options { allow, ..tls() }).await;
        let tcp = TcpStream::connect(d.local_addr()).await.unwrap();
        let mut conn = client().connect(ServerName::try_from("localhost").unwrap(), tcp).await.unwrap();
        let req = format!(
            "PUT {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        conn.write_all(req.as_bytes()).await.unwrap();
        conn.write_all(&body).await.unwrap();
        let mut reply = Vec::new();
        let _ = conn.read_to_end(&mut reply).await;
        let reply = String::from_utf8_lossy(&reply);
        assert!(reply.starts_with(&format!("HTTP/1.1 {status}")), "{reply}");
        assert!(reply.contains("grpc-status: "), "{reply}");
    }
}
