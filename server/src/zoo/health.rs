//! `turbo-kserve zoo-health`: the zoo container health check.
//!
//! Verifies both that the gRPC server answers `SERVING` (`turbo-kserve health`)
//! and that the demo page answers HTTP 2xx on loopback, over HTTPS when
//! `TURBO_DEMO_TLS_CERT` is set.

use std::net::SocketAddr;
use std::process::ExitCode;

use turbo_kserve::probe::{self, Probe};
use turbo_kserve::proto::health::health_check_response::ServingStatus;

use super::env::resolve;

const DEFAULT_DEMO_LISTEN: &str = "127.0.0.1:8080";
const ENV_DEMO_LISTEN: &str = "TURBO_DEMO_LISTEN";
const ENV_DEMO_TLS_CERT: &str = "TURBO_DEMO_TLS_CERT";

pub(super) async fn run(args: Vec<String>) -> ExitCode {
    if !args.is_empty() {
        eprintln!("zoo-health: takes no arguments");
        return ExitCode::from(2);
    }
    check_with(|k| std::env::var(k).ok()).await
}

async fn check_with(lookup: impl Fn(&str) -> Option<String>) -> ExitCode {
    let resolved = match resolve(&lookup) {
        Ok(env) => env,
        Err(e) => {
            eprintln!("zoo-env: {e}");
            return ExitCode::FAILURE;
        }
    };
    let merged = |key: &str| resolved.get(key).cloned().or_else(|| lookup(key));
    let probe = match Probe::parse_with(std::iter::empty(), merged) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("zoo-health: {e}");
            return ExitCode::from(2);
        }
    };
    match probe::check(&probe).await {
        Ok(ServingStatus::Serving) => println!("SERVING"),
        Ok(status) => {
            println!("{}", status.as_str_name());
            return ExitCode::FAILURE;
        }
        Err(e) => {
            eprintln!("zoo-health: {e}");
            return ExitCode::FAILURE;
        }
    }

    let (target, https) = match page_target(&lookup) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("zoo-health: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = probe_page(target, https).await {
        eprintln!("zoo-health: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn page_target(lookup: impl Fn(&str) -> Option<String>) -> Result<(SocketAddr, bool), String> {
    let raw = lookup(ENV_DEMO_LISTEN)
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DEFAULT_DEMO_LISTEN.to_owned());
    let listen: SocketAddr = raw.parse().map_err(|e| format!("{ENV_DEMO_LISTEN} {raw}: {e}"))?;
    let https = lookup(ENV_DEMO_TLS_CERT).is_some_and(|v| !v.trim().is_empty());
    Ok((probe::target(listen), https))
}

#[cfg(feature = "catalog-http")]
async fn probe_page(target: SocketAddr, https: bool) -> Result<(), String> {
    tokio::task::spawn_blocking(move || {
        let scheme = if https { "https" } else { "http" };
        let url = format!("{scheme}://{target}/");
        let mut builder = ureq::Agent::config_builder()
            .proxy(None)
            .max_redirects(0)
            .accept_encoding("identity")
            .timeout_global(Some(probe::DEFAULT_TIMEOUT));
        if https {
            // Loopback liveness check only: the page certificate names the
            // zoo's external host, which browsers verify, not 127.0.0.1.
            let tls = ureq::tls::TlsConfig::builder().disable_verification(true).build();
            builder = builder.tls_config(tls);
        }
        let agent: ureq::Agent = builder.build().into();
        let mut resp = agent.get(&url).call().map_err(|e| format!("{url}: {e}"))?;
        let _ = resp.body_mut().with_config().limit(64 * 1024).read_to_vec();
        Ok(())
    })
    .await
    .map_err(|e| format!("page probe task: {e}"))?
}

#[cfg(not(feature = "catalog-http"))]
async fn probe_page(_target: SocketAddr, _https: bool) -> Result<(), String> {
    Err("this turbo-kserve was built without the catalog-http feature".into())
}

#[cfg(all(test, feature = "catalog-http"))]
mod tests {
    use std::sync::Arc;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio_rustls::TlsAcceptor;
    use tokio_rustls::rustls::ServerConfig;
    use tokio_rustls::rustls::pki_types::pem::PemObject;
    use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer};
    use turbo_kserve::{Serve, Server};

    use super::*;

    const TEST_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----\n\
MIIBhDCCASugAwIBAgIUMDz+0icDFMibjkseSNPtVqwKB3MwCgYIKoZIzj0EAwIw\n\
HzEdMBsGA1UEAwwUdHVyYm8tdGVzdC1zZXJ2ZXItY2EwIBcNMjYxMDI2MDY0NjM3\n\
WhgPMjEyNjEwMDIwNjQ2MzdaMBQxEjAQBgNVBAMMCWxvY2FsaG9zdDBZMBMGByqG\n\
SM49AgEGCCqGSM49AwEHA0IABOi+ACPVzeNQULfW6uWG0DxJwUaESso6RcyeDKLF\n\
i2luj3jexu9SFUiZAXxeNuiYmD2NblgiY75pp/D2SNGD2BajTjBMMAkGA1UdEwQC\n\
MAAwDgYDVR0PAQH/BAQDAgWgMBMGA1UdJQQMMAoGCCsGAQUFBwMBMBoGA1UdEQQT\n\
MBGCCWxvY2FsaG9zdIcEfwAAATAKBggqhkjOPQQDAgNHADBEAiAyBIr/0nJz51E0\n\
0JHXUsmLmZKA3KgLL/mZu33rcPWbYwIgAgeirtnliflpsQbdl0I3BqLVt3Jo8PuU\n\
3eF5aNpH2eI=\n\
-----END CERTIFICATE-----\n";

    const TEST_KEY_PEM: &str = "-----BEGIN PRIVATE KEY-----\n\
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgXTzexqw3bQIvfaiE\n\
kQgrSussEkBOWa4hXiu6rEivRoahRANCAATovgAj1c3jUFC31urlhtA8ScFGhErK\n\
OkXMngyixYtpbo943sbvUhVImQF8XjbomJg9jW5YImO+aafw9kjRg9gW\n\
-----END PRIVATE KEY-----\n";

    #[test]
    fn page_target_maps_wildcard_binds_to_loopback_and_tracks_https() {
        let (addr, https) = page_target(|k| match k {
            ENV_DEMO_LISTEN => Some("0.0.0.0:9080".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(addr, "127.0.0.1:9080".parse().unwrap());
        assert!(!https);

        let (v6, tls) = page_target(|k| match k {
            ENV_DEMO_LISTEN => Some("[::]:9443".into()),
            ENV_DEMO_TLS_CERT => Some("/tls/page.crt".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(v6, "[::1]:9443".parse().unwrap());
        assert!(tls);
    }

    #[tokio::test]
    async fn check_with_verifies_both_grpc_server_and_http_or_https_page() {
        let server = Server::start(
            "127.0.0.1:0".parse().unwrap(),
            Vec::new(),
            4 * 1024 * 1024,
            Serve { admin: true, ..Serve::default() },
        )
        .await
        .unwrap();
        server.load().await.unwrap();
        let grpc_addr = server.local_addr().to_string();

        let http_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let http_addr = http_listener.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            let (mut stream, _) = http_listener.accept().await.unwrap();
            let mut req = [0u8; 512];
            let _ = stream.read(&mut req).await;
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok").await.unwrap();
        });

        let code = check_with(|k| match k {
            "TURBO_KSERVE_LISTEN" => Some(grpc_addr.clone()),
            ENV_DEMO_LISTEN => Some(http_addr.clone()),
            _ => None,
        })
        .await;
        assert_eq!(code, ExitCode::SUCCESS);

        let cert = CertificateDer::from_pem_slice(TEST_CERT_PEM.as_bytes()).unwrap();
        let key = PrivateKeyDer::from_pem_slice(TEST_KEY_PEM.as_bytes()).unwrap();
        let tls_cfg = ServerConfig::builder().with_no_client_auth().with_single_cert(vec![cert], key).unwrap();
        let acceptor = TlsAcceptor::from(Arc::new(tls_cfg));
        let https_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let https_addr = https_listener.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            let (tcp, _) = https_listener.accept().await.unwrap();
            let mut tls = acceptor.accept(tcp).await.unwrap();
            let mut req = [0u8; 512];
            let _ = tls.read(&mut req).await;
            tls.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok").await.unwrap();
            tls.shutdown().await.unwrap();
        });

        let https_code = check_with(|k| match k {
            "TURBO_KSERVE_LISTEN" => Some(grpc_addr.clone()),
            ENV_DEMO_LISTEN => Some(https_addr.clone()),
            ENV_DEMO_TLS_CERT => Some("/tls/page.crt".into()),
            _ => None,
        })
        .await;
        assert_eq!(https_code, ExitCode::SUCCESS);

        // Plain HTTP listener probed with TURBO_DEMO_TLS_CERT set must fail.
        let bad_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bad_addr = bad_listener.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            let (mut stream, _) = bad_listener.accept().await.unwrap();
            let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n").await;
        });
        let bad_code = check_with(|k| match k {
            "TURBO_KSERVE_LISTEN" => Some(grpc_addr.clone()),
            ENV_DEMO_LISTEN => Some(bad_addr.clone()),
            ENV_DEMO_TLS_CERT => Some("/tls/page.crt".into()),
            _ => None,
        })
        .await;
        assert_eq!(bad_code, ExitCode::FAILURE);

        server.drain();
        server.stop().await;
    }
}
