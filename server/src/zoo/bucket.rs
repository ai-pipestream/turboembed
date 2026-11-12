//! `turbo-kserve zoo-bucket`: ensures the catalog's S3 bucket exists.
//!
//! Reads `TURBO_KSERVE_CATALOG_STORE` (`s3://bucket[/prefix]`), signs a
//! `PUT /<bucket>` request with AWS SigV4 (`AWS_ACCESS_KEY_ID`,
//! `AWS_SECRET_ACCESS_KEY`, optional `AWS_SESSION_TOKEN`, `AWS_REGION` or
//! `AWS_DEFAULT_REGION`) against `AWS_ENDPOINT_URL_S3` or `AWS_ENDPOINT_URL`,
//! and waits up to a minute for the store to answer.

use std::process::ExitCode;
#[cfg(feature = "catalog-http")]
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const ENV_CATALOG_STORE: &str = "TURBO_KSERVE_CATALOG_STORE";
const ENV_ENDPOINT_S3: &str = "AWS_ENDPOINT_URL_S3";
const ENV_ENDPOINT: &str = "AWS_ENDPOINT_URL";
const ENV_REGION: &str = "AWS_REGION";
const ENV_DEFAULT_REGION: &str = "AWS_DEFAULT_REGION";
const ENV_ACCESS_KEY: &str = "AWS_ACCESS_KEY_ID";
const ENV_SECRET_KEY: &str = "AWS_SECRET_ACCESS_KEY";
const ENV_SESSION_TOKEN: &str = "AWS_SESSION_TOKEN";
const ENV_AWS_CA_BUNDLE: &str = "AWS_CA_BUNDLE";
const ENV_SSL_CERT_FILE: &str = "SSL_CERT_FILE";

const DEFAULT_REGION: &str = "us-east-1";
#[cfg(feature = "catalog-http")]
const MAX_ATTEMPTS: usize = 60;
#[cfg(feature = "catalog-http")]
const RETRY_DELAY: Duration = Duration::from_secs(1);
#[cfg(feature = "catalog-http")]
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

struct BucketConfig {
    bucket: String,
    endpoint_display: String,
    base: String,
    host: String,
    region: String,
    access_key: String,
    secret_key: String,
    session_token: Option<String>,
    ca_bundle: Option<Vec<u8>>,
}

impl BucketConfig {
    fn parse(lookup: impl Fn(&str) -> Option<String>) -> Result<BucketConfig, String> {
        let get = |key: &str| lookup(key).filter(|v| !v.is_empty());
        let store = get(ENV_CATALOG_STORE).ok_or_else(|| format!("set {ENV_CATALOG_STORE} to s3://bucket/prefix"))?;
        let rest = store.strip_prefix("s3://").ok_or_else(|| format!("{store} is not s3://bucket/prefix"))?;
        let bucket = rest.split('/').next().unwrap_or("");
        if !valid_bucket(bucket) {
            return Err(format!("{store}: invalid S3 bucket name"));
        }

        let endpoint_display = get(ENV_ENDPOINT_S3)
            .or_else(|| get(ENV_ENDPOINT))
            .ok_or_else(|| format!("set {ENV_ENDPOINT_S3} or {ENV_ENDPOINT}"))?;
        let (scheme, host) = split_endpoint(&endpoint_display).ok_or_else(|| {
            format!("endpoint {endpoint_display} must be http:// or https:// with host and optional port")
        })?;
        let base = format!("{scheme}://{host}");

        let region = get(ENV_REGION).or_else(|| get(ENV_DEFAULT_REGION)).unwrap_or_else(|| DEFAULT_REGION.to_owned());
        if region.is_empty() || !region.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-') {
            return Err(format!("{region:?} is not a valid region name"));
        }

        let access_key = get(ENV_ACCESS_KEY).ok_or_else(|| format!("set {ENV_ACCESS_KEY}"))?;
        let secret_key = get(ENV_SECRET_KEY).ok_or_else(|| format!("set {ENV_SECRET_KEY}"))?;
        let session_token = get(ENV_SESSION_TOKEN);

        let ca_bundle = if let Some(path) = get(ENV_AWS_CA_BUNDLE) {
            Some(std::fs::read(&path).map_err(|e| format!("{ENV_AWS_CA_BUNDLE} {path}: {e}"))?)
        } else if let Some(path) = get(ENV_SSL_CERT_FILE) {
            std::fs::read(&path).ok()
        } else {
            None
        };

        Ok(BucketConfig {
            bucket: bucket.to_owned(),
            endpoint_display,
            base,
            host,
            region,
            access_key,
            secret_key,
            session_token,
            ca_bundle,
        })
    }
}

pub(super) async fn run(args: Vec<String>) -> ExitCode {
    if !args.is_empty() {
        eprintln!("zoo-bucket: takes no arguments");
        return ExitCode::from(2);
    }
    let cfg = match BucketConfig::parse(|k| std::env::var(k).ok()) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("zoo-bucket: {e}");
            return ExitCode::from(2);
        }
    };
    ensure(cfg).await
}

#[cfg(feature = "catalog-http")]
async fn ensure(cfg: BucketConfig) -> ExitCode {
    tokio::task::spawn_blocking(move || ensure_sync(&cfg, MAX_ATTEMPTS, RETRY_DELAY)).await.unwrap_or(ExitCode::FAILURE)
}

#[cfg(not(feature = "catalog-http"))]
async fn ensure(cfg: BucketConfig) -> ExitCode {
    let _ = (
        &cfg.bucket,
        &cfg.endpoint_display,
        &cfg.base,
        &cfg.host,
        &cfg.region,
        &cfg.access_key,
        &cfg.secret_key,
        &cfg.session_token,
        &cfg.ca_bundle,
    );
    eprintln!("zoo-bucket: this turbo-kserve was built without the catalog-http feature");
    ExitCode::FAILURE
}

#[cfg(feature = "catalog-http")]
fn build_agent(ca_bundle: Option<&[u8]>) -> Result<ureq::Agent, String> {
    let mut builder = ureq::Agent::config_builder()
        .proxy(None)
        .max_redirects(0)
        .http_status_as_error(false)
        .accept_encoding("identity")
        .timeout_global(Some(REQUEST_TIMEOUT));
    if let Some(pem) = ca_bundle {
        let certs = parse_pem_certs(pem)?;
        let tls = ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::new_with_certs(&certs)).build();
        builder = builder.tls_config(tls);
    }
    Ok(builder.build().into())
}

#[cfg(feature = "catalog-http")]
fn parse_pem_certs(bundle: &[u8]) -> Result<Vec<ureq::tls::Certificate<'static>>, String> {
    const BEGIN: &[u8] = b"-----BEGIN CERTIFICATE-----";
    const END: &[u8] = b"-----END CERTIFICATE-----";
    let mut certs = Vec::new();
    let mut rest = bundle;
    while !rest.is_empty() {
        let Some(start) = rest.windows(BEGIN.len()).position(|w| w == BEGIN) else { break };
        let after = &rest[start..];
        let Some(finish) = after.windows(END.len()).position(|w| w == END) else { break };
        let end = finish + END.len();
        let cert = ureq::tls::Certificate::from_pem(&after[..end])
            .map_err(|e| format!("CA bundle certificate: {e}"))?
            .to_owned();
        certs.push(cert);
        rest = &after[end..];
    }
    if certs.is_empty() {
        return Err("no PEM certificate block in CA bundle".into());
    }
    Ok(certs)
}

#[cfg(feature = "catalog-http")]
fn ensure_sync(cfg: &BucketConfig, attempts: usize, retry_delay: Duration) -> ExitCode {
    use turbo_zoo::sigv4::{Credentials, EMPTY_SHA256, amz_date, sign};

    let agent = match build_agent(cfg.ca_bundle.as_deref()) {
        Ok(agent) => agent,
        Err(e) => {
            eprintln!("zoo-bucket: {e}");
            return ExitCode::from(2);
        }
    };
    let creds =
        Credentials { key: cfg.access_key.clone(), secret: cfg.secret_key.clone(), token: cfg.session_token.clone() };
    let path = format!("/{}", cfg.bucket);
    let url = format!("{}{path}", cfg.base);

    for attempt in 0..attempts {
        let secs = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_secs(),
            Err(e) => {
                eprintln!("zoo-bucket: system clock: {e}");
                return ExitCode::FAILURE;
            }
        };
        let headers = match sign("PUT", &creds, &cfg.region, (&cfg.host, &path), EMPTY_SHA256, &[], &amz_date(secs)) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("zoo-bucket: {e}");
                return ExitCode::FAILURE;
            }
        };
        let mut req = agent.put(&url);
        for (name, value) in headers {
            req = req.header(name, value);
        }
        if let Ok(resp) = req.send_empty() {
            match resp.status().as_u16() {
                200 => {
                    println!("zoo-bucket: made {}", cfg.bucket);
                    return ExitCode::SUCCESS;
                }
                409 => {
                    println!("zoo-bucket: {} is already there", cfg.bucket);
                    return ExitCode::SUCCESS;
                }
                500..=599 => {}
                code => {
                    eprintln!("zoo-bucket: making {} at {} answered HTTP {code}", cfg.bucket, cfg.endpoint_display);
                    return ExitCode::FAILURE;
                }
            }
        }
        if attempt + 1 < attempts {
            std::thread::sleep(retry_delay);
        }
    }
    eprintln!("zoo-bucket: {} did not answer in a minute", cfg.endpoint_display);
    ExitCode::FAILURE
}

fn valid_bucket(name: &str) -> bool {
    let b = name.as_bytes();
    (3..=63).contains(&b.len())
        && b.iter().all(|&c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'.' || c == b'-')
        && b[0].is_ascii_alphanumeric()
        && b[b.len() - 1].is_ascii_alphanumeric()
        && !name.contains("..")
}

fn split_endpoint(endpoint: &str) -> Option<(&'static str, String)> {
    let (scheme, rest) = if let Some(r) = endpoint.strip_prefix("https://") {
        ("https", r)
    } else {
        ("http", endpoint.strip_prefix("http://")?)
    };
    let authority = rest.strip_suffix('/').unwrap_or(rest);
    if authority.is_empty() || authority.contains(['/', '?', '#', '@']) {
        return None;
    }
    let default_port = if scheme == "https" { ":443" } else { ":80" };
    Some((scheme, authority.strip_suffix(default_port).unwrap_or(authority).to_ascii_lowercase()))
}

#[cfg(all(test, feature = "catalog-http"))]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::Arc;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_rustls::TlsAcceptor;
    use tokio_rustls::rustls::ServerConfig;
    use tokio_rustls::rustls::pki_types::pem::PemObject;
    use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer};
    use turbo_zoo::sigv4::EMPTY_SHA256;

    use super::*;

    const TEST_CA_PEM: &str = "-----BEGIN CERTIFICATE-----\n\
MIIBhDCCASqgAwIBAgIUAVogSpXO+xbeQ4x9o1xRs9pbHo8wCgYIKoZIzj0EAwIw\n\
HzEdMBsGA1UEAwwUdHVyYm8tdGVzdC1zZXJ2ZXItY2EwIBcNMjYxMDI2MDY0NjM3\n\
WhgPMjEyNjEwMDIwNjQ2MzdaMB8xHTAbBgNVBAMMFHR1cmJvLXRlc3Qtc2VydmVy\n\
LWNhMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAE9sFFiocjmxKXQ0kCtYZ7tmSY\n\
4hYeE1i66YfvRcbSiufRzGTY7h3xjboIymPNxoeEgvgk1231MfGd0vS3Xc7Qk6NC\n\
MEAwHQYDVR0OBBYEFOFpRbGCcyAKmJ0s5z3Ahii6GYDdMA8GA1UdEwEB/wQFMAMB\n\
Af8wDgYDVR0PAQH/BAQDAgEGMAoGCCqGSM49BAMCA0gAMEUCIQCYhj6qYO018b+k\n\
THo9t4rdisRMQVK5qXp1b3TTiqwu0wIgPFW9G8RUUEgj76ET9gx6m2PWA+7QmoHs\n\
czkEzHeFhYU=\n\
-----END CERTIFICATE-----\n";

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

    fn base_cfg(endpoint: String) -> BucketConfig {
        BucketConfig::parse(|k| match k {
            ENV_CATALOG_STORE => Some("s3://turbo-zoo/zoo".into()),
            ENV_ENDPOINT => Some(endpoint.clone()),
            ENV_ACCESS_KEY => Some("AKIAEXAMPLE".into()),
            ENV_SECRET_KEY => Some("secret-key".into()),
            _ => None,
        })
        .unwrap()
    }

    #[test]
    fn config_rejects_missing_or_malformed_store_endpoint_and_keys() {
        assert!(BucketConfig::parse(|_| None).is_err());
        assert!(BucketConfig::parse(|k| (k == ENV_CATALOG_STORE).then(|| "file:///tmp".into())).is_err());
        assert!(BucketConfig::parse(|k| (k == ENV_CATALOG_STORE).then(|| "s3://Bad_Bucket/zoo".into())).is_err());
        assert!(
            BucketConfig::parse(|k| match k {
                ENV_CATALOG_STORE => Some("s3://turbo-zoo/zoo".into()),
                ENV_ENDPOINT => Some("ftp://127.0.0.1:9000".into()),
                ENV_ACCESS_KEY => Some("k".into()),
                ENV_SECRET_KEY => Some("s".into()),
                _ => None,
            })
            .is_err()
        );
    }

    #[test]
    fn ensure_retries_5xx_signs_sigv4_accepts_200_and_409_and_fails_403() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let mut requests = Vec::new();
            for reply in ["503 Service Unavailable", "200 OK", "409 Conflict", "403 Forbidden"] {
                let (mut conn, _) = listener.accept().unwrap();
                let mut lines = Vec::new();
                for line in BufReader::new(conn.try_clone().unwrap()).lines() {
                    let line = line.unwrap();
                    if line.is_empty() {
                        break;
                    }
                    lines.push(line.to_ascii_lowercase());
                }
                requests.push(lines);
                write!(conn, "HTTP/1.1 {reply}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
            }
            requests
        });

        let cfg = base_cfg(addr);
        assert_eq!(ensure_sync(&cfg, 3, Duration::from_millis(5)), ExitCode::SUCCESS);
        assert_eq!(ensure_sync(&cfg, 2, Duration::from_millis(5)), ExitCode::SUCCESS);
        assert_eq!(ensure_sync(&cfg, 3, Duration::from_millis(5)), ExitCode::FAILURE);

        let seen = server.join().unwrap();
        assert_eq!(seen.len(), 4);
        assert_eq!(seen[0][0], "put /turbo-zoo http/1.1");
        assert!(seen[0].iter().any(|h| h == &format!("x-amz-content-sha256: {EMPTY_SHA256}")));
        assert!(seen[0].iter().any(|h| h.starts_with("authorization: aws4-hmac-sha256 credential=akiaexample/")));
    }

    #[tokio::test]
    async fn ensure_verifies_https_endpoint_against_ca_bundle() {
        let cert = CertificateDer::from_pem_slice(TEST_CERT_PEM.as_bytes()).unwrap();
        let key = PrivateKeyDer::from_pem_slice(TEST_KEY_PEM.as_bytes()).unwrap();
        let tls_cfg = ServerConfig::builder().with_no_client_auth().with_single_cert(vec![cert], key).unwrap();
        let acceptor = TlsAcceptor::from(Arc::new(tls_cfg));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut tls = acceptor.accept(tcp).await.unwrap();
            let mut req = [0u8; 1024];
            let _ = tls.read(&mut req).await;
            tls.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").await.unwrap();
            tls.shutdown().await.unwrap();
        });

        let mut cfg = base_cfg(format!("https://127.0.0.1:{port}"));
        cfg.ca_bundle = Some(TEST_CA_PEM.as_bytes().to_vec());
        let code = tokio::task::spawn_blocking(move || ensure_sync(&cfg, 1, Duration::from_millis(5))).await.unwrap();
        assert_eq!(code, ExitCode::SUCCESS);
    }
}
