//! The fetch transports: what serves the bytes. The zoo verifies whatever
//! a transport returns, so one needs no checks of its own.

use std::str::FromStr;

use crate::index::Url;

/// Fetches the bytes at an address. The zoo verifies whatever it returns,
/// so a transport needs no checks of its own; `size` is the most the zoo
/// takes from the address, so a transport can stop reading once more than
/// that arrives. For a file served compressed it is the larger of the
/// wire size and the unpacked size: a transport that decodes a
/// `Content-Encoding` hands over the file itself.
pub trait Transport {
    fn get(&self, url: &Url, size: u64) -> Result<Vec<u8>, String>;
}

/// A transport that fetches nothing: the zoo hands out only what is
/// already in its cache, still checking the licence and every digest.
pub struct Offline;

impl Transport for Offline {
    fn get(&self, _url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        Err("offline: the file is not in the zoo's cache".into())
    }
}

/// HTTPS through `ureq`. Redirects the server issues are followed, up to
/// 10, over HTTPS only: a redirect to a plain `http://` address is an
/// error. Where a redirect leads is not checked against the index (a
/// GitHub release, for one, redirects to another host). The zoo checks
/// the bytes by size, SHA-256 and SHA-512 ([`verify`]), so a file it keeps
/// holds the bytes the index records wherever a redirect led. Reads at most one byte more
/// than the index expects, enough for the size check to fail on a longer
/// body without holding all of it. Asks for the bytes unencoded
/// (`Accept-Encoding: identity`); a server that encodes them anyway may
/// have them decoded on the way, and the zoo keeps those when they verify
/// as the file ([`unpack`]).
#[cfg(feature = "http")]
pub struct Http {
    agent: ureq::Agent,
}

#[cfg(feature = "http")]
impl Http {
    /// HTTPS with the default (WebPki) root certificates.
    pub fn new() -> Http {
        Http { agent: default_agent().clone() }
    }

    /// HTTPS trusting only the PEM-encoded certificates in `bundle`: the CA
    /// of a self-hosted artifact mirror, for example. Every
    /// `-----BEGIN CERTIFICATE-----` block must parse. The public roots are
    /// not in the set, so a deployment mixing public and mirrored artifacts
    /// keeps the default transport for the public ones.
    pub fn with_root_ca_bundle_pem(bundle: &[u8]) -> Result<Http, String> {
        let mut certs = Vec::new();
        let mut rest = bundle;
        const BEGIN: &[u8] = b"-----BEGIN CERTIFICATE-----";
        const END: &[u8] = b"-----END CERTIFICATE-----";
        while !rest.is_empty() {
            let Some(b) = rest.windows(BEGIN.len()).position(|w| w == BEGIN) else { break };
            let after = &rest[b..];
            let Some(e) = after.windows(END.len()).position(|w| w == END) else { break };
            let end = e + END.len();
            certs.push(ureq::tls::Certificate::from_pem(&after[..end]).map_err(|e| e.to_string())?);
            rest = &after[end..];
        }
        if certs.is_empty() {
            return Err("no PEM certificate block in the CA bundle".into());
        }
        let tls = ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::new_with_certs(&certs)).build();
        Ok(Http { agent: config().https_only(true).tls_config(tls).build().into() })
    }
}

#[cfg(feature = "http")]
impl Default for Http {
    fn default() -> Http {
        Http::new()
    }
}

#[cfg(feature = "http")]
impl Transport for Http {
    fn get(&self, url: &Url, size: u64) -> Result<Vec<u8>, String> {
        read(&self.agent, url.as_str(), size)
    }
}

/// A transport that serves every fetch from one mirror: `<base>` joined
/// with the indexed URL's host and path, so a mirror of several origins
/// keeps them apart (`https://host/x` becomes `<base>/host/x`). The zoo's
/// digest checks still run on what the mirror serves.
pub struct Mirror<T> {
    base: String,
    inner: T,
}

impl<T> Mirror<T> {
    /// A mirror of every URL at `base` (an `https://` address), fetched
    /// through `inner`. The base must not contain a query or fragment.
    pub fn new(base: &str, inner: T) -> Result<Mirror<T>, String> {
        let base = Url::from_str(base).map_err(|e| e.to_string())?;
        if base.as_str().contains('?') || base.as_str().contains('#') {
            return Err("a mirror base must not contain a query or fragment".into());
        }
        Ok(Mirror { base: base.as_str().trim_end_matches('/').to_owned(), inner })
    }

    /// The mirror address of an indexed URL.
    pub fn rewrite(&self, url: &Url) -> Url {
        let stripped = url.as_str().strip_prefix("https://").expect("an https URL");
        Url::from_str(&format!("{}/{}", self.base, stripped)).expect("the mirror join is a valid URL")
    }
}

impl<T: Transport> Transport for Mirror<T> {
    fn get(&self, url: &Url, size: u64) -> Result<Vec<u8>, String> {
        self.inner.get(&self.rewrite(url), size)
    }
}

/// One GET through `agent`, reading at most `size` and one byte more.
#[cfg(feature = "http")]
fn read(agent: &ureq::Agent, url: &str, size: u64) -> Result<Vec<u8>, String> {
    let mut resp = agent.get(url).call().map_err(|e| e.to_string())?;
    resp.body_mut().with_config().limit(size.saturating_add(1)).read_to_vec().map_err(|e| e.to_string())
}

/// `ureq`'s defaults with at most 10 redirects followed and the bytes
/// asked for unencoded.
#[cfg(feature = "http")]
fn config() -> ureq::config::ConfigBuilder<ureq::typestate::AgentScope> {
    ureq::Agent::config_builder().max_redirects(10).accept_encoding("identity")
}

/// The default agent [`Http::new`] shares: [`config`], with every request
/// and every redirect restricted to HTTPS.
#[cfg(feature = "http")]
fn default_agent() -> &'static ureq::Agent {
    static AGENT: std::sync::OnceLock<ureq::Agent> = std::sync::OnceLock::new();
    AGENT.get_or_init(|| config().https_only(true).build().into())
}

#[cfg(all(test, feature = "http"))]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    use std::str::FromStr;

    use super::{Http, Mirror, Transport, config, default_agent, read};
    use crate::fetch::Zoo;
    use crate::{Acceptances, Index, Policy, Sha256, Sha512, Url, UserId};

    const HELLO: &[u8] = b"hello hello hello\n";

    /// HELLO as Python's `gzip.compress(..., mtime=0)` writes it.
    const HELLO_GZ: [u8; 29] = [
        0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40,
        0x90, 0x5c, 0x00, 0x3b, 0x7c, 0x8a, 0xdf, 0x12, 0x00, 0x00, 0x00,
    ];

    /// The production agent's settings over plain HTTP to a local server,
    /// whatever address the index gives.
    struct Local(ureq::Agent, String);

    impl Transport for Local {
        fn get(&self, _url: &Url, size: u64) -> Result<Vec<u8>, String> {
            read(&self.0, &self.1, size)
        }
    }

    #[test]
    fn a_file_a_server_gzip_encodes_anyway_is_kept_whether_or_not_the_transport_decodes_it() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = format!("http://{}/hello.txt.gz", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let mut head = Vec::new();
            for line in BufReader::new(conn.try_clone().unwrap()).lines() {
                let line = line.unwrap();
                if line.is_empty() {
                    break;
                }
                head.push(line.to_ascii_lowercase());
            }
            let reply = format!(
                "HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                HELLO_GZ.len()
            );
            conn.write_all(reply.as_bytes()).unwrap();
            conn.write_all(&HELLO_GZ).unwrap();
            head
        });
        let json = format!(
            r#"{{"zoo_version": 1, "artifacts": [{{"id": "hello", "kind": "KIND_LEXICON", "format": "FORMAT_TEXT",
              "version": "1", "license": "Apache-2.0", "license_terms": "https://example.org/LICENSE",
              "trained_on": [], "class": "CLASS_PERMISSIVE", "attribution": "Example attribution.",
              "files": [{{"path": "hello.txt", "url": "https://example.org/hello.txt.gz", "size": {}, "sha256": "{}",
                "sha512": "{}", "download": {{"compression": "gzip", "download_bytes": {}, "download_sha256": "{}"}}}}]}}]}}"#,
            HELLO.len(),
            Sha256::of(HELLO),
            Sha512::of(HELLO),
            HELLO_GZ.len(),
            Sha256::of(&HELLO_GZ),
        );
        let dir = std::env::temp_dir().join(format!("turbo-zoo-encoded-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let local = Local(config().build().into(), addr);
        let zoo = Zoo::new(Index::parse(json.as_bytes()).unwrap(), dir.clone(), Policy::default(), &local);
        let user: UserId = "alice".parse().unwrap();
        let got = zoo.fetch("hello", &user, &Acceptances::default()).map(|f| std::fs::read(&f.files[0]).unwrap());
        let head = server.join().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got.unwrap(), HELLO);
        assert!(head.contains(&"accept-encoding: identity".to_string()), "{head:?}");
    }

    #[test]
    fn the_http_agent_is_https_only_and_follows_at_most_10_redirects() {
        assert!(default_agent().config().https_only());
        assert_eq!(default_agent().config().max_redirects(), 10);
        assert!(format!("{:?}", default_agent().config().accept_encoding()).contains("identity"));
    }

    #[test]
    fn the_http_agent_refuses_a_plain_http_address_before_connecting() {
        // The check runs before any connection is made, so the test needs
        // no network.
        match default_agent().get("http://127.0.0.1:9/x").call() {
            Err(ureq::Error::RequireHttpsOnly(_)) => {}
            other => panic!("expected the https-only refusal, got {other:?}"),
        }
    }

    /// A small self-signed test CA, the shape a private mirror's bundle has.
    const TEST_CA_PEM: &[u8] = b"-----BEGIN CERTIFICATE-----\n\
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

    #[test]
    fn root_ca_bundle_parses_certificates_and_refuses_garbage() {
        assert!(Http::with_root_ca_bundle_pem(TEST_CA_PEM).is_ok());
        assert!(Http::with_root_ca_bundle_pem(b"no certificates here").is_err());
        assert!(Http::with_root_ca_bundle_pem(b"").is_err());
        // A bundle of the same CA twice parses both blocks.
        let mut two = TEST_CA_PEM.to_vec();
        two.extend_from_slice(TEST_CA_PEM);
        assert!(Http::with_root_ca_bundle_pem(&two).is_ok());
        // A broken block inside a bundle is an error, not a partial read.
        let mut broken = TEST_CA_PEM.to_vec();
        broken.extend_from_slice(b"-----BEGIN CERTIFICATE-----\n%!#*\n-----END CERTIFICATE-----\n");
        assert!(Http::with_root_ca_bundle_pem(&broken).is_err());
    }

    struct Record(std::sync::Mutex<Vec<String>>);

    impl Transport for Record {
        fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
            self.0.lock().unwrap().push(url.as_str().to_owned());
            Ok(b"bytes".to_vec())
        }
    }

    #[test]
    fn the_mirror_rewrites_urls_under_one_base() {
        let rec = Record(std::sync::Mutex::new(Vec::new()));
        let mirror = Mirror::new("https://mirror.internal/zoo/", rec).unwrap();
        let url = Url::from_str("https://huggingface.co/models/x/weights").unwrap();
        mirror.get(&url, 16).unwrap();
        let seen = mirror.inner.0.lock().unwrap();
        assert_eq!(seen.as_slice(), ["https://mirror.internal/zoo/huggingface.co/models/x/weights"]);
        assert!(Mirror::new("http://mirror.internal", Record(std::sync::Mutex::new(Vec::new()))).is_err());
        assert!(Mirror::new("https://mirror.internal/zoo?tenant=x", super::Offline).is_err());
        assert!(Mirror::new("https://mirror.internal/zoo?", super::Offline).is_err());
        assert!(Mirror::new("https://mirror.internal/zoo#tenant=x", super::Offline).is_err());
        assert!(Mirror::new("https://mirror.internal/zoo#", super::Offline).is_err());
        assert!(Mirror::new("https://mirror.internal/zoo%3Ftenant=x%23part", super::Offline).is_ok());
    }
}
