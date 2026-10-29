//! Verified TLS for the upstream gRPC server, separate from page HTTPS.
use std::path::{Path, PathBuf};
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint, Identity};

/// A client certificate chain and its private key, both PEM files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientIdentity {
    pub cert: PathBuf,
    pub key: PathBuf,
}

/// The trust and identity used for upstream gRPC. No plaintext fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamTls {
    pub ca: PathBuf,
    pub identity: Option<ClientIdentity>,
    /// Expected certificate name when connecting to a different address.
    /// This changes the verified name, not the trust checks.
    pub server_name: Option<String>,
}

impl UpstreamTls {
    pub fn from_parts(
        ca: Option<String>,
        cert: Option<String>,
        key: Option<String>,
        server_name: Option<String>,
    ) -> Result<Option<Self>, String> {
        if [&ca, &cert, &key, &server_name].iter().any(|v| v.as_ref().is_some_and(|s| s.trim().is_empty())) {
            return Err("upstream TLS values must not be empty".into());
        }
        let identity = match (cert, key) {
            (None, None) => None,
            (Some(cert), Some(key)) => Some(ClientIdentity { cert: cert.into(), key: key.into() }),
            _ => return Err("--server-tls-cert and --server-tls-key go together".into()),
        };
        match ca {
            Some(ca) => Ok(Some(Self { ca: ca.into(), identity, server_name })),
            None if identity.is_none() && server_name.is_none() => Ok(None),
            None => Err("upstream TLS needs --server-tls-ca".into()),
        }
    }

    fn config(&self) -> Result<ClientTlsConfig, String> {
        let mut tls = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(read(&self.ca)?));
        if let Some(name) = &self.server_name {
            if name.trim().is_empty() {
                return Err("upstream TLS server name must not be empty".into());
            }
            tls = tls.domain_name(name);
        }
        if let Some(identity) = &self.identity {
            tls = tls.identity(Identity::from_pem(read(&identity.cert)?, read(&identity.key)?));
        }
        Ok(tls)
    }
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("upstream TLS {}: {e}", path.display()))
}

pub(crate) fn endpoint(server: &str, tls: Option<&UpstreamTls>) -> Result<Endpoint, String> {
    let endpoint = Endpoint::from_shared(server.to_owned()).map_err(|e| format!("server {server}: {e}"))?;
    if endpoint.uri().authority().is_none() {
        return Err(format!("server {server}: must be an http:// or https:// URL"));
    }
    match (endpoint.uri().scheme_str(), tls) {
        (Some("http"), None) => Ok(endpoint),
        (Some("https"), Some(tls)) => endpoint.tls_config(tls.config()?).map_err(|e| format!("upstream TLS: {e}")),
        (Some("http"), Some(_)) => Err("upstream TLS requires an https:// server URL".into()),
        (Some("https"), None) => Err("https:// server requires --server-tls-ca".into()),
        _ => Err(format!("server {server}: must be an http:// or https:// URL")),
    }
}
