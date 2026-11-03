//! Optional TLS and mutual-TLS (mTLS) transport setup for `turbo-kserve`
//! (`--tls-cert`, `--tls-key`, `--tls-client-ca`, issue `#1300`).

use std::path::PathBuf;

/// Server TLS and optional mutual-TLS (mTLS) settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsConfig {
    /// Path to the PEM-encoded server certificate chain (`--tls-cert`).
    pub cert: PathBuf,
    /// Path to the PEM-encoded server private key (`--tls-key`).
    pub key: PathBuf,
    /// Optional path to the PEM-encoded client CA bundle for mTLS (`--tls-client-ca`).
    pub client_ca: Option<PathBuf>,
}

/// Applies `tls` to `builder` when configured, reading the PEM files and
/// validating the identity and optional client CA root before the server starts
/// listening.
#[cfg(feature = "tls")]
pub(crate) fn configure_server(
    builder: tonic::transport::Server,
    tls: Option<&TlsConfig>,
) -> Result<tonic::transport::Server, String> {
    let Some(tls) = tls else {
        return Ok(builder);
    };
    let cert_pem = std::fs::read(&tls.cert).map_err(|e| format!("--tls-cert {}: {e}", tls.cert.display()))?;
    let key_pem = std::fs::read(&tls.key).map_err(|e| format!("--tls-key {}: {e}", tls.key.display()))?;
    let identity = tonic::transport::Identity::from_pem(cert_pem, key_pem);
    let mut cfg = tonic::transport::ServerTlsConfig::new().identity(identity);
    if let Some(ca_path) = &tls.client_ca {
        let ca_pem = std::fs::read(ca_path).map_err(|e| format!("--tls-client-ca {}: {e}", ca_path.display()))?;
        cfg = cfg.client_ca_root(tonic::transport::Certificate::from_pem(ca_pem));
    }
    builder.tls_config(cfg).map_err(|e| format!("TLS configuration: {e}"))
}

/// Refuses `--tls-cert` when `turbo-kserve` is built without the `tls` feature.
#[cfg(not(feature = "tls"))]
pub(crate) fn configure_server(
    builder: tonic::transport::Server,
    tls: Option<&TlsConfig>,
) -> Result<tonic::transport::Server, String> {
    if tls.is_some() {
        return Err("this turbo-kserve was built without the tls feature, so it cannot serve TLS".into());
    }
    Ok(builder)
}
