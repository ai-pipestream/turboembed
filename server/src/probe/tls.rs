//! Client TLS for health checks. Listener TLS settings are not client credentials.
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientIdentity {
    pub cert: PathBuf,
    pub key: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeTls {
    pub ca: PathBuf,
    pub identity: Option<ClientIdentity>,
    pub server_name: Option<String>,
}

#[derive(Default)]
pub(super) struct Settings {
    ca: Option<String>,
    cert: Option<String>,
    key: Option<String>,
    name: Option<String>,
}

impl Settings {
    pub fn flag(&mut self, flag: &str, value: String) -> Result<(), String> {
        let slot = match flag {
            "--tls-ca" => &mut self.ca,
            "--tls-cert" => &mut self.cert,
            "--tls-key" => &mut self.key,
            "--tls-name" => &mut self.name,
            _ => return Err(format!("unknown argument {flag}")),
        };
        if value.trim().is_empty() {
            return Err(format!("{flag} needs a value"));
        }
        if slot.replace(value).is_some() {
            return Err(format!("{flag} given twice"));
        }
        Ok(())
    }

    pub fn finish(self, env: impl Fn(&str) -> Option<String>) -> Result<Option<ProbeTls>, String> {
        let get = |value: Option<String>, key| value.or_else(|| env(key).filter(|v| !v.is_empty()));
        let ca = get(self.ca, "TURBO_KSERVE_HEALTH_TLS_CA");
        let cert = get(self.cert, "TURBO_KSERVE_HEALTH_TLS_CERT");
        let key = get(self.key, "TURBO_KSERVE_HEALTH_TLS_KEY");
        let server_name = get(self.name, "TURBO_KSERVE_HEALTH_TLS_NAME");
        if [&ca, &cert, &key, &server_name].iter().any(|v| v.as_ref().is_some_and(|s| s.trim().is_empty())) {
            return Err("health TLS values must not be blank".into());
        }
        let identity = match (cert, key) {
            (None, None) => None,
            (Some(cert), Some(key)) => Some(ClientIdentity { cert: cert.into(), key: key.into() }),
            _ => return Err("--tls-cert and --tls-key go together".into()),
        };
        match ca {
            None if identity.is_none() && server_name.is_none() => Ok(None),
            None => Err("health TLS needs --tls-ca".into()),
            Some(ca) => Ok(Some(ProbeTls { ca: ca.into(), identity, server_name })),
        }
    }
}

#[cfg(feature = "tls")]
pub(super) fn configure(
    endpoint: tonic::transport::Endpoint,
    tls: &ProbeTls,
) -> Result<tonic::transport::Endpoint, String> {
    use tonic::transport::{Certificate, ClientTlsConfig, Identity};
    let read = |p: &PathBuf| std::fs::read(p).map_err(|e| format!("health TLS {}: {e}", p.display()));
    let mut config = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(read(&tls.ca)?));
    if let Some(name) = &tls.server_name {
        if name.trim().is_empty() {
            return Err("health TLS server name must not be empty".into());
        }
        config = config.domain_name(name);
    }
    if let Some(identity) = &tls.identity {
        config = config.identity(Identity::from_pem(read(&identity.cert)?, read(&identity.key)?));
    }
    endpoint.tls_config(config).map_err(|e| format!("health TLS: {e}"))
}

#[cfg(not(feature = "tls"))]
pub(super) fn configure(_: tonic::transport::Endpoint, _: &ProbeTls) -> Result<tonic::transport::Endpoint, String> {
    Err("health TLS requires the tls feature".into())
}
