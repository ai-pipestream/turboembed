//! A zoo server as a source: each file read with `ReadModelFile`.

use std::time::Duration;

use tokio::runtime::{Builder, Runtime};
use tonic::transport::{Channel, Endpoint};
use turbo_zoo::{Artifact, File, Transport, Url, UserId, most_bytes};

use crate::proto::ReadModelFileRequest;
use crate::proto::model_file_chunk::Frame;
use crate::proto::turbo_catalog_service_client::TurboCatalogServiceClient;

/// A turboembed zoo server, read for one user. Its calls block on a
/// runtime of its own, so a caller already inside an async runtime makes
/// them from a blocking task (`tokio::task::spawn_blocking`), as it would
/// any other blocking fetch.
pub struct ZooServer {
    address: String,
    user: UserId,
    runtime: Runtime,
    channel: Channel,
}

impl ZooServer {
    /// The server at `address`, `grpc://host:port` in plain text or
    /// `grpcs://host:port` over TLS with the public web roots, reading as
    /// `user`. Nothing is connected until the first read.
    pub fn new(address: &str, user: UserId) -> Result<ZooServer, String> {
        ZooServer::with_tls(address, user, None)
    }

    /// [`ZooServer::new`], trusting only the PEM-encoded CA `ca_pem` for
    /// `grpcs://`: a zoo served with a private CA.
    pub fn with_tls(address: &str, user: UserId, ca_pem: Option<&[u8]>) -> Result<ZooServer, String> {
        let (http, tls) = if let Some(rest) = address.strip_prefix("grpcs://") {
            (format!("https://{rest}"), true)
        } else if let Some(rest) = address.strip_prefix("grpc://") {
            if ca_pem.is_some() {
                return Err(format!("{address}: a CA is for grpcs:// only"));
            }
            (format!("http://{rest}"), false)
        } else {
            return Err(format!("{address}: a zoo server address starts grpc:// or grpcs://"));
        };
        let host = http.split_once("://").map_or("", |(_, h)| h);
        if host.is_empty() || host.contains(['/', '?', '#', '@']) {
            return Err(format!("{address}: give the host and port alone"));
        }
        let mut endpoint = Endpoint::from_shared(http).map_err(|e| format!("{address}: {e}"))?;
        endpoint = endpoint.connect_timeout(Duration::from_secs(10));
        if tls {
            endpoint = with_tls(endpoint, ca_pem).map_err(|e| format!("{address}: {e}"))?;
        }
        let runtime = Builder::new_current_thread().enable_all().build().map_err(|e| format!("{address}: {e}"))?;
        let channel = {
            let _entered = runtime.enter();
            endpoint.connect_lazy()
        };
        Ok(ZooServer { address: address.to_owned(), user, runtime, channel })
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    /// The file's bytes as the server streams them: its record first,
    /// which must pin the same SHA-256 and size, then at most `most`
    /// bytes and one more, so the zoo's size check fails on a longer
    /// file without holding all of it.
    async fn read(&self, artifact: &Artifact, file: &File, most: u64) -> Result<Vec<u8>, String> {
        let mut client = TurboCatalogServiceClient::new(self.channel.clone()).max_decoding_message_size(1 << 20);
        let req = ReadModelFileRequest {
            user: self.user.to_string(),
            artifact_id: artifact.id.to_string(),
            path: file.path.as_str().to_owned(),
            sha256: file.sha256.to_string(),
        };
        let what = |m: &dyn std::fmt::Display| format!("{}: {}/{}: {m}", self.address, artifact.id, file.path.as_str());
        let mut stream = client.read_model_file(req).await.map_err(|s| what(&s.message()))?.into_inner();
        let Some(Frame::File(head)) = stream.message().await.map_err(|s| what(&s.message()))?.and_then(|m| m.frame)
        else {
            return Err(what(&"the server sent no file record first"));
        };
        if head.sha256 != file.sha256.to_string() || head.size != file.size {
            return Err(what(&format!("the server's record is {} bytes at {}", head.size, head.sha256)));
        }
        let mut out = Vec::with_capacity(usize::try_from(file.size).unwrap_or(0).min(1 << 30));
        while let Some(m) = stream.message().await.map_err(|s| what(&s.message()))? {
            let Some(Frame::Data(d)) = m.frame else { return Err(what(&"a second file record")) };
            out.extend_from_slice(&d);
            if out.len() as u64 > most {
                break;
            }
        }
        Ok(out)
    }
}

/// `endpoint` over TLS, trusting `ca_pem` alone when given, else the public
/// web roots.
#[cfg(feature = "tls")]
fn with_tls(endpoint: Endpoint, ca_pem: Option<&[u8]>) -> Result<Endpoint, String> {
    use tonic::transport::{Certificate, ClientTlsConfig};
    let config = match ca_pem {
        Some(pem) => ClientTlsConfig::new().ca_certificate(Certificate::from_pem(pem)),
        None => ClientTlsConfig::new().with_webpki_roots(),
    };
    endpoint.tls_config(config).map_err(|e| e.to_string())
}

#[cfg(not(feature = "tls"))]
fn with_tls(_endpoint: Endpoint, _ca_pem: Option<&[u8]>) -> Result<Endpoint, String> {
    Err("grpcs:// needs the tls feature, which this build leaves out".into())
}

impl Transport for ZooServer {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        Err(format!("{url}: a zoo server is read by artifact and path, not by address"))
    }

    fn get_file(&self, artifact: &Artifact, file: &File) -> Result<Vec<u8>, String> {
        // The server holds the file unpacked, so the file's own size bounds it.
        let most = most_bytes(file);
        self.runtime.block_on(self.read(artifact, file, most))
    }
}
