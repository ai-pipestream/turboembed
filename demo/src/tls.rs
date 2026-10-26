//! TLS for the demo's listener: a certificate chain and key in PEM, off
//! unless both are given. The handshake runs off the accept loop, so a
//! client that never finishes one holds only its own connection, and only
//! for `HANDSHAKE`.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_rustls::TlsAcceptor;
use tokio_rustls::rustls::ServerConfig;
use tokio_rustls::rustls::crypto::ring;
use tokio_rustls::rustls::pki_types::pem::PemObject;
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer};
use tokio_rustls::server::TlsStream;

const HANDSHAKE: Duration = Duration::from_secs(10);

/// The listener's certificate chain and private key, each a PEM file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tls {
    pub cert: PathBuf,
    pub key: PathBuf,
}

/// The server side of TLS 1.2 and 1.3 with `tls`'s certificate, speaking
/// HTTP/1.1.
pub(crate) fn config(tls: &Tls) -> Result<Arc<ServerConfig>, String> {
    let file = |p: &Path, e: &dyn std::fmt::Display| format!("{}: {e}", p.display());
    let certs = CertificateDer::pem_file_iter(&tls.cert)
        .and_then(|i| i.collect::<Result<Vec<_>, _>>())
        .map_err(|e| file(&tls.cert, &e))?;
    if certs.is_empty() {
        return Err(file(&tls.cert, &"no certificate"));
    }
    let key = PrivateKeyDer::from_pem_file(&tls.key).map_err(|e| file(&tls.key, &e))?;
    let mut config = ServerConfig::builder_with_provider(Arc::new(ring::default_provider()))
        .with_safe_default_protocol_versions()
        .map_err(|e| e.to_string())?
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| format!("{} and {}: {e}", tls.cert.display(), tls.key.display()))?;
    config.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(Arc::new(config))
}

/// Connections that finished their handshake, for `axum::serve`.
pub(crate) struct Listener {
    ready: mpsc::Receiver<(TlsStream<TcpStream>, SocketAddr)>,
    addr: SocketAddr,
    accepting: JoinHandle<()>,
}

impl Listener {
    pub(crate) fn new(tcp: TcpListener, config: Arc<ServerConfig>) -> std::io::Result<Listener> {
        let addr = tcp.local_addr()?;
        let acceptor = TlsAcceptor::from(config);
        let (tx, ready) = mpsc::channel(64);
        let accepting = tokio::spawn(async move {
            loop {
                let (conn, peer) = match tcp.accept().await {
                    Ok(c) => c,
                    Err(_) => {
                        // Out of descriptors and the like: wait, then go on.
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        continue;
                    }
                };
                let (acceptor, tx) = (acceptor.clone(), tx.clone());
                tokio::spawn(async move {
                    if let Ok(Ok(conn)) = tokio::time::timeout(HANDSHAKE, acceptor.accept(conn)).await {
                        let _ = tx.send((conn, peer)).await;
                    }
                });
            }
        });
        Ok(Listener { ready, addr, accepting })
    }
}

impl axum::serve::Listener for Listener {
    type Io = TlsStream<TcpStream>;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        match self.ready.recv().await {
            Some(c) => c,
            None => std::future::pending().await,
        }
    }

    fn local_addr(&self) -> std::io::Result<SocketAddr> {
        Ok(self.addr)
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        self.accepting.abort();
    }
}
