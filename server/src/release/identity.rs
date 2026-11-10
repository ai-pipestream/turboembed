//! The calling service's identity: the verified client certificate of the
//! connection, never anything the request says about itself (#1909).
//! Metadata, headers and request fields confer nothing; a plaintext or
//! server-only TLS connection has no identity.

use super::caps::Fingerprint;

/// The SHA-256 of the DER bytes of the leaf certificate the TLS handshake
/// verified against `--tls-client-ca`, or `None` when the connection
/// presented none.
#[cfg(feature = "tls")]
pub fn of<T>(req: &tonic::Request<T>) -> Option<Fingerprint> {
    let certs = req.peer_certs()?;
    let leaf = certs.first()?;
    Some(Fingerprint(turbo_sha256::sha256(leaf.as_ref()).0))
}

/// Without the `tls` feature no connection has an identity.
#[cfg(not(feature = "tls"))]
pub fn of<T>(_: &tonic::Request<T>) -> Option<Fingerprint> {
    None
}
