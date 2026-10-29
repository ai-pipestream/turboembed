//! `TURBO_ZOO_SOURCE`: the places a model loads from, in the order tried.

use turbo_zoo::{Blobs, Sources, UserId};

use crate::server::ZooServer;

/// The environment variable [`sources`] reads its list from.
pub const ENV_SOURCE: &str = "TURBO_ZOO_SOURCE";

/// The sources `spec` names, separated by commas, in the order they are
/// tried:
///
/// - `file:///dir`: the content-addressed store under `/dir`;
/// - `grpc://host:port`, `grpcs://host:port`: a zoo server, read as `user`;
/// - `origin`: the index's own addresses, through `origin`, the transport
///   the caller fetches them with (HTTPS, for one). Without it in the list
///   nothing is fetched from an index address.
///
/// An empty spec, an unknown scheme, or `origin` named twice is refused.
pub fn sources<T>(spec: &str, user: &UserId, origin: T) -> Result<Sources, String>
where
    T: turbo_zoo::Transport + Send + Sync + 'static,
{
    let mut out = Sources::new();
    let mut origin = Some(origin);
    for item in spec.split(',').map(str::trim) {
        if item.is_empty() {
            return Err(format!("{ENV_SOURCE}={spec}: an empty entry"));
        }
        out = if let Some(dir) = item.strip_prefix("file://") {
            if !dir.starts_with('/') {
                return Err(format!("{item}: a file:// source names an absolute directory"));
            }
            out.then(item, Blobs::new(dir))
        } else if item.starts_with("grpc://") || item.starts_with("grpcs://") {
            out.then(item, ZooServer::new(item, user.clone())?)
        } else if item == "origin" {
            let t = origin.take().ok_or_else(|| format!("{ENV_SOURCE}={spec}: origin is named twice"))?;
            out.then(item, t)
        } else {
            return Err(format!("{item}: a source is file://, grpc://, grpcs:// or origin"));
        };
    }
    Ok(out)
}
