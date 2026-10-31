//! `TURBO_ZOO_SOURCE`: the places a model loads from, in the order tried.

use turbo_zoo::{Blobs, Sources, UserId};

use crate::server::ZooServer;

/// The environment variable [`sources`] reads its list from.
pub const ENV_SOURCE: &str = "TURBO_ZOO_SOURCE";

/// The sources `spec` names, separated by commas, in the order they are
/// tried:
///
/// - `file:///dir`: the content-addressed store under `/dir`. The first
///   one named also keeps every file a later source serves, once it
///   verifies, so the next read finds it there;
/// - `grpc://host:port`, `grpcs://host:port`: a zoo server, read as `user`;
/// - `s3://bucket/prefix`: an S3 bucket in the same layout, configured
///   from the `AWS_*` environment ([`crate::S3Bucket::new`]); a build
///   without the `s3` feature refuses it;
/// - `gdrive://<folder-id>[/name/...]`: a Google Drive folder in the
///   same layout, signed in from `GOOGLE_OAUTH_ACCESS_TOKEN` or
///   `GOOGLE_API_KEY` ([`crate::GoogleDrive::new`]); a build without the
///   `gdrive` feature refuses it;
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
    let mut kept = false;
    for item in spec.split(',').map(str::trim) {
        if item.is_empty() {
            return Err(format!("{ENV_SOURCE}={spec}: an empty entry"));
        }
        out = if let Some(dir) = item.strip_prefix("file://") {
            if !dir.starts_with('/') {
                return Err(format!("{item}: a file:// source names an absolute directory"));
            }
            if kept {
                out.then(item, Blobs::new(dir))
            } else {
                kept = true;
                out.then_keep(item, Blobs::new(dir))
            }
        } else if item.starts_with("s3://") {
            out.then(item, s3(item)?)
        } else if item.starts_with("gdrive://") {
            out.then(item, gdrive(item)?)
        } else if item.starts_with("grpc://") || item.starts_with("grpcs://") {
            out.then(item, ZooServer::new(item, user.clone())?)
        } else if item == "origin" {
            let t = origin.take().ok_or_else(|| format!("{ENV_SOURCE}={spec}: origin is named twice"))?;
            out.then(item, t)
        } else {
            return Err(format!("{item}: a source is file://, grpc://, grpcs://, s3://, gdrive:// or origin"));
        };
    }
    Ok(out)
}

#[cfg(feature = "s3")]
fn s3(item: &str) -> Result<crate::S3Bucket, String> {
    crate::S3Bucket::new(item)
}

#[cfg(not(feature = "s3"))]
fn s3(item: &str) -> Result<turbo_zoo::Offline, String> {
    Err(format!("{item}: s3:// needs the s3 feature, which this build leaves out"))
}

#[cfg(feature = "gdrive")]
fn gdrive(item: &str) -> Result<crate::GoogleDrive, String> {
    crate::GoogleDrive::new(item)
}

#[cfg(not(feature = "gdrive"))]
fn gdrive(item: &str) -> Result<turbo_zoo::Offline, String> {
    Err(format!("{item}: gdrive:// needs the gdrive feature, which this build leaves out"))
}
