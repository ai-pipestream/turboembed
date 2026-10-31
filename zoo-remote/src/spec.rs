//! `TURBO_ZOO_SOURCE`: the places a model loads from, in the order tried.

use turbo_zoo::{Holder, SignIn, Sources, UserId};

use crate::registry::{List, Opened, Registry};

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
/// Any other scheme [`Registry::builtin`] knows opens too. An empty
/// entry, an unknown scheme, or `origin` named twice is refused.
pub fn sources<T>(spec: &str, user: &UserId, origin: T) -> Result<Sources, String>
where
    T: turbo_zoo::Transport + Send + Sync + 'static,
{
    Registry::builtin().spec(spec, user, origin)
}

impl Registry {
    /// [`sources`] through this registry's kinds: each entry is a holder
    /// signed in from the environment, or `origin`.
    pub fn spec<T>(&self, spec: &str, user: &UserId, origin: T) -> Result<Sources, String>
    where
        T: turbo_zoo::Transport + Send + Sync + 'static,
    {
        let mut list = List::default();
        let mut origin = Some(origin);
        for item in spec.split(',').map(str::trim) {
            if item.is_empty() {
                return Err(format!("{ENV_SOURCE}={spec}: an empty entry"));
            }
            if item == "origin" {
                let t = origin.take().ok_or_else(|| format!("{ENV_SOURCE}={spec}: origin is named twice"))?;
                list.push(item, Opened::Source(Box::new(t)));
                continue;
            }
            let holder = Holder { name: item.to_owned(), address: item.to_owned(), sign_in: SignIn::Environment };
            list.push(item, self.open(&holder, user)?);
        }
        Ok(list.out)
    }
}
