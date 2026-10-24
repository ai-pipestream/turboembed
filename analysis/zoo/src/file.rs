//! One file of an artifact, as the index lists it.

use serde::{Deserialize, Serialize};

use super::{RelPath, Url};
use crate::digest::{Sha256, Sha512};
use crate::transfer::Download;

/// One file of an artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct File {
    /// Where the file goes in the artifact's directory.
    pub path: RelPath,
    /// Where it is fetched from. The address must serve the same bytes for
    /// as long as the entry exists: a release archive, or a repository
    /// path at a commit.
    pub url: Url,
    pub size: u64,
    pub sha256: Sha256,
    /// The digest OpenNLP's `DownloadUtil` checks.
    pub sha512: Sha512,
    /// For a file served compressed: the compression and the compressed
    /// bytes' own size and SHA-256. `size` and the digests above are the
    /// unpacked file's, which is what the cache holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub download: Option<Download>,
}
