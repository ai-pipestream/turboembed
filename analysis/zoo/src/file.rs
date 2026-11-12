//! One file of an artifact, as the index lists it.

use serde::{Deserialize, Serialize};

use super::{RelPath, Url};
use crate::digest::{Sha256, Sha512};
use crate::licence::LicenceClass;
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
    /// Further locations of the same bytes, tried in order after `url`
    /// when it fails or serves other bytes. Only a `CLASS_PERMISSIVE`
    /// artifact's file may list any.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub also: Vec<Url>,
    pub size: u64,
    pub sha256: Sha256,
    /// The SHA-512 digest of the artifact bytes.
    pub sha512: Sha512,
    /// For a file served compressed: the compression and the compressed
    /// bytes' own size and SHA-256. `size` and the digests above are the
    /// unpacked file's, which is what the cache holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub download: Option<Download>,
}

impl File {
    /// Every address the file is fetched from, in the order they are
    /// tried: `url`, then each of `also`.
    pub fn locations(&self) -> impl Iterator<Item = &Url> + '_ {
        std::iter::once(&self.url).chain(&self.also)
    }

    /// Why `also` cannot stand on an artifact of `class`: a further
    /// location on anything but a permissive artifact (one fetched after
    /// acceptance, or only under its origin, keeps its one address), or
    /// an address listed twice.
    pub(super) fn check_also(&self, class: LicenceClass) -> Result<(), String> {
        if self.also.is_empty() {
            return Ok(());
        }
        if class != LicenceClass::Permissive {
            return Err(format!(
                "{} lists a further location, which is only for {}; a {class} file has one address",
                self.path,
                LicenceClass::Permissive
            ));
        }
        for (i, u) in self.also.iter().enumerate() {
            if *u == self.url || self.also[..i].contains(u) {
                return Err(format!("{} lists {u} twice among its locations", self.path));
            }
        }
        Ok(())
    }
}
