//! A Hugging Face model repository at a pinned commit, read as an index
//! entry: the repository's own listing at that commit names the files
//! and their sizes, every file is fetched at the commit and held to it,
//! and the entry pins what was read.

use std::str::FromStr;

use super::{Described, described, manifest_record};
use crate::digest::{Sha256, Sha512};
use crate::error::{Check, ZooError};
use crate::fetch::Transport;
use crate::index::{Artifact, ArtifactId, File, Format, Kind, MANIFEST, ModelType, RelPath, Url};
use crate::licence::Licence;

/// Hugging Face's own address.
pub const HUGGING_FACE: &str = "https://huggingface.co";

/// The most the repository's listing may be.
const LISTING_BYTES: u64 = 8 << 20;

/// The most entries one page of the listing holds; a listing this long
/// may have been cut, so it is refused.
const LISTING_PAGE: usize = 1000;

/// A repository, a commit, and what an entry holds that neither says.
#[derive(Debug, Clone)]
pub struct RepoImport {
    pub id: ArtifactId,
    /// The model repository, `<owner>/<name>`.
    pub repo: String,
    /// The commit, as 40 lowercase hex digits. A branch or tag is refused:
    /// it may move.
    pub commit: String,
    /// The files to list, by their path in the repository, in this order;
    /// empty for every file of the commit, in the listing's order.
    pub files: Vec<String>,
    /// With `bundle`, the repository holds a sealed bundle directory and
    /// the entry is a `TYPE_ENCODER_BUNDLE` one, its `kind`, `format`,
    /// `version` and bundle record taken from the manifest. Otherwise the
    /// entry is an analysis artifact of this `kind` and `format`.
    pub bundle: bool,
    pub kind: Kind,
    pub format: Format,
    /// The entry's `version`; the commit when none is given.
    pub version: Option<String>,
    /// The artifact's own licence, as its publisher states it at
    /// `described.license_terms`.
    pub license: Licence,
    pub described: Described,
}

/// One file of the listing: its path, its size, and for a file kept in
/// large file storage the SHA-256 the repository records for it.
struct Listed {
    path: RelPath,
    size: u64,
    sha256: Option<Sha256>,
}

fn is_commit(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

fn is_repo(s: &str) -> bool {
    let part = |p: &str| {
        !p.is_empty()
            && !p.starts_with('.')
            && !p.contains("..")
            && p.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
    };
    matches!(s.split_once('/'), Some((owner, name)) if part(owner) && part(name))
}

/// A path the importer can name in an address as it is: no character
/// that an address reads as anything but part of the path.
fn plain_path(p: &str) -> bool {
    p.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.' | b'/' | b'+' | b'=' | b'@' | b','))
}

/// The files of the listing `bytes` hold, directories left out.
fn listing(fail: &dyn Fn(String) -> ZooError, bytes: &[u8]) -> Result<Vec<Listed>, ZooError> {
    let v: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| fail(format!("the repository listing is not JSON: {e}")))?;
    let entries = v.as_array().ok_or_else(|| fail("the repository listing is not a list".into()))?;
    if entries.len() >= LISTING_PAGE {
        return Err(fail(format!("the repository lists {LISTING_PAGE} entries or more, which may not be all of them")));
    }
    let mut out = Vec::new();
    for e in entries {
        let text = |name: &str| e.get(name).and_then(|s| s.as_str()).unwrap_or_default();
        if text("type") != "file" {
            continue;
        }
        let path = text("path");
        let bad = |what: &str| fail(format!("the repository listing gives {path:?} {what}"));
        let lfs = e.get("lfs").filter(|l| !l.is_null());
        let size = lfs.and_then(|l| l.get("size")).or_else(|| e.get("size")).and_then(|s| s.as_u64());
        let sha256 = match lfs {
            Some(l) => Some(
                l.get("oid")
                    .and_then(|o| o.as_str())
                    .and_then(|o| Sha256::from_str(o).ok())
                    .ok_or_else(|| bad("no SHA-256"))?,
            ),
            None => None,
        };
        let rel = RelPath::from_str(path).ok().filter(|_| plain_path(path));
        out.push(Listed {
            path: rel.ok_or_else(|| bad("a path the zoo does not take"))?,
            size: size.ok_or_else(|| bad("no size"))?,
            sha256,
        });
    }
    Ok(out)
}

/// The entry of the repository `spec.repo` at `spec.commit`, read through
/// `transport` from `endpoint` ([`HUGGING_FACE`]). Each file's address is
/// `<endpoint>/<repo>/resolve/<commit>/<path>`. A file is refused when it
/// is not the size the listing gives, or, for a file in large file
/// storage, not the SHA-256 the repository records; the entry then pins
/// the size, SHA-256 and SHA-512 of the bytes read. A revision that is
/// not a commit, a file the commit does not hold, and an entry the index
/// would not load are refused too.
pub fn hugging_face(transport: &dyn Transport, endpoint: &str, spec: RepoImport) -> Result<Artifact, ZooError> {
    let id = spec.id.clone();
    let fail = |message: String| ZooError::Import { id: id.clone(), message };
    if !is_repo(&spec.repo) {
        return Err(fail(format!("{:?} is not a repository, <owner>/<name>", spec.repo)));
    }
    if !is_commit(&spec.commit) {
        return Err(fail(format!(
            "{:?} is not a commit (40 lowercase hex digits); a branch or tag may move and is not imported",
            spec.commit
        )));
    }
    let endpoint = endpoint.trim_end_matches('/');
    let address = |s: String| Url::from_str(&s).map_err(&fail);
    let get = |url: &Url, size: u64| {
        transport.get(url, size).map_err(|message| ZooError::Transport { url: url.to_string(), message })
    };
    let tree = address(format!("{endpoint}/api/models/{}/tree/{}?recursive=true", spec.repo, spec.commit))?;
    let bytes = get(&tree, LISTING_BYTES)?;
    if bytes.len() as u64 > LISTING_BYTES {
        return Err(fail(format!("the repository listing is over {LISTING_BYTES} bytes")));
    }
    let mut listed = listing(&fail, &bytes)?;
    if !spec.files.is_empty() {
        let mut chosen = Vec::with_capacity(spec.files.len());
        for want in &spec.files {
            let at = listed
                .iter()
                .position(|l| l.path.as_str() == want)
                .ok_or_else(|| fail(format!("the commit holds no file {want:?} (or it is named twice)")))?;
            chosen.push(listed.remove(at));
        }
        listed = chosen;
    }
    if listed.is_empty() {
        return Err(fail("the commit holds no file".into()));
    }

    let mut files = Vec::with_capacity(listed.len());
    let mut manifest = None;
    for l in listed {
        let url = address(format!("{endpoint}/{}/resolve/{}/{}", spec.repo, spec.commit, l.path))?;
        let bytes = get(&url, l.size)?;
        let differs = |check, expected: String, actual: String| ZooError::Integrity {
            id: id.clone(),
            path: l.path.to_string(),
            check,
            expected,
            actual,
        };
        if bytes.len() as u64 != l.size {
            return Err(differs(Check::Size, l.size.to_string(), bytes.len().to_string()));
        }
        let sha256 = Sha256::of(&bytes);
        if let Some(recorded) = l.sha256.filter(|r| *r != sha256) {
            return Err(differs(Check::Sha256, recorded.to_string(), sha256.to_string()));
        }
        files.push(File {
            path: l.path,
            url,
            also: Vec::new(),
            size: l.size,
            sha256,
            sha512: Sha512::of(&bytes),
            download: None,
        });
        if files.last().is_some_and(|f| f.path.as_str() == MANIFEST) {
            manifest = Some(bytes);
        }
    }

    let (model_type, kind, format, version, record) = if spec.bundle {
        let bytes = manifest.as_deref().ok_or_else(|| fail(format!("a bundle needs {MANIFEST} among its files")))?;
        let (record, _) = manifest_record(&id, bytes)?;
        (ModelType::EncoderBundle, Kind::Model, Format::Bundle, record.revision.clone(), Some(record))
    } else {
        (ModelType::AnalysisArtifact, spec.kind, spec.format, spec.version.unwrap_or_else(|| spec.commit.clone()), None)
    };
    let task = record.as_ref().map(|r| r.task);
    let artifact = Artifact {
        id: spec.id,
        model_type,
        kind,
        component: spec.described.component.or_else(|| task.and_then(super::component_of)),
        language: spec.described.language,
        format,
        version,
        license: spec.license,
        license_terms: spec.described.license_terms,
        trained_on: spec.described.trained_on,
        class: spec.license.class(),
        attribution: spec.described.attribution,
        origin: spec.described.origin,
        bundle: record,
        files,
    };
    described(artifact, manifest.as_deref())
}
