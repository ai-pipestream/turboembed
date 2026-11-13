//! Importers: a model published in another layout, read and verified,
//! written as an index entry. The zoo itself reads only its own index;
//! an importer is how a model from elsewhere gets one.
//!
//! [`bundle`] reads a sealed bundle directory (docs/bundle.md), and
//! [`hugging_face`] a Hugging Face model repository at a pinned commit,
//! and [`oci`] an OCI artifact in a registry, pinned by digest.

use std::fs;
use std::io::Read;
use std::path::Path;
use std::str::FromStr;

use crate::digest::{Sha256, Sha512};
use crate::error::{Check, ZooError};
use crate::index::{
    Artifact, ArtifactId, Bundle, Component, File, Format, Index, Kind, Language, MANIFEST, ModelType, RelPath, Task,
    TrainingData, Url,
};
use crate::licence::Licence;

mod hugging_face;
#[cfg(feature = "http")]
mod oci;

pub use hugging_face::{HUGGING_FACE, RepoImport, hugging_face};
#[cfg(feature = "http")]
pub use oci::{OciImport, TITLE, oci};

/// What an entry holds that no layout says of the model itself.
#[derive(Debug, Clone)]
pub struct Described {
    /// Where the model's publisher states its licence.
    pub license_terms: Url,
    /// What the model was trained on. The entry's class is the most
    /// restrictive of its own licence and these.
    pub trained_on: Vec<TrainingData>,
    pub attribution: Option<String>,
    /// For a model whose class is `CLASS_LOCAL_ONLY`: its publisher's own
    /// location, which every file's address must lie under.
    pub origin: Option<Url>,
    /// The component the model serves; for a bundle, by default the one
    /// its task gives ([`component_of`]).
    pub component: Option<Component>,
    pub language: Option<Language>,
}

/// What an entry holds that a sealed bundle does not say of itself.
#[derive(Debug, Clone)]
pub struct BundleImport {
    /// The entry's id.
    pub id: ArtifactId,
    /// Where the bundle directory is published: each file's address is
    /// `<base>/<path>`.
    pub base: Url,
    pub described: Described,
}

/// What a bundle's manifest records of its model, and the model's
/// licence as the manifest names it.
pub(crate) fn manifest_record(id: &ArtifactId, manifest: &[u8]) -> Result<(Bundle, Licence), ZooError> {
    let fail = |message: String| ZooError::Bundle { id: id.clone(), message };
    let v: serde_json::Value =
        serde_json::from_slice(manifest).map_err(|e| fail(format!("{MANIFEST} is not JSON: {e}")))?;
    let text = |pointer: &str| {
        v.pointer(pointer)
            .and_then(|s| s.as_str())
            .ok_or_else(|| fail(format!("{MANIFEST} has no string at {pointer}")))
    };
    let parsed = |pointer: &str, e: String| fail(format!("{MANIFEST} at {pointer}: {e}"));
    let task = Task::from_str(text("/task")?).map_err(|e| parsed("/task", e))?;
    let license = Licence::from_str(text("/model/license")?).map_err(|e| parsed("/model/license", e))?;
    Ok((Bundle { model: text("/model/id")?.to_owned(), revision: text("/model/revision")?.to_owned(), task }, license))
}

/// `artifact` with the class its licences give, held to every rule of the
/// index, and, for a bundle, to its `manifest`, before it is handed out.
pub(crate) fn described(mut artifact: Artifact, manifest: Option<&[u8]>) -> Result<Artifact, ZooError> {
    artifact.class = artifact.computed_class();
    let index = Index::parse(Index::to_canonical_json(std::slice::from_ref(&artifact)).as_bytes())?;
    index.get(artifact.id.as_str())?;
    if let Some(manifest) = manifest.filter(|_| artifact.model_type == ModelType::EncoderBundle) {
        crate::index::check_manifest(&artifact, manifest)?;
    }
    Ok(artifact)
}

/// The component a bundle of `task` serves when the importer names none:
/// a dense encoder for an embedding bundle, extractive QA for a QA one,
/// and none for a reranker.
pub fn component_of(task: Task) -> Option<Component> {
    match task {
        Task::Embed => Some(Component::DenseEncoder),
        Task::ExtractiveQa => Some(Component::ExtractiveQa),
        Task::Rerank => None,
    }
}

/// The `TYPE_ENCODER_BUNDLE` entry of the sealed bundle in `dir`: the
/// manifest and every file it names, each read whole and held to the
/// size and SHA-256 the manifest pins, with the model id, revision, task
/// and licence the manifest records. A file that is absent, a symlink, or
/// not the bytes the manifest pins refuses the import, and so does an
/// entry the index would not load (a class that needs an `origin`, a
/// licence that asks for attribution, a refused licence).
pub fn bundle(dir: &Path, spec: BundleImport) -> Result<Artifact, ZooError> {
    let id = spec.id.clone();
    let fail = |message: String| ZooError::Bundle { id: id.clone(), message };
    let manifest_bytes = read_plain(dir, MANIFEST)?;
    let (record, license) = manifest_record(&id, &manifest_bytes)?;
    let task = record.task;
    let v: serde_json::Value =
        serde_json::from_slice(&manifest_bytes).map_err(|e| fail(format!("{MANIFEST} is not JSON: {e}")))?;

    let base = spec.base.as_str().trim_end_matches('/');
    let at = |path: &RelPath| Url::from_str(&format!("{base}/{path}")).map_err(|e| fail(format!("{path}: {e}")));
    let manifest_path = RelPath::from_str(MANIFEST).expect("a plain file name");
    let mut files = vec![File {
        url: at(&manifest_path)?,
        path: manifest_path,
        also: Vec::new(),
        size: manifest_bytes.len() as u64,
        sha256: Sha256::of(&manifest_bytes),
        sha512: Sha512::of(&manifest_bytes),
        download: None,
    }];
    let named = v
        .pointer("/files")
        .and_then(|f| f.as_array())
        .ok_or_else(|| fail(format!("{MANIFEST} has no list at /files")))?;
    for (i, f) in named.iter().enumerate() {
        let field = |name: &str| f.get(name).ok_or_else(|| fail(format!("{MANIFEST} files[{i}] has no {name}")));
        let at_file = |name: &str, e: String| fail(format!("{MANIFEST} files[{i}].{name}: {e}"));
        let path = RelPath::from_str(field("path")?.as_str().unwrap_or_default()).map_err(|e| at_file("path", e))?;
        let size = field("size")?.as_u64().ok_or_else(|| at_file("size", "is not a size".into()))?;
        let sha256 =
            Sha256::from_str(field("sha256")?.as_str().unwrap_or_default()).map_err(|e| at_file("sha256", e))?;
        let bytes = read_plain(dir, path.as_str())?;
        let differs = |check, expected: String, actual: String| ZooError::Integrity {
            id: id.clone(),
            path: path.to_string(),
            check,
            expected,
            actual,
        };
        if bytes.len() as u64 != size {
            return Err(differs(Check::Size, size.to_string(), bytes.len().to_string()));
        }
        let held = Sha256::of(&bytes);
        if held != sha256 {
            return Err(differs(Check::Sha256, sha256.to_string(), held.to_string()));
        }
        files.push(File {
            url: at(&path)?,
            path,
            also: Vec::new(),
            size,
            sha256,
            sha512: Sha512::of(&bytes),
            download: None,
        });
    }

    let d = spec.described;
    let artifact = Artifact {
        id: spec.id,
        model_type: ModelType::EncoderBundle,
        kind: Kind::Model,
        component: d.component.or_else(|| component_of(task)),
        language: d.language,
        format: Format::Bundle,
        version: record.revision.clone(),
        license,
        license_terms: d.license_terms,
        trained_on: d.trained_on,
        class: license.class(),
        attribution: d.attribution,
        origin: d.origin,
        bundle: Some(record),
        files,
    };
    described(artifact, Some(&manifest_bytes))
}

/// The index text `index` with `entry` added last, or in the place of
/// the entry of the same id, in the canonical form. Every other entry is
/// carried over, one the zoo refuses and does not list among them, and
/// the result is held to every rule of the index.
pub fn add_to(index: &[u8], entry: Artifact) -> Result<String, ZooError> {
    let mut artifacts = Index::written(index)?;
    match artifacts.iter_mut().find(|a| a.id == entry.id) {
        Some(held) => *held = entry,
        None => artifacts.push(entry),
    }
    let written = Index::to_canonical_json(&artifacts);
    Index::parse(written.as_bytes())?;
    Ok(written)
}

/// Writes an index's canonical text to `path` through a temporary name
/// beside it, so an interrupted write leaves the file as it was.
pub fn write(path: &Path, canonical: &str) -> Result<(), ZooError> {
    crate::fetch::write_atomic(path, canonical.as_bytes())
}

/// The whole of the plain file at `path` (`/`-separated) under `dir`.
/// Every directory on the way is a plain directory and the file a plain
/// file: a symlink anywhere below `dir` is refused, so no byte is read
/// from outside the bundle directory.
fn read_plain(dir: &Path, path: &str) -> Result<Vec<u8>, ZooError> {
    let mut at = dir.to_path_buf();
    let mut parts = path.split('/').peekable();
    while let Some(part) = parts.next() {
        at.push(part);
        let io = |message: String| ZooError::Io { path: at.clone(), message };
        let m = fs::symlink_metadata(&at).map_err(|e| io(e.to_string()))?;
        if parts.peek().is_some() && !m.is_dir() {
            return Err(io("is not a plain directory".into()));
        }
        if parts.peek().is_none() && !m.is_file() {
            return Err(io("is not a plain file".into()));
        }
    }
    let path = at.as_path();
    let io = |message: String| ZooError::Io { path: path.to_path_buf(), message };
    let mut out = Vec::new();
    fs::File::open(path).and_then(|mut f| f.read_to_end(&mut out)).map_err(|e| io(e.to_string()))?;
    Ok(out)
}
