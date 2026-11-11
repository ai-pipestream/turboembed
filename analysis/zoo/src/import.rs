//! Importers: a model published in another layout, read and verified,
//! written as an index entry. The zoo itself reads only its own index;
//! an importer is how a model from elsewhere gets one.
//!
//! [`bundle`] reads a sealed bundle directory (docs/bundle.md).

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

/// What an entry holds that a sealed bundle does not say of itself.
#[derive(Debug, Clone)]
pub struct BundleImport {
    /// The entry's id.
    pub id: ArtifactId,
    /// Where the bundle directory is published: each file's address is
    /// `<base>/<path>`.
    pub base: Url,
    /// Where the model's publisher states its licence.
    pub license_terms: Url,
    /// What the model was trained on. The entry's class is the most
    /// restrictive of the manifest's licence and these.
    pub trained_on: Vec<TrainingData>,
    pub attribution: Option<String>,
    /// For a model whose class is `CLASS_LOCAL_ONLY`: its publisher's own
    /// location, which `base` must lie under.
    pub origin: Option<Url>,
    /// The component the model serves; by default the one its task gives
    /// ([`component_of`]).
    pub component: Option<Component>,
    pub language: Option<Language>,
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
    let manifest_bytes = read_plain(&dir.join(MANIFEST))?;
    let v: serde_json::Value =
        serde_json::from_slice(&manifest_bytes).map_err(|e| fail(format!("{MANIFEST} is not JSON: {e}")))?;
    let text = |pointer: &str| {
        v.pointer(pointer)
            .and_then(|s| s.as_str())
            .ok_or_else(|| fail(format!("{MANIFEST} has no string at {pointer}")))
    };
    let parsed = |pointer: &str, e: String| fail(format!("{MANIFEST} at {pointer}: {e}"));
    let task = Task::from_str(text("/task")?).map_err(|e| parsed("/task", e))?;
    let license = Licence::from_str(text("/model/license")?).map_err(|e| parsed("/model/license", e))?;
    let record = Bundle { model: text("/model/id")?.to_owned(), revision: text("/model/revision")?.to_owned(), task };

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
        let bytes = read_plain(&dir.join(path.as_str()))?;
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

    let mut artifact = Artifact {
        id: spec.id,
        model_type: ModelType::EncoderBundle,
        kind: Kind::Model,
        component: spec.component.or_else(|| component_of(task)),
        language: spec.language,
        format: Format::Bundle,
        version: record.revision.clone(),
        license,
        license_terms: spec.license_terms,
        trained_on: spec.trained_on,
        class: license.class(),
        attribution: spec.attribution,
        origin: spec.origin,
        bundle: Some(record),
        files,
    };
    artifact.class = artifact.computed_class();
    // The entry is held to every rule of the index, and to its manifest,
    // before it is handed out.
    let index = Index::parse(Index::to_canonical_json(std::slice::from_ref(&artifact)).as_bytes())?;
    index.get(artifact.id.as_str())?;
    crate::index::check_manifest(&artifact, &manifest_bytes)?;
    Ok(artifact)
}

/// `index` with `entry` added last, or in the place of the entry of the
/// same id, in the canonical form.
pub fn add_to(index: &Index, entry: Artifact) -> String {
    let mut artifacts = index.list().to_vec();
    match artifacts.iter_mut().find(|a| a.id == entry.id) {
        Some(held) => *held = entry,
        None => artifacts.push(entry),
    }
    Index::to_canonical_json(&artifacts)
}

/// Writes an index's canonical text to `path` through a temporary name
/// beside it, so an interrupted write leaves the file as it was.
pub fn write(path: &Path, canonical: &str) -> Result<(), ZooError> {
    crate::fetch::write_atomic(path, canonical.as_bytes())
}

/// The whole of the plain file at `path`; a symlink, a directory or
/// anything else is refused.
fn read_plain(path: &Path) -> Result<Vec<u8>, ZooError> {
    let io = |message: String| ZooError::Io { path: path.to_path_buf(), message };
    let m = fs::symlink_metadata(path).map_err(|e| io(e.to_string()))?;
    if !m.is_file() {
        return Err(io("is not a plain file".into()));
    }
    let mut out = Vec::new();
    fs::File::open(path).and_then(|mut f| f.read_to_end(&mut out)).map_err(|e| io(e.to_string()))?;
    Ok(out)
}
