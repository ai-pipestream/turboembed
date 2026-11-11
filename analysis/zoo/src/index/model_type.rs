//! The schema an entry follows beside the fields every entry has: an
//! analysis artifact (the default, nothing more), or an encoder bundle,
//! a sealed bundle directory (docs/bundle.md) with its manifest.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{Artifact, Format, Kind};
use crate::digest::Sha256;
use crate::error::ZooError;
use crate::licence::Licence;

/// Where a bundle keeps its manifest, among the entry's files.
pub const MANIFEST: &str = "manifest.json";

constant_enum! {
    /// Which schema an entry follows. An entry that names none is an
    /// analysis artifact.
    ModelType, "a model type" {
        /// A model, head, lexicon, centroid set or dataset an analysis
        /// add-on reads: the shared fields and nothing more.
        AnalysisArtifact => "TYPE_ANALYSIS_ARTIFACT",
        /// A sealed encoder bundle: the manifest and every file it names.
        EncoderBundle => "TYPE_ENCODER_BUNDLE",
    }
}

impl ModelType {
    /// Whether this is the type an entry without the field has.
    pub fn is_default(&self) -> bool {
        *self == ModelType::AnalysisArtifact
    }
}

#[allow(clippy::derivable_impls)]
impl Default for ModelType {
    fn default() -> ModelType {
        ModelType::AnalysisArtifact
    }
}

constant_enum! {
    /// What a bundle computes, as its manifest's `task` names it.
    Task, "a bundle task" {
        Embed => "TASK_EMBED",
        Rerank => "TASK_RERANK",
        ExtractiveQa => "TASK_EXTRACTIVE_QA",
    }
}

/// What an encoder bundle's entry records of its manifest, so a reader
/// picks a bundle without fetching it. Each field is checked against the
/// manifest once the files are fetched ([`check_manifest`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bundle {
    /// The manifest's `model.id`, the publisher's name for the model
    /// (`BAAI/bge-small-en-v1.5`).
    pub model: String,
    /// The manifest's `model.revision`, which counts builds of the model.
    pub revision: String,
    /// The manifest's `task`.
    pub task: Task,
}

fn plain(s: &str) -> bool {
    !s.is_empty() && s.trim() == s && !s.chars().any(char::is_control)
}

/// Why `a` does not follow the schema of its type.
pub(super) fn check(a: &Artifact) -> Result<(), String> {
    let encoder = ModelType::EncoderBundle;
    match (a.model_type, &a.bundle) {
        (ModelType::AnalysisArtifact, None) => Ok(()),
        (ModelType::AnalysisArtifact, Some(_)) => Err(format!("bundle is only for {encoder}")),
        (ModelType::EncoderBundle, None) => Err(format!("{encoder} needs bundle, what its manifest records")),
        (ModelType::EncoderBundle, Some(b)) => {
            if a.kind != Kind::Model {
                return Err(format!("{encoder} is {}, not {}", Kind::Model, a.kind));
            }
            if a.format != Format::Bundle {
                return Err(format!("{encoder} is {}, not {}", Format::Bundle, a.format));
            }
            if !a.files.iter().any(|f| f.path.as_str() == MANIFEST) {
                return Err(format!("{encoder} lists no {MANIFEST}"));
            }
            if !plain(&b.model) || !plain(&b.revision) {
                return Err(format!(
                    "bundle model {:?} or revision {:?} is empty, padded or holds a control character",
                    b.model, b.revision
                ));
            }
            Ok(())
        }
    }
}

/// Checks an encoder bundle's manifest, already verified as the entry's
/// own file, against the entry: the model id, revision, task and licence
/// are the entry's, every file the manifest names is listed with that
/// size and SHA-256, and the entry lists no file the manifest does not
/// name. An entry of another type passes unread.
pub fn check_manifest(a: &Artifact, manifest: &[u8]) -> Result<(), ZooError> {
    let Some(b) = a.bundle.as_ref().filter(|_| a.model_type == ModelType::EncoderBundle) else {
        return Ok(());
    };
    let fail = |m: String| ZooError::Bundle { id: a.id.clone(), message: m };
    let v: serde_json::Value =
        serde_json::from_slice(manifest).map_err(|e| fail(format!("{MANIFEST} is not JSON: {e}")))?;
    let text = |pointer: &str| {
        v.pointer(pointer)
            .and_then(|s| s.as_str())
            .ok_or_else(|| fail(format!("{MANIFEST} has no string at {pointer}")))
    };
    let same = |pointer: &str, want: &str| match text(pointer)? {
        got if got == want => Ok(()),
        got => Err(fail(format!("{MANIFEST} has {got:?} at {pointer}, the index says {want:?}"))),
    };
    same("/model/id", &b.model)?;
    same("/model/revision", &b.revision)?;
    same("/task", b.task.name())?;
    let licence = text("/model/license")?;
    if Licence::from_str(licence).ok() != Some(a.license) {
        return Err(fail(format!("{MANIFEST} has {licence:?} at /model/license, the index says {}", a.license)));
    }
    let named = v
        .pointer("/files")
        .and_then(|f| f.as_array())
        .ok_or_else(|| fail(format!("{MANIFEST} has no list at /files")))?;
    let mut paths = Vec::with_capacity(named.len());
    for (i, f) in named.iter().enumerate() {
        let field = |name: &str| f.get(name).ok_or_else(|| fail(format!("{MANIFEST} files[{i}] has no {name}")));
        let path = field("path")?.as_str().unwrap_or_default();
        let listed = a
            .files
            .iter()
            .find(|l| l.path.as_str() == path)
            .ok_or_else(|| fail(format!("{MANIFEST} names {path:?}, which the index does not list")))?;
        let size = field("size")?.as_u64();
        let sha256 = field("sha256")?.as_str().and_then(|s| Sha256::from_str(s).ok());
        if size != Some(listed.size) || sha256.as_ref() != Some(&listed.sha256) {
            return Err(fail(format!(
                "{MANIFEST} gives {path} another size or SHA-256 than the index ({} bytes, {})",
                listed.size, listed.sha256
            )));
        }
        paths.push(path);
    }
    match a.files.iter().map(|f| f.path.as_str()).find(|p| *p != MANIFEST && !paths.contains(p)) {
        Some(extra) => Err(fail(format!("the index lists {extra}, which {MANIFEST} does not name"))),
        None => Ok(()),
    }
}
