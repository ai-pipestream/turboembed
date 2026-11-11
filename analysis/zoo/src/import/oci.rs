//! An OCI artifact in a registry, pinned by its manifest's digest, read
//! as an index entry: each layer that names a file is one file of the
//! entry, at its blob's own address.

use std::str::FromStr;

use super::{Described, described, manifest_record};
use crate::digest::{Sha256, Sha512};
use crate::error::{Check, ZooError};
use crate::fetch::{OciRegistry, Transport};
use crate::index::{Artifact, ArtifactId, File, Format, Kind, MANIFEST, ModelType, RelPath, Url};
use crate::licence::Licence;

/// The annotation a layer names its file with, as ORAS writes it.
pub const TITLE: &str = "org.opencontainers.image.title";

/// An artifact in a registry, and what an entry holds that it does not say.
#[derive(Debug, Clone)]
pub struct OciImport {
    pub id: ArtifactId,
    /// `<registry>/<name>@sha256:<64 hex>`. A tag is refused: it may move.
    pub reference: String,
    /// With `bundle`, the layers are a sealed bundle directory and the
    /// entry is a `TYPE_ENCODER_BUNDLE` one. Otherwise an analysis
    /// artifact of this `kind` and `format`.
    pub bundle: bool,
    pub kind: Kind,
    pub format: Format,
    /// The entry's `version`; the manifest's digest when none is given.
    pub version: Option<String>,
    pub license: Licence,
    pub described: Described,
}

/// `sha256:<64 hex>` as its digest.
fn digest(s: &str) -> Option<Sha256> {
    s.strip_prefix("sha256:").and_then(|hex| Sha256::from_str(hex).ok())
}

/// The registry, repository name and manifest digest `reference` names.
fn parse(reference: &str) -> Result<(&str, &str, Sha256), String> {
    let (repository, pin) = reference
        .split_once('@')
        .ok_or_else(|| format!("{reference:?} names no digest; a tag may move and is not imported"))?;
    let pin =
        digest(pin).ok_or_else(|| format!("{reference:?}: the digest is not sha256:<64 lowercase hex digits>"))?;
    let (registry, name) = repository.split_once('/').unwrap_or((repository, ""));
    let host = !registry.is_empty()
        && (registry.contains('.') || registry.contains(':'))
        && registry.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'-' | b':'));
    let part = |p: &str| {
        !p.is_empty()
            && p.bytes().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, b'.' | b'_' | b'-'))
            && p.bytes().next().is_some_and(|c| c.is_ascii_alphanumeric())
    };
    if !host || name.is_empty() || !name.split('/').all(part) {
        return Err(format!("{reference:?} is not <registry>/<name>@sha256:<digest>, with the registry's host named"));
    }
    Ok((registry, name, pin))
}

/// The entry of the artifact `spec.reference` names, read from
/// `registry`. The manifest is held to the digest of the reference, and
/// each layer to the size and digest the manifest gives it; a layer's
/// file is the path its `org.opencontainers.image.title` annotation
/// names, and its address is `https://<registry>/v2/<name>/blobs/<digest>`.
/// An image index, a layer that names no file (a container image's file
/// system layer), a digest that is not SHA-256, and an entry the index
/// would not load are refused.
pub fn oci(registry: &OciRegistry, spec: OciImport) -> Result<Artifact, ZooError> {
    let id = spec.id.clone();
    let fail = |message: String| ZooError::Import { id: id.clone(), message };
    let (host, name, pin) = parse(&spec.reference).map_err(&fail)?;
    let at = format!("https://{host}/v2/{name}");
    let manifest = registry
        .manifest(host, name, &pin)
        .map_err(|message| ZooError::Transport { url: format!("{at}/manifests/sha256:{pin}"), message })?;
    let v: serde_json::Value =
        serde_json::from_slice(&manifest).map_err(|e| fail(format!("the artifact's manifest is not JSON: {e}")))?;
    if v.get("manifests").is_some() {
        return Err(fail("the digest names an image index; name one manifest of it".into()));
    }
    let layers = v
        .get("layers")
        .and_then(|l| l.as_array())
        .filter(|l| !l.is_empty())
        .ok_or_else(|| fail("the artifact's manifest lists no layers".into()))?;

    let mut files = Vec::with_capacity(layers.len());
    let mut bundle_manifest = None;
    for (i, layer) in layers.iter().enumerate() {
        let bad = |what: &str| fail(format!("layers[{i}] {what}"));
        let title = layer.get("annotations").and_then(|a| a.get(TITLE)).and_then(|t| t.as_str()).ok_or_else(|| {
            bad("names no file (no org.opencontainers.image.title); a container image's layers are not model files")
        })?;
        let path = RelPath::from_str(title).map_err(|e| bad(&format!("names a file the zoo does not take: {e}")))?;
        let size = layer.get("size").and_then(|s| s.as_u64()).ok_or_else(|| bad("has no size"))?;
        let named = layer.get("digest").and_then(|d| d.as_str()).unwrap_or_default();
        let sha256 = digest(named).ok_or_else(|| bad(&format!("has the digest {named:?}, not a SHA-256 one")))?;
        let url = Url::from_str(&format!("{at}/blobs/sha256:{sha256}")).map_err(&fail)?;
        let bytes =
            registry.get(&url, size).map_err(|message| ZooError::Transport { url: url.to_string(), message })?;
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
        let is_manifest = path.as_str() == MANIFEST;
        files.push(File { path, url, also: Vec::new(), size, sha256, sha512: Sha512::of(&bytes), download: None });
        if is_manifest {
            bundle_manifest = Some(bytes);
        }
    }

    let (model_type, kind, format, version, record) = if spec.bundle {
        let bytes =
            bundle_manifest.as_deref().ok_or_else(|| fail(format!("a bundle needs {MANIFEST} among its layers")))?;
        let (record, _) = manifest_record(&id, bytes)?;
        (ModelType::EncoderBundle, Kind::Model, Format::Bundle, record.revision.clone(), Some(record))
    } else {
        let version = spec.version.unwrap_or_else(|| format!("sha256:{pin}"));
        (ModelType::AnalysisArtifact, spec.kind, spec.format, version, None)
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
    described(artifact, bundle_manifest.as_deref())
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn a_reference_names_a_registry_a_repository_and_a_digest() {
        let hex = "a".repeat(64);
        let pinned = format!("registry.example.org:5000/models/tiny-bert@sha256:{hex}");
        let (registry, name, pin) = parse(&pinned).unwrap();
        assert_eq!((registry, name, pin.to_string()), ("registry.example.org:5000", "models/tiny-bert", hex.clone()));
        for moving in ["registry.example.org/models/tiny:latest", "registry.example.org/models/tiny"] {
            assert!(parse(moving).unwrap_err().contains("a tag may move"), "{moving}");
        }
        for bad in [
            format!("registry.example.org/models/tiny@sha512:{hex}"),
            format!("registry.example.org/models/tiny@sha256:{}", "A".repeat(64)),
            format!("registry.example.org/models/tiny@sha256:{}", &hex[1..]),
        ] {
            assert!(parse(&bad).unwrap_err().contains("the digest is not"), "{bad}");
        }
        for bad in [
            "models/tiny",
            "registry.example.org",
            "registry.example.org/",
            "registry.example.org/Models/x",
            "registry.example.org/a//b",
            "registry.example.org/../x",
            "user@registry.example.org/x",
            "/x",
        ] {
            assert!(parse(&format!("{bad}@sha256:{hex}")).is_err(), "{bad}");
        }
    }
}
