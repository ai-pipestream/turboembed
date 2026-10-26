use crate::{Config, Model};
use serde::{Deserialize, Serialize};
use std::{fs, io::Read, path::Path};
use turbo::bundle::Bundle;
use turbo::deberta::package::{MAX_WEIGHT_BYTES, PackageManifest, digest};
use turbo::safetensors::{Dtype, File};
use turbo_analysis_embed::EncoderOptions;

/// Digests bind the head, tokenizer, encoder manifest and GLiNER configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub config: Config,
    pub encoder_sha256: String,
    pub heads_sha256: String,
    pub tokenizer_sha256: String,
}
pub fn read(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let f = fs::File::open(path).map_err(|e| e.to_string())?;
    if f.metadata().map_err(|e| e.to_string())?.len() > limit {
        return Err("gliner package byte limit exceeded".into());
    }
    let mut b = Vec::new();
    f.take(limit.saturating_add(1)).read_to_end(&mut b).map_err(|e| e.to_string())?;
    if b.len() as u64 > limit {
        return Err("gliner package byte limit exceeded".into());
    }
    Ok(b)
}
/// What `read_checked` returns: the manifest, its bytes, the heads and
/// the tokenizer.
type Checked = (Manifest, Vec<u8>, Vec<u8>, Vec<u8>);

impl Manifest {
    /// The package's heads and tokenizer over its encoder as the
    /// FAMILY_DEBERTA_V2 core bundle at `bundle`, on the device `options`
    /// names, through the backend table (Model::open_on). The package is
    /// checked as `load` checks it, and the bundle's weights must be the
    /// package encoder's, tensor for tensor.
    pub fn open_on(dir: &Path, bundle: &Path, options: &EncoderOptions) -> Result<Model, String> {
        let (m, meta, heads, tokenizer) = Self::read_checked(dir)?;
        let e: PackageManifest =
            serde_json::from_slice(&read(&dir.join("encoder/deberta.json"), 65536)?).map_err(|e| e.to_string())?;
        let ours = read(&dir.join("encoder/weights.safetensors"), MAX_WEIGHT_BYTES)?;
        let core = Bundle::open(bundle).map_err(|e| e.message)?;
        let file = core.manifest.artifacts.first().and_then(|a| a.files.first()).ok_or("encoder bundle: no weights")?;
        let theirs = core.read_verified(file).map_err(|e| e.message)?;
        let (ours, theirs) = (
            File::parse("gliner encoder", &ours).map_err(|e| e.message)?,
            File::parse("encoder bundle", &theirs).map_err(|e| e.message)?,
        );
        for (name, shape) in e.config.tensor_shapes().map_err(|e| e.message)? {
            let a = ours.get(&name, Dtype::F32, shape.len()).map_err(|e| e.message)?;
            let b = theirs.get(&name, Dtype::F32, shape.len()).map_err(|e| e.message)?;
            if a.shape != b.shape || a.data != b.data {
                return Err(format!("encoder bundle: {name} is not the package encoder's"));
            }
        }
        let identity =
            turbo_analysis::ModelRef { id: e.source_repository, revision: e.source_revision, sha256: digest(&meta) };
        Model::open_on(bundle, options, m.config, &heads, &tokenizer, identity)
    }

    /// gliner.json, its bytes, the heads and the tokenizer, every digest
    /// checked.
    fn read_checked(dir: &Path) -> Result<Checked, String> {
        let meta = read(&dir.join("gliner.json"), 65536)?;
        let m: Self = serde_json::from_slice(&meta).map_err(|e| e.to_string())?;
        if m.version != 1 {
            return Err("gliner unsupported package version".into());
        }
        let enc_meta = read(&dir.join("encoder/deberta.json"), 65536)?;
        let heads = read(&dir.join("heads.safetensors"), 256 << 20)?;
        let tokenizer = read(&dir.join("spm.model"), 16 << 20)?;
        if digest(&enc_meta) != m.encoder_sha256
            || digest(&heads) != m.heads_sha256
            || digest(&tokenizer) != m.tokenizer_sha256
        {
            return Err("gliner package SHA-256 mismatch".into());
        }
        Ok((m, meta, heads, tokenizer))
    }

    pub fn load(dir: &Path) -> Result<Model, String> {
        let (m, meta, heads, tokenizer) = Self::read_checked(dir)?;
        let (e, encoder) = PackageManifest::load(&dir.join("encoder")).map_err(|e| e.message)?;
        let identity =
            turbo_analysis::ModelRef { id: e.source_repository, revision: e.source_revision, sha256: digest(&meta) };
        Model::new(encoder, m.config, &heads, &tokenizer, identity)
    }
}
