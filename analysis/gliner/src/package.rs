use crate::{Config, Model};
use serde::{Deserialize, Serialize};
use std::{fs, io::Read, path::Path};
use turbo::deberta::package::{PackageManifest, digest};

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
impl Manifest {
    pub fn load(dir: &Path) -> Result<Model, String> {
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
        let (e, encoder) = PackageManifest::load(&dir.join("encoder")).map_err(|e| e.message)?;
        let identity =
            turbo_analysis::ModelRef { id: e.source_repository, revision: e.source_revision, sha256: digest(&meta) };
        Model::new(encoder, m.config, &heads, &tokenizer, identity)
    }
}
