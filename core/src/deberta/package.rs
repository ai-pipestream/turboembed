use super::{Config, Encoder};
use crate::status::{Result, invalid};
use serde::{Deserialize, Serialize};
use std::{fs, io::Read, path::Path};

/// A separate, versioned encoder package for the Rust extraction API.
/// This is not a general libturbo session manifest.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageManifest {
    pub version: u32,
    pub config: Config,
    pub source_repository: String,
    pub source_revision: String,
    pub source_license: String,
    pub weights_sha256: String,
}
impl PackageManifest {
    pub fn check(&self) -> Result<()> {
        self.config.check()?;
        if self.version != 1
            || self.source_repository.is_empty()
            || self.source_repository.len() > 512
            || self.source_license.is_empty()
            || self.source_license.len() > 128
            || !hex(&self.source_revision, 40)
            || !hex(&self.weights_sha256, 64)
        {
            return Err(invalid("deberta package version/source/digest invalid"));
        }
        Ok(())
    }
    pub fn load(directory: &Path) -> Result<(Self, Encoder)> {
        let manifest = read_limited(&directory.join("deberta.json"), 65536)?;
        let meta: Self = serde_json::from_slice(&manifest).map_err(|e| invalid(format!("deberta manifest: {e}")))?;
        meta.check()?;
        let weights = read_limited(&directory.join("weights.safetensors"), MAX_WEIGHT_BYTES)?;
        if digest(&weights) != meta.weights_sha256 {
            return Err(invalid("deberta weights SHA-256 mismatch"));
        }
        let encoder = Encoder::from_safetensors(meta.config.clone(), &weights)?;
        Ok((meta, encoder))
    }
}

pub const MAX_WEIGHT_BYTES: u64 = 2_147_483_648;
pub fn digest(bytes: &[u8]) -> String {
    turbo_sha256::sha256(bytes).to_hex()
}
fn hex(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let f = fs::File::open(path).map_err(|e| invalid(format!("deberta package read: {e}")))?;
    let n = f.metadata().map_err(|e| invalid(e.to_string()))?.len();
    if n > limit {
        return Err(invalid("deberta package byte limit exceeded"));
    }
    let mut data = Vec::new();
    data.try_reserve_exact(n as usize).map_err(|_| invalid("deberta package allocation failed"))?;
    f.take(limit + 1).read_to_end(&mut data).map_err(|e| invalid(e.to_string()))?;
    if data.len() as u64 > limit {
        return Err(invalid("deberta package byte limit exceeded"));
    }
    Ok(data)
}
