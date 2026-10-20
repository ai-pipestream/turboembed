//! Convert a local F32 HF DeBERTa-v3 checkpoint to a checked native encoder
//! package. Prefix removal is explicit, so GLiNER2 uses `encoder.` and a bare
//! DebertaV2Model uses an empty prefix. Non-encoder heads are not copied.
mod config;
mod files;
use crate::Result;
pub use config::from_huggingface;
pub use files::convert_files;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};
use turbo::{
    deberta::{
        Config, Encoder,
        package::{MAX_WEIGHT_BYTES, PackageManifest, digest},
    },
    safetensors::{Dtype, File},
};

/// Canonical sorted F32 safetensors. Every required tensor is checked before
/// output; unsupported dtype, missing keys and wrong shapes are errors.
pub fn convert(config: &Config, checkpoint: &[u8], prefix: &str) -> Result<Vec<u8>> {
    let file = File::parse("upstream checkpoint", checkpoint).map_err(|e| e.message)?;
    let mut header = BTreeMap::<String, Value>::new();
    let mut values = Vec::new();
    for (name, shape) in config.tensor_shapes().map_err(|e| e.message)? {
        let tensor = file.get(&format!("{prefix}{name}"), Dtype::F32, shape.len()).map_err(|e| e.message)?;
        if tensor.shape != shape {
            return Err(format!("deberta {name}: checkpoint shape mismatch"));
        }
        if tensor.data.as_chunks::<4>().0.iter().any(|b| !f32::from_le_bytes(*b).is_finite()) {
            return Err(format!("deberta {name}: non-finite checkpoint value"));
        }
        let start = values.len();
        let end = start
            .checked_add(tensor.data.len())
            .filter(|&n| n as u64 <= MAX_WEIGHT_BYTES)
            .ok_or("deberta weight byte limit exceeded")?;
        values.try_reserve(tensor.data.len()).map_err(|_| "deberta conversion allocation failed")?;
        values.extend_from_slice(tensor.data);
        header.insert(name, json!({"dtype": "F32", "shape": shape, "data_offsets": [start, end]}));
    }
    let mut h = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(values);
    // Use the same loader the consumer uses; no independent permissive path.
    Encoder::from_safetensors(config.clone(), &out).map_err(|e| e.message)?;
    Ok(out)
}

/// Publish to a new directory only, preserving existing packages on error.
/// Source revision and licence are required provenance, not inferred from names.
pub fn write_package(
    config: Config,
    checkpoint: &[u8],
    prefix: &str,
    source: (&str, &str, &str),
    directory: &Path,
) -> Result<()> {
    let bytes = convert(&config, checkpoint, prefix)?;
    let manifest = PackageManifest {
        version: 1,
        config,
        source_repository: source.0.into(),
        source_revision: source.1.into(),
        source_license: source.2.into(),
        weights_sha256: digest(&bytes),
    };
    manifest.check().map_err(|e| e.message)?;
    let json = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    fs::create_dir(directory).map_err(|e| format!("deberta output directory must be new: {e}"))?;
    // A failed write leaves an incomplete directory for inspection; load
    // requires a complete manifest and a matching digest, never partial weights.
    fs::write(directory.join("weights.safetensors"), bytes).map_err(|e| e.to_string())?;
    fs::write(directory.join("deberta.json"), json).map_err(|e| e.to_string())?;
    Ok(())
}
