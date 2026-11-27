//! `--bundle-dir`: every bundle directory under one directory, served
//! under its directory name with the defaults, on `--bundle-device`
//! (runtime device 0 when absent).

use super::{Device, ModelConfig};

/// Adds each directory of `dir` holding a manifest.json, in name order,
/// unless a model of its name is already served.
pub(super) fn discover(dir: &str, device: Device, models: &mut Vec<ModelConfig>) -> Result<(), String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("--bundle-dir {dir}: {e}"))?;
    let mut paths = Vec::new();
    for entry in rd {
        let entry = entry.map_err(|e| format!("--bundle-dir {dir}: {e}"))?;
        let path = entry.path();
        if path.is_dir() && path.join("manifest.json").is_file() {
            paths.push(path);
        }
    }
    paths.sort();
    let existing_names: Vec<String> = models.iter().filter_map(|m| m.name().map(str::to_owned)).collect();
    for path in paths {
        let bundle = path.to_string_lossy().into_owned();
        let cfg = ModelConfig { bundle, name: None, device, precision: 0, max_batch: 0, max_seq: 0, sessions: 1 };
        if let Some(n) = cfg.name()
            && !existing_names.iter().any(|en| en == n)
        {
            models.push(cfg);
        }
    }
    Ok(())
}
