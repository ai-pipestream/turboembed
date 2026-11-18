//! `--model` configuration parsing and model-name validation.

use super::refused::{MODEL_SETTINGS, number, refused};
use crate::api::names;

/// Where the model runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// A runtime device index, for turbo_context_create.
    Index(u32),
    /// The index turbo_runtime_select gives for TURBO_TASK_EMBED.
    Select,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    pub bundle: String,
    /// The model name a request uses; None is the bundle's directory name.
    pub name: Option<String>,
    pub device: Device,
    /// turbo_session_desc.precision, TURBO_PRECISION_*.
    pub precision: u32,
    /// turbo_session_desc.max_batch; 0 is the model's.
    pub max_batch: u32,
    /// turbo_session_desc.max_seq; 0 is the model's.
    pub max_seq: u32,
    /// How many sessions turbo_session_create makes; at least 1.
    pub sessions: u32,
}

impl ModelConfig {
    /// The model name: the `name` setting, or the last component of the
    /// bundle path, exactly as it is spelled. None for a path whose last
    /// component is empty, `.` or `..`, which names no bundle.
    pub fn name(&self) -> Option<&str> {
        if let Some(n) = &self.name {
            return Some(n);
        }
        let last = self.bundle.rsplit(std::path::is_separator).next().unwrap_or("");
        (!matches!(last, "" | "." | "..")).then_some(last)
    }

    /// One `--model` value.
    pub fn parse(s: &str) -> Result<ModelConfig, String> {
        let (mut bundle, mut name, mut device, mut sessions) = (None, None, None, None);
        let (mut precision, mut max_batch, mut max_seq) = (None, None, None);
        for part in s.split(',') {
            let (k, v) = part
                .split_once('=')
                .ok_or_else(|| format!("--model: a setting ({} bytes, not shown) is not key=value", part.len()))?;
            let seen = match k {
                "bundle" => bundle.replace(v.to_string()).is_some(),
                "name" => {
                    if v.is_empty() {
                        return Err("--model: name is empty".into());
                    }
                    name.replace(v.to_string()).is_some()
                }
                "device" => {
                    let d = if v == "select" { Device::Select } else { Device::Index(number(k, v)?) };
                    device.replace(d).is_some()
                }
                "precision" => {
                    let p = names::PRECISION.iter().find(|(_, n)| *n == v).map(|(p, _)| *p).ok_or_else(|| {
                        refused("--model precision", v, "PRECISION_MODEL, PRECISION_FASTEST or PRECISION_EXACT")
                    })?;
                    precision.replace(p).is_some()
                }
                "max_batch" => max_batch.replace(number(k, v)?).is_some(),
                "max_seq" => max_seq.replace(number(k, v)?).is_some(),
                "sessions" => sessions.replace(number(k, v)?).is_some(),
                _ => return Err(format!("--model: a setting ({} bytes, not shown) is not {MODEL_SETTINGS}", k.len())),
            };
            if seen {
                return Err(format!("--model: {k} given twice"));
            }
        }
        let sessions = sessions.ok_or("--model: sessions is required")?;
        if sessions == 0 {
            return Err("--model: sessions must be at least 1".into());
        }
        Ok(ModelConfig {
            bundle: bundle.ok_or("--model: bundle is required")?,
            name,
            device: device.ok_or("--model: device is required")?,
            precision: precision.unwrap_or(0),
            max_batch: max_batch.unwrap_or(0),
            max_seq: max_seq.unwrap_or(0),
            sessions,
        })
    }
}

/// Every model's name, refusing an empty model list, zero sessions, a path
/// that names no bundle, and two models of one name.
pub fn names(models: &[ModelConfig]) -> Result<Vec<String>, String> {
    names_with(models, false)
}

/// Every model's name, allowing an empty model list when `empty_ok` is true:
/// an admin server starts empty and accepts LoadBundle calls, and a zoo
/// serves only its catalog.
pub fn names_with(models: &[ModelConfig], empty_ok: bool) -> Result<Vec<String>, String> {
    if models.is_empty() && !empty_ok {
        return Err("at least one --model is required".into());
    }
    let mut out: Vec<String> = Vec::new();
    for m in models {
        let name = m.name().ok_or_else(|| format!("bundle {}: the path names no bundle", m.bundle))?;
        if m.sessions == 0 {
            return Err(format!("bundle {}: sessions must be at least 1", m.bundle));
        }
        if out.iter().any(|n| n == name) {
            return Err(format!("bundle {}: a model named {name} is already served", m.bundle));
        }
        out.push(name.to_string());
    }
    Ok(out)
}
