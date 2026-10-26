//! The server's configuration (docs/kserve.md, Serving a bundle), from the
//! command line and the environment: one `--model` per model,
//! `bundle=PATH,device=INDEX|select,sessions=N[,name=NAME][,precision=P][,max_batch=N][,max_seq=N]`.

use std::net::SocketAddr;

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
            let (k, v) = part.split_once('=').ok_or_else(|| format!("--model {s}: `{part}` is not key=value"))?;
            let seen = match k {
                "bundle" => bundle.replace(v.to_string()).is_some(),
                "name" => {
                    if v.is_empty() {
                        return Err(format!("--model {s}: name is empty"));
                    }
                    name.replace(v.to_string()).is_some()
                }
                "device" => {
                    let d = if v == "select" { Device::Select } else { Device::Index(number(k, v)?) };
                    device.replace(d).is_some()
                }
                "precision" => {
                    let p = names::PRECISION.iter().find(|(_, n)| *n == v).map(|(p, _)| *p).ok_or_else(|| {
                        format!("precision {v}: not PRECISION_MODEL, PRECISION_FASTEST or PRECISION_EXACT")
                    })?;
                    precision.replace(p).is_some()
                }
                "max_batch" => max_batch.replace(number(k, v)?).is_some(),
                "max_seq" => max_seq.replace(number(k, v)?).is_some(),
                "sessions" => sessions.replace(number(k, v)?).is_some(),
                _ => return Err(format!("--model {s}: unknown setting `{k}`")),
            };
            if seen {
                return Err(format!("--model {s}: `{k}` given twice"));
            }
        }
        let sessions = sessions.ok_or_else(|| format!("--model {s}: sessions is required"))?;
        if sessions == 0 {
            return Err(format!("--model {s}: sessions must be at least 1"));
        }
        Ok(ModelConfig {
            bundle: bundle.ok_or_else(|| format!("--model {s}: bundle is required"))?,
            name,
            device: device.ok_or_else(|| format!("--model {s}: device is required"))?,
            precision: precision.unwrap_or(0),
            max_batch: max_batch.unwrap_or(0),
            max_seq: max_seq.unwrap_or(0),
            sessions,
        })
    }
}

fn number(k: &str, v: &str) -> Result<u32, String> {
    v.parse().map_err(|_| format!("{k} {v}: not a number from 0 to 4294967295"))
}

/// Every model's name, refusing an empty model list, zero sessions, a path
/// that names no bundle, and two models of one name.
pub fn names(models: &[ModelConfig]) -> Result<Vec<String>, String> {
    names_with(models, false)
}

/// Every model's name, allowing an empty model list when `admin` is true so
/// an admin-enabled server can start empty and accept LoadBundle calls.
pub fn names_with(models: &[ModelConfig], admin: bool) -> Result<Vec<String>, String> {
    if models.is_empty() && !admin {
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

fn discover_bundles(dir: &str, models: &mut Vec<ModelConfig>) -> Result<(), String> {
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
        let cfg = ModelConfig {
            bundle,
            name: None,
            device: Device::Index(0),
            precision: 0,
            max_batch: 0,
            max_seq: 0,
            sessions: 1,
        };
        if let Some(n) = cfg.name()
            && !existing_names.iter().any(|en| en == n)
        {
            models.push(cfg);
        }
    }
    Ok(())
}

/// The command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub listen: SocketAddr,
    pub models: Vec<ModelConfig>,
    /// The largest request message gRPC reads.
    pub max_message_bytes: usize,
    /// The most bytes one call's request carries in all.
    pub max_stream_bytes: u64,
    /// Whether the admin calls, LoadBundle and UnloadBundle, are served.
    pub admin: bool,
    /// The vocabulary store's directory, made when absent.
    pub vocabulary_dir: Option<String>,
    /// Where completed analysis documents are forwarded, and the queue's
    /// bounds.
    pub sink: Option<crate::sink::SinkConfig>,
    /// The model catalog's zoo home, and the index it serves when not the
    /// built-in one.
    pub catalog_dir: Option<String>,
    pub catalog_index: Option<String>,
    /// The directory static models are trained into, made when absent.
    pub training_dir: Option<String>,
}

/// `--max-message-bytes` when absent: 64 MiB.
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 64 << 20;

/// `--max-stream-bytes` when absent: 2 GiB, the most one call's request
/// carries in all, frames and their prefixes included. Past one message's
/// limit, a request arrives as a client stream of frames.
pub const DEFAULT_MAX_STREAM_BYTES: u64 = 2 << 30;

/// The payload one frame of a large client-streaming upload should carry
/// (`IndexDocuments`, `ImportDictionary`, `LearnVocabulary`,
/// `AnalyzeStream`, `StreamDocuments`): well under the message limit, and
/// what the server holds of a stream at a time.
pub const FRAME_BYTES: usize = 4 << 20;

/// The largest reply message sent: 256 MiB, or the request limit when
/// that is larger. One reply can outgrow its request (a typed document or
/// formatted document for a long text, a large inference batch); every
/// limit stays under protobuf's 2 GiB.
pub fn max_reply_bytes(max_message_bytes: usize) -> usize {
    max_message_bytes.max(256 << 20)
}

/// The variable `--admin` reads when absent.
pub const ENV_ADMIN: &str = "TURBO_KSERVE_ADMIN";

/// The variable `--bundle-dir` reads when absent.
pub const ENV_BUNDLE_DIR: &str = "TURBO_KSERVE_BUNDLE_DIR";

/// The environment variables, each what the flag of the same name is when
/// the flag is absent. `TURBO_KSERVE_MODELS` holds `--model` values
/// separated by `;`, and is read only when no `--model` is given.
pub const ENV_LISTEN: &str = "TURBO_KSERVE_LISTEN";
pub const ENV_MODELS: &str = "TURBO_KSERVE_MODELS";
pub const ENV_MAX_MESSAGE_BYTES: &str = "TURBO_KSERVE_MAX_MESSAGE_BYTES";
pub const ENV_MAX_STREAM_BYTES: &str = "TURBO_KSERVE_MAX_STREAM_BYTES";
pub const ENV_VOCABULARY_DIR: &str = "TURBO_KSERVE_VOCABULARY_DIR";
pub const ENV_CATALOG_DIR: &str = "TURBO_KSERVE_CATALOG_DIR";
pub const ENV_CATALOG_INDEX: &str = "TURBO_KSERVE_CATALOG_INDEX";
pub const ENV_TRAINING_DIR: &str = "TURBO_KSERVE_TRAINING_DIR";
pub const ENV_SINK_ENDPOINT: &str = "TURBO_KSERVE_SINK_ENDPOINT";
pub const ENV_SINK_QUEUE_ITEMS: &str = "TURBO_KSERVE_SINK_QUEUE_ITEMS";
pub const ENV_SINK_QUEUE_BYTES: &str = "TURBO_KSERVE_SINK_QUEUE_BYTES";

pub const USAGE: &str = "usage: turbo-kserve --listen ADDR:PORT [--model bundle=PATH,device=INDEX|select,sessions=N\
[,name=NAME][,precision=PRECISION_MODEL|PRECISION_FASTEST|PRECISION_EXACT][,max_batch=N][,max_seq=N] ...] \
[--bundle-dir DIR] [--max-message-bytes N] [--max-stream-bytes N] [--admin] [--vocabulary-dir PATH] [--catalog-dir PATH [--catalog-index FILE]] \
[--training-dir PATH] [--sink-endpoint URI [--sink-queue-items N] [--sink-queue-bytes N]]\n\
environment: TURBO_KSERVE_LISTEN, TURBO_KSERVE_MODELS (--model values separated by `;`), \
TURBO_KSERVE_BUNDLE_DIR, TURBO_KSERVE_MAX_MESSAGE_BYTES, TURBO_KSERVE_MAX_STREAM_BYTES, TURBO_KSERVE_ADMIN, TURBO_KSERVE_VOCABULARY_DIR, \
TURBO_KSERVE_CATALOG_DIR, TURBO_KSERVE_CATALOG_INDEX, TURBO_KSERVE_TRAINING_DIR, TURBO_KSERVE_SINK_ENDPOINT, TURBO_KSERVE_SINK_QUEUE_ITEMS, \
TURBO_KSERVE_SINK_QUEUE_BYTES; a flag on the command line replaces its variable";

/// Sets a path flag's value, given once and not empty.
fn path_once(flag: &str, v: String, slot: &mut Option<String>) -> Result<(), String> {
    if v.is_empty() || slot.replace(v).is_some() {
        return Err(format!("{flag} needs a path and may be given once"));
    }
    Ok(())
}

/// A flag's value, given once.
fn once(flag: &str, v: String, slot: &mut Option<String>) -> Result<(), String> {
    if v.is_empty() || slot.replace(v).is_some() {
        return Err(format!("{flag} needs a value and may be given once"));
    }
    Ok(())
}

/// The sink settings from their flags or variables: the queue bounds only
/// with an endpoint, each a count over 0.
fn sink(
    endpoint: Option<String>,
    items: Option<String>,
    bytes: Option<String>,
) -> Result<Option<crate::sink::SinkConfig>, String> {
    let count = |name: &str, v: Option<String>, default: u64| -> Result<u64, String> {
        match v {
            None => Ok(default),
            Some(v) => match v.parse::<u64>() {
                Ok(n) if n > 0 => Ok(n),
                _ => Err(format!("{name} {v}: not a count over 0")),
            },
        }
    };
    let Some(endpoint) = endpoint else {
        if items.is_some() || bytes.is_some() {
            return Err("--sink-queue-items and --sink-queue-bytes need --sink-endpoint".into());
        }
        return Ok(None);
    };
    Ok(Some(crate::sink::SinkConfig {
        endpoint,
        max_queue_items: count("--sink-queue-items", items, crate::sink::DEFAULT_SINK_QUEUE_ITEMS as u64)? as usize,
        max_queue_bytes: count("--sink-queue-bytes", bytes, crate::sink::DEFAULT_SINK_QUEUE_BYTES)?,
    }))
}

impl Args {
    /// The command line, with the process environment filling in what it
    /// leaves out.
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
        Args::parse_with(args, |k| std::env::var(k).ok())
    }

    /// `parse` with `env` as the environment.
    pub fn parse_with(
        args: impl IntoIterator<Item = String>,
        env: impl Fn(&str) -> Option<String>,
    ) -> Result<Args, String> {
        let mut listen = None;
        let mut max_message_bytes = None;
        let mut max_stream_bytes = None;
        let mut admin = None;
        let mut vocabulary_dir = None;
        let (mut sink_endpoint, mut sink_items, mut sink_bytes) = (None, None, None);
        let (mut catalog_dir, mut catalog_index) = (None, None);
        let mut training_dir = None;
        let mut bundle_dir = None;
        let mut models = Vec::new();
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            let (flag, inline) = match a.split_once('=') {
                Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
                _ => (a.clone(), None),
            };
            let mut value = || inline.clone().or_else(|| it.next()).ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--listen" => {
                    let v = value()?;
                    let addr = v.parse().map_err(|_| format!("--listen {v}: not ADDR:PORT"))?;
                    if listen.replace(addr).is_some() {
                        return Err("--listen given twice".into());
                    }
                }
                "--model" => models.push(ModelConfig::parse(&value()?)?),
                "--bundle-dir" => {
                    let v = value()?;
                    if v.is_empty() {
                        return Err("--bundle-dir: path is empty".into());
                    }
                    if bundle_dir.replace(v).is_some() {
                        return Err("--bundle-dir given twice".into());
                    }
                }
                "--admin" => {
                    if inline.is_some() || admin.replace(true).is_some() {
                        return Err("--admin takes no value and may be given once".into());
                    }
                }
                "--vocabulary-dir" => path_once(&flag, value()?, &mut vocabulary_dir)?,
                "--sink-endpoint" => once(&flag, value()?, &mut sink_endpoint)?,
                "--sink-queue-items" => once(&flag, value()?, &mut sink_items)?,
                "--sink-queue-bytes" => once(&flag, value()?, &mut sink_bytes)?,
                "--catalog-dir" => path_once(&flag, value()?, &mut catalog_dir)?,
                "--catalog-index" => path_once(&flag, value()?, &mut catalog_index)?,
                "--training-dir" => path_once(&flag, value()?, &mut training_dir)?,
                "--max-message-bytes" => {
                    let v = value()?;
                    let n: usize = v.parse().map_err(|_| format!("--max-message-bytes {v}: not a byte count"))?;
                    if max_message_bytes.replace(n).is_some() {
                        return Err("--max-message-bytes given twice".into());
                    }
                }
                "--max-stream-bytes" => {
                    let v = value()?;
                    let n: u64 = v.parse().map_err(|_| format!("--max-stream-bytes {v}: not a byte count"))?;
                    if max_stream_bytes.replace(n).is_some() {
                        return Err("--max-stream-bytes given twice".into());
                    }
                }
                _ => return Err(format!("unknown argument {a}")),
            }
        }
        if listen.is_none()
            && let Some(v) = env(ENV_LISTEN).filter(|v| !v.is_empty())
        {
            listen = Some(v.parse().map_err(|_| format!("{ENV_LISTEN}={v}: not ADDR:PORT"))?);
        }
        if models.is_empty()
            && let Some(v) = env(ENV_MODELS)
        {
            for m in v.split(';').map(str::trim).filter(|m| !m.is_empty()) {
                models.push(ModelConfig::parse(m).map_err(|e| format!("{ENV_MODELS}: {e}"))?);
            }
        }
        if bundle_dir.is_none()
            && let Some(v) = env(ENV_BUNDLE_DIR).filter(|v| !v.is_empty())
        {
            bundle_dir = Some(v);
        }
        if max_message_bytes.is_none()
            && let Some(v) = env(ENV_MAX_MESSAGE_BYTES).filter(|v| !v.is_empty())
        {
            max_message_bytes = Some(v.parse().map_err(|_| format!("{ENV_MAX_MESSAGE_BYTES}={v}: not a byte count"))?);
        }
        let max_message_bytes = max_message_bytes.unwrap_or(DEFAULT_MAX_MESSAGE_BYTES);
        if max_stream_bytes.is_none()
            && let Some(v) = env(ENV_MAX_STREAM_BYTES).filter(|v| !v.is_empty())
        {
            max_stream_bytes = Some(v.parse().map_err(|_| format!("{ENV_MAX_STREAM_BYTES}={v}: not a byte count"))?);
        }
        let max_stream_bytes = max_stream_bytes.unwrap_or(DEFAULT_MAX_STREAM_BYTES);
        if max_stream_bytes < max_message_bytes as u64 {
            return Err(format!("max stream bytes {max_stream_bytes} is under max message bytes {max_message_bytes}"));
        }
        if admin.is_none()
            && let Some(v) = env(ENV_ADMIN).filter(|v| !v.is_empty())
        {
            admin = Some(match v.as_str() {
                "1" | "true" | "yes" => true,
                "0" | "false" | "no" => false,
                _ => return Err(format!("{ENV_ADMIN}={v}: not one of 1, true, yes, 0, false, no")),
            });
        }
        vocabulary_dir = vocabulary_dir.or_else(|| env(ENV_VOCABULARY_DIR).filter(|v| !v.is_empty()));
        catalog_dir = catalog_dir.or_else(|| env(ENV_CATALOG_DIR).filter(|v| !v.is_empty()));
        catalog_index = catalog_index.or_else(|| env(ENV_CATALOG_INDEX).filter(|v| !v.is_empty()));
        training_dir = training_dir.or_else(|| env(ENV_TRAINING_DIR).filter(|v| !v.is_empty()));
        if catalog_index.is_some() && catalog_dir.is_none() {
            return Err("--catalog-index needs --catalog-dir".into());
        }
        let from_env = |slot: Option<String>, k: &str| slot.or_else(|| env(k).filter(|v| !v.is_empty()));
        let sink = sink(
            from_env(sink_endpoint, ENV_SINK_ENDPOINT),
            from_env(sink_items, ENV_SINK_QUEUE_ITEMS),
            from_env(sink_bytes, ENV_SINK_QUEUE_BYTES),
        )?;
        if let Some(dir) = &bundle_dir {
            discover_bundles(dir, &mut models)?;
        }
        let admin = admin.unwrap_or(false);
        let listen = listen.ok_or("--listen is required")?;
        names_with(&models, admin)?;
        Ok(Args {
            listen,
            models,
            max_message_bytes,
            max_stream_bytes,
            admin,
            vocabulary_dir,
            sink,
            catalog_dir,
            catalog_index,
            training_dir,
        })
    }
}

#[cfg(test)]
mod tests;
