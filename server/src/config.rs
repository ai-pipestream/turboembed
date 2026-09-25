//! The per-model configuration (docs/kserve.md, Serving a bundle), from the
//! command line: one `--model` per model,
//! `bundle=PATH,device=INDEX|select,sessions=N[,precision=P][,max_batch=N][,max_seq=N]`.

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
    /// The model name: the last component of the bundle path, exactly as it
    /// is spelled. None for a path whose last component is empty, `.` or
    /// `..`, which names no bundle.
    pub fn name(&self) -> Option<&str> {
        let last = self.bundle.rsplit(std::path::is_separator).next().unwrap_or("");
        (!matches!(last, "" | "." | "..")).then_some(last)
    }

    /// One `--model` value.
    pub fn parse(s: &str) -> Result<ModelConfig, String> {
        let (mut bundle, mut device, mut sessions) = (None, None, None);
        let (mut precision, mut max_batch, mut max_seq) = (None, None, None);
        for part in s.split(',') {
            let (k, v) = part.split_once('=').ok_or_else(|| format!("--model {s}: `{part}` is not key=value"))?;
            let seen = match k {
                "bundle" => bundle.replace(v.to_string()).is_some(),
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

/// Every model's name, refusing a path that names no bundle and two models
/// of one name.
pub fn names(models: &[ModelConfig]) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for m in models {
        let name = m.name().ok_or_else(|| format!("bundle {}: the path names no bundle", m.bundle))?;
        if out.iter().any(|n| n == name) {
            return Err(format!("bundle {}: a model named {name} is already served", m.bundle));
        }
        out.push(name.to_string());
    }
    Ok(out)
}

/// The command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    pub listen: SocketAddr,
    pub models: Vec<ModelConfig>,
    /// The largest request message gRPC reads.
    pub max_message_bytes: usize,
}

/// `--max-message-bytes` when absent: 64 MiB.
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 64 << 20;

pub const USAGE: &str = "usage: turbo-kserve --listen ADDR:PORT --model bundle=PATH,device=INDEX|select,sessions=N\
[,precision=PRECISION_MODEL|PRECISION_FASTEST|PRECISION_EXACT][,max_batch=N][,max_seq=N] [--model ...] [--max-message-bytes N]";

impl Args {
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
        let mut listen = None;
        let mut max_message_bytes = None;
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
                    listen = Some(v.parse().map_err(|_| format!("--listen {v}: not ADDR:PORT"))?);
                }
                "--model" => models.push(ModelConfig::parse(&value()?)?),
                "--max-message-bytes" => {
                    let v = value()?;
                    let n: usize = v.parse().map_err(|_| format!("--max-message-bytes {v}: not a byte count"))?;
                    if max_message_bytes.replace(n).is_some() {
                        return Err("--max-message-bytes given twice".into());
                    }
                }
                _ => return Err(format!("unknown argument {a}")),
            }
        }
        let listen = listen.ok_or("--listen is required")?;
        if models.is_empty() {
            return Err("at least one --model is required".into());
        }
        names(&models)?;
        Ok(Args { listen, models, max_message_bytes: max_message_bytes.unwrap_or(DEFAULT_MAX_MESSAGE_BYTES) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Result<Args, String> {
        Args::parse(s.split_whitespace().map(String::from))
    }

    #[test]
    fn a_model() {
        let a = args("--listen 127.0.0.1:0 --model bundle=/b/tiny,device=0,sessions=2").unwrap();
        assert_eq!(a.max_message_bytes, 64 * 1024 * 1024);
        let b = args("--listen 127.0.0.1:0 --max-message-bytes 1024 --model bundle=/b/tiny,device=0,sessions=2");
        assert_eq!(b.unwrap().max_message_bytes, 1024);
        assert_eq!(
            a.models,
            [ModelConfig {
                bundle: "/b/tiny".into(),
                device: Device::Index(0),
                precision: 0,
                max_batch: 0,
                max_seq: 0,
                sessions: 2
            }]
        );
        let a = args(
            "--listen=[::1]:8001 --model=bundle=x,device=select,sessions=1,precision=PRECISION_EXACT,max_batch=8,max_seq=128",
        )
        .unwrap();
        let m = &a.models[0];
        assert_eq!((m.device, m.precision, m.max_batch, m.max_seq), (Device::Select, 2, 8, 128));
        assert_eq!(m.name(), Some("x"));
    }

    #[test]
    fn refusals() {
        for bad in [
            "--model bundle=x,device=0,sessions=1",
            "--listen 127.0.0.1:0",
            "--listen 127.0.0.1:0 --model bundle=x,device=0",
            "--listen 127.0.0.1:0 --model bundle=x,sessions=1",
            "--listen 127.0.0.1:0 --model device=0,sessions=1",
            "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=0",
            "--listen 127.0.0.1:0 --model bundle=x,device=cpu,sessions=1",
            "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1,precision=TURBO_PRECISION_EXACT",
            "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1,precision=precision_exact",
            "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1,max_batch=-1",
            "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1,sessions=2",
            "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1,colour=red",
            "--listen 127.0.0.1:0 --model bundle=a/x,device=0,sessions=1 --model bundle=b/x,device=0,sessions=1",
            "--listen 127.0.0.1:0 --model bundle=a/x/,device=0,sessions=1",
            "--listen 127.0.0.1:0 --model bundle=a/.,device=0,sessions=1",
            "--listen 127.0.0.1:0 --model bundle=..,device=0,sessions=1",
            "--listen nowhere --model bundle=x,device=0,sessions=1",
            "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1 --max-message-bytes -1",
            "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1 --max-message-bytes 64MiB",
            "--listen 127.0.0.1:0 --model bundle=x,device=0,sessions=1 --max-message-bytes 1 --max-message-bytes 2",
        ] {
            assert!(args(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_name_is_the_last_component_as_spelled() {
        let m =
            |b: &str| ModelConfig { bundle: b.into(), ..ModelConfig::parse("bundle=x,device=0,sessions=1").unwrap() };
        assert_eq!(m("testdata/Tiny-Bert").name(), Some("Tiny-Bert"));
        assert_eq!(m("tiny").name(), Some("tiny"));
        assert_eq!(m("/a/b/").name(), None);
        assert_eq!(m("").name(), None);
        assert_eq!(m("a/..").name(), None);
    }
}
