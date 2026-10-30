//! The command line of `turbo-train-head`: its arguments, the encoder
//! settings they give, and the output directory a run claims.

use std::path::PathBuf;

use turbo::manifest::WordPooling;
use turbo_analysis_embed::{Device, Precision};

use turbo::manifest::HashEmbed;
use turbo_trainer::Config;

use crate::run::HASH_ROWS;
use crate::{Error, Result};

pub struct Args {
    pub rest: Vec<String>,
}

impl Args {
    pub fn flag(&mut self, name: &str) -> bool {
        let at = self.rest.iter().position(|a| a == name);
        at.map(|i| self.rest.remove(i)).is_some()
    }

    pub fn value(&mut self, name: &str) -> Result<Option<String>> {
        let Some(i) = self.rest.iter().position(|a| a == name) else { return Ok(None) };
        if i + 1 >= self.rest.len() {
            return Err(Error(format!("{name} needs a value")));
        }
        let v = self.rest.remove(i + 1);
        self.rest.remove(i);
        Ok(Some(v))
    }

    pub fn required(&mut self, name: &str) -> Result<String> {
        self.value(name)?.ok_or_else(|| Error(format!("{name} is required")))
    }

    pub fn parsed<T: std::str::FromStr>(&mut self, name: &str, default: T) -> Result<T> {
        match self.value(name)? {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| Error(format!("{name} {v:?} is not a valid value"))),
        }
    }

    pub fn done(self) -> Result<()> {
        if self.rest.is_empty() { Ok(()) } else { Err(Error(format!("unknown arguments: {}", self.rest.join(" ")))) }
    }
}

/// --device, --precision and --pooling.
pub fn encoder_args(a: &mut Args) -> Result<(Device, Precision, WordPooling)> {
    let device = match a.value("--device")?.as_deref() {
        None | Some("cpu") => Device::Cpu,
        Some(d) => Device::Backend(d.to_owned()),
    };
    let precision = match a.value("--precision")?.as_deref() {
        None | Some("model") => Precision::Model,
        Some("exact") => Precision::Exact,
        Some(p) => return Err(Error(format!("--precision {p:?}: model or exact"))),
    };
    let pooling = match a.value("--pooling")?.as_deref() {
        None | Some("first") => WordPooling::First,
        Some("mean") => WordPooling::Mean,
        Some(p) => return Err(Error(format!("--pooling {p:?}: first or mean"))),
    };
    Ok((device, precision, pooling))
}

/// A run's output directory, claimed before the run reads or writes
/// anything: `create_dir` fails when the directory exists, so two runs
/// given one `--out` cannot both have it. Dropped unkept, as when the run
/// fails, it is removed with what the run wrote in it.
pub struct OutDir {
    pub path: PathBuf,
    pub keep: bool,
}

impl OutDir {
    pub fn claim(p: &str) -> Result<OutDir> {
        let path = PathBuf::from(p);
        if let Some(parent) = path.parent().filter(|q| !q.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| Error(format!("{}: {e}", parent.display())))?;
        }
        match std::fs::create_dir(&path) {
            Ok(()) => Ok(OutDir { path, keep: false }),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                Err(Error(format!("{}: already there; a run writes a new directory", path.display())))
            }
            Err(e) => Err(Error(format!("{}: {e}", path.display()))),
        }
    }
}

impl Drop for OutDir {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

pub fn commit(given: Option<String>) -> String {
    given
        .or_else(|| {
            let o = std::process::Command::new("git").args(["rev-parse", "HEAD"]).output().ok()?;
            o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".into())
}

/// The head's shape and the training settings a run's arguments give.
pub struct HeadArgs {
    pub crf: bool,
    pub bilstm: Option<(u32, u32)>,
    pub hash: Option<HashEmbed>,
    pub config: Config,
}

/// --crf, --bilstm, --hash-embed, --hash-rows and the training settings.
pub fn head_args(a: &mut Args) -> Result<HeadArgs> {
    let crf = a.flag("--crf");
    let bilstm = match a.value("--bilstm")? {
        None => None,
        Some(v) => {
            let parse = |s: &str| s.parse::<u32>().map_err(|_| Error(format!("--bilstm {v:?}: <hidden>x<layers>")));
            let (h, l) = v.split_once('x').ok_or_else(|| Error(format!("--bilstm {v:?}: <hidden>x<layers>")))?;
            Some((parse(h)?, parse(l)?))
        }
    };
    let hash_dim: Option<u32> =
        a.value("--hash-embed")?.map(|v| v.parse()).transpose().map_err(|_| Error("--hash-embed <dim>".into()))?;
    let rows = match a.value("--hash-rows")? {
        None => HASH_ROWS,
        Some(v) => {
            let r: Vec<u32> = v.split(',').map(str::parse).collect::<std::result::Result<_, _>>().unwrap_or_default();
            r.try_into().map_err(|_| Error(format!("--hash-rows {v:?}: four counts, comma separated")))?
        }
    };
    let hash = hash_dim.map(|dim| HashEmbed { dim, rows });
    let d = Config::default();
    let c = Config {
        seed: a.parsed("--seed", d.seed)?,
        epochs: a.parsed("--epochs", d.epochs)?,
        batch: a.parsed("--batch", d.batch)?,
        lr: a.parsed("--lr", d.lr)?,
        patience: a.parsed("--patience", if bilstm.is_some() { BILSTM_PATIENCE } else { d.patience })?,
        tolerance: a.parsed("--tolerance", d.tolerance)?,
        threads: a.parsed("--threads", d.threads)?,
        verbose: true,
        ..d
    };
    Ok(HeadArgs { crf, bilstm, hash, config: c })
}

/// The default patience of a head with a BiLSTM adapter: its dev score
/// keeps creeping up by less than the tolerance an epoch for many epochs
/// (MASC POS, hashed words and a 256x1 BiLSTM: 0.9733 at epoch 32, 0.9755
/// at 57), and each better epoch is saved as it comes, so waiting costs
/// only time.
const BILSTM_PATIENCE: u32 = 15;
