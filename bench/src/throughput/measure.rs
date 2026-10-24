//! What the throughput modes share: their options, the
//! committed text they time over, the median-pass timing, and
//! the provenance every record carries.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use turbo::bundle::sha256_hex;
use turbo::record::{Device, Library, Machine};

use super::{PhaseRun, RecordText, Subject, THROUGHPUT_RECORD_VERSION, ThroughputRecord};
use crate::api::{Runtime, field};
use crate::git::Provenance;
use crate::{Result, utc};

/// `--name value` options plus bare `--flag`s, each at most once.
pub(super) struct Options {
    values: BTreeMap<String, String>,
    flags: Vec<String>,
    pub(super) positional: Vec<String>,
}

impl Options {
    pub(super) fn parse(args: &[String], valued: &[&str], flags: &[&str], usage: &str) -> Result<Options> {
        let mut o = Options { values: BTreeMap::new(), flags: Vec::new(), positional: Vec::new() };
        let mut it = args.iter();
        while let Some(name) = it.next() {
            if flags.contains(&name.as_str()) {
                o.flags.push(name.clone());
            } else if valued.contains(&name.as_str()) {
                let value = it.next().ok_or_else(|| format!("{name}: no value\n{usage}"))?;
                if o.values.insert(name.clone(), value.clone()).is_some() {
                    return Err(format!("{name} given twice"));
                }
            } else if name.starts_with("--") {
                return Err(format!("{name}: not an option\n{usage}"));
            } else {
                o.positional.push(name.clone());
            }
        }
        Ok(o)
    }

    pub(super) fn take(&mut self, name: &str) -> Option<String> {
        self.values.remove(name)
    }

    pub(super) fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|f| f == name)
    }

    pub(super) fn rounds(&mut self) -> Result<u32> {
        let rounds: u32 =
            self.take("--rounds").map_or(Ok(5), |v| v.parse().map_err(|e| format!("--rounds {v}: {e}")))?;
        if rounds == 0 {
            return Err("--rounds 0: nothing would be timed".into());
        }
        Ok(rounds)
    }

    pub(super) fn repo(&mut self) -> PathBuf {
        self.take("--repo").map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."), PathBuf::from)
    }

    pub(super) fn out(&mut self) -> Option<PathBuf> {
        self.take("--out").map(PathBuf::from)
    }
}

/// The goldens inputs of `sets` under the repository at `top`,
/// set by set in name order, each file followed by a line feed,
/// repeated to at least `floor` bytes: the same bytes on every
/// machine.
pub(super) fn goldens_text(top: &Path, sets: &[&str], floor: usize) -> Result<(String, String)> {
    let mut one = String::new();
    for set in sets {
        let dir = top.join("analysis/goldens/data/inputs").join(set);
        let mut names: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::io::Result<_>>()
            .map_err(|e| format!("{}: {e}", dir.display()))?;
        names.retain(|p| p.extension().is_some_and(|x| x == "txt"));
        names.sort();
        if names.is_empty() {
            return Err(format!("{}: no .txt inputs", dir.display()));
        }
        for p in &names {
            one.push_str(&std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?);
            one.push('\n');
        }
    }
    let mut text = String::with_capacity(floor + one.len());
    while text.len() < floor {
        text.push_str(&one);
    }
    Ok((
        text,
        format!(
            "analysis/goldens/data/inputs/{{{}}} in name order, repeated to at least {floor} bytes",
            sets.join(",")
        ),
    ))
}

/// One warm pass, then `rounds` timed passes of `f`; the warm
/// pass's unit count and the median pass's seconds.
pub(super) fn median_pass(rounds: u32, mut f: impl FnMut() -> Result<u64>) -> Result<(u64, f64)> {
    let units = f()?;
    let mut secs: Vec<f64> = Vec::with_capacity(rounds as usize);
    for _ in 0..rounds {
        let t = Instant::now();
        f()?;
        secs.push(t.elapsed().as_secs_f64());
    }
    secs.sort_by(f64::total_cmp);
    Ok((units, secs[secs.len() / 2]))
}

/// A timed phase as a run of the record, over `text`.
pub(super) fn run(phase: &str, units: u64, seconds: f64, text: &str) -> PhaseRun {
    PhaseRun {
        phase: phase.to_owned(),
        units,
        seconds,
        chars_per_second: text.chars().count() as f64 / seconds,
        bytes_per_second: text.len() as f64 / seconds,
    }
}

/// The record around measured runs: machine, CPU device and
/// library read from the runtime, provenance agreed before and
/// after by the caller.
#[allow(clippy::too_many_arguments)]
pub(super) fn record(
    mode: &str,
    subject: Subject,
    source: String,
    text: &str,
    procedure: String,
    command: String,
    rounds: u32,
    runs: Vec<PhaseRun>,
    after: &Provenance,
    recorded_at: String,
) -> Result<ThroughputRecord> {
    let rt = Runtime::create()?;
    let d = rt.cpu()?.ok_or("this build lists no CPU device")?;
    Ok(ThroughputRecord {
        record_version: THROUGHPUT_RECORD_VERSION,
        recorded_at,
        machine: Machine { arch: field(&d.arch), host_cpu: field(&d.name), os: std::env::consts::OS.into() },
        device: Device {
            backend: field(&d.backend),
            kind: crate::kind_name(d.kind).into(),
            name: field(&d.name),
            vendor: field(&d.vendor),
            driver_version: field(&d.driver_version),
            runtime_version: field(&d.runtime_version),
            memory_total: d.memory_total,
        },
        library: Library {
            version: crate::api::version().split(' ').next().unwrap_or("").into(),
            build: crate::api::version(),
            commit: after.commit.clone(),
            pushed_to: after.pushed_to.clone(),
            settings: Vec::new(),
        },
        procedure,
        command,
        mode: mode.to_owned(),
        subject,
        text: RecordText {
            source,
            bytes: text.len() as u64,
            chars: text.chars().count() as u64,
            sha256: sha256_hex(text.as_bytes()),
        },
        rounds,
        runs,
    })
}

/// The timestamp a mode stamps before it measures.
pub(super) fn now() -> String {
    utc(SystemTime::now())
}

/// The out directory a record goes to: `--out`, or the
/// throughput records directory of the repository measured.
pub(super) fn out_dir(out: Option<PathBuf>, after: &Provenance) -> PathBuf {
    out.unwrap_or_else(|| Path::new(&after.top).join(super::THROUGHPUT_DIR))
}

/// Refuse a run whose tree moved while it measured.
pub(super) fn same_tree(before: &Provenance, after: &Provenance) -> Result<()> {
    if after != before {
        return Err(format!("the working tree moved from {} to {} during the run", before.commit, after.commit));
    }
    Ok(())
}
