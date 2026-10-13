//! The two commands of `turbo-eval` as functions: run one pipeline over
//! named sets and write a record per set, and read two directories of
//! records for the gate.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::Result;
use crate::gate::{GateReport, gate};
use crate::pipeline::{Pipeline, Source};
use crate::record::EvalRecord;
use crate::set::LoadedSet;
use crate::source::utc;

/// Reads the set of this name from under the cache root; the binary's
/// loader is `sets::find(name)?.load(cache)`.
pub type SetLoader = fn(&str, &Path) -> Result<LoadedSet>;

/// The file a set's record is written to: one directory holds one
/// pipeline's records, one per set.
pub fn record_path(out: &Path, set: &str) -> PathBuf {
    out.join(format!("{set}.json"))
}

/// Runs the pipeline over each set in turn and writes its record under
/// `out`, which is made if missing. A record already there is never
/// replaced: the run stops with an error before the set is loaded.
pub fn run_sets(
    pipeline: &mut Pipeline,
    sets: &[String],
    cache: &Path,
    load: SetLoader,
    out: &Path,
    source: &Source,
) -> Result<Vec<PathBuf>> {
    fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut written = Vec::new();
    for name in sets {
        let path = record_path(out, name);
        if path.exists() {
            return Err(format!("{}: a record is never replaced", path.display()));
        }
        let set = load(name, cache).map_err(|e| format!("set {name}: {e}"))?;
        if set.name != *name {
            return Err(format!("set {name}: the reader returned set {}", set.name));
        }
        if !set.unplaced.is_empty() {
            eprintln!(
                "set {name}: {} judgements could not be placed in their document and are left out",
                set.unplaced.len()
            );
        }
        let record = pipeline.record(&set, source, utc(SystemTime::now())).map_err(|e| format!("set {name}: {e}"))?;
        record.write(&path)?;
        written.push(path);
    }
    Ok(written)
}

/// Every record in `dir` (each `*.json` file), in file name order.
pub fn read_records(dir: &Path) -> Result<Vec<EvalRecord>> {
    let mut paths: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .map(|e| e.map(|e| e.path()).map_err(|e| format!("{}: {e}", dir.display())))
        .collect::<Result<Vec<_>>>()?;
    paths.retain(|p| p.extension().is_some_and(|x| x == "json"));
    paths.sort();
    if paths.is_empty() {
        return Err(format!("{}: no records (*.json)", dir.display()));
    }
    paths.iter().map(|p| EvalRecord::read(p)).collect()
}

/// The gate over two directories of records.
pub fn gate_dirs(candidate: &Path, baseline: &Path) -> Result<GateReport> {
    gate(&read_records(candidate)?, &read_records(baseline)?)
}
