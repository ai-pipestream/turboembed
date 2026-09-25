//! The benchmark tool: measures the library on one device through its C
//! interface, runs the pinned reference programs on the same token rows,
//! and writes the record docs/benchmarks.md describes. Nothing in a
//! record is typed by hand: every figure is measured, every hash read,
//! and the commit is the clean, pushed one the library was built from.

pub mod api;
pub mod docker;
pub mod git;
pub mod measure;
pub mod tei;
pub mod tensorrt;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use turbo::record::{self, BundleId, Device, Library, Machine, Record, ReferenceRun};
use turbo::{TURBO_DEVICE_CPU, TURBO_DEVICE_GPU, TURBO_DEVICE_IGPU, TURBO_DEVICE_NPU, TURBO_TASK_EMBED};

use api::field;
use git::Provenance;
use measure::Measurement;

pub type Result<T> = std::result::Result<T, String>;

/// A time as `YYYY-MM-DDTHH:MM:SSZ`, UTC.
pub fn utc(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let (days, rem) = ((secs / 86400) as i64, secs % 86400);
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn kind_name(kind: u32) -> &'static str {
    match kind {
        TURBO_DEVICE_CPU => "DEVICE_CPU",
        TURBO_DEVICE_GPU => "DEVICE_GPU",
        TURBO_DEVICE_IGPU => "DEVICE_IGPU",
        TURBO_DEVICE_NPU => "DEVICE_NPU",
        _ => "",
    }
}

/// The record of a measurement, its references and where the library
/// came from, checked as the core will check it.
pub fn record(m: &Measurement, p: &Provenance, references: Vec<ReferenceRun>, recorded_at: String) -> Result<Record> {
    let d = &m.device;
    let (speed_ratio, speed_reference) = record::speed(m.timing.p50_ms, &references);
    let r = Record {
        record_version: record::RECORD_VERSION,
        recorded_at,
        machine: Machine { arch: field(&d.arch), host_cpu: m.host_cpu.clone(), os: std::env::consts::OS.into() },
        device: Device {
            backend: field(&d.backend),
            kind: kind_name(d.kind).into(),
            name: field(&d.name),
            vendor: field(&d.vendor),
            driver_version: field(&d.driver_version),
            runtime_version: field(&d.runtime_version),
            memory_total: d.memory_total,
        },
        library: Library {
            version: m.build.split(' ').next().unwrap_or("").into(),
            build: m.build.clone(),
            commit: p.commit.clone(),
            pushed_to: p.pushed_to.clone(),
        },
        task: record::task_name(TURBO_TASK_EMBED).unwrap().into(),
        precision: record::precision_name(m.precision).ok_or("precision is not a TURBO_PRECISION_* value")?.into(),
        compute_dtype: record::dtype_name(m.compute_dtype)
            .ok_or_else(|| format!("compute dtype {} has no name in a record", m.compute_dtype))?
            .into(),
        bundle: BundleId {
            model_id: field(&m.model.model_id),
            revision: field(&m.model.revision),
            manifest_sha256: field(&m.model.manifest_sha256),
            artifact_sha256: field(&m.model.artifact_sha256),
            tokenizer_sha256: field(&m.model.tokenizer_sha256),
        },
        rows: record::Rows {
            batch: m.rows.batch,
            seq: m.rows.seq,
            live_tokens: m.rows.live_tokens(),
            cases: m.rows.cases.clone(),
            sha256: m.rows.sha256(),
        },
        timing: m.timing.clone(),
        conformance: m.conformance.clone(),
        references,
        speed_ratio,
        speed_reference,
    };
    r.check()?;
    Ok(r)
}

/// Write the record under its own name in `dir`, through the core's
/// parser first; an existing record of that name is never replaced.
pub fn write(r: &Record, dir: &Path) -> Result<PathBuf> {
    let name = record::file_name(r)?;
    let mut bytes = serde_json::to_vec_pretty(r).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    Record::parse(&name, &bytes)?;
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(&name);
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|e| format!("{}: {e}; a record is never replaced", path.display()))?;
    std::io::Write::write_all(&mut f, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(path)
}
