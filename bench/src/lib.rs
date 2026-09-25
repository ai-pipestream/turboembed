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

/// The commit this tool and the library in it were built from (bench/build.rs);
/// empty when the build was not in a git tree.
pub const BUILD_COMMIT: &str = env!("TURBO_BENCH_BUILD_COMMIT");

/// The working tree's changes when this was built, as `git status
/// --porcelain` gives them; empty when it was clean.
pub const BUILD_CHANGES: &str = env!("TURBO_BENCH_BUILD_CHANGES");

/// A record may name `commit` only when this build is that commit, with
/// no changes: otherwise the library measured is not the one it names.
pub fn check_build(commit: &str, built_from: &str, changes: &str) -> Result<()> {
    if built_from.is_empty() {
        return Err("this tool was not built in a git working tree, so no commit names the library in it".into());
    }
    if built_from != commit {
        return Err(format!(
            "this tool and its library were built from {built_from}, and --repo is at {commit}: build and run it \
             from that tree (cargo run --release -p turbo-bench)"
        ));
    }
    if !changes.is_empty() {
        return Err(format!(
            "this tool was built from {built_from} with changes in the working tree ({changes}): the library \
             measured is not that commit; rebuild from the clean tree"
        ));
    }
    Ok(())
}

/// A bundle under the repository's testdata/ is a test fixture, not a
/// model anyone deploys: no record is made of it. `bundle` and each
/// `testdata` are compared as canonical paths, and so is the manifest,
/// so a link into testdata is refused too.
pub fn check_not_testdata(bundle: &Path, testdata: &[PathBuf]) -> Result<()> {
    let dir = fs::canonicalize(bundle).map_err(|e| format!("--bundle {}: {e}", bundle.display()))?;
    let manifest = fs::canonicalize(bundle.join("manifest.json")).unwrap_or_else(|_| dir.clone());
    for t in testdata {
        let Ok(t) = fs::canonicalize(t) else { continue };
        if dir.starts_with(&t) || manifest.starts_with(&t) {
            return Err(format!(
                "--bundle {}: a test fixture under {}; a record is of a bundle made for use (bundle/README.md)",
                bundle.display(),
                t.display()
            ));
        }
    }
    Ok(())
}

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

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use super::utc;

    #[test]
    fn utc_gives_the_civil_time_of_known_epochs() {
        for (secs, want) in [
            (0, "1970-01-01T00:00:00Z"),
            (951_782_399, "2000-02-28T23:59:59Z"),
            (951_782_400, "2000-02-29T00:00:00Z"),
            (1_790_294_723, "2026-09-25T00:05:23Z"),
            (4_107_542_400, "2100-03-01T00:00:00Z"),
        ] {
            assert_eq!(utc(UNIX_EPOCH + Duration::from_secs(secs)), want, "{secs}");
        }
    }
}
