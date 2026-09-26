//! Load-time kernel choice, the core's side (docs/autotune.md): which
//! numeric classes a precision allows a backend's kernels, whether and for
//! how long a session is tuned, and what the choices are called in a
//! record. The backend owns its variants, its timing and its choices
//! string; the core never parses the string.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::backend::{TURBO_NUMERIC_F16_CHUNKACC, TURBO_NUMERIC_F16_F32ACC, TURBO_NUMERIC_F32_FMA};
use crate::bundle::sha256_hex;
use crate::status::{Error, INVALID_ARGUMENT, INVALID_ENUM, Result};
use crate::{
    TURBO_AUTOTUNE_OFF, TURBO_AUTOTUNE_ON, TURBO_AUTOTUNE_RETUNE, TURBO_AUTOTUNE_RUNTIME, TURBO_PRECISION_EXACT,
    TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL, TURBO_TUNED_CACHE, TURBO_TUNED_DEFAULT, TURBO_TUNED_FORCED,
    TURBO_TUNED_MEASURED,
};

/// The numeric classes each precision allows, as TURBO_NUMERIC_* bits.
/// Widened only by a decision recorded with its measurements: F16 sums
/// within a chunk at FASTEST were added on 2026-09-26, when a whole-k
/// F16-sums tile measured 8-12% faster than the F32-sums one on an RTX
/// 4080 SUPER with the reference cosine at 0.999998, three orders of
/// magnitude inside FASTEST's bound; TF32 at MODEL is not in it. A
/// backend cannot widen it; an environment variable of the backend's
/// widens the backend's own copy for the experiment it names, and such a
/// session is never cached.
pub const NUMERICS_ALLOWED: [(u32, u32); 3] = [
    (TURBO_PRECISION_EXACT, TURBO_NUMERIC_F32_FMA),
    (TURBO_PRECISION_MODEL, TURBO_NUMERIC_F32_FMA),
    (TURBO_PRECISION_FASTEST, TURBO_NUMERIC_F16_F32ACC | TURBO_NUMERIC_F16_CHUNKACC),
];

/// The classes `precision` allows.
pub fn numerics_allowed(precision: u32) -> u32 {
    NUMERICS_ALLOWED.iter().find(|(p, _)| *p == precision).map_or(0, |&(_, n)| n)
}

/// The budget without a disk cache: a process pays it at its first
/// session of each shape, so the backend times the GEMM tiles alone.
pub const BUDGET_MEMORY_MS: u32 = 150;
/// The budget with a disk cache directory, where a measurement outlives
/// the process.
pub const BUDGET_DISK_MS: u32 = 750;

/// A session's tuning mode: turbo_session_desc.tuning, or for RUNTIME the
/// TURBO_AUTOTUNE variable (`off`, `on`, `retune`), unset being OFF.
pub fn mode(asked: u32) -> Result<u32> {
    match asked {
        TURBO_AUTOTUNE_OFF | TURBO_AUTOTUNE_ON | TURBO_AUTOTUNE_RETUNE => Ok(asked),
        TURBO_AUTOTUNE_RUNTIME => match std::env::var("TURBO_AUTOTUNE") {
            Err(_) => Ok(TURBO_AUTOTUNE_OFF),
            Ok(v) => match v.as_str() {
                "off" => Ok(TURBO_AUTOTUNE_OFF),
                "on" => Ok(TURBO_AUTOTUNE_ON),
                "retune" => Ok(TURBO_AUTOTUNE_RETUNE),
                _ => Err(Error::field(
                    INVALID_ARGUMENT,
                    4,
                    format!("tuning: TURBO_AUTOTUNE is {v:?}, not off, on or retune"),
                )),
            },
        },
        _ => Err(Error::new(INVALID_ENUM, format!("tuning: {asked} is not a TURBO_AUTOTUNE_* value"))),
    }
}

/// The time a session in `mode` may spend measuring: turbo_session_desc's
/// tuning_budget_ms, else the TURBO_AUTOTUNE_BUDGET_MS variable, else
/// BUDGET_DISK_MS with a disk cache and BUDGET_MEMORY_MS without. 0 when
/// OFF.
pub fn budget(mode: u32, asked: u32, disk: bool) -> Result<u32> {
    if mode == TURBO_AUTOTUNE_OFF {
        return Ok(0);
    }
    if asked != 0 {
        return Ok(asked);
    }
    match std::env::var("TURBO_AUTOTUNE_BUDGET_MS") {
        Ok(v) => v.parse::<u32>().ok().filter(|&n| n > 0).ok_or_else(|| {
            Error::field(
                INVALID_ARGUMENT,
                5,
                format!("tuning_budget_ms: TURBO_AUTOTUNE_BUDGET_MS is {v:?}, not a count of milliseconds"),
            )
        }),
        Err(_) => Ok(if disk { BUDGET_DISK_MS } else { BUDGET_MEMORY_MS }),
    }
}

/// turbo_session_info.tuned as a record's settings spell it.
pub fn tuned_name(tuned: u32) -> Option<&'static str> {
    match tuned {
        TURBO_TUNED_DEFAULT => Some("default"),
        TURBO_TUNED_FORCED => Some("forced"),
        TURBO_TUNED_MEASURED => Some("measured"),
        TURBO_TUNED_CACHE => Some("cache"),
        _ => None,
    }
}

/// The knobs a choices string names as forced: its `forced=` item's
/// list, empty when it has none or the item is empty.
pub fn forced_knobs(choices: &str) -> Vec<&str> {
    choices
        .split(';')
        .find_map(|item| item.strip_prefix("forced="))
        .map(|l| l.split(',').filter(|k| !k.is_empty()).collect())
        .unwrap_or_default()
}

/// This build's kernels: the crate's version and a hash of the backends'
/// kernel sources, so an entry measured on other kernels is never taken.
pub const LIBRARY_BUILD: &str = concat!(env!("CARGO_PKG_VERSION"), "+", env!("TURBO_KERNELS_ID"));

/// The packed-token bins' upper edges, as the backends bin a run's tokens;
/// past the last is one bin more.
const BIN_EDGES: [u32; 4] = [256, 1024, 4096, 16384];

/// The highest token bin a session of `tokens` packed tokens at most has.
pub fn tcap_bin(tokens: u32) -> u32 {
    BIN_EDGES.iter().take_while(|&&e| e < tokens).count() as u32
}

/// max_seq rounded up to a power of two, 64 at least: attention's chunk.
pub fn max_seq_bin(max_seq: u32) -> u32 {
    max_seq.max(64).next_power_of_two()
}

/// Everything that decides a session's kernel choice or changes its
/// bits: any field different is a different entry.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuneKey {
    pub backend: String,
    pub device_name: String,
    pub arch: String,
    pub driver_version: String,
    pub runtime_version: String,
    pub library_build: String,
    pub manifest_sha256: String,
    pub artifact_sha256: String,
    pub precision: u32,
    pub compute_dtype: u32,
    /// The core's classes for the precision (NUMERICS_ALLOWED); a session
    /// whose backend widened them is never cached.
    pub numerics_allowed: u32,
    pub tcap_bin: u32,
    pub max_seq_bin: u32,
}

/// A measured choice: the backend's choices string, opaque to the core,
/// and what it was measured on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuneEntry {
    pub key: TuneKey,
    pub choices: String,
    /// TURBO_TUNED_MEASURED: how the choice was made.
    pub source: u32,
    /// When, as `YYYY-MM-DDTHH:MM:SSZ`.
    pub tuned_at: String,
    pub tune_ms: u32,
    /// `<bin>/<knob>/<variant>` and its least time in milliseconds, as
    /// the backend reported them.
    pub timings: Vec<(String, f32)>,
}

impl TuneEntry {
    /// The entry for a session measured now; `timings` as the backend
    /// writes them, `<name>=<ms>` lines.
    pub fn measured(key: TuneKey, choices: &str, tune_ms: u32, timings: &str) -> TuneEntry {
        let timings = timings
            .lines()
            .filter_map(|l| l.split_once('='))
            .filter_map(|(n, ms)| ms.parse::<f32>().ok().map(|ms| (n.to_owned(), ms)))
            .collect();
        TuneEntry {
            key,
            choices: choices.to_owned(),
            source: TURBO_TUNED_MEASURED,
            tuned_at: utc(SystemTime::now()),
            tune_ms,
            timings,
        }
    }
}

/// A time as `YYYY-MM-DDTHH:MM:SSZ`, UTC.
fn utc(t: SystemTime) -> String {
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

/// What the tests set in place of TURBO_AUTOTUNE_CACHE: `Some(None)` for
/// memory only.
static CACHE_DIR: Mutex<Option<Option<PathBuf>>> = Mutex::new(None);

/// The disk cache's directory: TURBO_AUTOTUNE_CACHE, or the tests' in its
/// place; `None`, memory only. Never a default path: the library writes
/// nowhere it was not told to.
pub fn cache_dir() -> Option<PathBuf> {
    if let Some(d) = CACHE_DIR.lock().unwrap_or_else(|e| e.into_inner()).clone() {
        return d;
    }
    std::env::var_os("TURBO_AUTOTUNE_CACHE").filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// Sets the disk cache's directory in place of TURBO_AUTOTUNE_CACHE for
/// the sessions made next: `Some(None)` memory only, `None` the variable
/// again. Built only with `internals`.
#[cfg(feature = "internals")]
pub fn use_cache_dir(dir: Option<Option<&Path>>) {
    *CACHE_DIR.lock().unwrap_or_else(|e| e.into_inner()) = dir.map(|d| d.map(Path::to_path_buf));
}

/// The file an entry of `key` lives in under `dir`: the backend's name,
/// then the first 16 hex digits of the SHA-256 of the key's JSON.
pub fn cache_file(dir: &Path, key: &TuneKey) -> PathBuf {
    let json = serde_json::to_vec(key).unwrap_or_default();
    dir.join(format!("{}-{}.json", key.backend, &sha256_hex(&json)[..16]))
}

/// A runtime's measured choices: in memory for the runtime's life, and
/// one JSON file per entry in a directory when one is named.
#[derive(Default)]
pub struct Cache {
    map: Mutex<HashMap<TuneKey, TuneEntry>>,
}

impl Cache {
    /// The entry for `key`: the runtime's own, else the file's under
    /// `dir`. A file that is missing, does not parse or holds another key
    /// is no entry.
    pub fn get(&self, key: &TuneKey, dir: Option<&Path>) -> Option<TuneEntry> {
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(e) = map.get(key) {
            return Some(e.clone());
        }
        let bytes = std::fs::read(cache_file(dir?, key)).ok()?;
        let e: TuneEntry = serde_json::from_slice(&bytes).ok()?;
        if e.key != *key || e.source != TURBO_TUNED_MEASURED {
            return None;
        }
        map.insert(key.clone(), e.clone());
        Some(e)
    }

    /// `e` into memory, and into its file under `dir` when one is named:
    /// written beside it and renamed over it, so a reader sees the old
    /// file or the new one whole.
    pub fn put(&self, e: TuneEntry, dir: Option<&Path>) -> std::io::Result<()> {
        let written = match dir {
            None => Ok(()),
            Some(d) => {
                let file = cache_file(d, &e.key);
                let tmp = file.with_extension(format!("json.{}.tmp", std::process::id()));
                std::fs::create_dir_all(d)
                    .and_then(|_| std::fs::write(&tmp, serde_json::to_vec_pretty(&e).unwrap_or_default()))
                    .and_then(|_| std::fs::rename(&tmp, &file))
                    .inspect_err(|_| {
                        let _ = std::fs::remove_file(&tmp);
                    })
            }
        };
        self.map.lock().unwrap_or_else(|e| e.into_inner()).insert(e.key.clone(), e);
        written
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TURBO_NUMERIC_TF32;

    /// The table a decision changes: EXACT and MODEL allow F32 FMAs only,
    /// FASTEST F16 operands with F32 sums and, by the decision of
    /// 2026-09-26, F16 sums within a chunk; TF32 is in no precision's set.
    #[test]
    fn the_classes_are_the_decided_ones() {
        assert_eq!(
            NUMERICS_ALLOWED,
            [
                (TURBO_PRECISION_EXACT, TURBO_NUMERIC_F32_FMA),
                (TURBO_PRECISION_MODEL, TURBO_NUMERIC_F32_FMA),
                (TURBO_PRECISION_FASTEST, TURBO_NUMERIC_F16_F32ACC | TURBO_NUMERIC_F16_CHUNKACC),
            ]
        );
        for p in [TURBO_PRECISION_EXACT, TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
            assert_eq!(numerics_allowed(p) & TURBO_NUMERIC_TF32, 0, "precision {p}");
        }
        assert_eq!(numerics_allowed(TURBO_PRECISION_MODEL) & TURBO_NUMERIC_F16_CHUNKACC, 0);
    }

    #[test]
    fn a_mode_asked_for_is_taken_and_an_unknown_one_refused() {
        for m in [TURBO_AUTOTUNE_OFF, TURBO_AUTOTUNE_ON, TURBO_AUTOTUNE_RETUNE] {
            assert_eq!(mode(m).unwrap(), m);
        }
        assert_eq!(mode(4).unwrap_err().code, INVALID_ENUM);
    }

    #[test]
    fn the_budget_is_what_was_asked_else_by_where_the_cache_is() {
        assert_eq!(budget(TURBO_AUTOTUNE_OFF, 500, true).unwrap(), 0);
        assert_eq!(budget(TURBO_AUTOTUNE_ON, 40, false).unwrap(), 40);
        if std::env::var_os("TURBO_AUTOTUNE_BUDGET_MS").is_none() {
            assert_eq!(budget(TURBO_AUTOTUNE_ON, 0, false).unwrap(), BUDGET_MEMORY_MS);
            assert_eq!(budget(TURBO_AUTOTUNE_RETUNE, 0, true).unwrap(), BUDGET_DISK_MS);
        }
    }

    fn key() -> TuneKey {
        TuneKey {
            backend: "cuda".into(),
            device_name: "a device".into(),
            arch: "sm89".into(),
            driver_version: "1.0".into(),
            runtime_version: "12.4".into(),
            library_build: LIBRARY_BUILD.into(),
            manifest_sha256: "00".into(),
            artifact_sha256: "11".into(),
            precision: TURBO_PRECISION_FASTEST,
            compute_dtype: crate::TURBO_DTYPE_F16,
            numerics_allowed: TURBO_NUMERIC_F16_F32ACC,
            tcap_bin: tcap_bin(8192),
            max_seq_bin: max_seq_bin(256),
        }
    }

    #[test]
    fn the_bins_of_a_key_are_the_backends() {
        assert_eq!([tcap_bin(1), tcap_bin(256), tcap_bin(257), tcap_bin(8192), tcap_bin(16385)], [0, 0, 1, 3, 4]);
        assert_eq!([max_seq_bin(1), max_seq_bin(64), max_seq_bin(65), max_seq_bin(512)], [64, 64, 128, 512]);
        assert_eq!(utc(UNIX_EPOCH + std::time::Duration::from_secs(1_790_294_400)), "2026-09-25T00:00:00Z");
    }

    /// An entry is found in memory and in its file, which holds it as
    /// JSON; a file of another key, or not JSON, is no entry.
    #[test]
    fn the_cache_keeps_an_entry_in_memory_and_in_its_file() {
        let dir = std::env::temp_dir().join(format!("turbo-tune-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let k = key();
        let e = TuneEntry::measured(k.clone(), "le256:qkv=8w/sk4;pool=groups;forced=", 12, "le256/qkv/8w=0.0500\nx\n");
        assert_eq!(e.timings, [("le256/qkv/8w".to_owned(), 0.05)]);
        let c = Cache::default();
        assert_eq!(c.get(&k, Some(&dir)), None);
        c.put(e.clone(), Some(&dir)).unwrap();
        assert_eq!(c.get(&k, None), Some(e.clone()));
        let files: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|f| f.unwrap().path()).collect();
        assert_eq!(files, [cache_file(&dir, &k)]);
        let name = files[0].file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("cuda-") && name.ends_with(".json") && name.len() == 5 + 16 + 5, "{name}");
        // Another runtime reads the file.
        assert_eq!(Cache::default().get(&k, Some(&dir)), Some(e.clone()));
        // A file of another key where this key's would be, or not JSON.
        let mut other = e.clone();
        other.key.arch = "sm75".into();
        std::fs::write(&files[0], serde_json::to_vec(&other).unwrap()).unwrap();
        assert_eq!(Cache::default().get(&k, Some(&dir)), None);
        std::fs::write(&files[0], b"{ not json").unwrap();
        assert_eq!(Cache::default().get(&k, Some(&dir)), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_forced_knobs_are_read_from_the_string() {
        assert!(forced_knobs("le256:qkv=8w/sk4;pool=groups;forced=").is_empty());
        assert!(forced_knobs("").is_empty());
        assert_eq!(forced_knobs("le256:qkv=8w/sk4;pool=groups;forced=tile,sk"), ["tile", "sk"]);
        assert_eq!(tuned_name(TURBO_TUNED_CACHE), Some("cache"));
        assert_eq!(tuned_name(4), None);
    }
}
