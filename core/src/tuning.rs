//! Load-time kernel choice, the core's side (docs/autotune.md): which
//! numeric classes a precision allows a backend's kernels, whether and for
//! how long a session is tuned, and what the choices are called in a
//! record. The backend owns its variants, its timing and its choices
//! string; the core never parses the string.

use crate::backend::{TURBO_NUMERIC_F16_F32ACC, TURBO_NUMERIC_F32_FMA};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_ENUM, Result};
use crate::{
    TURBO_AUTOTUNE_OFF, TURBO_AUTOTUNE_ON, TURBO_AUTOTUNE_RETUNE, TURBO_AUTOTUNE_RUNTIME, TURBO_PRECISION_EXACT,
    TURBO_PRECISION_FASTEST, TURBO_PRECISION_MODEL, TURBO_TUNED_CACHE, TURBO_TUNED_DEFAULT, TURBO_TUNED_FORCED,
    TURBO_TUNED_MEASURED,
};

/// The numeric classes each precision allows, as TURBO_NUMERIC_* bits.
/// Widened only by a decision recorded with its measurements: TF32 at
/// MODEL and F16 sums within a chunk at FASTEST are not in it. A backend
/// cannot widen it; an environment variable of the backend's widens the
/// backend's own copy for the experiment it names, and such a session is
/// never cached.
pub const NUMERICS_ALLOWED: [(u32, u32); 3] = [
    (TURBO_PRECISION_EXACT, TURBO_NUMERIC_F32_FMA),
    (TURBO_PRECISION_MODEL, TURBO_NUMERIC_F32_FMA),
    (TURBO_PRECISION_FASTEST, TURBO_NUMERIC_F16_F32ACC),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{TURBO_NUMERIC_F16_CHUNKACC, TURBO_NUMERIC_TF32};

    /// The table a decision changes: no accuracy-changing class is allowed
    /// by default in any precision.
    #[test]
    fn no_accuracy_changing_class_is_allowed_by_default() {
        assert_eq!(
            NUMERICS_ALLOWED,
            [
                (TURBO_PRECISION_EXACT, TURBO_NUMERIC_F32_FMA),
                (TURBO_PRECISION_MODEL, TURBO_NUMERIC_F32_FMA),
                (TURBO_PRECISION_FASTEST, TURBO_NUMERIC_F16_F32ACC),
            ]
        );
        for p in [TURBO_PRECISION_EXACT, TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
            assert_eq!(numerics_allowed(p) & (TURBO_NUMERIC_TF32 | TURBO_NUMERIC_F16_CHUNKACC), 0, "precision {p}");
        }
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

    #[test]
    fn the_forced_knobs_are_read_from_the_string() {
        assert!(forced_knobs("le256:qkv=8w/sk4;pool=groups;forced=").is_empty());
        assert!(forced_knobs("").is_empty());
        assert_eq!(forced_knobs("le256:qkv=8w/sk4;pool=groups;forced=tile,sk"), ["tile", "sk"]);
        assert_eq!(tuned_name(TURBO_TUNED_CACHE), Some("cache"));
        assert_eq!(tuned_name(4), None);
    }
}
