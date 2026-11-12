//! The GPU clock a levelzero record carries (`device.clock`), read from
//! the xe kernel driver's sysfs files: `min_freq`, `max_freq` and
//! `act_freq` in MHz under `/sys/class/drm/card<N>/device/tile0/gt0/freq0`.
//! A record does not say how the clock was set, so a pinned clock
//! (`min_freq` = `max_freq`) and a run where the GPU dropped below it
//! (`act_freq`) show only here. `act_freq` reads 0 while the GPU is idle,
//! so a reading taken between runs can be 0.

use std::path::{Path, PathBuf};

use turbo::record::ClockReading;

const DRM: &str = "/sys/class/drm";
const FREQ: &str = "device/tile0/gt0/freq0";

/// The clock of the one xe GPU on the machine, for `backend`; None for
/// any other backend, off Linux, or where there is no xe GPU or more than
/// one, since the files do not say which is the measured device.
pub fn read(backend: &str) -> Option<ClockReading> {
    if backend != "levelzero" || !cfg!(target_os = "linux") {
        return None;
    }
    let mut dirs = xe_freq_dirs(Path::new(DRM));
    if dirs.len() != 1 {
        return None;
    }
    reading(&dirs.pop()?)
}

/// Each `card<N>` under `drm` (not its connectors, `card<N>-DP-1`) that
/// has the xe frequency directory.
fn xe_freq_dirs(drm: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(drm) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_str().and_then(|n| n.strip_prefix("card")).is_some_and(is_digits))
        .map(|e| e.path().join(FREQ))
        .filter(|d| d.is_dir())
        .collect();
    dirs.sort();
    dirs
}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// The three frequencies under `dir`; None when one is missing or not a
/// number.
pub(crate) fn reading(dir: &Path) -> Option<ClockReading> {
    let mhz = |name: &str| std::fs::read_to_string(dir.join(name)).ok()?.trim().parse::<u32>().ok();
    Some(ClockReading { min_mhz: mhz("min_freq")?, max_mhz: mhz("max_freq")?, act_mhz: mhz("act_freq")? })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("turbo-bench-gpu-clock-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn write_card(drm: &Path, card: &str, min: &str, max: &str, act: &str) {
        let dir = drm.join(card).join(FREQ);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, v) in [("min_freq", min), ("max_freq", max), ("act_freq", act)] {
            std::fs::write(dir.join(name), format!("{v}\n")).unwrap();
        }
    }

    #[test]
    fn a_card_directory_reads_as_its_three_frequencies() {
        let drm = scratch("one");
        write_card(&drm, "card0", "2200", "2200", "2150");
        std::fs::create_dir_all(drm.join("card0-DP-1")).unwrap();
        std::fs::create_dir_all(drm.join("renderD128")).unwrap();
        let dirs = xe_freq_dirs(&drm);
        assert_eq!(dirs, [drm.join("card0").join(FREQ)]);
        assert_eq!(reading(&dirs[0]), Some(ClockReading { min_mhz: 2200, max_mhz: 2200, act_mhz: 2150 }));
        std::fs::remove_dir_all(&drm).unwrap();
    }

    #[test]
    fn a_missing_or_unreadable_frequency_is_no_reading() {
        let drm = scratch("bad");
        write_card(&drm, "card1", "1200", "2800", "idle");
        assert_eq!(reading(&drm.join("card1").join(FREQ)), None);
        std::fs::remove_file(drm.join("card1").join(FREQ).join("act_freq")).unwrap();
        assert_eq!(reading(&drm.join("card1").join(FREQ)), None);
        assert!(xe_freq_dirs(&drm.join("absent")).is_empty());
        std::fs::remove_dir_all(&drm).unwrap();
    }

    #[test]
    fn other_backends_read_no_clock() {
        for backend in ["cpu", "cuda", "metal", "npu", "hailo"] {
            assert_eq!(read(backend), None, "{backend}");
        }
    }
}
