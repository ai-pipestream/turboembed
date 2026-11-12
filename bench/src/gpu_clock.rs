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

/// Intel's PCI vendor id, as `device/vendor` reads it: every GPU the
/// levelzero backend can list is one of these.
const INTEL: &str = "0x8086";

/// The clock of the measured GPU, for `backend`; None for any other
/// backend or off Linux. The files do not say which device was measured,
/// so the reading is taken only when the machine has exactly one Intel GPU
/// and that GPU has the xe frequency files. With a second Intel GPU, under
/// xe or i915, or one GPU without the files, there is no reading.
pub fn read(backend: &str) -> Option<ClockReading> {
    if backend != "levelzero" || !cfg!(target_os = "linux") {
        return None;
    }
    xe_freq_dir(Path::new(DRM)).as_deref().and_then(reading)
}

/// The xe frequency directory of the one Intel `card<N>` under `drm` (not
/// its connectors, `card<N>-DP-1`); None when there is no Intel card, more
/// than one, or the one has no xe frequency directory.
fn xe_freq_dir(drm: &Path) -> Option<PathBuf> {
    let mut intel = std::fs::read_dir(drm)
        .ok()?
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_str().and_then(|n| n.strip_prefix("card")).is_some_and(is_digits))
        .map(|e| e.path())
        .filter(|card| std::fs::read_to_string(card.join("device/vendor")).is_ok_and(|v| v.trim() == INTEL));
    let card = intel.next()?;
    if intel.next().is_some() {
        return None;
    }
    Some(card.join(FREQ)).filter(|d| d.is_dir())
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

    fn write_card(drm: &Path, card: &str, vendor: &str, freqs: Option<[&str; 3]>) {
        let device = drm.join(card).join("device");
        std::fs::create_dir_all(&device).unwrap();
        std::fs::write(device.join("vendor"), format!("{vendor}\n")).unwrap();
        let Some(freqs) = freqs else { return };
        let dir = drm.join(card).join(FREQ);
        std::fs::create_dir_all(&dir).unwrap();
        for (name, v) in ["min_freq", "max_freq", "act_freq"].into_iter().zip(freqs) {
            std::fs::write(dir.join(name), format!("{v}\n")).unwrap();
        }
    }

    #[test]
    fn the_one_intel_card_reads_as_its_three_frequencies() {
        let drm = scratch("one");
        write_card(&drm, "card0", INTEL, Some(["2200", "2200", "2150"]));
        std::fs::create_dir_all(drm.join("card0-DP-1")).unwrap();
        std::fs::create_dir_all(drm.join("renderD128")).unwrap();
        // A GPU of another vendor is not one the levelzero backend lists.
        write_card(&drm, "card1", "0x10de", None);
        let dir = xe_freq_dir(&drm);
        assert_eq!(dir, Some(drm.join("card0").join(FREQ)));
        assert_eq!(reading(&dir.unwrap()), Some(ClockReading { min_mhz: 2200, max_mhz: 2200, act_mhz: 2150 }));
        std::fs::remove_dir_all(&drm).unwrap();
    }

    #[test]
    fn a_second_intel_card_is_no_reading_even_without_the_xe_files() {
        let drm = scratch("two");
        write_card(&drm, "card0", INTEL, None);
        write_card(&drm, "card1", INTEL, Some(["2200", "2200", "2150"]));
        assert_eq!(xe_freq_dir(&drm), None);
        std::fs::remove_dir_all(&drm).unwrap();
    }

    #[test]
    fn a_missing_or_unreadable_frequency_is_no_reading() {
        let drm = scratch("bad");
        write_card(&drm, "card1", INTEL, Some(["1200", "2800", "idle"]));
        assert_eq!(reading(&drm.join("card1").join(FREQ)), None);
        std::fs::remove_file(drm.join("card1").join(FREQ).join("act_freq")).unwrap();
        assert_eq!(reading(&drm.join("card1").join(FREQ)), None);
        let lone = scratch("lone");
        write_card(&lone, "card0", INTEL, None);
        assert_eq!(xe_freq_dir(&lone), None);
        assert_eq!(xe_freq_dir(&drm.join("absent")), None);
        std::fs::remove_dir_all(&drm).unwrap();
        std::fs::remove_dir_all(&lone).unwrap();
    }

    #[test]
    fn other_backends_read_no_clock() {
        for backend in ["cpu", "cuda", "metal", "npu", "hailo"] {
            assert_eq!(read(backend), None, "{backend}");
        }
    }
}
