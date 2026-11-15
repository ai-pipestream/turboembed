//! The GPU clock a levelzero record carries (`device.clock`), read from
//! the xe kernel driver's sysfs files: `min_freq`, `max_freq` and
//! `act_freq` in MHz under `/sys/class/drm/card<N>/device/tile0/gt0/freq0`.
//! A record does not say how the clock was set, so a pinned clock
//! (`min_freq` = `max_freq`) and a run where the GPU dropped below it
//! (`act_freq`) show only here. `act_freq` reads 0 while the GPU is idle,
//! so a reading taken between runs can be 0.

use std::path::{Path, PathBuf};

use turbo::record::{ClockReading, GpuClock};

const DRM: &str = "/sys/class/drm";
const FREQ: &str = "device/tile0/gt0/freq0";
const LEVELZERO: &str = "levelzero";

/// Intel's PCI vendor id, as `device/vendor` reads it: every GPU the
/// levelzero backend can list is one of these.
const INTEL: &str = "0x8086";

/// The clock of the measured GPU, for `backend`; None for any other
/// backend or off Linux. The files do not say which device was measured,
/// so the reading is taken only when the machine has exactly one Intel GPU
/// and that GPU has the xe frequency files. With a second Intel GPU, under
/// xe or i915, or one GPU without the files, there is no reading.
pub fn read(backend: &str) -> Option<ClockReading> {
    read_from(backend, Path::new(DRM))
}

fn read_from(backend: &str, drm: &Path) -> Option<ClockReading> {
    if backend != LEVELZERO || !cfg!(target_os = "linux") {
        return None;
    }
    xe_freq_dir(drm).as_deref().and_then(reading)
}

/// Pair the readings taken before the first timed run and after the last
/// into `device.clock`; None when either reading is absent.
pub fn span(before: Option<ClockReading>, after: Option<ClockReading>) -> Option<GpuClock> {
    before.zip(after).map(|(before, after)| GpuClock { before, after })
}

/// Samples `device.clock` across the `steps` timed loops of one record:
/// `before_timed` reads once before the first timed loop (after its warmup)
/// and `after_timed` reads once after the last timed loop.
pub(crate) struct Window {
    backend: String,
    drm: PathBuf,
    remaining: usize,
    started: bool,
    before: Option<ClockReading>,
    after: Option<ClockReading>,
}

impl Window {
    pub(crate) fn new(backend: impl Into<String>, steps: usize) -> Self {
        Self {
            backend: backend.into(),
            drm: PathBuf::from(DRM),
            remaining: steps,
            started: false,
            before: None,
            after: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_drm(backend: impl Into<String>, steps: usize, drm: impl Into<PathBuf>) -> Self {
        Self { backend: backend.into(), drm: drm.into(), remaining: steps, started: false, before: None, after: None }
    }

    pub(crate) fn none() -> Self {
        Self::new(String::new(), 0)
    }

    pub(crate) fn before_timed(&mut self) {
        if !self.started && self.remaining > 0 {
            self.started = true;
            self.before = read_from(&self.backend, &self.drm);
        }
    }

    pub(crate) fn after_timed(&mut self) {
        if self.remaining > 0 {
            self.remaining -= 1;
            if self.remaining == 0 {
                self.after = read_from(&self.backend, &self.drm);
            }
        }
    }

    pub(crate) fn finish(self) -> Option<GpuClock> {
        debug_assert_eq!(self.remaining, 0);
        span(self.before, self.after)
    }
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
pub(crate) mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("turbo-bench-gpu-clock-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// A temporary tree of the xe driver's files with one Intel card, for
    /// a builder's timing function to read as a levelzero run reads
    /// `/sys/class/drm`. Removed when dropped.
    pub(crate) struct Tree(PathBuf);

    impl Tree {
        pub(crate) const MIN: u32 = 1200;
        pub(crate) const MAX: u32 = 2800;

        /// The tree for the test `name`, with the GPU idle (`act_freq` 0).
        pub(crate) fn new(name: &str) -> Self {
            let drm = scratch(name);
            write_card(&drm, "card0", INTEL, Some(["1200", "2800", "0"]));
            Self(drm)
        }

        /// Write `act_freq`, as the driver does while the GPU works.
        pub(crate) fn set_act(&self, mhz: u32) {
            std::fs::write(self.0.join("card0").join(FREQ).join("act_freq"), format!("{mhz}\n")).unwrap();
        }

        /// A levelzero window of `steps` timed loops over this tree.
        pub(crate) fn window(&self, steps: usize) -> Window {
            Window::with_drm(LEVELZERO, steps, &self.0)
        }

        /// What a window over this tree finishes with when `act_freq` read
        /// `before` and `after` at its two readings: off Linux, no clock.
        pub(crate) fn clock(before: u32, after: u32) -> Option<GpuClock> {
            let at = |act_mhz| ClockReading { min_mhz: Self::MIN, max_mhz: Self::MAX, act_mhz };
            cfg!(target_os = "linux").then(|| GpuClock { before: at(before), after: at(after) })
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
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
    fn a_second_intel_card_is_no_reading_with_or_without_the_xe_files() {
        // Both cards have the files, so the answer cannot depend on which
        // one the directory lists first.
        let drm = scratch("two");
        write_card(&drm, "card0", INTEL, Some(["1200", "2800", "2400"]));
        write_card(&drm, "card1", INTEL, Some(["2200", "2200", "2150"]));
        assert_eq!(xe_freq_dir(&drm), None);
        // One of them without the files, as under i915, is still two.
        std::fs::remove_dir_all(drm.join("card0").join("device/tile0")).unwrap();
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

    #[test]
    fn window_reads_once_before_first_step_and_once_after_last() {
        let b = ClockReading { min_mhz: 2200, max_mhz: 2200, act_mhz: 0 };
        let a = ClockReading { min_mhz: 2200, max_mhz: 2200, act_mhz: 2200 };
        assert_eq!(span(Some(b), Some(a)), Some(GpuClock { before: b, after: a }));
        assert_eq!(span(Some(b), None), None);
        assert_eq!(span(None, Some(a)), None);

        let drm = scratch("window");
        write_card(&drm, "card0", INTEL, Some(["2200", "2200", "0"]));
        let act = drm.join("card0").join(FREQ).join("act_freq");

        assert_eq!(read_from("cpu", &drm), None);
        let mut w = Window::with_drm("levelzero", 2, &drm);
        w.before_timed();
        // Mutating act_freq between the first and last timed steps must not
        // show up in `before` or `after` if intermediate calls do not read.
        std::fs::write(&act, "9999\n").unwrap();
        w.after_timed();
        w.before_timed();
        std::fs::write(&act, "2200\n").unwrap();
        w.after_timed();
        let clock = w.finish();
        if cfg!(target_os = "linux") {
            assert_eq!(clock, Some(GpuClock { before: b, after: a }));
        } else {
            assert_eq!(clock, None);
        }

        // An unreadable frequency at the end yields None for the whole span.
        std::fs::write(&act, "idle\n").unwrap();
        let mut incomplete = Window::with_drm("levelzero", 1, &drm);
        std::fs::write(&act, "0\n").unwrap();
        incomplete.before_timed();
        std::fs::write(&act, "idle\n").unwrap();
        incomplete.after_timed();
        assert_eq!(incomplete.finish(), None);
        std::fs::remove_dir_all(&drm).unwrap();
    }
}
