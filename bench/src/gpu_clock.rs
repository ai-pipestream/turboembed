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
    read_at(backend, Path::new(DRM))
}

fn read_at(backend: &str, drm: &Path) -> Option<ClockReading> {
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
        Self::at(backend, steps, PathBuf::from(DRM))
    }

    fn at(backend: impl Into<String>, steps: usize, drm: PathBuf) -> Self {
        Self { backend: backend.into(), drm, remaining: steps, started: false, before: None, after: None }
    }

    pub(crate) fn none() -> Self {
        Self::new(String::new(), 0)
    }

    pub(crate) fn before_timed(&mut self) {
        if !self.started && self.remaining > 0 {
            self.started = true;
            self.before = read_at(&self.backend, &self.drm);
        }
    }

    pub(crate) fn after_timed(&mut self) {
        if self.remaining > 0 {
            self.remaining -= 1;
            if self.remaining == 0 {
                self.after = read_at(&self.backend, &self.drm);
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
#[path = "gpu_clock_tests.rs"]
pub(crate) mod tests;
