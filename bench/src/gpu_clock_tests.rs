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

/// Driver files shared by the reader and timed-loop tests, never a reader override.
#[cfg(target_os = "linux")]
pub(crate) struct Driver {
    root: turbo_scratch::Scratch,
}

#[cfg(target_os = "linux")]
impl Driver {
    pub(crate) fn new(name: &str) -> Self {
        let root = turbo_scratch::Scratch::at(scratch(name));
        write_card(&root, "card0", INTEL, Some(["1200", "2800", "0"]));
        Self { root }
    }

    pub(crate) fn window(&self, steps: usize) -> Window {
        Window::at(LEVELZERO, steps, self.root.to_path_buf())
    }

    pub(crate) fn set(&self, act: u32) {
        std::fs::write(self.root.join("card0").join(FREQ).join("act_freq"), act.to_string()).unwrap();
    }
}

#[cfg(target_os = "linux")]
#[test]
fn window_reads_driver_files_before_first_and_after_last_step() {
    let driver = Driver::new("window");
    let mut w = driver.window(2);
    driver.set(100);
    w.before_timed();
    driver.set(200);
    w.after_timed();
    driver.set(300);
    w.before_timed();
    driver.set(400);
    w.after_timed();
    assert_eq!(
        w.finish(),
        Some(GpuClock {
            before: ClockReading { min_mhz: 1200, max_mhz: 2800, act_mhz: 100 },
            after: ClockReading { min_mhz: 1200, max_mhz: 2800, act_mhz: 400 },
        })
    );
    assert_eq!(read_at("cpu", &driver.root), None);
    assert_eq!(read_at(LEVELZERO, &driver.root).unwrap().act_mhz, 400);
}

#[cfg(target_os = "linux")]
#[test]
fn either_missing_driver_reading_omits_the_window_clock() {
    for before in [true, false] {
        let driver = Driver::new(if before { "missing-before" } else { "missing-after" });
        let path = driver.root.join("card0").join(FREQ).join("act_freq");
        let mut w = driver.window(1);
        if before {
            std::fs::remove_file(&path).unwrap();
        }
        w.before_timed();
        driver.set(100);
        if !before {
            std::fs::remove_file(&path).unwrap();
        }
        w.after_timed();
        assert_eq!(w.finish(), None);
    }
}

static LEVELZERO_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Explicit device selection opts into hardware work; required mode fails rather than skipping.
pub(crate) fn levelzero_turn(test: &str) -> Option<std::sync::MutexGuard<'static, ()>> {
    let required = std::env::var("TURBO_TEST_REQUIRE_LEVELZERO").is_ok_and(|v| !v.is_empty() && v != "0");
    if let Ok(device) = std::env::var("TURBO_TEST_DEVICE")
        && device != LEVELZERO
    {
        assert!(!required, "{test}: required levelzero test selected a different device: {device}");
        eprintln!("{test}: skipped hardware clock check; a different device was selected");
        return None;
    }
    let selected = std::env::var("TURBO_TEST_DEVICE").as_deref() == Ok(LEVELZERO);
    if !selected && !required {
        eprintln!("{test}: skipped hardware clock check; select TURBO_TEST_DEVICE=levelzero");
        return None;
    }
    let turn = LEVELZERO_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let runtime = crate::api::Runtime::create().unwrap();
    let available = runtime.find(LEVELZERO).is_ok() && read(LEVELZERO).is_some();
    if !available {
        assert!(!required, "{test}: required levelzero device and readable xe clock are unavailable");
        eprintln!("{test}: skipped hardware clock check; levelzero device or readable xe clock unavailable");
        return None;
    }
    Some(turn)
}

pub(crate) fn assert_device_clock(clock: Option<GpuClock>) {
    let clock = clock.expect("the measured levelzero record must contain both xe clock readings");
    for reading in [clock.before, clock.after] {
        assert!(reading.min_mhz > 0 && reading.max_mhz >= reading.min_mhz, "{reading:?}");
    }
}
