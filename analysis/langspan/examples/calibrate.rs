//! Fit on the first 20 mixed-dev documents, reserve the last 10 for
//! conformal scores, then evaluate mixed-test without changing settings.
//! Usage: calibrate MODEL DEV TEST OUTPUT
mod calibration_support;
use calibration_support::{collect, report};
use std::{fs, path::Path};
use turbo_calibration::{Calibration, digest};
use turbo_langdetect::{LanguageDetector, LanguageModel};
use turbo_langspan::{CalibratedSpanDetector, Settings, SpanDetector, calibration_identity, gold::parse};
use turbo_maxent::ModelFile;
use turbo_train::calibration::{Method, fit};

const DOMAIN: &str = "mixed-authored-v1";
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: calibrate MODEL DEV TEST OUTPUT".into());
    }
    if Path::new(&args[3]).exists() {
        return Err("output already exists".into());
    }
    let detector =
        SpanDetector::new(LanguageDetector::new(LanguageModel::open(ModelFile::read(&args[0])?)?), Settings::default());
    let dev_bytes = fs::read(&args[1])?;
    let test_bytes = fs::read(&args[2])?;
    let dev = parse(std::str::from_utf8(&dev_bytes)?)?;
    let test = parse(std::str::from_utf8(&test_bytes)?)?;
    if dev.len() != 30 || test.len() != 40 {
        return Err("expected 30 development and 40 test documents".into());
    }
    for d in &dev {
        if test.iter().any(|t| t.id == d.id || t.text == d.text) {
            return Err("development and test documents overlap".into());
        }
    }
    let fitting = collect(&detector, &dev[..20]);
    let conformal = collect(&detector, &dev[20..]);
    let held = collect(&detector, &test);
    let identity = calibration_identity(&detector, DOMAIN);
    println!("model_sha256={} dev_sha256={} test_sha256={}", identity.sha256, digest(&dev_bytes), digest(&test_bytes));
    let profile = fit(
        identity,
        Method::Temperature,
        fitting.samples("mixed-dev-first20"),
        conformal.samples("mixed-dev-last10"),
    )?;
    println!(
        "transform={:?} profile_sha256={} fit_rows={} conformal_rows={}",
        profile.transform(),
        profile.id(),
        fitting.rows.len(),
        conformal.rows.len()
    );
    // Fixed before evaluating test results. This is a label-confidence
    // cutoff, not a guarantee that every boundary or letter is correct.
    for (name, data) in [("fitting", &fitting), ("conformal", &conformal), ("test", &held)] {
        report(name, data, &profile, 0.9)?;
    }
    let bytes = profile.to_bytes();
    let restored = Calibration::from_bytes(&bytes, profile.identity())?;
    let calibrated = CalibratedSpanDetector::new(detector, restored, DOMAIN)?;
    calibration_support::verify(&calibrated, &test)?;
    fs::write(&args[3], bytes)?;
    Ok(())
}
