//! Bounded scaling checks for the place scanners (`state`, `county`, `city`)
//! at `n` and `8n` (#2112). Counts scanner steps through [`super::PLACE_STEPS`],
//! never wall-clock time, so any quadratic regression fails in milliseconds
//! and names the scanner and input shape that went super-linear.

use std::cell::Cell;

use turbo_analysis::ViewId;
use turbo_analysis_layers::Pii;

use super::{Extractor, PLACE_STEPS, Scanner};
use crate::{Pack, PiiDetector};

fn steps_for(detector: &PiiDetector, text: &str) -> usize {
    PLACE_STEPS.with(|c| c.set(0));
    let mut pii = Pii::new(ViewId::SOURCE);
    detector.detect(text, &mut pii);
    PLACE_STEPS.with(Cell::get)
}

#[test]
fn place_scanners_scale_linearly_across_adversarial_shapes() {
    type Shape = (&'static str, fn(usize) -> String);
    let shapes: &[Shape] = &[
        ("runs of ':'", |n| ":".repeat(n)),
        ("City before ':' run", |n| format!("City{}", ":".repeat(n))),
        ("City and ': ' run before city", |n| format!("City{}Phoenix", ": ".repeat(n / 2))),
        ("runs of ' : '", |n| " : ".repeat(n / 3)),
        ("runs of 'City #= '", |n| "City #= ".repeat(n / 8)),
        ("city of and ' #= ' run before city", |n| format!("city of{}Phoenix", " #= ".repeat(n / 4))),
        ("', IL' repeated", |n| ", IL".repeat(n / 4)),
        ("city and state repeated", |n| "Springfield, IL ".repeat(n / 16)),
        ("one long word before ', IL'", |n| format!("{}, IL", "A".repeat(n))),
        ("one long dotted word before ', IL'", |n| format!("A{}, IL", "a.".repeat(n / 2))),
        ("capital words before ', IL'", |n| format!("{}, IL", "A ".repeat(n / 2).trim_end())),
        ("City: before capital words", |n| format!("City: {}", "A ".repeat(n / 2))),
        ("City: before one long word", |n| format!("City: {}", "A".repeat(n))),
        ("Cook County County County", |n| format!("Cook{}", " County".repeat(n / 7))),
        ("one long word before County", |n| format!("{} County", "A".repeat(n))),
        ("State and ':' run before code", |n| format!("State{} IL", ":".repeat(n))),
        ("New New New York", |n| format!("{}York", "New ".repeat(n / 4))),
        ("Washington DC repeated", |n| "Washington DC ".repeat(n / 14)),
    ];

    let detectors = [
        ("state", PiiDetector::new(vec![Scanner::all(Extractor::UsState)])),
        ("county", PiiDetector::new(vec![Scanner::all(Extractor::UsCounty)])),
        ("city", PiiDetector::new(vec![Scanner::all(Extractor::City)])),
        ("places pack", PiiDetector::pack(Pack::Places)),
    ];

    // 336 = 3 * 7 * 16 is evenly divisible by every repeat unit above (1, 2,
    // 3, 4, 7, 8, 14, 16), so 8n has 8 times as many units as n.
    let n = 336;
    for (scanner, detector) in &detectors {
        // Warm lazy tables (`BundledGazetteer`, `us_counties`) before counting.
        let _ = steps_for(detector, "Cook County, Springfield, IL");
        for &(shape, make) in shapes {
            let (t1, t8) = (make(n), make(8 * n));
            let (s1, s8) = (steps_for(detector, &t1), steps_for(detector, &t8));
            let ratio = s8 as f64 / s1.max(1) as f64;
            assert!(
                s1 > 0 && s8 <= s1 * 12,
                "{scanner} scanner went quadratic on {shape:?}: n={s1} steps ({} B), 8n={s8} steps ({} B), ratio={ratio:.2}x",
                t1.len(),
                t8.len(),
            );
            assert!(
                s8 <= t8.len() * 32,
                "{scanner} scanner exceeded 32 steps/byte on {shape:?}: {s8} steps for {} B",
                t8.len(),
            );
        }
    }
}
