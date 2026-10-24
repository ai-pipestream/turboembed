//! Reading paired timings of the dispatched step and the step it runs on.

use super::middle;

/// The part of a time below which paired rounds on the Hailo-8 boards do
/// not tell two runs of the same step apart.
pub const RESOLUTION: f64 = 0.005;

/// The fewest quiet pairs out of six that give a reading.
const QUIET: usize = 4;

/// How many sets of six pairs a submission is timed with before it is
/// reported inconclusive.
pub const ATTEMPTS: usize = 3;

/// From pairs of (dispatched, fixed) ms, what dispatch adds and the fixed
/// time, both in ms, or None when the pairs are too noisy to tell. A pair
/// is quiet when each of its times is within RESOLUTION of the lowest time
/// on its own side: a slow stretch of the machine lifts one or both times
/// of a pair well past that. A steady slowdown lifts every dispatched time
/// alike, so it stays among the quiet pairs and shows in the middle of
/// their differences. One that comes and goes is taken for noise.
pub fn added(pairs: &[(f64, f64)]) -> Option<(f64, f64)> {
    let floor = |side: fn(&(f64, f64)) -> f64| pairs.iter().map(side).fold(f64::INFINITY, f64::min);
    let (low_x, low_y) = (floor(|p| p.0), floor(|p| p.1));
    let quiet: Vec<&(f64, f64)> =
        pairs.iter().filter(|(x, y)| *x <= low_x * (1.0 + RESOLUTION) && *y <= low_y * (1.0 + RESOLUTION)).collect();
    if quiet.len() < QUIET {
        return None;
    }
    let fixed = middle(&quiet.iter().map(|p| p.1).collect::<Vec<_>>());
    Some((middle(&quiet.iter().map(|(x, y)| x - y).collect::<Vec<_>>()), fixed))
}

/// Pairs from the Hailo-8 boards' paired run at 2787fa1e. The Compute
/// Module 5's one sentence at 16 x 8 had three pairs lifted by 3 to 7 ms;
/// the Raspberry Pi 5's whole set at 16 x 1 had one, by about 80 ms.
#[test]
fn noisy_pairs_are_inconclusive_and_a_steady_slowdown_is_measured() {
    let cm5 = [(7.574, 7.546), (14.672, 8.283), (7.656, 10.584), (10.173, 7.569), (11.698, 7.579), (7.535, 7.532)];
    assert_eq!(added(&cm5), None);
    let pi5 =
        [(1614.1, 1616.0), (1612.0, 1608.1), (1618.8, 1610.6), (1691.8, 1608.5), (1613.5, 1614.6), (1614.5, 1608.0)];
    let (d, fixed) = added(&pi5).unwrap();
    assert!(d.abs() <= RESOLUTION * fixed, "{d} of {fixed}");
    // The same pairs with every dispatched time 2% slower: still quiet, and
    // the 2% shows.
    let slower: Vec<(f64, f64)> = pi5.iter().map(|&(x, y)| (x * 1.02, y)).collect();
    let (d, fixed) = added(&slower).unwrap();
    assert!(d > 0.015 * fixed, "{d} of {fixed}");
}
