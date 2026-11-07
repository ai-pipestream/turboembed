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

/// The widest a board's own window may be: a step that cannot tell two of
/// its own runs apart within this is too noisy to read at all.
pub const WIDEST: f64 = 0.05;

/// How far each pair's times sit above the lowest time on their side, as a
/// part of it, the larger of the two.
fn above(pairs: &[(f64, f64)]) -> Vec<f64> {
    let floor = |side: fn(&(f64, f64)) -> f64| pairs.iter().map(side).fold(f64::INFINITY, f64::min);
    let (low_x, low_y) = (floor(|p| p.0), floor(|p| p.1));
    pairs.iter().map(|(x, y)| (x / low_x - 1.0).max(y / low_y - 1.0)).collect()
}

/// From pairs of the fixed step timed against itself, in ms, the window
/// within which the board tells two runs of it apart: the narrowest part
/// of the lowest time that holds QUIET of the pairs on both sides, never
/// narrower than RESOLUTION, or None when it is wider than WIDEST. The
/// Hailo-8 boards' own runs sit within RESOLUTION; the Hailo-10H's spread
/// is wider, and a reading held to RESOLUTION there is noise.
pub fn window(own: &[(f64, f64)]) -> Option<f64> {
    let mut a = above(own);
    a.sort_by(f64::total_cmp);
    let w = a.get(QUIET - 1).copied()?.max(RESOLUTION);
    (w <= WIDEST).then_some(w)
}

/// From pairs of (dispatched, fixed) ms, what dispatch adds and the fixed
/// time, both in ms, or None when the pairs are too noisy to tell. A pair
/// is quiet when each of its times is within `window` of the lowest time
/// on its own side: a slow stretch of the machine lifts one or both times
/// of a pair well past that. A steady slowdown lifts every dispatched time
/// alike, so it stays among the quiet pairs and shows in the middle of
/// their differences. One that comes and goes is taken for noise.
pub fn added(pairs: &[(f64, f64)], window: f64) -> Option<(f64, f64)> {
    let quiet: Vec<&(f64, f64)> =
        pairs.iter().zip(above(pairs)).filter(|(_, a)| *a <= window).map(|(p, _)| p).collect();
    if quiet.len() < QUIET {
        return None;
    }
    let fixed = middle(&quiet.iter().map(|p| p.1).collect::<Vec<_>>());
    Some((middle(&quiet.iter().map(|(x, y)| x - y).collect::<Vec<_>>()), fixed))
}

/// Whether a run with these inconclusive submissions passes. A run under
/// TURBO_TEST_REQUIRE_HAILO=1 is evidence that dispatch adds no more than
/// the planning and the board's window, and a submission left inconclusive after
/// ATTEMPTS sets is a gap in that evidence: the run fails and names each
/// one. Without it they are only reported.
pub fn settle(inconclusive: &[String], require: bool) -> Result<(), String> {
    if require && !inconclusive.is_empty() {
        return Err(format!(
            "what dispatch adds is unchecked after {ATTEMPTS} sets of pairs for {}: time them again",
            inconclusive.join(", ")
        ));
    }
    Ok(())
}

#[test]
fn an_inconclusive_submission_fails_only_a_required_run() {
    let left = vec!["16x1 all".to_string()];
    assert_eq!(settle(&left, false), Ok(()));
    assert_eq!(settle(&[], true), Ok(()));
    let why = settle(&left, true).unwrap_err();
    assert!(why.contains("16x1 all") && why.contains("3 sets"), "{why}");
}

/// Pairs from the Hailo-8 boards' paired run at 2787fa1e. The Compute
/// Module 5's one sentence at 16 x 8 had three pairs lifted by 3 to 7 ms;
/// the Raspberry Pi 5's whole set at 16 x 1 had one, by about 80 ms.
#[test]
fn noisy_pairs_are_inconclusive_and_a_steady_slowdown_is_measured() {
    let cm5 = [(7.574, 7.546), (14.672, 8.283), (7.656, 10.584), (10.173, 7.569), (11.698, 7.579), (7.535, 7.532)];
    assert_eq!(added(&cm5, RESOLUTION), None);
    let pi5 =
        [(1614.1, 1616.0), (1612.0, 1608.1), (1618.8, 1610.6), (1691.8, 1608.5), (1613.5, 1614.6), (1614.5, 1608.0)];
    let (d, fixed) = added(&pi5, RESOLUTION).unwrap();
    assert!(d.abs() <= RESOLUTION * fixed, "{d} of {fixed}");
    // The same pairs with every dispatched time 2% slower: still quiet, and
    // the 2% shows.
    let slower: Vec<(f64, f64)> = pi5.iter().map(|&(x, y)| (x * 1.02, y)).collect();
    let (d, fixed) = added(&slower, RESOLUTION).unwrap();
    assert!(d > 0.015 * fixed, "{d} of {fixed}");
}

/// Pairs from the Hailo-10H's run at 9edfef03, 2026-11-07: one sentence at
/// window 32 and cap 32 read +0.524% held to RESOLUTION, and one document
/// at window 16 and cap 1 spread over 28.10 to 29.06 ms.
#[test]
fn a_board_noisier_than_resolution_is_read_within_its_own_window() {
    // The Hailo-8 boards' own pairs sit within RESOLUTION: the window stays.
    let pi5 =
        [(1614.1, 1616.0), (1612.0, 1608.1), (1618.8, 1610.6), (1691.8, 1608.5), (1613.5, 1614.6), (1614.5, 1608.0)];
    assert_eq!(window(&pi5), Some(RESOLUTION));
    // A step whose own runs spread over 3%: four of six fit 2.1%.
    let own = [(28.10, 28.31), (28.52, 28.20), (28.69, 28.15), (29.06, 28.12), (28.33, 28.95), (28.20, 28.40)];
    let w = window(&own).unwrap();
    assert!(w > 0.02 && w < 0.022, "{w}");
    let pairs = [(28.31, 28.12), (28.49, 28.40), (28.20, 28.33), (29.01, 28.15), (28.40, 28.69), (28.66, 28.22)];
    assert_eq!(added(&pairs, RESOLUTION), None);
    let (d, fixed) = added(&pairs, w).unwrap();
    assert!(d.abs() <= w * fixed, "{d} of {fixed}");
    // A dispatch that adds 5% stays over the window, so it still fails.
    let slower: Vec<(f64, f64)> = pairs.iter().map(|&(x, y)| (x * 1.05, y)).collect();
    let (d, fixed) = added(&slower, w).unwrap();
    assert!(d > w * fixed, "{d} of {fixed}");
    // A step that cannot tell its own runs apart within WIDEST is not read.
    let wild = [(10.0, 10.0), (11.0, 10.0), (10.0, 11.5), (12.0, 10.0), (10.0, 13.0), (14.0, 10.0)];
    assert_eq!(window(&wild), None);
}
