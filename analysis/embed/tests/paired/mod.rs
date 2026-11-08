//! Reading paired timings of the dispatched step and the step it runs on.

use super::middle;
pub mod receipt;

#[test]
fn the_exact_five_percent_window_is_inclusive() {
    let own: [(f64, f64); 6] =
        [(100.0, 100.0), (105.0, 105.0), (105.0, 105.0), (105.0, 105.0), (106.0, 106.0), (106.0, 106.0)];
    assert_eq!(window(&own), Some(WIDEST));
    let over: Vec<_> = own.iter().map(|&(x, y)| if x == 105.0 { (x.next_up(), y.next_up()) } else { (x, y) }).collect();
    assert_eq!(window(&over), None);
}

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
    pairs.iter().map(|(x, y)| ((x - low_x) / low_x).max((y - low_y) / low_y)).collect()
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

/// From every set's pairs of (dispatched, fixed) ms and every set's pairs
/// of the fixed step against itself, when no set alone was quiet: what
/// dispatch adds, the middle difference of all the pairs, the fixed time,
/// and what the board allows, the middle gap between two runs of the same
/// step (never under RESOLUTION of the fixed time), all in ms; None when
/// that gap is over WIDEST of the fixed time. A step whose times swing in
/// bursts seldom gives four quiet pairs in a set of six, but the middle of
/// eighteen still shows a steady slowdown larger than its runs' own gap.
pub fn pooled(pairs: &[(f64, f64)], own: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    if pairs.is_empty() || own.is_empty() {
        return None;
    }
    let fixed = middle(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());
    let gap = middle(&own.iter().map(|(a, b)| (a - b).abs()).collect::<Vec<_>>()).max(RESOLUTION * fixed);
    (gap <= WIDEST * fixed).then(|| (middle(&pairs.iter().map(|(x, y)| x - y).collect::<Vec<_>>()), fixed, gap))
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

/// The first set is the Hailo-10H's whole set at window 64 and cap 32 at
/// 5ddf3192, 2026-11-07; the other sets and the self pairs are made up in
/// the same 72.5 to 80.7 ms swing. No set is quiet, and all of them at
/// once are read.
#[test]
fn bursty_sets_are_read_together_and_a_steady_slowdown_still_fails() {
    let pairs = [
        (75.962, 74.451),
        (76.091, 80.525),
        (76.401, 75.735),
        (73.564, 78.523),
        (75.870, 78.020),
        (77.293, 79.888),
        (72.540, 76.310),
        (80.700, 74.900),
        (74.120, 77.950),
        (78.330, 73.010),
        (75.400, 75.900),
        (79.100, 76.800),
        (76.900, 72.800),
        (73.300, 79.400),
        (77.800, 75.100),
        (74.600, 80.100),
        (78.900, 74.300),
        (75.100, 77.700),
    ];
    let own = [
        (75.10, 77.90),
        (74.20, 76.00),
        (79.80, 75.30),
        (73.90, 74.60),
        (76.50, 80.20),
        (77.20, 75.80),
        (72.90, 75.50),
        (78.40, 76.10),
        (74.80, 79.30),
        (76.00, 73.20),
        (80.10, 77.40),
        (75.60, 74.90),
        (73.70, 76.80),
        (77.90, 75.20),
        (75.30, 78.60),
        (79.20, 74.10),
        (74.40, 75.70),
        (76.70, 79.90),
    ];
    for set in pairs.chunks(6) {
        assert_eq!(window(set).and_then(|w| added(set, w)), None);
    }
    let (d, fixed, gap) = pooled(&pairs, &own).unwrap();
    assert!(gap > 0.02 * fixed && gap < WIDEST * fixed, "{gap} of {fixed}");
    assert!(d <= gap, "{d} over {gap}");
    // The same pairs with every dispatched time 6% slower: over the gap.
    let slower: Vec<(f64, f64)> = pairs.iter().map(|&(x, y)| (x * 1.06, y)).collect();
    let (d, _, gap) = pooled(&slower, &own).unwrap();
    assert!(d > gap, "{d} within {gap}");
    // A step whose own runs are further apart than WIDEST is not read.
    let wild: Vec<(f64, f64)> = own.iter().map(|&(a, b)| (a, b * 1.2)).collect();
    assert_eq!(pooled(&pairs, &wild), None);
}

#[test]
fn pooled_cap_accepts_its_boundary_but_not_an_excess() {
    let own = [(100.0, 105.0); 18];
    let pairs = [(104.9, 100.0); 18];
    let (delta, fixed, allowance) = pooled(&pairs, &own).unwrap();
    assert_eq!(allowance, WIDEST * fixed);
    assert!(delta < allowance, "the approved window permits a sub-window difference");
    let (delta, _, allowance) = pooled(&[(105.1, 100.0); 18], &own).unwrap();
    assert!(delta > allowance, "a steady excess must fail the gate");
    assert_eq!(pooled(&pairs, &[(100.0, 105.0_f64.next_up()); 18]), None);
}
