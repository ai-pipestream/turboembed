//! How the calibration times two sides against each other: each side in a
//! block of its own runs, so no timed run of one follows a run of the other
//! straight away (a CPU run timed just after a device run on the same host
//! is slower than one after another CPU run). The blocks go a, b, b, a,
//! each after one untimed run of its side, so drift over the measurement
//! falls on both.

use std::time::Instant;

use turbo::status::{Error as CoreError, INVALID_ARGUMENT};

/// The fastest times of `reps` runs each of `a` and `b`, in ms: what
/// another process on the machine cannot add to, since it only ever slows
/// a run.
pub(super) fn fastest(
    reps: usize,
    a: impl FnMut() -> Result<(), CoreError>,
    b: impl FnMut() -> Result<(), CoreError>,
) -> Result<(f64, f64), CoreError> {
    let (ta, tb) = times(reps, a, b)?;
    Ok((ta[0], tb[0]))
}

/// The times of `reps` runs each of `a` and `b`, in ms, sorted. The runs
/// are in [`blocks`] order: each block one untimed run of its side, then
/// its timed runs.
pub(super) fn times(
    reps: usize,
    mut a: impl FnMut() -> Result<(), CoreError>,
    mut b: impl FnMut() -> Result<(), CoreError>,
) -> Result<(Vec<f64>, Vec<f64>), CoreError> {
    if reps == 0 {
        return Err(CoreError::new(INVALID_ARGUMENT, "calibration needs at least one timed run"));
    }
    let (mut ta, mut tb) = (Vec::with_capacity(reps), Vec::with_capacity(reps));
    for (first, n) in blocks(reps) {
        let (f, t): (&mut dyn FnMut() -> Result<(), CoreError>, &mut Vec<f64>) =
            if first { (&mut a, &mut ta) } else { (&mut b, &mut tb) };
        f()?;
        for _ in 0..n {
            let s = Instant::now();
            f()?;
            t.push(s.elapsed().as_secs_f64() * 1e3);
        }
    }
    ta.sort_by(f64::total_cmp);
    tb.sort_by(f64::total_cmp);
    Ok((ta, tb))
}

/// The blocks of `reps` timed runs a side, as (whether it is `a`, its
/// timed runs): `a` and `b` the first half each (the larger when `reps` is
/// odd), then `b` and `a` the rest; a block of no runs is left out.
fn blocks(reps: usize) -> Vec<(bool, usize)> {
    let (first, rest) = (reps.div_ceil(2), reps / 2);
    [(true, first), (false, first), (false, rest), (true, rest)].into_iter().filter(|&(_, n)| n > 0).collect()
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    /// The sides' runs in the order `times` makes them, `a` and `b` for
    /// each.
    fn order(reps: usize) -> String {
        let log = RefCell::new(String::new());
        let run = |side: char| {
            log.borrow_mut().push(side);
            Ok(())
        };
        let (ta, tb) = times(reps, || run('a'), || run('b')).unwrap();
        assert_eq!((ta.len(), tb.len()), (reps, reps));
        assert!(ta.is_sorted() && tb.is_sorted());
        log.into_inner()
    }

    #[test]
    fn each_side_runs_in_blocks_of_its_own_a_b_b_a() {
        // Each block is one untimed run and its timed runs.
        assert_eq!(order(1), "aabb");
        assert_eq!(order(2), "aabbbbaa");
        assert_eq!(order(5), "aaaabbbbbbbaaa");
        assert_eq!(order(10), "aaaaaabbbbbbbbbbbbaaaaaa");
        assert_eq!(blocks(5), [(true, 3), (false, 3), (false, 2), (true, 2)]);
        assert_eq!(blocks(1), [(true, 1), (false, 1)]);
    }

    #[test]
    fn no_timed_run_of_one_side_follows_a_run_of_the_other() {
        for reps in 1..12 {
            let runs = order(reps);
            let mut timed = 0;
            for (i, (block, n)) in blocks(reps).into_iter().enumerate() {
                let side = if block { 'a' } else { 'b' };
                let start: usize = blocks(reps)[..i].iter().map(|(_, n)| n + 1).sum();
                assert!(runs[start..start + n + 1].chars().all(|c| c == side), "{reps}: {runs}");
                timed += n;
            }
            assert_eq!(timed, 2 * reps);
        }
    }

    #[test]
    fn no_timed_run_is_an_error_and_a_failing_run_stops_the_timing() {
        assert!(times(0, || Ok(()), || Ok(())).is_err());
        let e = times(3, || Ok(()), || Err(CoreError::new(INVALID_ARGUMENT, "b failed"))).unwrap_err();
        assert!(e.to_string().contains("b failed"), "{e}");
    }
}
