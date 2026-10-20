//! Score-level CPU benchmark paired with tests/crf-reference/Speed.java.
use std::{hint::black_box, time::Instant};
use turbo_sequence::crf::{CrfConstraints, CrfLattice, CrfScores};
use turbo_sequence::namefind::NameCodec;
fn main() {
    let repeats: usize = std::env::args().nth(1).unwrap_or("1000".into()).parse().unwrap();
    let labels: Vec<_> = (0..9).map(|k| format!("t{k}-unit")).collect();
    let mask = CrfConstraints::new(&labels, NameCodec::Bilou).unwrap();
    let emissions: Vec<Vec<_>> =
        (0..32).map(|t| (0..9).map(|k| ((t * 9 + k + 1) as f64 * 0.37).sin()).collect()).collect();
    let transitions: Vec<_> = (0..81).map(|i| ((i + 1) as f64 * 0.21).cos() * 0.2).collect();
    let start: Vec<_> = (0..9).map(|k| k as f64 * 0.03).collect();
    let end: Vec<_> = (0..9).map(|k| -(k as f64) * 0.02).collect();
    let run = || {
        let lattice = CrfLattice::new(
            CrfScores { emissions: &emissions, transitions: &transitions, start: &start, end: &end },
            &mask,
        )
        .unwrap();
        (lattice.marginals(), lattice.best_paths(1).unwrap())
    };
    let (p, path) = run();
    println!("path={:?}\nmarginal={:?}", path[0].labels, p[0]);
    for _ in 0..200 {
        black_box(run());
    }
    let time = Instant::now();
    for _ in 0..repeats {
        black_box(run());
    }
    println!("repeats={repeats} seconds={:.9}", time.elapsed().as_secs_f64());
}
