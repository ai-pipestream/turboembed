//! Patterns that make a backtracking matcher exponential, run here within
//! the bound the state set simulation promises: a start that reads `k`
//! tokens costs at most `2 * k * states` state visits.

mod common;

use common::Doc;
use turbo_matcher::{Matcher, Tok};

/// The most visits a call over `n` tokens may take: each start `s` reads
/// at most `n - s` tokens.
fn bound(m: &Matcher, n: usize) -> u64 {
    let reads: u64 = (1..=n as u64).sum();
    2 * reads * m.states() as u64
}

fn run(m: &Matcher, doc: &Doc) -> (usize, u64) {
    let input = doc.input();
    let (found, work) = m.find_in_counting(&input, input.all()).unwrap();
    assert!(work <= bound(m, input.len()), "{work} visits, over the bound {}", bound(m, input.len()));
    (found.len(), work)
}

fn a_times(n: usize) -> Doc {
    Doc::words(&vec!["a"; n])
}

#[test]
fn nested_optionals() {
    // `a?{n} a{n}` over `a{n}`: a backtracking matcher tries 2^n ways to
    // spend the optional tokens before the required ones fit.
    let n = 40;
    let mut pattern = vec![Tok::text("a").opt(); n];
    pattern.extend(vec![Tok::text("a"); n]);
    let mut m = Matcher::new();
    m.add("EXP", &pattern).unwrap();
    let doc = a_times(n);
    let found = m.find(&doc.input()).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!((found[0].tokens.start.get(), found[0].tokens.end.get()), (0, n));
    run(&m, &doc);
}

#[test]
fn adjacent_stars_with_no_end() {
    // Ten `a*` in a row, then a `b` the text never has: a backtracking
    // matcher tries every way to share the `a`s among the stars.
    let mut pattern = vec![Tok::text("a").star(); 10];
    pattern.push(Tok::text("b"));
    let mut m = Matcher::new();
    m.add("NONE", &pattern).unwrap();
    let (found, _) = run(&m, &a_times(400));
    assert_eq!(found, 0);
}

#[test]
fn nested_repeats_and_wildcards() {
    // Overlapping repeats, a plus and wildcard stars that can all take
    // the same `a`s, then a `z` the text never has.
    let pattern = vec![
        Tok::text("a").repeat(1, Some(5)),
        Tok::text("a").repeat(1, Some(5)),
        Tok::text("a").plus(),
        Tok::any().star(),
        Tok::any().star(),
        Tok::text("z"),
    ];
    let mut m = Matcher::new();
    m.add("NONE", &pattern).unwrap();
    let (found, _) = run(&m, &a_times(300));
    assert_eq!(found, 0);
}

#[test]
fn work_grows_linearly_with_one_start() {
    // "x" then `a*` eight times and a `b`, over "x" and `n` `a`s: only the
    // first token can start a match, so twice the tokens take at most
    // about twice the visits.
    let mut pattern = vec![Tok::text("x")];
    pattern.extend(vec![Tok::text("a").star(); 8]);
    pattern.push(Tok::text("b"));
    let mut m = Matcher::new();
    m.add("NONE", &pattern).unwrap();
    let work = |n: usize| {
        let mut words = vec!["x"];
        words.extend(vec!["a"; n]);
        run(&m, &Doc::words(&words)).1
    };
    let (w1, w2) = (work(2000), work(4000));
    assert!(w2 <= 2 * w1 + 2 * m.states() as u64, "{w1} then {w2}");
    assert!(w1 <= 2 * 2001 * m.states() as u64 + 2001, "{w1}");
}
