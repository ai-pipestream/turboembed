//! The matcher against a direct reading of what a pattern means: random
//! patterns over a three-word alphabet, random documents, and every run of
//! tokens checked by a recursive matcher written from the operators'
//! definitions. The alignment of every match is checked to be a parse of
//! the run by the pattern, and the phrase matcher is checked against a
//! search at every start.

mod common;

use common::doc;
use turbo_matcher::{Attr, Greedy, Matcher, Op, Pattern, PhraseMatcher, PhraseOn, Tok};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

const WORDS: [&str; 3] = ["a", "b", "c"];

/// A token test as the reference reads it.
#[derive(Clone)]
enum Want {
    Any,
    Is(&'static str),
    In(Vec<&'static str>),
    NotIn(Vec<&'static str>),
}

impl Want {
    fn ok(&self, w: &str) -> bool {
        match self {
            Want::Any => true,
            Want::Is(x) => *x == w,
            Want::In(xs) => xs.contains(&w),
            Want::NotIn(xs) => !xs.contains(&w),
        }
    }
}

fn random_pattern(rng: &mut Rng) -> (Pattern, Vec<(Want, Op)>) {
    let len = 1 + rng.below(4) as usize;
    let mut pattern = Vec::new();
    let mut reference = Vec::new();
    for _ in 0..len {
        let pick = |rng: &mut Rng| WORDS[rng.below(3) as usize];
        let (tok, want) = match rng.below(4) {
            0 => (Tok::any(), Want::Any),
            1 => {
                let w = pick(rng);
                (Tok::text(w), Want::Is(w))
            }
            2 => {
                let (a, b) = (pick(rng), pick(rng));
                (Tok::any().is_in(Attr::Text, &[a, b]), Want::In(vec![a, b]))
            }
            _ => {
                let a = pick(rng);
                (Tok::any().not_in(Attr::Text, &[a]), Want::NotIn(vec![a]))
            }
        };
        let op = match rng.below(7) {
            0 => Op::Optional,
            1 => Op::ZeroOrMore,
            2 => Op::OneOrMore,
            3 => Op::Not,
            4 => {
                let min = rng.below(3) as u32;
                let max = if rng.below(3) == 0 { None } else { Some(min.max(1) + rng.below(2) as u32) };
                Op::Repeat { min, max }
            }
            _ => Op::One,
        };
        // `!` on a test that passes everything never matches: keep it
        // only with a real test.
        let op = if op == Op::Not && matches!(want, Want::Any) { Op::One } else { op };
        pattern.push(tok.op(op));
        reference.push((want, op));
    }
    (pattern, reference)
}

/// Whether the pattern accepts exactly these words.
fn accepts(p: &[(Want, Op)], w: &[&str]) -> bool {
    let Some(((want, op), rest)) = p.split_first() else { return w.is_empty() };
    let ok0 = !w.is_empty() && want.ok(w[0]);
    match *op {
        Op::One => ok0 && accepts(rest, &w[1..]),
        Op::Not => !w.is_empty() && !want.ok(w[0]) && accepts(rest, &w[1..]),
        Op::Optional => accepts(rest, w) || (ok0 && accepts(rest, &w[1..])),
        Op::ZeroOrMore => accepts(rest, w) || (ok0 && accepts(p, &w[1..])),
        Op::OneOrMore => {
            ok0 && {
                let star = [(want.clone(), Op::ZeroOrMore)];
                let tail: Vec<(Want, Op)> = star.iter().cloned().chain(rest.iter().cloned()).collect();
                accepts(&tail, &w[1..])
            }
        }
        Op::Repeat { min, max } => {
            let max = max.map_or(w.len(), |m| (m as usize).min(w.len()));
            (min as usize..=max).any(|k| k <= w.len() && w[..k].iter().all(|x| want.ok(x)) && accepts(rest, &w[k..]))
        }
    }
}

/// Whether an alignment is a parse of the words by the pattern.
fn parses(p: &[(Want, Op)], w: &[&str], align: &[u16]) -> bool {
    if align.len() != w.len() || !align.windows(2).all(|x| x[0] <= x[1]) {
        return false;
    }
    p.iter().enumerate().all(|(i, (want, op))| {
        let taken: Vec<&str> = w.iter().zip(align).filter(|&(_, &a)| a as usize == i).map(|(x, _)| *x).collect();
        let n = taken.len() as u32;
        let tests = match op {
            Op::Not => taken.iter().all(|x| !want.ok(x)),
            _ => taken.iter().all(|x| want.ok(x)),
        };
        let count = match *op {
            Op::One | Op::Not => n == 1,
            Op::Optional => n <= 1,
            Op::ZeroOrMore => true,
            Op::OneOrMore => n >= 1,
            Op::Repeat { min, max } => n >= min && max.is_none_or(|m| n <= m),
        };
        tests && count
    })
}

/// spaCy's greedy filter, written out again.
fn greedy(mut spans: Vec<(u32, u32)>, g: Greedy) -> Vec<(u32, u32)> {
    match g {
        Greedy::First => spans.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1))),
        Greedy::Longest => spans.sort_by(|a, b| (b.1 - b.0).cmp(&(a.1 - a.0)).then(a.0.cmp(&b.0))),
    }
    let mut taken = [false; 64];
    let mut kept = Vec::new();
    for (s, e) in spans {
        if (s..e).all(|i| !taken[i as usize]) {
            (s..e).for_each(|i| taken[i as usize] = true);
            kept.push((s, e));
        }
    }
    kept.sort();
    kept
}

#[test]
fn every_run_the_pattern_accepts_and_nothing_else() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for round in 0..3000 {
        let (pattern, reference) = random_pattern(&mut rng);
        let filter = match rng.below(3) {
            0 => Some(Greedy::First),
            1 => Some(Greedy::Longest),
            _ => None,
        };
        let mut b = Matcher::builder();
        if b.add("R", vec![pattern], filter).is_err() {
            continue;
        }
        let m = b.build().unwrap();
        let n = rng.below(9) as usize;
        let words: Vec<&str> = (0..n).map(|_| WORDS[rng.below(3) as usize]).collect();
        let d = doc(&words);
        let mut want = Vec::new();
        for s in 0..n {
            for e in s + 1..=n {
                if accepts(&reference, &words[s..e]) {
                    want.push((s as u32, e as u32));
                }
            }
        }
        if let Some(g) = filter {
            want = greedy(want, g);
        }
        let got = m.find(&d.seq()).unwrap();
        let spans: Vec<(u32, u32)> = got.iter().map(|x| (x.start, x.end)).collect();
        assert_eq!(spans, want, "round {round}: {words:?}");
        for x in &got {
            let a = m.align(&d.seq(), x).unwrap().expect("an alignment for every match");
            assert!(parses(&reference, &words[x.start as usize..x.end as usize], &a), "round {round}: {x:?} {a:?}");
        }
    }
}

#[test]
fn phrases_found_at_every_start() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for round in 0..500 {
        let mut b = PhraseMatcher::builder(PhraseOn::Attr(Attr::Text));
        let mut phrases = Vec::new();
        for _ in 0..1 + rng.below(6) {
            let len = 1 + rng.below(3) as usize;
            let p: Vec<&str> = (0..len).map(|_| WORDS[rng.below(3) as usize]).collect();
            let label = ["X", "Y"][rng.below(2) as usize];
            b.add(label, &p).unwrap();
            phrases.push((p, label));
        }
        let m = b.build().unwrap();
        let n = rng.below(12) as usize;
        let words: Vec<&str> = (0..n).map(|_| WORDS[rng.below(3) as usize]).collect();
        let mut want = Vec::new();
        for (p, label) in &phrases {
            for s in 0..n {
                if words[s..].starts_with(p) {
                    want.push((s as u32, (s + p.len()) as u32, label.to_string()));
                }
            }
        }
        want.sort();
        want.dedup();
        let mut got: Vec<(u32, u32, String)> =
            common::phrase_found(&m, &doc(&words)).into_iter().map(|(l, s, e)| (s, e, l)).collect();
        got.sort();
        assert_eq!(got, want, "round {round}: {words:?}");
    }
}
