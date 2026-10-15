//! The matcher against a direct reading of what a pattern means: random
//! patterns over a three-word alphabet, random documents, and every run
//! of tokens checked by a recursive matcher written from the operators'
//! definitions, with spaCy's greedy filters written out again. The phrase
//! matcher is checked against a search at every start.

mod common;

use common::Doc;
use turbo_matcher::{Attr, Field, Greedy, Matcher, PhraseMatcher, Tok};

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

/// One pattern token as the reference reads it: a test, whether it is
/// negated, and how many tokens in a row it takes.
#[derive(Clone)]
struct Step {
    want: Want,
    negated: bool,
    min: usize,
    max: Option<usize>,
}

impl Step {
    fn ok(&self, w: &str) -> bool {
        self.want.ok(w) != self.negated
    }
}

fn random_pattern(rng: &mut Rng) -> (Vec<Tok>, Vec<Step>) {
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
                (Tok::attr_in(Attr::Text, [a, b]), Want::In(vec![a, b]))
            }
            _ => {
                let a = pick(rng);
                (Tok::attr_not_in(Attr::Text, [a]), Want::NotIn(vec![a]))
            }
        };
        // `!` here may take a repeat too, which spaCy's one `OP` cannot say.
        let negated = !matches!(want, Want::Any) && rng.below(5) == 0;
        let tok = if negated { !tok } else { tok };
        let (min, max) = match rng.below(6) {
            0 => (0, Some(1)),
            1 => (0, None),
            2 => (1, None),
            3 => {
                let min = rng.below(3) as u32;
                let max = if rng.below(3) == 0 { None } else { Some(min.max(1) + rng.below(2) as u32) };
                (min, max)
            }
            _ => (1, Some(1)),
        };
        pattern.push(tok.repeat(min, max));
        reference.push(Step { want, negated, min: min as usize, max: max.map(|m| m as usize) });
    }
    (pattern, reference)
}

/// Whether the pattern accepts exactly these words.
fn accepts(p: &[Step], w: &[&str]) -> bool {
    let Some((step, rest)) = p.split_first() else { return w.is_empty() };
    let max = step.max.map_or(w.len(), |m| m.min(w.len()));
    (step.min..=max).any(|k| w[..k].iter().all(|x| step.ok(x)) && accepts(rest, &w[k..]))
}

/// spaCy's greedy filter, written out again.
fn greedy(mut spans: Vec<(usize, usize)>, g: Greedy) -> Vec<(usize, usize)> {
    match g {
        Greedy::First => spans.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1))),
        Greedy::Longest => spans.sort_by(|a, b| (b.1 - b.0).cmp(&(a.1 - a.0)).then(a.0.cmp(&b.0))),
        Greedy::All => return spans,
    }
    let mut taken = [false; 64];
    let mut kept = Vec::new();
    for (s, e) in spans {
        if (s..e).all(|i| !taken[i]) {
            (s..e).for_each(|i| taken[i] = true);
            kept.push((s, e));
        }
    }
    kept.sort();
    kept
}

#[test]
fn every_run_the_patterns_accept_and_nothing_else() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut checked = 0;
    for round in 0..3000 {
        // One to three patterns under one label: a run is reported once.
        let mut m = Matcher::new();
        let mut references = Vec::new();
        for _ in 0..1 + rng.below(3) {
            let (pattern, reference) = random_pattern(&mut rng);
            if m.add("R", &pattern).is_ok() {
                references.push(reference);
            }
        }
        if references.is_empty() {
            continue;
        }
        let filter = [Greedy::First, Greedy::Longest, Greedy::All][rng.below(3) as usize];
        m.set_greedy(m.label("R").unwrap(), filter);
        let n = rng.below(10) as usize;
        let words: Vec<&str> = (0..n).map(|_| WORDS[rng.below(3) as usize]).collect();
        let mut want = Vec::new();
        for s in 0..n {
            for e in s + 1..=n {
                if references.iter().any(|r| accepts(r, &words[s..e])) {
                    want.push((s, e));
                }
            }
        }
        let want = greedy(want, filter);
        let got: Vec<(usize, usize)> = Doc::words(&words).run(&m).iter().map(|x| (x.1, x.2)).collect();
        assert_eq!(got, want, "round {round}: {words:?}");
        checked += 1;
    }
    assert!(checked > 2900, "{checked} rounds checked");
}

#[test]
fn phrases_found_at_every_start() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for round in 0..500 {
        let mut m = PhraseMatcher::new(Field::Attr(Attr::Text));
        let mut phrases = Vec::new();
        for _ in 0..1 + rng.below(6) {
            let len = 1 + rng.below(3) as usize;
            let p: Vec<&str> = (0..len).map(|_| WORDS[rng.below(3) as usize]).collect();
            let label = ["X", "Y"][rng.below(2) as usize];
            m.add(label, &p).unwrap();
            phrases.push((p, label));
        }
        let n = rng.below(12) as usize;
        let words: Vec<&str> = (0..n).map(|_| WORDS[rng.below(3) as usize]).collect();
        let mut want = Vec::new();
        for (p, label) in &phrases {
            for s in 0..n {
                if words[s..].starts_with(p) {
                    want.push((label.to_string(), s, s + p.len()));
                }
            }
        }
        want.sort();
        want.dedup();
        let mut got = Doc::words(&words).run_phrases(&m);
        got.sort();
        assert_eq!(got, want, "round {round}: {words:?}");
    }
}
