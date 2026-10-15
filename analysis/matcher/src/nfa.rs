//! Patterns compiled to a nondeterministic automaton over tokens, and its
//! simulation.
//!
//! Each pattern becomes a Thompson automaton: a state that takes one
//! token passing a token's tests, states that split into two without
//! taking a token, and one accepting state. The simulation keeps the set
//! of states the automaton can be in after each token, each state at most
//! once, rather than trying one path and backing up on failure. From one
//! start, a token costs at most one visit per state, so a run of `k`
//! tokens costs `O(k * states)` visits whatever the operators: no pattern
//! can make it exponential, as nested or adjacent quantifiers make a
//! backtracking matcher.

use crate::attr::{Attr, Cmp, Flag, Iob};
use crate::error::PatternError;
use crate::input::{Columns, Input, Needs};
use crate::pattern::{Test, TextTest, Tok};
use crate::vocab::Vocab;

/// A test with its strings replaced by vocabulary ids.
pub(crate) enum Compiled {
    Is(Attr, u32),
    /// Sorted ids.
    In(Attr, Box<[u32]>),
    NotIn(Attr, Box<[u32]>),
    Flag(Flag, bool),
    Iob(Iob),
    Length(Cmp, u32),
    Text(TextTest),
}

/// A token's compiled tests.
pub(crate) struct TokTests {
    tests: Box<[Compiled]>,
    negated: bool,
}

impl TokTests {
    /// Whether token `abs` (at `rel` in the columns) passes.
    #[inline]
    fn passes(&self, cols: &Columns, input: &Input<'_>, rel: usize, abs: usize) -> bool {
        let all = self.tests.iter().all(|t| match t {
            Compiled::Is(a, id) => cols.values[*a as usize][rel] == *id,
            Compiled::In(a, ids) => ids.binary_search(&cols.values[*a as usize][rel]).is_ok(),
            Compiled::NotIn(a, ids) => ids.binary_search(&cols.values[*a as usize][rel]).is_err(),
            Compiled::Flag(f, on) => (cols.flags[rel] & f.bit() != 0) == *on,
            Compiled::Iob(i) => cols.iob[rel] == *i,
            Compiled::Length(c, n) => c.holds(cols.chars[rel], *n),
            Compiled::Text(f) => f(input.token(abs)),
        });
        all != self.negated
    }
}

/// The sorted, distinct ids of `values`. A pattern's values are compared
/// as written: `Tok::attr(Attr::Lower, "Boston")` matches no token, as
/// spaCy's `{"LOWER": "Boston"}` does not.
fn ids(values: &[Box<str>], vocab: &mut Vocab) -> Box<[u32]> {
    let mut v: Vec<u32> = values.iter().map(|s| vocab.intern(s)).collect();
    v.sort_unstable();
    v.dedup();
    v.into()
}

fn compile_tests(tok: &Tok, vocab: &mut Vocab, needs: &mut Needs) -> TokTests {
    let tests = tok
        .tests
        .iter()
        .map(|t| match t {
            Test::Is(a, s) => {
                needs.attr(*a);
                Compiled::Is(*a, vocab.intern(s))
            }
            Test::In(a, s) => {
                needs.attr(*a);
                Compiled::In(*a, ids(s, vocab))
            }
            Test::NotIn(a, s) => {
                needs.attr(*a);
                Compiled::NotIn(*a, ids(s, vocab))
            }
            Test::Flag(f, on) => {
                needs.flags |= f.bit();
                Compiled::Flag(*f, *on)
            }
            Test::Iob(i) => {
                needs.iob = true;
                Compiled::Iob(*i)
            }
            Test::Length(c, n) => {
                needs.length = true;
                Compiled::Length(*c, *n)
            }
            Test::Text(f) => Compiled::Text(f.clone()),
        })
        .collect();
    TokTests { tests, negated: tok.negated }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum State {
    /// Takes one token that passes `toks[tok]`, then goes to `next`.
    Take { tok: u32, next: u32 },
    /// Goes to both, taking nothing.
    Split(u32, u32),
    /// Pattern `.0` has matched.
    Accept(u32),
}

/// The automata of all of a matcher's patterns, side by side.
#[derive(Default)]
pub(crate) struct Nfa {
    pub(crate) states: Vec<State>,
    /// The pattern each state belongs to.
    pub(crate) owner: Vec<u32>,
    toks: Vec<TokTests>,
}

impl Nfa {
    pub(crate) fn len(&self) -> usize {
        self.states.len()
    }

    /// The values taking state `s` needs its token to have, when one of
    /// its tests says: the attribute and ids of its first `Is` test, or
    /// else of its first `In` test. `None` for a negated token or one
    /// with neither test.
    pub(crate) fn required_values(&self, s: u32) -> Option<(Attr, &[u32])> {
        let State::Take { tok, .. } = self.states[s as usize] else { return None };
        let tests = &self.toks[tok as usize];
        if tests.negated {
            return None;
        }
        let is = tests.tests.iter().find_map(|t| match t {
            Compiled::Is(a, id) => Some((*a, std::slice::from_ref(id))),
            _ => None,
        });
        is.or_else(|| {
            tests.tests.iter().find_map(|t| match t {
                Compiled::In(a, ids) => Some((*a, &ids[..])),
                _ => None,
            })
        })
    }

    fn push(&mut self, s: State, pattern: u32) -> u32 {
        self.states.push(s);
        self.owner.push(pattern);
        (self.states.len() - 1) as u32
    }

    /// Adds pattern number `pattern` and returns its entry state, with what
    /// it reads added to `needs`. Nothing is added when it is refused.
    pub(crate) fn add(
        &mut self,
        toks: &[Tok],
        pattern: u32,
        max_states: usize,
        vocab: &mut Vocab,
        needs: &mut Needs,
    ) -> Result<u32, PatternError> {
        if toks.is_empty() {
            return Err(PatternError::Empty);
        }
        let mut states = 1usize;
        for t in toks {
            if let Some(max) = t.max
                && (max < t.min || max == 0)
            {
                return Err(PatternError::Repeat { min: t.min, max });
            }
            let copies = t.min as usize + t.max.map_or(1, |m| (m - t.min) as usize);
            states = states.saturating_add(copies.saturating_mul(2));
        }
        if states > max_states {
            return Err(PatternError::TooLarge { states });
        }
        let mut exit = self.push(State::Accept(pattern), pattern);
        for t in toks.iter().rev() {
            let tok = self.toks.len() as u32;
            self.toks.push(compile_tests(t, vocab, needs));
            match t.max {
                None => {
                    // `a*`: a split whose first branch takes `a` and comes back.
                    let split = self.push(State::Split(0, 0), pattern);
                    let take = self.push(State::Take { tok, next: split }, pattern);
                    self.states[split as usize] = State::Split(take, exit);
                    exit = split;
                }
                Some(max) => {
                    // `a{0,m}` nested: after each optional `a`, either another
                    // or out, so a run of `a`s has one path through.
                    let out = exit;
                    for _ in t.min..max {
                        let take = self.push(State::Take { tok, next: exit }, pattern);
                        exit = self.push(State::Split(take, out), pattern);
                    }
                }
            }
            for _ in 0..t.min {
                exit = self.push(State::Take { tok, next: exit }, pattern);
            }
        }
        Ok(exit)
    }
}

/// The simulation's working sets, reused across starts and calls.
#[derive(Default)]
pub(crate) struct Sim {
    cur: Vec<u32>,
    next: Vec<u32>,
    stack: Vec<u32>,
    /// The step at which each state was last added, so one step adds it
    /// once.
    mark: Vec<u32>,
    step: u32,
    accepted: Vec<u32>,
    /// State visits: closure steps and token tests.
    pub(crate) work: u64,
}

impl Sim {
    fn new_step(&mut self, states: usize) {
        if self.mark.len() < states {
            self.mark.resize(states, 0);
        }
        self.step = self.step.wrapping_add(1);
        if self.step == 0 {
            self.mark.fill(0);
            self.step = 1;
        }
    }

    /// Adds the states `from` reaches without taking a token to `next`
    /// (the taking ones) and to `accepted`.
    fn close(&mut self, nfa: &Nfa, from: u32) {
        self.stack.push(from);
        while let Some(s) = self.stack.pop() {
            if self.mark[s as usize] == self.step {
                continue;
            }
            self.mark[s as usize] = self.step;
            self.work += 1;
            match nfa.states[s as usize] {
                State::Take { .. } => self.next.push(s),
                State::Split(a, b) => {
                    self.stack.push(b);
                    self.stack.push(a);
                }
                State::Accept(p) => self.accepted.push(p),
            }
        }
    }

    /// The taking states the entries reach without taking a token.
    pub(crate) fn entry(&mut self, nfa: &Nfa, entries: impl IntoIterator<Item = u32>) -> Vec<u32> {
        self.new_step(nfa.len());
        self.next.clear();
        for e in entries {
            self.close(nfa, e);
        }
        // An accepting state reached before any token is an empty match,
        // which is never reported.
        self.accepted.clear();
        std::mem::take(&mut self.next)
    }

    /// Every match starting at token `start` (relative to the columns, at
    /// `base + start` in the input) and ending at or before `end`, as
    /// `(end, pattern)` pairs pushed to `out`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &mut self,
        nfa: &Nfa,
        entry: &[u32],
        cols: &Columns,
        input: &Input<'_>,
        base: usize,
        start: usize,
        end: usize,
        out: &mut Vec<(u32, u32)>,
    ) {
        self.cur.clear();
        self.cur.extend_from_slice(entry);
        for i in start..end {
            self.new_step(nfa.len());
            self.next.clear();
            self.accepted.clear();
            let cur = std::mem::take(&mut self.cur);
            for &s in &cur {
                self.work += 1;
                let State::Take { tok, next } = nfa.states[s as usize] else { unreachable!("a taking state") };
                if nfa.toks[tok as usize].passes(cols, input, i, base + i) {
                    self.close(nfa, next);
                }
            }
            self.cur = cur;
            for &p in &self.accepted {
                out.push(((base + i + 1) as u32, p));
            }
            if self.next.is_empty() {
                break;
            }
            std::mem::swap(&mut self.cur, &mut self.next);
        }
    }
}
