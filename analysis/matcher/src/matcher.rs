//! The token pattern matcher, spaCy's `Matcher`.

use std::sync::Mutex;

use turbo_analysis::IxRange;
use turbo_analysis_layers::Tokens;

use crate::align::{Step, align};
use crate::entries::Entries;
use crate::error::{MatchError, PatternError};
use crate::input::{Columns, Input, Needs};
use crate::nfa::{Nfa, Sim};
use crate::pattern::Tok;
use crate::vocab::{Label, Labels, Vocab};

/// Which of a label's matches are reported, spaCy's `greedy`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Greedy {
    /// Every match: every start, and every end a pattern can reach from it.
    #[default]
    All,
    /// Matches taken by start, the longest first at each start, skipping
    /// any that shares a token with one taken: spaCy's `FIRST`.
    First,
    /// Matches taken longest first, the earlier first among equals,
    /// skipping any that shares a token with one taken: spaCy's `LONGEST`.
    Longest,
}

/// A pattern's number in its matcher, in the order patterns were added.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PatternId(pub u32);

/// One match of a token pattern: the tokens it covers, the label it was
/// added under, and the pattern that matched (the first added, when
/// several patterns of one label match the same tokens).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Match {
    pub tokens: IxRange<Tokens>,
    pub label: Label,
    pub pattern: PatternId,
}

struct PatternEntry {
    label: Label,
    needs: Needs,
    /// Its tokens, in order, for alignments.
    steps: Box<[Step]>,
}

/// Token patterns under labels, matched over a document's tokens.
///
/// A match is a run of tokens that the whole pattern matches; every start
/// and every end is reported, as spaCy's `Matcher` does without `greedy`,
/// except empty runs. Matches come in token order (by start, then end,
/// then label), one per label and run of tokens.
#[derive(Default)]
pub struct Matcher {
    vocab: Vocab,
    labels: Labels,
    greedy: Vec<Greedy>,
    nfa: Nfa,
    patterns: Vec<PatternEntry>,
    /// The taking states the active patterns' entries reach.
    entry: Entries,
    needs: Needs,
    /// Working sets reused between calls; a call that finds it in use
    /// makes its own.
    sim: Mutex<Sim>,
}

impl Matcher {
    /// The most states one pattern may compile to. A repeat `{n,m}` takes
    /// about `2m` states.
    pub const MAX_STATES: usize = 1 << 16;

    pub fn new() -> Matcher {
        Matcher::default()
    }

    /// Adds `pattern` under `label`, making the label when it is new.
    pub fn add(&mut self, label: &str, pattern: &[Tok]) -> Result<PatternId, PatternError> {
        let id = self.patterns.len() as u32;
        let mut needs = Needs::default();
        let first_tok = self.nfa.toks();
        let entry = self.nfa.add(pattern, id, Matcher::MAX_STATES, &mut self.vocab, &mut needs)?;
        let n = pattern.len() as u32;
        let steps = (0..n)
            .map(|i| {
                let t = &pattern[i as usize];
                Step { tok: first_tok + n - 1 - i, min: t.min, max: t.max }
            })
            .collect();
        let label = self.labels.get_or_add(label);
        if self.greedy.len() <= label.0 as usize {
            self.greedy.resize(label.0 as usize + 1, Greedy::All);
        }
        self.patterns.push(PatternEntry { label, needs, steps });
        let sim = self.sim.get_mut().unwrap_or_else(|e| e.into_inner());
        let reached = sim.entry(&self.nfa, [entry]);
        self.entry.extend(&self.nfa, reached);
        self.needs = union(self.needs, needs);
        Ok(PatternId(id))
    }

    /// Sets which matches of `label` are reported.
    pub fn set_greedy(&mut self, label: Label, greedy: Greedy) {
        self.greedy[label.0 as usize] = greedy;
    }

    /// Removes a label and every pattern under it.
    pub fn remove(&mut self, label: &str) -> Result<(), MatchError> {
        self.labels.remove(label)?;
        let Matcher { patterns, nfa, entry, labels, .. } = self;
        entry.retain(|s| labels.is_active(patterns[nfa.owner[s as usize] as usize].label));
        self.needs = self
            .patterns
            .iter()
            .filter(|p| self.labels.is_active(p.label))
            .fold(Needs::default(), |n, p| union(n, p.needs));
        Ok(())
    }

    /// The label called `name`.
    pub fn label(&self, name: &str) -> Option<Label> {
        self.labels.get(name)
    }

    pub fn label_name(&self, label: Label) -> &str {
        self.labels.name(label)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.labels.get(name).is_some()
    }

    /// How many labels the matcher has, as spaCy's `len(matcher)`.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.len() == 0
    }

    /// How many automaton states the patterns compiled to, removed ones
    /// included.
    pub fn states(&self) -> usize {
        self.nfa.len()
    }

    /// The label a pattern was added under.
    pub fn pattern_label(&self, p: PatternId) -> Label {
        self.patterns[p.0 as usize].label
    }

    /// The matches over all of `input`'s tokens.
    pub fn find(&self, input: &Input<'_>) -> Result<Vec<Match>, MatchError> {
        self.find_in(input, input.all())
    }

    /// The matches that lie inside tokens `range`, as spaCy matches a
    /// `Span`.
    pub fn find_in(&self, input: &Input<'_>, range: IxRange<Tokens>) -> Result<Vec<Match>, MatchError> {
        self.find_in_counting(input, range).map(|(m, _)| m)
    }

    /// [`Matcher::find_in`], with the number of automaton state visits it
    /// took. A start that reads `k` tokens costs at most `2 * k *
    /// states()` visits (each state tested once and added once per
    /// token), whatever the operators, which is what keeps quantified
    /// patterns from going exponential.
    pub fn find_in_counting(&self, input: &Input<'_>, range: IxRange<Tokens>) -> Result<(Vec<Match>, u64), MatchError> {
        let range = input.range(range)?;
        if self.entry.is_empty() || range.is_empty() {
            return Ok((Vec::new(), 0));
        }
        let cols = Columns::build(input, &range, self.needs, &self.vocab)?;
        let mut own = None;
        let mut guard = self.sim.try_lock().ok();
        let sim = match guard.as_deref_mut() {
            Some(sim) => sim,
            None => own.insert(Sim::default()),
        };
        sim.work = 0;
        let mut ends = Vec::new();
        let mut entry = Vec::new();
        let mut found: Vec<(u32, u32, Label, u32)> = Vec::new();
        let n = range.len();
        for start in 0..n {
            self.entry.at(&cols, start, &mut entry);
            if entry.is_empty() {
                continue;
            }
            ends.clear();
            sim.run(&self.nfa, &entry, &cols, input, range.start, start, n, &mut ends);
            let s = (range.start + start) as u32;
            found.extend(ends.iter().map(|&(e, p)| (s, e, self.patterns[p as usize].label, p)));
        }
        let work = sim.work;
        drop(guard);
        found.sort_unstable();
        found.dedup_by_key(|m| (m.0, m.1, m.2));
        let found = self.filter_greedy(found, &range);
        let matches = found
            .into_iter()
            .map(|(s, e, label, p)| Match { tokens: IxRange::new(s, e), label, pattern: PatternId(p) })
            .collect();
        Ok((matches, work))
    }

    /// Which token of its pattern took each token of match `m`, as
    /// spaCy's `with_alignments`: entry `i` is the index in the pattern
    /// (as added) of the [`Tok`] that took token `m.tokens.start + i`.
    /// Where the pattern can take the tokens more than one way, the way
    /// is a backtracking regular expression's with greedy operators: each
    /// `Tok`, from the first, takes as many tokens as it can while the
    /// rest of the pattern can still take the rest. `None` when `m`'s
    /// pattern cannot take exactly those tokens (a match made by hand).
    ///
    /// This is a pass of its own over one match, after [`Matcher::find`],
    /// so matching itself keeps no paths; it costs a few words of memory
    /// per pattern token and matched token.
    pub fn align(&self, input: &Input<'_>, m: &Match) -> Result<Option<Vec<u32>>, MatchError> {
        let range = input.range(m.tokens)?;
        let Some(entry) = self.patterns.get(m.pattern.0 as usize) else { return Ok(None) };
        if range.is_empty() {
            return Ok(None);
        }
        let cols = Columns::build(input, &range, entry.needs, &self.vocab)?;
        Ok(align(&self.nfa, &entry.steps, &cols, input, range.start, range.len()))
    }

    /// Applies each label's [`Greedy`] to `found`, sorted by start, end
    /// and label, and keeps that order.
    fn filter_greedy(
        &self,
        found: Vec<(u32, u32, Label, u32)>,
        range: &std::ops::Range<usize>,
    ) -> Vec<(u32, u32, Label, u32)> {
        let greedy = |l: Label| self.greedy[l.0 as usize];
        if found.iter().all(|m| greedy(m.2) == Greedy::All) {
            return found;
        }
        let (mut kept, mut filtered): (Vec<_>, Vec<_>) = found.into_iter().partition(|m| greedy(m.2) == Greedy::All);
        // By label, keeping the order of start and end within each.
        filtered.sort_by_key(|m| m.2);
        let mut taken = Vec::new();
        for group in filtered.chunk_by(|a, b| a.2 == b.2) {
            keep_greedy(group, |m| (m.0, m.1), greedy(group[0].2), range.start, &mut taken, &mut kept);
        }
        kept.sort_unstable();
        kept
    }
}

/// spaCy's greedy sweep over runs of tokens (`tokens(item)` is `(start,
/// end)`, both at or after `base`): the items sorted by start, longest
/// first (`First`), or longest, earliest first (`Longest`), each kept,
/// appended to `out`, when none of its tokens is in one kept before it.
/// The sort is stable, so of two items over the same tokens the earlier
/// in `items` is kept, as spaCy's sorts keep their input order. `All`
/// keeps every item. `taken` is working space.
pub(crate) fn keep_greedy<T: Copy>(
    items: &[T],
    tokens: impl Fn(&T) -> (u32, u32),
    greedy: Greedy,
    base: usize,
    taken: &mut Vec<bool>,
    out: &mut Vec<T>,
) {
    let mut order: Vec<T> = items.to_vec();
    match greedy {
        Greedy::All => {
            out.extend(order);
            return;
        }
        Greedy::First => order.sort_by_key(|m| {
            let (s, e) = tokens(m);
            (s, std::cmp::Reverse(e))
        }),
        Greedy::Longest => order.sort_by_key(|m| {
            let (s, e) = tokens(m);
            (std::cmp::Reverse(e - s), s)
        }),
    }
    let top = order.iter().map(|m| tokens(m).1 as usize).max().unwrap_or(base);
    taken.clear();
    taken.resize(top - base, false);
    for m in order {
        let (s, e) = tokens(&m);
        let span = &mut taken[s as usize - base..e as usize - base];
        if !span.contains(&true) {
            span.fill(true);
            out.push(m);
        }
    }
}

fn union(a: Needs, b: Needs) -> Needs {
    Needs { attrs: a.attrs | b.attrs, flags: a.flags | b.flags, iob: a.iob || b.iob, length: a.length || b.length }
}
