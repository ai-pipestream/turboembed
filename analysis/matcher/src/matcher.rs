//! The token pattern matcher: spaCy's `Matcher`.
//!
//! Each pattern compiles to a list of elements, one per token test, with
//! `+` and `{min,max}` written out as copies (`a+` is `a a*`, `a{2,3}` is
//! `a a a?`). Matching runs the pattern as a set of states from every
//! start the pattern can begin at: the states are bits, element `k` meaning
//! "the next token is tested by element `k`", and each token moves every
//! state at once. A start whose states die stops there, so most starts
//! cost one test. A match is every run of tokens `[start, end)` the
//! pattern accepts, so a pattern with `*` or `?` matches all the lengths it
//! can, as spaCy's does; `Greedy` keeps the leftmost or longest of
//! overlapping ones.

use std::collections::HashMap;
use std::fmt;
use std::hash::BuildHasherDefault;
use std::ops::Range;
use std::sync::Arc;

use regex::Regex;
use turbo_analysis::{Tag, TagSet};

use crate::attr::{Attr, Flag};
use crate::fuzzy;
use crate::pattern::{Cmp, MAX_ELEMENTS, Op, Pattern, PatternError, Test, Tok};
use crate::seq::{Buf, Sequence};
use crate::vocab::{Fnv, FoldHasher, NONE, Sym, Vocab};

/// A run of tokens `[start, end)` a rule matched, with the rule's label.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Match {
    pub start: u32,
    pub end: u32,
    pub label: Tag,
}

/// Which of a rule's overlapping matches to keep, as spaCy's `greedy`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Greedy {
    /// The leftmost, and of those starting together the longest; then the
    /// leftmost of what does not overlap the ones kept, and so on.
    First,
    /// The longest, and of equally long ones the leftmost; then the
    /// longest of what does not overlap the ones kept, and so on.
    Longest,
}

/// Why a matcher could not run on a sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MatchError {
    /// A rule reads an attribute the sequence does not have.
    Missing(Attr),
    /// A rule tests a flag the sequence cannot give.
    MissingFlag(Flag),
}

impl fmt::Display for MatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchError::Missing(a) => write!(f, "the rules read {a:?}, which the sequence does not have"),
            MatchError::MissingFlag(flag) => write!(f, "the rules test {flag:?}, which the sequence cannot give"),
        }
    }
}

impl std::error::Error for MatchError {}

struct Rule {
    label: String,
    patterns: Vec<Pattern>,
    greedy: Option<Greedy>,
}

/// Collects rules: each a label, its patterns and how to keep overlapping
/// matches. `build` compiles them.
#[derive(Default)]
pub struct MatcherBuilder {
    rules: Vec<Rule>,
    /// Each rule's position in `rules`, by label.
    index: HashMap<String, usize>,
}

impl MatcherBuilder {
    pub fn new() -> MatcherBuilder {
        MatcherBuilder::default()
    }

    /// Adds patterns under `label`. Adding to a label that has patterns
    /// adds to them, and sets its `greedy` to this one.
    pub fn add(&mut self, label: &str, patterns: Vec<Pattern>, greedy: Option<Greedy>) -> Result<(), PatternError> {
        if label.is_empty() {
            return Err(PatternError::EmptyLabel);
        }
        for p in &patterns {
            // Checked here so a bad pattern is reported where it is added.
            compile(p, &mut Vocab::default(), &mut Vec::new(), &mut Vec::new())?;
        }
        match self.index.get(label) {
            Some(&i) => {
                let r = &mut self.rules[i];
                r.patterns.extend(patterns);
                r.greedy = greedy;
            }
            None => {
                self.index.insert(label.into(), self.rules.len());
                self.rules.push(Rule { label: label.into(), patterns, greedy });
            }
        }
        Ok(())
    }

    /// Removes a label and its patterns; false when there was none.
    pub fn remove(&mut self, label: &str) -> bool {
        let Some(i) = self.index.remove(label) else { return false };
        self.rules.remove(i);
        for r in &self.rules[i..] {
            *self.index.get_mut(&r.label).unwrap() -= 1;
        }
        true
    }

    pub fn contains(&self, label: &str) -> bool {
        self.index.contains_key(label)
    }

    /// How many labels have patterns.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    pub fn build(self) -> Result<Matcher, PatternError> {
        if self.rules.len() > usize::from(u16::MAX) + 1 {
            return Err(PatternError::Labels);
        }
        let mut vocab = Vocab::default();
        let mut cols = Vec::new();
        let mut slow = Vec::new();
        let mut programs = Vec::new();
        let mut greedy = Vec::new();
        let mut config = Fnv::default();
        for (l, rule) in self.rules.iter().enumerate() {
            config.str(&rule.label);
            config.u64(rule.greedy.map_or(0, |g| g as u64 + 1));
            greedy.push(rule.greedy);
            for p in &rule.patterns {
                hash_pattern(&mut config, p);
                let elems = compile(p, &mut vocab, &mut cols, &mut slow)?;
                programs.push(Program::new(l as u16, elems));
            }
        }
        let labels: Vec<&str> = self.rules.iter().map(|r| r.label.as_str()).collect();
        let mut m = Matcher {
            labels: TagSet::new("matcher", &labels),
            greedy,
            vocab,
            cols,
            slow,
            programs,
            index: HashMap::default(),
            index_cols: Vec::new(),
            always: Vec::new(),
            reads: Vec::new(),
            flags: 0,
            config: config.0,
        };
        m.index_starts();
        m.collect_reads();
        Ok(m)
    }
}

fn hash_pattern(h: &mut Fnv, p: &Pattern) {
    h.u64(p.len() as u64);
    for t in p {
        h.str(&format!("{:?}", t.op));
        h.u64(t.tests.len() as u64);
        for test in &t.tests {
            h.str(&format!("{test:?}"));
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Quant {
    One,
    Opt,
    Star,
    Not,
}

/// A test compiled against the matcher's symbol columns.
#[derive(Clone)]
enum Check {
    Eq {
        col: u16,
        sym: Sym,
    },
    In {
        col: u16,
        syms: Box<[Sym]>,
    },
    NotIn {
        col: u16,
        syms: Box<[Sym]>,
    },
    Length(Cmp, u32),
    /// A regular expression or fuzzy test, by its index in `Matcher::slow`;
    /// its results are kept per token.
    Slow(u32),
}

enum Slow {
    Regex { attr: Attr, res: Vec<Regex>, none: bool },
    Fuzzy { attr: Attr, targets: Vec<Vec<char>>, max: Option<u32>, none: bool },
}

impl Slow {
    fn attr(&self) -> Attr {
        match self {
            Slow::Regex { attr: a, .. } | Slow::Fuzzy { attr: a, .. } => *a,
        }
    }
}

#[derive(Clone)]
struct Elem {
    quant: Quant,
    checks: Vec<Check>,
    flag_mask: u32,
    flag_want: u32,
    /// Two of its tests contradict each other, so no token passes.
    never: bool,
    /// The index of the `Tok` it came from.
    origin: u16,
}

struct Program {
    label: u16,
    elems: Vec<Elem>,
    /// u64 words in a state set: one bit per element and one for the end.
    words: usize,
}

impl Program {
    fn new(label: u16, elems: Vec<Elem>) -> Program {
        let words = (elems.len() + 1).div_ceil(64);
        Program { label, elems, words }
    }

    fn skippable(&self, k: usize) -> bool {
        matches!(self.elems[k].quant, Quant::Opt | Quant::Star)
    }

    /// Adds every state reachable without taking a token.
    fn close(&self, set: &mut [u64]) {
        for k in 0..self.elems.len() {
            if get(set, k) && self.skippable(k) {
                put(set, k + 1);
            }
        }
    }
}

fn get(set: &[u64], k: usize) -> bool {
    set[k / 64] >> (k % 64) & 1 != 0
}

fn put(set: &mut [u64], k: usize) {
    set[k / 64] |= 1 << (k % 64);
}

/// Compiles one pattern, interning its literals and noting the columns and
/// slow tests it needs.
fn compile(
    pattern: &Pattern,
    vocab: &mut Vocab,
    cols: &mut Vec<Attr>,
    slow: &mut Vec<Slow>,
) -> Result<Vec<Elem>, PatternError> {
    let mut elems: Vec<Elem> = Vec::new();
    for (origin, tok) in pattern.iter().enumerate() {
        // (quantifier, count): the steps this token writes out.
        let quants: [(Quant, usize); 2] = match tok.op {
            Op::One => [(Quant::One, 1), (Quant::One, 0)],
            Op::Optional => [(Quant::Opt, 1), (Quant::One, 0)],
            Op::ZeroOrMore => [(Quant::Star, 1), (Quant::One, 0)],
            Op::OneOrMore => [(Quant::One, 1), (Quant::Star, 1)],
            Op::Not => [(Quant::Not, 1), (Quant::One, 0)],
            Op::Repeat { min, max } => {
                if let Some(max) = max
                    && (max == 0 || max < min)
                {
                    return Err(PatternError::Repeat { min, max });
                }
                match max {
                    Some(max) => [(Quant::One, min as usize), (Quant::Opt, (max - min) as usize)],
                    None => [(Quant::One, min as usize), (Quant::Star, 1)],
                }
            }
        };
        let steps = quants[0].1.saturating_add(quants[1].1);
        if elems.len().saturating_add(steps) > MAX_ELEMENTS {
            return Err(PatternError::TooLong);
        }
        // The tests are compiled once and shared by the token's steps, so
        // a regex or fuzzy test in a repeat runs once per token.
        let compiled = compile_tok(tok, Quant::One, origin as u16, vocab, cols, slow)?;
        for (quant, count) in quants {
            for _ in 0..count {
                elems.push(Elem { quant, ..compiled.clone() });
            }
        }
    }
    // A pattern must be able to take a token: one with no elements, or
    // only `!` tokens that test nothing, never matches.
    let can_match = elems.iter().any(|e| e.quant != Quant::Not || !e.checks.is_empty() || e.flag_mask != 0);
    if !can_match {
        return Err(PatternError::Empty);
    }
    Ok(elems)
}

fn col_of(cols: &mut Vec<Attr>, attr: Attr) -> u16 {
    match cols.iter().position(|&a| a == attr) {
        Some(i) => i as u16,
        None => {
            cols.push(attr);
            (cols.len() - 1) as u16
        }
    }
}

fn compile_tok(
    tok: &Tok,
    quant: Quant,
    origin: u16,
    vocab: &mut Vocab,
    cols: &mut Vec<Attr>,
    slow: &mut Vec<Slow>,
) -> Result<Elem, PatternError> {
    let mut e = Elem { quant, checks: Vec::new(), flag_mask: 0, flag_want: 0, never: false, origin };
    let syms = |vocab: &mut Vocab, values: &[String]| -> Box<[Sym]> {
        let mut s: Vec<Sym> = values.iter().map(|v| vocab.intern(v)).collect();
        s.sort_unstable();
        s.dedup();
        s.into()
    };
    for test in &tok.tests {
        match test {
            Test::Eq(a, v) => e.checks.push(Check::Eq { col: col_of(cols, *a), sym: vocab.intern(v) }),
            Test::In(a, vs) => e.checks.push(Check::In { col: col_of(cols, *a), syms: syms(vocab, vs) }),
            Test::NotIn(a, vs) => e.checks.push(Check::NotIn { col: col_of(cols, *a), syms: syms(vocab, vs) }),
            Test::Length(c, n) => e.checks.push(Check::Length(*c, *n)),
            Test::Flag(f, want) => {
                // Two tests of one flag that disagree can never both pass.
                if e.flag_mask & f.bit() != 0 && (e.flag_want & f.bit() != 0) != *want {
                    e.never = true;
                }
                e.flag_mask |= f.bit();
                if *want {
                    e.flag_want |= f.bit();
                }
            }
            Test::Regex(a, ps, none) => {
                let res = ps
                    .iter()
                    .map(|p| {
                        Regex::new(p).map_err(|err| PatternError::Regex { pattern: p.clone(), error: err.to_string() })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                e.checks.push(Check::Slow(slow.len() as u32));
                slow.push(Slow::Regex { attr: *a, res, none: *none });
            }
            Test::Fuzzy(a, vs, max, none) => {
                e.checks.push(Check::Slow(slow.len() as u32));
                let targets = vs.iter().map(|v| v.chars().collect()).collect();
                slow.push(Slow::Fuzzy { attr: *a, targets, max: *max, none: *none });
            }
        }
    }
    Ok(e)
}

/// Compiled rules, ready to run over any number of sequences.
pub struct Matcher {
    labels: Arc<TagSet>,
    greedy: Vec<Option<Greedy>>,
    vocab: Vocab,
    /// The string attributes the rules compare with literals; each is one
    /// column of symbols per sequence.
    cols: Vec<Attr>,
    slow: Vec<Slow>,
    programs: Vec<Program>,
    /// The programs that can begin only at a token with this symbol in
    /// this column, by `key(col, sym)`.
    index: HashMap<u64, Vec<u32>, BuildHasherDefault<FoldHasher>>,
    index_cols: Vec<u16>,
    /// The programs that can begin anywhere.
    always: Vec<u32>,
    /// Every attribute the rules read.
    reads: Vec<Attr>,
    /// Every flag the rules test.
    flags: u32,
    config: u64,
}

fn key(col: u16, sym: Sym) -> u64 {
    (u64::from(col) << 32) | u64::from(sym)
}

/// What one run needs per token: the symbol columns, the flags and the
/// kept results of slow tests.
struct Ctx<'s, 'a> {
    seq: &'s Sequence<'a>,
    /// The first token the columns, flags and slow results cover; they
    /// are indexed by token minus `base`.
    base: usize,
    /// How many tokens they cover.
    width: usize,
    cols: Vec<Vec<Sym>>,
    flags: Vec<u32>,
    /// Per slow test and token: 0 not yet run, 1 failed, 2 passed.
    slow: Vec<u8>,
    buf: Buf,
    row: Vec<u32>,
}

impl Matcher {
    pub fn builder() -> MatcherBuilder {
        MatcherBuilder::new()
    }

    /// The labels, in the order they were added: a match's `label` is a
    /// tag of this set.
    pub fn labels(&self) -> &Arc<TagSet> {
        &self.labels
    }

    /// A hash of the rules, for provenance: equal rules give equal hashes.
    pub fn config_hash(&self) -> u64 {
        self.config
    }

    /// The attributes the rules read.
    pub fn reads(&self) -> &[Attr] {
        &self.reads
    }

    /// Whether the rules test `flag`.
    pub fn tests_flag(&self, flag: Flag) -> bool {
        self.flags & flag.bit() != 0
    }

    fn collect_reads(&mut self) {
        let mut reads = self.cols.clone();
        reads.extend(self.slow.iter().map(Slow::attr));
        reads.sort_unstable();
        reads.dedup();
        self.reads = reads;
        self.flags = self.programs.iter().flat_map(|p| &p.elems).fold(0, |f, e| f | e.flag_mask);
    }

    /// Notes, per program, the symbols a first token must have, when its
    /// first tokens all compare a column with literals.
    fn index_starts(&mut self) {
        for (p, prog) in self.programs.iter().enumerate() {
            let mut first = vec![0u64; prog.words];
            put(&mut first, 0);
            prog.close(&mut first);
            let mut keys = Vec::new();
            let mut indexable = true;
            for k in 0..prog.elems.len() {
                if !get(&first, k) {
                    continue;
                }
                let e = &prog.elems[k];
                let found = (e.quant != Quant::Not)
                    .then(|| {
                        e.checks.iter().find_map(|c| match c {
                            Check::Eq { col, sym } => Some(vec![key(*col, *sym)]),
                            Check::In { col, syms } => Some(syms.iter().map(|&s| key(*col, s)).collect()),
                            _ => None,
                        })
                    })
                    .flatten();
                match found {
                    Some(k) => keys.extend(k),
                    None => {
                        indexable = false;
                        break;
                    }
                }
            }
            if indexable {
                keys.sort_unstable();
                keys.dedup();
                for k in keys {
                    let col = (k >> 32) as u16;
                    if !self.index_cols.contains(&col) {
                        self.index_cols.push(col);
                    }
                    self.index.entry(k).or_default().push(p as u32);
                }
            } else {
                self.always.push(p as u32);
            }
        }
    }

    fn check_inputs(&self, seq: &Sequence<'_>) -> Result<(), MatchError> {
        if let Some(&a) = self.reads.iter().find(|&&a| !seq.has(a)) {
            return Err(MatchError::Missing(a));
        }
        for f in [Flag::Stop, Flag::SentStart] {
            if self.tests_flag(f) && !seq.has_flag(f) {
                return Err(MatchError::MissingFlag(f));
            }
        }
        Ok(())
    }

    /// The values the tests read, for the tokens in `range`.
    fn ctx<'s, 'a>(&self, seq: &'s Sequence<'a>, range: Range<usize>) -> Result<Ctx<'s, 'a>, MatchError> {
        self.check_inputs(seq)?;
        let mut buf = Buf::default();
        let cols = self
            .cols
            .iter()
            .map(|&a| range.clone().map(|i| seq.with_value(a, i, &mut buf, |v| self.vocab.get(v))).collect())
            .collect();
        let flags = if self.flags != 0 { seq.flags_in(range.clone()) } else { Vec::new() };
        let slow = vec![0; self.slow.len() * range.len()];
        Ok(Ctx { seq, base: range.start, width: range.len(), cols, flags, slow, buf, row: Vec::new() })
    }

    /// Every match of every rule in the sequence, after each rule's
    /// `Greedy` filter, in order of start, end and label. A rule's match
    /// over one run of tokens is reported once, however many of its
    /// patterns or paths through a pattern reach it.
    pub fn find(&self, seq: &Sequence<'_>) -> Result<Vec<Match>, MatchError> {
        let n = seq.len();
        let mut ctx = self.ctx(seq, 0..n)?;
        let mut found = Vec::new();
        let mut stamp = vec![u32::MAX; self.programs.len()];
        let words = self.programs.iter().map(|p| p.words).max().unwrap_or(1);
        let (mut cur, mut next) = (vec![0u64; words], vec![0u64; words]);
        let mut candidates = Vec::new();
        for i in 0..n {
            candidates.clear();
            candidates.extend_from_slice(&self.always);
            for &c in &self.index_cols {
                let sym = ctx.cols[c as usize][i];
                if sym != NONE
                    && let Some(ps) = self.index.get(&key(c, sym))
                {
                    candidates.extend(ps.iter().copied().filter(|&p| {
                        let fresh = stamp[p as usize] != i as u32;
                        stamp[p as usize] = i as u32;
                        fresh
                    }));
                }
            }
            for &p in &candidates {
                let prog = &self.programs[p as usize];
                self.run(prog, i, &mut ctx, &mut cur[..prog.words], &mut next[..prog.words], |end| {
                    found.push(Match { start: i as u32, end: end as u32, label: Tag(prog.label) })
                });
            }
        }
        Ok(self.filter(found))
    }

    /// Runs one program from token `start`, calling `emit` with the end of
    /// every match.
    fn run(
        &self,
        prog: &Program,
        start: usize,
        ctx: &mut Ctx,
        cur: &mut [u64],
        next: &mut [u64],
        mut emit: impl FnMut(usize),
    ) {
        let m = prog.elems.len();
        cur.fill(0);
        put(cur, 0);
        prog.close(cur);
        for j in start..ctx.seq.len() {
            next.fill(0);
            let mut any = false;
            for k in 0..m {
                if !get(cur, k) {
                    continue;
                }
                let e = &prog.elems[k];
                let pass = self.passes(e, j, ctx);
                match e.quant {
                    Quant::One | Quant::Opt if pass => {
                        put(next, k + 1);
                        any = true;
                    }
                    Quant::Star if pass => {
                        put(next, k);
                        any = true;
                    }
                    Quant::Not if !pass => {
                        put(next, k + 1);
                        any = true;
                    }
                    _ => {}
                }
            }
            if !any {
                return;
            }
            prog.close(next);
            if get(next, m) {
                emit(j + 1);
            }
            cur.copy_from_slice(next);
        }
    }

    /// Whether token `j` passes every test of `e`, before `!` negates it.
    fn passes(&self, e: &Elem, j: usize, ctx: &mut Ctx) -> bool {
        if e.never {
            return false;
        }
        let at = j - ctx.base;
        if e.flag_mask != 0 && ctx.flags[at] & e.flag_mask != e.flag_want {
            return false;
        }
        for c in &e.checks {
            let ok = match c {
                Check::Eq { col, sym } => ctx.cols[*col as usize][at] == *sym,
                Check::In { col, syms } => syms.binary_search(&ctx.cols[*col as usize][at]).is_ok(),
                Check::NotIn { col, syms } => syms.binary_search(&ctx.cols[*col as usize][at]).is_err(),
                Check::Length(cmp, n) => cmp.holds(ctx.seq.text(j).chars().count() as u32, *n),
                Check::Slow(s) => self.slow_test(*s as usize, j, ctx),
            };
            if !ok {
                return false;
            }
        }
        true
    }

    fn slow_test(&self, s: usize, j: usize, ctx: &mut Ctx) -> bool {
        let slot = s * ctx.width + (j - ctx.base);
        if ctx.slow[slot] != 0 {
            return ctx.slow[slot] == 2;
        }
        let row = &mut ctx.row;
        let ok = match &self.slow[s] {
            Slow::Regex { attr, res, none } => {
                ctx.seq.with_value(*attr, j, &mut ctx.buf, |v| res.iter().any(|re| re.is_match(v)) != *none)
            }
            Slow::Fuzzy { attr, targets, max, none } => ctx.seq.with_value(*attr, j, &mut ctx.buf, |v| {
                let v: Vec<char> = v.chars().collect();
                let close = targets.iter().any(|t| {
                    let max = max.unwrap_or_else(|| fuzzy::default_max(t.len()));
                    fuzzy::within(&v, t, max, row)
                });
                close != *none
            }),
        };
        ctx.slow[slot] = if ok { 2 } else { 1 };
        ok
    }

    /// Sorts and removes repeats, applies each rule's `Greedy`, and puts
    /// the matches in order of start, end and label.
    fn filter(&self, mut found: Vec<Match>) -> Vec<Match> {
        found.sort_unstable_by_key(|m| (m.label, m.start, m.end));
        found.dedup();
        let mut out = Vec::with_capacity(found.len());
        for group in found.chunk_by(|a, b| a.label == b.label) {
            match self.greedy[group[0].label.0 as usize] {
                None => out.extend_from_slice(group),
                Some(g) => out.extend(keep_greedy(group, g)),
            }
        }
        out.sort_unstable();
        out
    }

    /// Which pattern element took each token of a match: for each token,
    /// the index of the `Tok` in the first of the rule's patterns that
    /// matches exactly that run. Where a pattern can take the run more than
    /// one way, the way chosen is a backtracking regular expression's with
    /// greedy operators: each `*`, `+`, `?` and repeat takes as many tokens
    /// as it can while the rest still matches, the earliest first. None
    /// when no pattern of the rule matches that run.
    pub fn align(&self, seq: &Sequence<'_>, m: &Match) -> Result<Option<Vec<u16>>, MatchError> {
        let (start, end) = (m.start as usize, m.end as usize);
        let mut ctx = self.ctx(seq, start..end)?;
        Ok(self.programs.iter().filter(|p| p.label == m.label.0).find_map(|p| self.walk(p, start, end, &mut ctx)))
    }

    /// The first parse of tokens `[start, end)` by one program, in the
    /// order a backtracking matcher tries the ways: take the token, then
    /// skip the element. A table of which states (element, token) can
    /// still reach the end is filled from the last token back, then the
    /// path is read forward, taking the first way whose state can.
    fn walk(&self, prog: &Program, start: usize, end: usize, ctx: &mut Ctx) -> Option<Vec<u16>> {
        let m = prog.elems.len();
        let width = end - start + 1;
        // live[j * (m + 1) + k]: from element k at token start + j, the
        // rest of the run can be taken.
        let mut live = vec![false; width * (m + 1)];
        let at = |j: usize, k: usize| j * (m + 1) + k;
        let mut step = vec![None; width * (m + 1)];
        for j in (0..width).rev() {
            for k in (0..=m).rev() {
                if k == m {
                    live[at(j, k)] = j + 1 == width;
                    continue;
                }
                let e = &prog.elems[k];
                let token = start + j;
                let next = if token < end {
                    let pass = self.passes(e, token, ctx);
                    match e.quant {
                        Quant::Star if pass => Some(k),
                        Quant::One | Quant::Opt if pass => Some(k + 1),
                        Quant::Not if !pass => Some(k + 1),
                        _ => None,
                    }
                } else {
                    None
                };
                if let Some(n) = next.filter(|&n| live[at(j + 1, n)]) {
                    live[at(j, k)] = true;
                    step[at(j, k)] = Some((n, true));
                } else if prog.skippable(k) && live[at(j, k + 1)] {
                    live[at(j, k)] = true;
                    step[at(j, k)] = Some((k + 1, false));
                }
            }
        }
        if !live[at(0, 0)] {
            return None;
        }
        let (mut j, mut k) = (0, 0);
        let mut path = Vec::with_capacity(end - start);
        while let Some((n, took)) = step[at(j, k)] {
            if took {
                path.push(prog.elems[k].origin);
                j += 1;
            }
            k = n;
        }
        Some(path)
    }
}

/// spaCy's greedy filter over one rule's matches: sorted by start then
/// longest first (`First`) or by length then leftmost (`Longest`), each
/// kept when none of its tokens is in one kept before it.
pub(crate) fn keep_greedy(group: &[Match], g: Greedy) -> Vec<Match> {
    let mut order: Vec<Match> = group.to_vec();
    match g {
        Greedy::First => order.sort_by_key(|m| (m.start, std::cmp::Reverse(m.end))),
        Greedy::Longest => order.sort_by_key(|m| (std::cmp::Reverse(m.end - m.start), m.start)),
    }
    // Tokens already covered by a kept match, from the first start on.
    let base = order.iter().map(|m| m.start).min().unwrap_or(0);
    let top = order.iter().map(|m| m.end).max().unwrap_or(0);
    let mut taken = vec![false; (top - base) as usize];
    let mut kept: Vec<Match> = Vec::new();
    for m in order {
        let run = &mut taken[(m.start - base) as usize..(m.end - base) as usize];
        if !run.contains(&true) {
            run.fill(true);
            kept.push(m);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeats_that_match_nothing_are_refused() {
        let mut b = Matcher::builder();
        assert_eq!(
            b.add("R", vec![vec![Tok::text("a").repeat(3, Some(2))]], None),
            Err(PatternError::Repeat { min: 3, max: 2 })
        );
        assert_eq!(
            b.add("R", vec![vec![Tok::text("a").repeat(0, Some(0))]], None),
            Err(PatternError::Repeat { min: 0, max: 0 })
        );
        assert_eq!(b.add("R", vec![vec![]], None), Err(PatternError::Empty));
        assert_eq!(b.add("R", vec![vec![Tok::any().negated()]], None), Err(PatternError::Empty));
        assert!(matches!(
            b.add("R", vec![vec![Tok::any().regex(Attr::Text, "(")]], None),
            Err(PatternError::Regex { .. })
        ));
        assert_eq!(b.add("", vec![vec![Tok::any()]], None), Err(PatternError::EmptyLabel));
    }

    #[test]
    fn starts_are_indexed_when_every_first_token_names_a_literal() {
        let mut b = Matcher::builder();
        b.add("A", vec![vec![Tok::text("a"), Tok::any()]], None).unwrap();
        b.add("B", vec![vec![Tok::text("x").optional(), Tok::any().is_in(Attr::Lower, &["y", "z"])]], None).unwrap();
        b.add("C", vec![vec![Tok::text("x").optional(), Tok::any()]], None).unwrap();
        b.add("D", vec![vec![Tok::text("x").negated()]], None).unwrap();
        let m = b.build().unwrap();
        assert_eq!(m.always, vec![2, 3]);
        assert_eq!(m.index.values().map(Vec::len).sum::<usize>(), 4);
    }

    #[test]
    fn contradicting_flag_tests_never_pass() {
        let mut b = Matcher::builder();
        b.add("R", vec![vec![Tok::any().flag(Flag::Alpha, true).flag(Flag::Alpha, false)]], None).unwrap();
        let m = b.build().unwrap();
        let seq = Sequence::new("a b", &[0, 2], &[1, 3]).unwrap();
        assert_eq!(m.find(&seq).unwrap(), vec![]);
    }
}
