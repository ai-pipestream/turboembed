//! The scope of a cue: its governing predicate's clause in the
//! dependency tree, or a token window when there are no dependencies.

use turbo_analysis::Span;
use turbo_analysis_layers::Dependencies;

use super::cues::{Class, Cue};

/// Relations that head a clause of their own.
const CLAUSAL: [&str; 7] = ["conj", "advcl", "ccomp", "xcomp", "parataxis", "acl", "csubj"];
/// Clauses cut out of a predicate's scope: coordinated, adverbial and
/// paratactic clauses carry their own modality, unless the cue reaches
/// into them (`prohibited from reselling`).
const CUT: [&str; 3] = ["conj", "advcl", "parataxis"];
/// Words that end a window scope.
const STOPS: [&str; 13] = [",", ";", ":", ".", "!", "?", "and", "but", "or", "because", "although", "while", "whereas"];

pub(super) struct Input<'a> {
    pub text: &'a str,
    pub spans: Vec<Span>,
    deps: Option<&'a Dependencies>,
    children: Vec<Vec<usize>>,
}

/// A cue's scope: tokens `lo..=hi`, grouped with the other cues that
/// share its `key` (the predicate, or the window's end).
#[derive(Debug, Clone, Copy)]
pub(super) struct Scope {
    pub key: usize,
    pub lo: usize,
    pub hi: usize,
    /// The governing predicate, when scoped by dependencies.
    pub predicate: Option<usize>,
}

impl<'a> Input<'a> {
    pub(super) fn new(text: &'a str, spans: Vec<Span>, deps: Option<&'a Dependencies>) -> Self {
        let mut children = vec![Vec::new(); spans.len()];
        if let Some(d) = deps {
            for (t, h) in d.head.iter().enumerate() {
                if let Some(h) = h {
                    children[h.get()].push(t);
                }
            }
        }
        Input { text, spans, deps, children }
    }

    pub(super) fn parsed(&self) -> bool {
        self.deps.is_some()
    }

    pub(super) fn lower(&self, t: usize) -> String {
        let s = self.spans[t];
        self.text[s.start as usize..s.end as usize].to_lowercase()
    }

    fn base(&self, t: usize) -> &str {
        match self.deps {
            Some(d) => {
                let r = d.relations.label(d.relation[t]);
                r.split(':').next().unwrap_or(r)
            }
            None => "",
        }
    }

    fn head(&self, t: usize) -> Option<usize> {
        self.deps.and_then(|d| d.head[t].map(|h| h.get()))
    }

    fn punct(&self, t: usize) -> bool {
        self.base(t) == "punct" || self.lower(t).chars().all(|c| c.is_ascii_punctuation())
    }

    /// The bytes of tokens `lo..=hi`.
    pub(super) fn bytes(&self, lo: usize, hi: usize) -> Span {
        Span::new(self.spans[lo].start, self.spans[hi].end)
    }

    pub(super) fn scope(&self, cue: &Cue, window: usize) -> Scope {
        if self.parsed() { self.dep_scope(cue) } else { self.window_scope(cue, window) }
    }

    /// The token a cue hangs from: the last word of a conditional or
    /// factive (`if`, `that`), else the first.
    fn anchor(cue: &Cue) -> usize {
        if matches!(cue.class, Class::Cond | Class::Factive) { cue.end - 1 } else { cue.start }
    }

    /// The clause head that governs `cue`: the nearest clausal node at
    /// or above its anchor, climbing over function words and nominals.
    pub(super) fn predicate(&self, cue: &Cue) -> usize {
        let a = Self::anchor(cue);
        // A negating preposition (`without approval`) scopes its object.
        if matches!(cue.class, Class::Neg) && self.base(a) == "case" {
            return self.head(a).unwrap_or(a);
        }
        let mut x = a;
        loop {
            let function = matches!(
                self.base(x),
                "aux" | "neg" | "advmod" | "mark" | "case" | "det" | "cop" | "fixed" | "cc" | "discourse" | "punct"
            );
            if !function && (self.head(x).is_none() || CLAUSAL.contains(&self.base(x))) {
                return x;
            }
            match self.head(x) {
                Some(h) => x = h,
                None => return x,
            }
        }
    }

    /// Whether `t` is `c` or below it.
    fn under(&self, t: usize, c: usize) -> bool {
        let mut x = Some(t);
        let mut steps = 0;
        while let Some(y) = x {
            if y == c {
                return true;
            }
            steps += 1;
            if steps > self.spans.len() {
                return false;
            }
            x = self.head(y);
        }
        false
    }

    fn dep_scope(&self, cue: &Cue) -> Scope {
        let p = self.predicate(cue);
        let mut kept = vec![false; self.spans.len()];
        let mut stack = vec![p];
        while let Some(x) = stack.pop() {
            kept[x] = true;
            for &c in &self.children[x] {
                let cut = CUT.contains(&self.base(c)) && !(cue.start..cue.end).any(|t| self.under(t, c));
                if !cut && !(x == p && matches!(self.base(c), "cc")) {
                    stack.push(c);
                }
            }
        }
        // The contiguous run around the predicate: a cut clause in the
        // middle ends the scope, so a trailing period past it is dropped.
        let (mut lo, mut hi) = (p, p);
        while lo > 0 && kept[lo - 1] {
            lo -= 1;
        }
        while hi + 1 < kept.len() && kept[hi + 1] {
            hi += 1;
        }
        // The cue is in scope even when it hangs outside, from its anchor:
        // a factive's verb (`confirmed that`) governs, it is not presupposed.
        lo = lo.min(if cue.class == Class::Factive { Self::anchor(cue) } else { cue.start });
        hi = hi.max(cue.end - 1);
        while lo < hi && self.punct(hi) {
            hi -= 1;
        }
        while lo < hi && self.punct(lo) {
            lo += 1;
        }
        Scope { key: p, lo, hi, predicate: Some(p) }
    }

    /// From the cue (a factive's complementizer) to the next stop word or punctuation, at most
    /// `window` tokens past the cue.
    fn window_scope(&self, cue: &Cue, window: usize) -> Scope {
        let n = self.spans.len();
        let end = cue.end.saturating_add(window).min(n);
        let mut hi = cue.end - 1;
        while hi + 1 < end && !STOPS.contains(&self.lower(hi + 1).as_str()) {
            hi += 1;
        }
        let lo = if cue.class == Class::Factive { Self::anchor(cue) } else { cue.start };
        Scope { key: hi, lo, hi, predicate: None }
    }

    /// Whether `may` at token `t` has a party for its subject (`Users
    /// may cancel`), which makes it a permission.
    pub(super) fn party_subject(&self, t: usize, parties: &[String], predicate: Option<usize>) -> bool {
        let is_party = |w: usize| parties.iter().any(|p| *p == self.lower(w));
        match predicate {
            Some(p) => self.children[p].iter().any(|&c| self.base(c) == "nsubj" && is_party(c)),
            None => (t.saturating_sub(2)..t).any(is_party),
        }
    }
}
