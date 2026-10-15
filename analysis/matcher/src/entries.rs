//! The taking states the patterns start in, indexed by the value their
//! first token needs, so a start tests only the patterns its token can
//! begin.
//!
//! Most patterns begin with a token of a set text, lowercase or tag. With
//! thousands of patterns, testing every pattern's first token at every
//! start would cost one test per pattern per token even where no pattern
//! begins; here a start looks its token's values up once per attribute
//! the index uses and tests only the states found, and the states whose
//! first token has no such value.

use std::collections::HashMap;

use crate::attr::Attr;
use crate::hash::BuildFoldHasher;
use crate::input::Columns;
use crate::nfa::Nfa;
use crate::vocab::NONE;

#[derive(Default)]
pub(crate) struct Entries {
    /// States whose token needs no single value: tested at every start.
    any: Vec<u32>,
    /// States by `(attribute, value id)` their token needs.
    keyed: HashMap<u64, Vec<u32>, BuildFoldHasher>,
    /// The attributes `keyed` uses.
    attrs: Vec<Attr>,
}

fn key(a: Attr, id: u32) -> u64 {
    (u64::from(a as u8) << 32) | u64::from(id)
}

impl Entries {
    pub(crate) fn is_empty(&self) -> bool {
        self.any.is_empty() && self.keyed.is_empty()
    }

    /// Adds taking states of the automaton.
    pub(crate) fn extend(&mut self, nfa: &Nfa, states: impl IntoIterator<Item = u32>) {
        for s in states {
            match nfa.required_values(s) {
                None => self.any.push(s),
                Some((a, ids)) => {
                    // A token has one value per attribute, so a state
                    // listed under several of `In`'s values is found once.
                    for &id in ids {
                        self.keyed.entry(key(a, id)).or_default().push(s);
                    }
                    if !self.attrs.contains(&a) {
                        self.attrs.push(a);
                    }
                }
            }
        }
    }

    /// Keeps the states `keep` says.
    pub(crate) fn retain(&mut self, mut keep: impl FnMut(u32) -> bool) {
        self.any.retain(|&s| keep(s));
        self.keyed.retain(|_, v| {
            v.retain(|&s| keep(s));
            !v.is_empty()
        });
        self.attrs.retain(|&a| self.keyed.keys().any(|&k| k >> 32 == u64::from(a as u8)));
    }

    /// The states token `rel` of the columns may start, written over `out`.
    pub(crate) fn at(&self, cols: &Columns, rel: usize, out: &mut Vec<u32>) {
        out.clear();
        out.extend_from_slice(&self.any);
        for &a in &self.attrs {
            let id = cols.values[a as usize][rel];
            if id != NONE
                && let Some(states) = self.keyed.get(&key(a, id))
            {
                out.extend_from_slice(states);
            }
        }
    }
}
