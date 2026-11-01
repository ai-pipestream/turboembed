//! The label of each training token's split, for the token split head
//! (#1798): the trees a word may take from its token are its cuts
//! ([`EditTree::cuts`]), which keep the word by its own length, and the
//! mined tree ([`EditTree::mine`]), which names what it drops. A clitic
//! cut from its verb by `(t 2 ())` serves every verb; the mined tree
//! `(m 5 0 (s 'hacer' '') ())` serves one. Which is which is not known
//! from one token, so each token takes, of the labels that give its
//! words, the one that gives the most distinct training tokens their
//! words too, a tie to the fewer cuts (a cut fits any token long
//! enough, where a named tree fits only its own kind), then to the
//! shorter label, then to the smaller.

use std::collections::{BTreeMap, BTreeSet};

use turbo::head::edit_tree::EditTree;
use turbo::head::split::{MAX_SPLIT_WORDS, parse as parse_split};

use crate::ud_split::TokenSentence;

/// The longest tree label a head carries (manifest rule), as the
/// trainer's `mine_trees` keeps.
pub const MAX_LABEL: usize = 128;

/// The most labels one token's split is weighed over; past that, only
/// the mined trees stand.
const MAX_CANDIDATES: usize = 256;

/// The trees that give `word` from `token`: its cuts, then the mined
/// tree where that is not among them.
fn trees(token: &str, word: &str) -> Vec<EditTree> {
    let mut out = EditTree::cuts(token, word);
    let mined = EditTree::mine(token, word);
    if !out.contains(&mined) {
        out.push(mined);
    }
    out
}

/// The labels that give `words` from `token`, each a split the core
/// takes that gives every word back: none for a token of one word, or
/// of more words than a split holds.
pub fn candidates(token: &str, words: &[String]) -> Vec<String> {
    if words.len() < 2 || words.len() > MAX_SPLIT_WORDS {
        return Vec::new();
    }
    let per_word: Vec<Vec<EditTree>> = words.iter().map(|w| trees(token, w)).collect();
    let mut labels: Vec<Vec<&EditTree>> = vec![Vec::new()];
    for ts in &per_word {
        labels = labels.iter().flat_map(|l| ts.iter().map(move |t| l.iter().copied().chain([t]).collect())).collect();
        if labels.len() > MAX_CANDIDATES {
            labels = vec![per_word.iter().map(|ts| ts.last().expect("a word has its mined tree")).collect()];
            break;
        }
    }
    labels
        .iter()
        .map(|ts| ts.iter().map(|t| t.label()).collect::<Vec<_>>().join(" "))
        .filter(|l| l.len() <= MAX_LABEL && gives(l, token, words))
        .collect()
}

/// Whether the split `label` gives `words` from `token`, read as the
/// core reads it.
fn gives(label: &str, token: &str, words: &[String]) -> bool {
    parse_split(label).is_ok_and(|trees| {
        trees.len() == words.len()
            && trees.iter().zip(words).all(|(t, w)| t.apply(token).is_some_and(|got| !got.is_empty() && got == *w))
    })
}

/// How many cuts a label holds.
fn cuts(label: &str) -> usize {
    label.matches("(h ").count() + label.matches("(t ").count()
}

/// Each distinct split of training (a token and its words) with its
/// label: of its candidates, the one that is a candidate of the most
/// distinct training tokens, ties as the module says. A split with no
/// candidate has no entry.
pub fn choose(train: &[TokenSentence]) -> BTreeMap<(String, Vec<String>), String> {
    let mut splits: BTreeMap<(String, Vec<String>), Vec<String>> = BTreeMap::new();
    for s in train {
        for (t, ws) in s.tokens.iter().zip(&s.words) {
            if ws.len() >= 2 && !splits.contains_key(&(t.clone(), ws.clone())) {
                splits.insert((t.clone(), ws.clone()), candidates(t, ws));
            }
        }
    }
    let mut tokens_of: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for ((t, _), ls) in &splits {
        for l in ls {
            tokens_of.entry(l).or_default().insert(t);
        }
    }
    splits
        .iter()
        .filter_map(|(k, ls)| {
            let best = ls.iter().max_by(|a, b| {
                let covers = |l: &str| tokens_of[l].len();
                covers(a)
                    .cmp(&covers(b))
                    .then_with(|| cuts(b).cmp(&cuts(a)))
                    .then_with(|| b.len().cmp(&a.len()))
                    .then_with(|| b.cmp(a))
            })?;
            Some((k.clone(), best.clone()))
        })
        .collect()
}
