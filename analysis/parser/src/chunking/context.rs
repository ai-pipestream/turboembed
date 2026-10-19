//! The shift-reduce build and check context generators
//! (`opennlp.tools.parser.chunking.BuildContextGenerator` and `CheckContextGenerator`).

use std::collections::{HashMap, HashSet};
use std::fmt;

use turbo_sequence::Features;

use crate::context::{
    ConsRef, DisplayFn, FeatureSink, check_cons_into as check_cons, check_cons_pair_into as check_cons_pair,
    cons2_into as cons2, cons3_into as cons3, surround_into as surround,
};
use crate::parse::{Parse, START};

// Decode length-prefixed dictionary keys once, so n-gram lookup needs no temporary key.
#[derive(Clone, Debug, Default)]
struct DictionaryNode {
    terminal: bool,
    children: HashMap<String, DictionaryNode>,
}

impl DictionaryNode {
    fn from_keys(keys: HashSet<String>) -> Self {
        let mut root = Self::default();
        for key in keys {
            let mut rest = key.as_str();
            let mut words = Vec::new();
            while let Some((length, tail)) = rest.split_once(':') {
                let Ok(n) = length.parse::<usize>() else {
                    break;
                };
                if n.to_string() != length {
                    break;
                }
                let Some(word) = tail.get(..n) else {
                    break;
                };
                words.push(word);
                rest = &tail[n..];
            }
            if !rest.is_empty() {
                continue;
            }
            let mut node = &mut root;
            for word in words {
                node = node.children.entry(word.to_owned()).or_default();
            }
            node.terminal = true;
        }
        root
    }

    fn contains(&self, words: &[&Parse]) -> bool {
        let mut node = self;
        for word in words {
            let Some(next) = node.children.get(word.head_text()) else {
                return false;
            };
            node = next;
        }
        node.terminal
    }
}

/// Context generator for the shift-reduce build phase (`opennlp.tools.parser.chunking.BuildContextGenerator`).
#[derive(Clone, Debug, Default)]
pub struct BuildContextGenerator {
    dict: Option<DictionaryNode>,
}

impl BuildContextGenerator {
    /// Creates a [`BuildContextGenerator`] without a dictionary, so every lexical feature is
    /// used (`new BuildContextGenerator()`).
    pub fn new() -> Self {
        Self { dict: None }
    }

    /// Creates a [`BuildContextGenerator`] whose lexical features are limited to the n-grams of
    /// `dict`, keyed by [`crate::bottom_up::dictionary_key`] (`new BuildContextGenerator(Dictionary)`).
    pub fn with_dictionary(dict: HashSet<String>) -> Self {
        Self { dict: Some(DictionaryNode::from_keys(dict)) }
    }

    /// Generates build features for `constituents[index]` (`BuildContextGenerator.getContext(Parse[], int)`).
    pub fn get_context(&self, constituents: &[Parse], index: usize) -> Vec<String> {
        let mut features = Vec::with_capacity(100);
        self.populate(constituents, index, &mut features);
        features
    }

    /// Replaces `out` with build features, reusing its byte storage.
    pub fn get_features(&self, constituents: &[Parse], index: usize, out: &mut Features) {
        out.clear();
        self.populate(constituents, index, out);
    }

    fn populate<S: FeatureSink>(&self, constituents: &[Parse], index: usize, features: &mut S) {
        let ps = constituents.len();

        let p_2 = index.checked_sub(2).map(|j| &constituents[j]);
        let p_1 = index.checked_sub(1).map(|j| &constituents[j]);
        let punct_2s = p_1.map(Parse::previous_punctuation);
        let p0 = &constituents[index];
        let punct_1s = Some(p0.previous_punctuation());
        let punct1s = Some(p0.next_punctuation());
        let p1 = constituents.get(index + 1);
        let punct2s = p1.map(Parse::next_punctuation);
        let p2 = constituents.get(index + 2);

        let mut u_2 = true;
        let mut u_1 = true;
        let mut u0 = true;
        let mut u1 = true;
        let mut u2 = true;
        let mut b_2_1 = true;
        let mut b_10 = true;
        let mut b01 = true;
        let mut b12 = true;
        let mut t_2_10 = true;
        let mut t_101 = true;
        let mut t012 = true;

        if let Some(dict) = &self.dict {
            let has = |ngram: &[&Parse]| dict.contains(ngram);
            if let Some(p_2) = p_2 {
                u_2 = has(&[p_2]);
            }
            if let Some(p2) = p2 {
                u2 = has(&[p2]);
            }
            u0 = has(&[p0]);
            if let (Some(p_2), Some(p_1)) = (p_2, p_1) {
                b_2_1 = has(&[p_2, p_1]);
                t_2_10 = has(&[p_2, p_1, p0]);
            }
            if let (Some(p_1), Some(p1)) = (p_1, p1) {
                t_101 = has(&[p_1, p0, p1]);
            }
            if let Some(p_1) = p_1 {
                u_1 = has(&[p_1]);
                // OpenNLP's "extra check for 2==null case"; `u1` is still true here.
                b_2_1 = b_2_1 && u_1 && u_2;
                t_2_10 = t_2_10 && u_1 && u_2 && u0;
                t_101 = t_101 && u_1 && u0 && u1;
                b_10 = has(&[p_1, p0]) && u_1 && u0;
            }
            if let (Some(p1), Some(p2)) = (p1, p2) {
                b12 = has(&[p1, p2]);
                t012 = has(&[p0, p1, p2]);
            }
            if let Some(p1) = p1 {
                u1 = has(&[p1]);
                b12 = b12 && u1 && u2;
                t012 = t012 && u1 && u2 && u0;
                t_101 = t_101 && u0 && u_1 && u1;
                b01 = has(&[p0, p1]) && u0 && u1;
            }
        }

        let c_2 = ConsRef::new(p_2, -2, true, u_2, features);
        let c_1 = ConsRef::new(p_1, -1, true, u_1, features);
        let c0 = ConsRef::new(Some(p0), 0, true, u0, features);
        let c1 = ConsRef::new(p1, 1, true, u1, features);
        let c2 = ConsRef::new(p2, 2, true, u2, features);

        features.emit(format_args!("default"));
        for c in [&c0, &c_2, &c_1, &c1, &c2] {
            if c.unigram {
                features.emit(format_args!("{}", c.cons));
            }
            features.emit(format_args!("{}", c.cons_bo));
        }

        cons2(features, &c0, &c1, punct1s, b01);
        cons2(features, &c_1, &c0, punct_1s, b_10);
        cons3(features, &c0, &c1, &c2, punct1s, punct2s, t012, b01, b12);
        cons3(features, &c_2, &c_1, &c0, punct_2s, punct_1s, t_2_10, b_2_1, b_10);
        cons3(features, &c_1, &c0, &c1, punct_1s, punct1s, t_101, b_10, b01);

        // Walks back to the closest start; a node before `index` always has a label here, where
        // OpenNLP would throw `NullPointerException` on one without.
        let starts_constituent = |p: &Parse| p.label().is_some_and(|l| l.starts_with(START));
        let matches_back = |open: &str, feature: &str, features: &mut S| {
            for p in constituents[..index].iter().rev() {
                if p.kind() == open {
                    features.emit(format_args!("{feature}"));
                    break;
                }
                if starts_constituent(p) {
                    break;
                }
            }
        };
        match p0.kind() {
            "-RRB-" => matches_back("-LRB-", "bracketsmatch", features),
            "-RCB-" => matches_back("-LCB-", "bracketsmatch", features),
            "''" => matches_back("``", "quotesmatch", features),
            "'" => matches_back("`", "quotesmatch", features),
            "," => matches_back(",", "iscomma", features),
            "." if index + 1 == ps => {
                if let Some(pi) = constituents[..index].iter().rposition(starts_constituent)
                    && pi == 0
                {
                    features.emit(format_args!("endofsentence"));
                }
            }
            _ => {}
        }
    }
}

/// Context generator for the shift-reduce check phase (`opennlp.tools.parser.chunking.CheckContextGenerator`).
#[derive(Clone, Debug, Default)]
pub struct CheckContextGenerator;

impl CheckContextGenerator {
    /// Creates a new [`CheckContextGenerator`].
    pub fn new() -> Self {
        Self
    }

    /// Generates completeness check features for reducing `constituents[start..=end]` into `kind`
    /// (`CheckContextGenerator.getContext(Parse[], String, int, int)`).
    pub fn get_context(&self, constituents: &[Parse], kind: &str, start: usize, end: usize) -> Vec<String> {
        let mut features = Vec::with_capacity(100);
        self.populate(constituents, kind, start, end, constituents[0].label().unwrap_or("null"), &mut features);
        features
    }

    /// Replaces `out` with check features, reusing its byte storage.
    pub fn get_features(&self, constituents: &[Parse], kind: &str, start: usize, end: usize, out: &mut Features) {
        out.clear();
        self.populate(constituents, kind, start, end, constituents[0].label().unwrap_or("null"), out);
    }

    pub(super) fn populate(
        &self,
        constituents: &[Parse],
        kind: &str,
        start: usize,
        end: usize,
        first_label: &str,
        features: &mut impl FeatureSink,
    ) {
        let ps = constituents.len();
        features.emit(format_args!("default"));
        features.emit(format_args!("fl={first_label}"));
        let pstart = &constituents[start];
        let pend = &constituents[end];
        check_cons(pstart, "begin", kind, features);
        check_cons(pend, "last", kind, features);
        for p in &constituents[start..end] {
            check_cons_pair(p, pend, kind, features);
        }
        for (prefix, include_punctuation) in [("p", false), ("pp", true)] {
            let production = DisplayFn(|f: &mut fmt::Formatter<'_>| {
                write!(f, "{prefix}={kind}->")?;
                for p in &constituents[start..end] {
                    write!(f, "{},", p.kind())?;
                    if include_punctuation {
                        for punct in p.next_punctuation() {
                            write!(f, "{},", punct.kind())?;
                        }
                    }
                }
                f.write_str(pend.kind())
            });
            features.emit(format_args!("{production}"));
        }

        let p1s = Some(constituents[end].next_punctuation());
        let p_1s = Some(constituents[start].previous_punctuation());
        let p_2 = start.checked_sub(2).map(|j| &constituents[j]);
        let p_1 = start.checked_sub(1).map(|j| &constituents[j]);
        let p_2s = p_1.map(Parse::previous_punctuation);
        let p1 = (end + 1 < ps).then(|| &constituents[end + 1]);
        let p2s = p1.map(Parse::next_punctuation);
        let p2 = (end + 2 < ps).then(|| &constituents[end + 2]);

        surround(p_1, -1, kind, p_1s, features);
        surround(p_2, -2, kind, p_2s, features);
        surround(p1, 1, kind, p1s, features);
        surround(p2, 2, kind, p2s, features);
    }
}
