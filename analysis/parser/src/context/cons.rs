//! Constituent features (`opennlp.tools.parser.Cons` and the `AbstractContextGenerator`
//! helpers `cons`, `consbo`, `production`, `cons2`, `cons3`, `surround`, and `checkcons`).

use std::collections::HashSet;
use std::fmt::{self, Display};

use crate::parse::Parse;

use super::*;

/// End-of-sentence boundary marker (`AbstractContextGenerator.EOS`).
pub const EOS: &str = "eos";

/// Holds feature strings and unigram visibility for a constituent slot
/// (`opennlp.tools.parser.Cons`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cons {
    /// Lexicalized constituent feature string (`cons`).
    pub cons: String,
    /// Backoff constituent feature string (`consbo`).
    pub cons_bo: String,
    /// Relative slot index (`-2..=2`).
    pub index: i32,
    /// Whether the lexicalized head token resides in the training dictionary (`unigram`).
    pub unigram: bool,
}

impl Cons {
    /// Creates a new [`Cons`] descriptor.
    pub fn new(cons: String, cons_bo: String, index: i32, unigram: bool) -> Self {
        Self { cons, cons_bo, index, unigram }
    }
}

/// Treats an empty punctuation set as Java's `null`: `Parse` creates the set on the first mark
/// it adds, so a set OpenNLP reads is either `null` or not empty.
fn punct_set_of(punctuation: Option<&[Parse]>) -> Option<&[Parse]> {
    punctuation.filter(|s| !s.is_empty())
}

fn cons_with_label(node: Option<&Parse>, i: i32, use_label: bool) -> String {
    let mut feat = String::with_capacity(24);
    feat.push_str(&i.to_string());
    feat.push('=');
    if let Some(p) = node {
        if use_label && i < 0 {
            feat.push_str(p.label().unwrap_or("null"));
            feat.push('|');
        }
        feat.push_str(p.kind());
        feat.push('|');
        feat.push_str(p.head_text());
    } else {
        feat.push_str(EOS);
    }
    feat
}

fn cons_bo_with_label(node: Option<&Parse>, i: i32, use_label: bool) -> String {
    let mut feat = String::with_capacity(16);
    feat.push_str(&i.to_string());
    feat.push_str("*=");
    if let Some(p) = node {
        if use_label && i < 0 {
            feat.push_str(p.label().unwrap_or("null"));
            feat.push('|');
        }
        feat.push_str(p.kind());
    } else {
        feat.push_str(EOS);
    }
    feat
}

/// Builds a lexicalized constituent feature (`AbstractContextGenerator.cons(Parse, int)`) with
/// `useLabel` false, as every generator but the chunking build generator has it.
pub fn cons(node: Option<&Parse>, i: i32) -> String {
    cons_with_label(node, i, false)
}

/// Builds a back-off constituent feature (`AbstractContextGenerator.consbo(Parse, int)`) with
/// `useLabel` false.
pub fn cons_bo(node: Option<&Parse>, i: i32) -> String {
    cons_bo_with_label(node, i, false)
}

/// [`cons`] with `useLabel` true (`chunking.BuildContextGenerator`): a node before the current
/// one carries its label, `null` when it has none.
pub fn cons_labeled(node: Option<&Parse>, i: i32) -> String {
    cons_with_label(node, i, true)
}

/// [`cons_bo`] with `useLabel` true (`chunking.BuildContextGenerator`).
pub fn cons_bo_labeled(node: Option<&Parse>, i: i32) -> String {
    cons_bo_with_label(node, i, true)
}

/// The index `cons2` and `cons3` give a mark before a node at `index`: one less at or before the
/// current node, the node's own index after it.
fn punct_index(index: i32) -> i32 {
    if index <= 0 { index - 1 } else { index }
}

/// The grammar production `p` starts (`AbstractContextGenerator.production`): its type, `->` and
/// the types of its children with punctuation collapsed by `punct_set`. With
/// `include_punctuation`, the types of each child's following marks come after it, except after
/// the last child.
pub fn production_in(p: &Parse, include_punctuation: bool, punct_set: &HashSet<String>) -> String {
    let mut prod = String::with_capacity(32);
    prod.push_str(p.kind());
    prod.push_str("->");
    let children = crate::bottom_up::collapse_punctuation(p.children(), punct_set);
    for (ci, child) in children.iter().enumerate() {
        prod.push_str(child.kind());
        if ci + 1 != children.len() {
            prod.push(',');
            if include_punctuation {
                for punct in child.next_punctuation() {
                    prod.push_str(punct.kind());
                    prod.push(',');
                }
            }
        }
    }
    prod
}

/// [`production_in`] without collapsing punctuation: it is OpenNLP's production only when `p`
/// has no punctuation children. Kept for the tree-insertion generators until they pass their
/// punctuation set to [`production_in`].
pub fn production(p: &Parse, include_punctuation: bool) -> String {
    production_in(p, include_punctuation, &HashSet::new())
}

/// Adds the features of two nodes and the punctuation between them
/// (`AbstractContextGenerator.cons2`, with `zeroBackOff` false as every generator has it).
pub(crate) fn cons2_into(
    features: &mut impl FeatureSink,
    c0: &ConsRef<'_>,
    c1: &ConsRef<'_>,
    punct1s: Option<&[Parse]>,
    bigram: bool,
) {
    if let Some(puncts) = punct_set_of(punct1s) {
        for p in puncts {
            let pbo = features
                .prepare(DisplayFn(|f: &mut fmt::Formatter<'_>| write!(f, "{}={}", punct_index(c1.index), p.kind())));
            features.emit(format_args!("{pbo}"));
            if c0.index == 0 {
                if c0.unigram {
                    features.emit(format_args!("{},{pbo}", c0.cons));
                }
                features.emit(format_args!("{},{pbo}", c0.cons_bo));
            }
            if c1.index == 0 {
                if c1.unigram {
                    features.emit(format_args!("{pbo},{}", c1.cons));
                }
                features.emit(format_args!("{pbo},{}", c1.cons_bo));
            }
            if bigram {
                features.emit(format_args!("{},{pbo},{}", c0.cons, c1.cons));
            }
            if c1.unigram {
                features.emit(format_args!("{},{pbo},{}", c0.cons_bo, c1.cons));
            }
            if c0.unigram {
                features.emit(format_args!("{},{pbo},{}", c0.cons, c1.cons_bo));
            }
            features.emit(format_args!("{},{pbo},{}", c0.cons_bo, c1.cons_bo));
        }
    } else {
        if bigram {
            features.emit(format_args!("{},{}", c0.cons, c1.cons));
        }
        if c1.unigram {
            features.emit(format_args!("{},{}", c0.cons_bo, c1.cons));
        }
        if c0.unigram {
            features.emit(format_args!("{},{}", c0.cons, c1.cons_bo));
        }
        features.emit(format_args!("{},{}", c0.cons_bo, c1.cons_bo));
    }
}

/// Adds the eight (or fewer) three-node features for one choice of separators, in the order of
/// `AbstractContextGenerator.cons3`. `s1` goes between `c0` and `c1`, `s2` between `c1` and `c2`.
#[allow(clippy::too_many_arguments)]
fn cons3_with(
    features: &mut impl FeatureSink,
    c0: &ConsRef<'_>,
    c1: &ConsRef<'_>,
    c2: &ConsRef<'_>,
    s1: impl Display,
    s2: impl Display,
    trigram: bool,
    bigram1: bool,
    bigram2: bool,
) {
    if trigram {
        features.emit(format_args!("{}{s1}{}{s2}{}", c0.cons, c1.cons, c2.cons));
    }
    if bigram2 {
        features.emit(format_args!("{}{s1}{}{s2}{}", c0.cons_bo, c1.cons, c2.cons));
    }
    if c0.unigram && c2.unigram {
        features.emit(format_args!("{}{s1}{}{s2}{}", c0.cons, c1.cons_bo, c2.cons));
    }
    if bigram1 {
        features.emit(format_args!("{}{s1}{}{s2}{}", c0.cons, c1.cons, c2.cons_bo));
    }
    if c2.unigram {
        features.emit(format_args!("{}{s1}{}{s2}{}", c0.cons_bo, c1.cons_bo, c2.cons));
    }
    if c1.unigram {
        features.emit(format_args!("{}{s1}{}{s2}{}", c0.cons_bo, c1.cons, c2.cons_bo));
    }
    if c0.unigram {
        features.emit(format_args!("{}{s1}{}{s2}{}", c0.cons, c1.cons_bo, c2.cons_bo));
    }
    features.emit(format_args!("{}{s1}{}{s2}{}", c0.cons_bo, c1.cons_bo, c2.cons_bo));
}

/// Adds the features of three nodes and the punctuation between them
/// (`AbstractContextGenerator.cons3`, with `zeroBackOff` false as every generator has it).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cons3_into(
    features: &mut impl FeatureSink,
    c0: &ConsRef<'_>,
    c1: &ConsRef<'_>,
    c2: &ConsRef<'_>,
    punct1s: Option<&[Parse]>,
    punct2s: Option<&[Parse]>,
    trigram: bool,
    bigram1: bool,
    bigram2: bool,
) {
    let punct1s = punct_set_of(punct1s);
    let punct2s = punct_set_of(punct2s);
    if let Some(p1s) = punct1s
        && c0.index == -2
    {
        for p in p1s {
            features.emit(format_args!("{}={}", punct_index(c1.index), p.kind()));
        }
    }
    if let Some(p2s) = punct2s {
        if c2.index == 2 {
            for p in p2s {
                features.emit(format_args!("{}={}", c2.index, p.kind()));
            }
        }
        for punct2 in p2s {
            let pbo2 = DisplayFn(|f: &mut fmt::Formatter<'_>| write!(f, "{}={}", punct_index(c2.index), punct2.kind()));
            if let Some(p1s) = punct1s {
                for punct1 in p1s {
                    let pbo1 = DisplayFn(|f: &mut fmt::Formatter<'_>| {
                        write!(f, "{}={}", punct_index(c1.index), punct1.kind())
                    });
                    let s1 = features.prepare(DisplayFn(|f: &mut fmt::Formatter<'_>| write!(f, ",{pbo1},")));
                    let s2 = features.prepare(DisplayFn(|f: &mut fmt::Formatter<'_>| write!(f, ",{pbo2},")));
                    cons3_with(features, c0, c1, c2, &s1, &s2, trigram, bigram1, bigram2);
                }
            } else {
                let s2 = features.prepare(DisplayFn(|f: &mut fmt::Formatter<'_>| write!(f, ",{pbo2},")));
                cons3_with(features, c0, c1, c2, ",", &s2, trigram, bigram1, bigram2);
            }
        }
    } else if let Some(p1s) = punct1s {
        for punct1 in p1s {
            let pbo1 = DisplayFn(|f: &mut fmt::Formatter<'_>| write!(f, "{}={}", punct_index(c1.index), punct1.kind()));
            let s1 = features.prepare(DisplayFn(|f: &mut fmt::Formatter<'_>| write!(f, ",{pbo1},")));
            cons3_with(features, c0, c1, c2, &s1, ",", trigram, bigram1, bigram2);
        }
    } else {
        cons3_with(features, c0, c1, c2, ",", ",", trigram, bigram1, bigram2);
    }
}

/// Adds the features of a node next to a completed node of type `kind`
/// (`AbstractContextGenerator.surround`).
///
/// With punctuation, Java writes three features per punctuation node into one `StringBuilder`
/// and clears it before the second and the third, not before the first. The first feature of the
/// second and later nodes therefore starts with the last feature of the node before it, as in
/// `s2*=VP|''she|VP|NP|,` for the set `'' ,` after `she`. A set holds more than one node only when
/// their probabilities differ, so this shows when parsing, not on gold trees.
pub(crate) fn surround_into(
    node: Option<&Parse>,
    i: i32,
    kind: &str,
    punctuation: Option<&[Parse]>,
    features: &mut impl FeatureSink,
) {
    if let Some(puncts) = punct_set_of(punctuation) {
        let mut previous = None;
        for punct in puncts {
            let ptype = punct.kind();
            let prefix = DisplayFn(|f: &mut fmt::Formatter<'_>| match previous {
                None => write!(f, "s{i}="),
                Some(ptype) => write!(f, "s{i}*={kind}|{ptype}"),
            });
            match node {
                Some(n) => features.emit(format_args!("{prefix}{}|{kind}|{}|{ptype}", n.head_text(), n.kind())),
                None => features.emit(format_args!("{prefix}{kind}|{EOS}|{ptype}")),
            }
            features.emit(format_args!("s{i}*={kind}|{}|{ptype}", node.map_or(EOS, Parse::kind)));
            features.emit(format_args!("s{i}*={kind}|{ptype}"));
            previous = Some(ptype);
        }
    } else {
        match node {
            Some(n) => features.emit(format_args!("s{i}={}|{kind}|{}", n.head_text(), n.kind())),
            None => features.emit(format_args!("s{i}={kind}|{EOS}")),
        }
        features.emit(format_args!("s{i}*={kind}|{}", node.map_or(EOS, Parse::kind)));
    }
}

/// Adds the features of one child of a constituent of type `kind` being checked
/// (`AbstractContextGenerator.checkcons(Parse, String, String, List)`).
pub(crate) fn check_cons_into(child: &Parse, index: &str, kind: &str, features: &mut impl FeatureSink) {
    features.emit(format_args!("c{index}={}|{}|{kind}", child.kind(), child.head_text()));
    features.emit(format_args!("c{index}*={}|{kind}", child.kind()));
}

/// Adds the features of two children of a constituent of type `kind` being checked
/// (`AbstractContextGenerator.checkcons(Parse, Parse, String, List)`).
pub(crate) fn check_cons_pair_into(p1: &Parse, p2: &Parse, kind: &str, features: &mut impl FeatureSink) {
    let (t1, h1, t2, h2) = (p1.kind(), p1.head_text(), p2.kind(), p2.head_text());
    features.emit(format_args!("cil={kind},{t1}|{h1},{t2}|{h2}"));
    features.emit(format_args!("ci*l={kind},{t1},{t2}|{h2}"));
    features.emit(format_args!("cil*={kind},{t1}|{h1},{t2}"));
    features.emit(format_args!("ci*l*={kind},{t1},{t2}"));
}

/// Adds two-node features to a string context.
pub fn cons2(features: &mut Vec<String>, c0: &Cons, c1: &Cons, punct1s: Option<&[Parse]>, bigram: bool) {
    cons2_into(features, &c0.into(), &c1.into(), punct1s, bigram);
}

/// Adds three-node features to a string context.
#[allow(clippy::too_many_arguments)]
pub fn cons3(
    features: &mut Vec<String>,
    c0: &Cons,
    c1: &Cons,
    c2: &Cons,
    punct1s: Option<&[Parse]>,
    punct2s: Option<&[Parse]>,
    trigram: bool,
    bigram1: bool,
    bigram2: bool,
) {
    cons3_into(features, &c0.into(), &c1.into(), &c2.into(), punct1s, punct2s, trigram, bigram1, bigram2);
}

/// Adds surrounding-node features to a string context.
pub fn surround(node: Option<&Parse>, i: i32, kind: &str, punctuation: Option<&[Parse]>, features: &mut Vec<String>) {
    surround_into(node, i, kind, punctuation, features);
}

/// Adds one-child check features to a string context.
pub fn check_cons(child: &Parse, index: &str, kind: &str, features: &mut Vec<String>) {
    check_cons_into(child, index, kind, features);
}

/// Adds two-child check features to a string context.
pub fn check_cons_pair(p1: &Parse, p2: &Parse, kind: &str, features: &mut Vec<String>) {
    check_cons_pair_into(p1, p2, kind, features);
}

/// Extracts the 5-constituent window `[c_2, c_1, c0, c1, c2]` around `index` in `chunks`
/// (`AbstractContextGenerator.getNodes`).
pub fn get_nodes(chunks: &[Parse], index: usize) -> [Option<&Parse>; 5] {
    [
        index.checked_sub(2).and_then(|j| chunks.get(j)),
        index.checked_sub(1).and_then(|j| chunks.get(j)),
        chunks.get(index),
        chunks.get(index + 1),
        chunks.get(index + 2),
    ]
}
