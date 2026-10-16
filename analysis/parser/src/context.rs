//! Context generators and sequence validators shared across the constituency parsers
//! (`opennlp.tools.parser.Cons`, `opennlp.tools.parser.AbstractContextGenerator`,
//! `opennlp.tools.parser.ChunkContextGenerator`, and
//! `opennlp.tools.parser.ParserChunkerSequenceValidator`).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{self, Display, Write};

use turbo_sequence::chunker::TokenTag;
use turbo_sequence::{ContextGenerator, Features, SequenceValidator};

use crate::bottom_up::dictionary_key;
use crate::parse::{CONT, OTHER, Parse, START};

// Both public string contexts and inference buffers use the same feature order.
pub(crate) trait FeatureSink {
    fn emit(&mut self, value: fmt::Arguments<'_>);

    fn prepare<T: Display>(&self, value: T) -> Prepared<T> {
        Prepared::Direct(value)
    }
}

impl FeatureSink for Vec<String> {
    fn prepare<T: Display>(&self, value: T) -> Prepared<T> {
        let mut rendered = String::with_capacity(32);
        write!(rendered, "{value}").expect("string writes cannot fail");
        Prepared::Rendered(rendered)
    }

    fn emit(&mut self, value: fmt::Arguments<'_>) {
        // Reserve room for a typical combined feature instead of growing per piece.
        let mut rendered = String::with_capacity(64);
        rendered.write_fmt(value).expect("string writes cannot fail");
        self.push(rendered);
    }
}

impl FeatureSink for Features {
    fn emit(&mut self, value: fmt::Arguments<'_>) {
        struct Writer<'a>(&'a mut Features);
        impl Write for Writer<'_> {
            fn write_str(&mut self, s: &str) -> fmt::Result {
                self.0.push_str(s);
                Ok(())
            }
        }
        Writer(self).write_fmt(value).expect("feature buffer writes cannot fail");
        self.finish();
    }
}

// String contexts reuse rendered pieces; inference keeps streaming into Features.
pub(crate) enum Prepared<T> {
    Direct(T),
    Rendered(String),
}

impl<T: Display> Display for Prepared<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direct(value) => value.fmt(f),
            Self::Rendered(value) => f.write_str(value),
        }
    }
}

pub(crate) struct DisplayFn<F>(pub F);
impl<F: Fn(&mut fmt::Formatter<'_>) -> fmt::Result> Display for DisplayFn<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self.0)(f)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ConsFeature<'a> {
    Text(&'a str),
    Node { node: Option<&'a Parse>, index: i32, labeled: bool, backoff: bool },
}

impl Display for ConsFeature<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Text(s) => f.write_str(s),
            Self::Node { node, index, labeled, backoff } => {
                write!(f, "{index}{}=", if backoff { "*" } else { "" })?;
                if let Some(p) = node {
                    if labeled && index < 0 {
                        write!(f, "{}|", p.label().unwrap_or("null"))?;
                    }
                    f.write_str(p.kind())?;
                    if !backoff {
                        write!(f, "|{}", p.head_text())?;
                    }
                    Ok(())
                } else {
                    f.write_str(EOS)
                }
            }
        }
    }
}

pub(crate) struct ConsRef<'a> {
    pub cons: Prepared<ConsFeature<'a>>,
    pub cons_bo: Prepared<ConsFeature<'a>>,
    pub index: i32,
    pub unigram: bool,
}

impl<'a> ConsRef<'a> {
    pub fn new(node: Option<&'a Parse>, index: i32, labeled: bool, unigram: bool, sink: &impl FeatureSink) -> Self {
        Self {
            cons: sink.prepare(ConsFeature::Node { node, index, labeled, backoff: false }),
            cons_bo: sink.prepare(ConsFeature::Node { node, index, labeled, backoff: true }),
            index,
            unigram,
        }
    }
}

impl<'a> From<&'a Cons> for ConsRef<'a> {
    fn from(c: &'a Cons) -> Self {
        Self {
            cons: Prepared::Direct(ConsFeature::Text(&c.cons)),
            cons_bo: Prepared::Direct(ConsFeature::Text(&c.cons_bo)),
            index: c.index,
            unigram: c.unigram,
        }
    }
}

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

/// Builds the Cons window `(c_2, c_1, c0, c1, c2)` and unigram/bigram/trigram dictionary flags.
pub struct ConsWindow {
    pub c_2: Cons,
    pub c_1: Cons,
    pub c0: Cons,
    pub c1: Cons,
    pub c2: Cons,
    pub u_2: bool,
    pub u_1: bool,
    pub u0: bool,
    pub u1: bool,
    pub u2: bool,
    pub b_2_1: bool,
    pub b_10: bool,
    pub b01: bool,
    pub b12: bool,
    pub t_2_10: bool,
    pub t_101: bool,
    pub t012: bool,
}

impl ConsWindow {
    /// Computes the constituent window and dictionary membership flags around `nodes`
    /// (`[c_2, c_1, c0, c1, c2]`). When `dict` is `None`, all unigram/bigram/trigram flags are
    /// `true`.
    pub fn build<'a>(nodes: [Option<&'a Parse>; 5], dict: Option<&HashSet<String>>) -> Self {
        let [cf_2, cf_1, cf0, cf1, cf2] = nodes;
        let head_word = |n: Option<&'a Parse>| -> &'a str { n.map_or(EOS, Parse::head_text) };
        let w_2 = head_word(cf_2);
        let w_1 = head_word(cf_1);
        let w0 = head_word(cf0);
        let w1 = head_word(cf1);
        let w2 = head_word(cf2);

        let (u_2, u_1, u0, u1, u2, b_2_1, b_10, b01, b12, t_2_10, t_101, t012) = match dict {
            Some(words) => (
                words.contains(&dictionary_key(&[w_2])),
                words.contains(&dictionary_key(&[w_1])),
                words.contains(&dictionary_key(&[w0])),
                words.contains(&dictionary_key(&[w1])),
                words.contains(&dictionary_key(&[w2])),
                words.contains(&dictionary_key(&[w_2, w_1])),
                words.contains(&dictionary_key(&[w_1, w0])),
                words.contains(&dictionary_key(&[w0, w1])),
                words.contains(&dictionary_key(&[w1, w2])),
                words.contains(&dictionary_key(&[w_2, w_1, w0])),
                words.contains(&dictionary_key(&[w_1, w0, w1])),
                words.contains(&dictionary_key(&[w0, w1, w2])),
            ),
            None => (true, true, true, true, true, true, true, true, true, true, true, true),
        };

        let c_2 = Cons::new(cons(cf_2, -2), cons_bo(cf_2, -2), -2, u_2);
        let c_1 = Cons::new(cons(cf_1, -1), cons_bo(cf_1, -1), -1, u_1);
        let c0 = Cons::new(cons(cf0, 0), cons_bo(cf0, 0), 0, u0);
        let c1 = Cons::new(cons(cf1, 1), cons_bo(cf1, 1), 1, u1);
        let c2 = Cons::new(cons(cf2, 2), cons_bo(cf2, 2), 2, u2);

        Self { c_2, c_1, c0, c1, c2, u_2, u_1, u0, u1, u2, b_2_1, b_10, b01, b12, t_2_10, t_101, t012 }
    }
}

/// Context generator for the parser's basal chunking phase (`opennlp.tools.parser.ChunkContextGenerator`).
///
/// Reproduces `ChunkContextGenerator`'s 19 features and its optional cache, keyed as OpenNLP keys
/// it by `i + t_2 + t1 + t0 + t1 + t2 + p_2 + p_1`.
#[derive(Clone, Debug, Default)]
pub struct ChunkContextGenerator {
    cache_size: usize,
    words_key: Vec<String>,
    tags_key: Vec<String>,
    cache: VecDeque<(String, Vec<String>)>,
}

impl ChunkContextGenerator {
    /// Creates an uncached [`ChunkContextGenerator`].
    pub fn new() -> Self {
        Self::with_cache(0)
    }

    /// Creates a [`ChunkContextGenerator`] with `cache_size` cached contexts.
    pub fn with_cache(cache_size: usize) -> Self {
        Self { cache_size, words_key: Vec::new(), tags_key: Vec::new(), cache: VecDeque::with_capacity(cache_size) }
    }

    /// Empties the cache, as a new `words` array does in OpenNLP.
    pub fn reset(&mut self) {
        self.cache.clear();
        self.words_key.clear();
        self.tags_key.clear();
    }

    /// Generates the 19 chunking features at token `i` (`ChunkContextGenerator.getContext`):
    /// `default`, the word and tag of each of the five positions with and without the word,
    /// then four pairs of positions -1 and 0, and four of 0 and 1. A position outside the
    /// sentence has `eos` for its word, tag and chunk tag.
    ///
    /// OpenNLP keeps the cache while `words` is the same array (`words == wordsKey`). A slice
    /// address can be reused by a later sentence, so the cache is kept here while `words` and
    /// `tags` hold the same strings, which gives the features a fresh call would.
    pub fn get_context(&mut self, i: usize, words: &[&str], tags: &[&str], preds: &[&str]) -> Vec<String> {
        self.context_strings(i, tags.len(), |j| (words[j], tags[j]), preds)
    }

    fn context_strings<'a>(
        &mut self,
        i: usize,
        len: usize,
        token: impl Fn(usize) -> (&'a str, &'a str),
        preds: &[&str],
    ) -> Vec<String> {
        let (t_2, p_2) = i.checked_sub(2).map_or((EOS, EOS), |j| (token(j).1, preds[j]));
        let p_1 = i.checked_sub(1).map_or(EOS, |j| preds[j]);
        let t0 = token(i).1;
        let t1 = if i + 1 < len { token(i + 1).1 } else { EOS };
        let t2 = if i + 2 < len { token(i + 2).1 } else { EOS };

        let cache_key = if self.cache_size > 0 {
            let same = self.words_key.len() == len
                && self.tags_key.len() == len
                && (0..len).all(|j| {
                    let (word, tag) = token(j);
                    self.words_key[j] == word && self.tags_key[j] == tag
                });
            if !same {
                self.cache.clear();
                self.words_key = (0..len).map(|j| token(j).0.to_owned()).collect();
                self.tags_key = (0..len).map(|j| token(j).1.to_owned()).collect();
            }
            // OpenNLP's key: `t1` twice and no `t_1`.
            let key = format!("{i}{t_2}{t1}{t0}{t1}{t2}{p_2}{p_1}");
            if let Some(pos) = self.cache.iter().position(|(k, _)| k == &key) {
                let entry = self.cache.remove(pos).expect("pos in bounds");
                let out = entry.1.clone();
                self.cache.push_front(entry);
                return out;
            }
            Some(key)
        } else {
            None
        };

        let mut features = Vec::with_capacity(19);
        Self::populate(i, len, token, preds, &mut features);

        if let Some(key) = cache_key {
            if self.cache.len() >= self.cache_size {
                self.cache.pop_back();
            }
            self.cache.push_front((key, features.clone()));
        }
        features
    }

    /// Replaces `out` with the uncached chunk features, reusing its byte storage.
    pub fn get_features(&self, i: usize, words: &[&str], tags: &[&str], preds: &[&str], out: &mut Features) {
        out.clear();
        Self::populate(i, tags.len(), |j| (words[j], tags[j]), preds, out);
    }

    fn populate<'a>(
        i: usize,
        len: usize,
        token: impl Fn(usize) -> (&'a str, &'a str),
        preds: &[&str],
        out: &mut impl FeatureSink,
    ) {
        assert!(i < len, "chunk token index out of bounds");
        let slots = [-2, -1, 0, 1, 2].map(|offset| {
            let at = i.checked_add_signed(offset);
            let (word, tag) = at.filter(|&j| j < len).map_or((EOS, EOS), &token);
            let prior = if offset < 0 { at.map_or(EOS, |j| preds[j]) } else { "" };
            (offset, word, tag, prior)
        });
        let lexical = |n: usize, backoff: bool| {
            let (offset, word, tag, prior) = slots[n];
            DisplayFn(move |f: &mut fmt::Formatter<'_>| {
                write!(f, "{offset}{}=", if backoff { "*" } else { "" })?;
                if !backoff {
                    write!(f, "{word}|")?;
                }
                f.write_str(tag)?;
                if offset < 0 {
                    write!(f, "|{prior}")?;
                }
                Ok(())
            })
        };
        let pieces =
            std::array::from_fn::<_, 5, _>(|n| [out.prepare(lexical(n, false)), out.prepare(lexical(n, true))]);
        let lexical = |n: usize, backoff: bool| &pieces[n][usize::from(backoff)];
        out.emit(format_args!("default"));
        for n in 0..5 {
            out.emit(format_args!("{}", lexical(n, false)));
            out.emit(format_args!("{}", lexical(n, true)));
        }
        for (a, b) in [(1, 2), (2, 3)] {
            for (ab, bb) in [(false, false), (true, false), (false, true), (true, true)] {
                out.emit(format_args!("{},{}", lexical(a, ab), lexical(b, bb)));
            }
        }
    }
}

impl<'a> ContextGenerator<TokenTag<'a>> for ChunkContextGenerator {
    fn start_sequence(&mut self) {
        self.reset();
    }

    fn context(&mut self, index: usize, sequence: &[TokenTag<'a>], prior: &[&str], out: &mut Features) {
        let token = |j: usize| (sequence[j].token, sequence[j].tag);
        if self.cache_size == 0 {
            Self::populate(index, sequence.len(), token, prior, out);
        } else {
            // Preserve the opt-in JVM cache's key and eviction behavior. The parser's
            // default uncached beam path writes directly from TokenTag without collections.
            for feature in self.context_strings(index, sequence.len(), token, prior) {
                out.push_str(&feature);
                out.finish();
            }
        }
    }
}

/// Sequence validator for the parser's basal chunker (`opennlp.tools.parser.ParserChunkerSequenceValidator`).
///
/// Outcomes use `S-<type>`, `C-<type>`, and `O`. A continuation outcome `C-<type>` is valid only
/// after `S-<type>` or `C-<type>`.
#[derive(Clone, Debug, Default)]
pub struct ParserChunkerSequenceValidator {
    continue_start_map: HashMap<String, String>,
}

impl ParserChunkerSequenceValidator {
    /// Creates a validator from the chunker model's outcome labels.
    pub fn new<S: AsRef<str>>(outcomes: &[S]) -> Self {
        let mut continue_start_map = HashMap::with_capacity(outcomes.len());
        for outcome in outcomes {
            let outcome = outcome.as_ref();
            if let Some(rest) = outcome.strip_prefix(CONT) {
                continue_start_map.insert(outcome.to_string(), format!("{START}{rest}"));
            }
        }
        Self { continue_start_map }
    }

    /// Validates whether `outcome` can occur at `i` after `outcomes`.
    pub fn valid_outcome(&self, outcomes: &[&str], outcome: &str) -> bool {
        if let Some(start_tag) = self.continue_start_map.get(outcome) {
            if outcomes.is_empty() {
                return false;
            }
            let last_tag = outcomes[outcomes.len() - 1];
            if last_tag == outcome || last_tag == start_tag {
                return true;
            }
            if last_tag == OTHER {
                return false;
            }
            return false;
        }
        true
    }
}

impl<'a> SequenceValidator<TokenTag<'a>> for ParserChunkerSequenceValidator {
    fn valid_sequence(&self, _index: usize, _sequence: &[TokenTag<'a>], prior: &[&str], outcome: &str) -> bool {
        self.valid_outcome(prior, outcome)
    }
}
