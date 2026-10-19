//! The tree-insertion build, attach, and check context generators
//! (`opennlp.tools.parser.treeinsert.BuildContextGenerator`, `AttachContextGenerator`, and
//! `CheckContextGenerator`).

use std::collections::HashSet;

use turbo_sequence::Features;

use crate::context::{
    EOS, FeatureSink, check_cons_into as check_cons, cons2_into as cons2, cons3_into as cons3,
    surround_into as surround,
};
use crate::parse::{Parse, ParseError};

use super::*;

/// Context generator for the tree-insertion build phase (`opennlp.tools.parser.treeinsert.BuildContextGenerator`).
#[derive(Clone, Debug, Default)]
pub struct BuildContextGenerator;

impl BuildContextGenerator {
    /// Creates a new [`BuildContextGenerator`].
    pub fn new() -> Self {
        Self
    }

    /// Generates the features for deciding what to build over `constituents[index]`
    /// (`BuildContextGenerator.getContext(Parse[], int)`).
    ///
    /// # Panics
    ///
    /// When `index` is out of bounds, where OpenNLP throws `ArrayIndexOutOfBoundsException`.
    pub fn get_context(&self, constituents: &[&Parse], index: usize) -> Vec<String> {
        let mut features = Vec::with_capacity(64);
        self.populate(constituents, index, &mut features);
        features
    }

    /// Replaces `out` with build features, reusing its byte storage.
    pub fn get_features(&self, constituents: &[&Parse], index: usize, out: &mut Features) {
        out.clear();
        self.populate(constituents, index, out);
    }

    fn populate(&self, constituents: &[&Parse], index: usize, features: &mut impl FeatureSink) {
        let p0 = constituents[index];
        let p1 = constituents.get(index + 1).copied();
        let p2 = constituents.get(index + 2).copied();

        let punct_1s = punct(p0.previous_punctuation());
        let punct1s = punct(p0.next_punctuation());
        let punct2s = p1.and_then(|p| punct(p.next_punctuation()));

        let ([p_1, p_2], has_frontier) = if index == 0 {
            ([None, None], false)
        } else {
            frontier_nodes(constituents[0], &HashSet::new(), None).expect("no frontier trim")
        };
        let punct_2s = p_1.and_then(|p| punct(p.previous_punctuation()));

        let c_2 = cons_slot(p_2, -2, features);
        let c_1 = cons_slot(p_1, -1, features);
        let c0 = cons_slot(Some(p0), 0, features);
        let c1 = cons_slot(p1, 1, features);
        let c2 = cons_slot(p2, 2, features);

        features.emit(format_args!("default"));
        for c in [&c_2, &c_1, &c0, &c1, &c2] {
            features.emit(format_args!("{}", c.cons));
            features.emit(format_args!("{}", c.cons_bo));
        }

        cons2(features, &c0, &c1, punct1s, true);
        cons2(features, &c_1, &c0, punct_1s, true);
        cons3(features, &c0, &c1, &c2, punct1s, punct2s, true, true, true);
        cons3(features, &c_2, &c_1, &c0, punct_2s, punct_1s, true, true, true);
        cons3(features, &c_1, &c0, &c1, punct_1s, punct_1s, true, true, true);

        if !has_frontier {
            features.emit(format_args!("{EOS},{}", c0.cons));
            features.emit(format_args!("{EOS},{}", c0.cons_bo));
        }
    }
}

/// Context generator for the tree-insertion attachment phase (`opennlp.tools.parser.treeinsert.AttachContextGenerator`).
#[derive(Clone, Debug)]
pub struct AttachContextGenerator {
    punct_set: HashSet<String>,
}

impl AttachContextGenerator {
    /// Creates a new [`AttachContextGenerator`] with `punct_set`.
    pub fn new(punct_set: HashSet<String>) -> Self {
        Self { punct_set }
    }

    /// Generates the features for attaching `constituents[index]` to `right_frontier[rfi]`
    /// (`AttachContextGenerator.getContext(Parse[], int, List<Parse>, int)`).
    ///
    /// # Panics
    ///
    /// When `index` or `rfi` is out of bounds, where OpenNLP throws.
    pub fn get_context(
        &self,
        constituents: &[&Parse],
        index: usize,
        right_frontier: &[&Parse],
        rfi: usize,
    ) -> Vec<String> {
        let mut features = Vec::with_capacity(100);
        self.populate(constituents, index, right_frontier, rfi, &mut features);
        features
    }

    /// Replaces `out` with attachment features, reusing its byte storage.
    pub fn get_features(
        &self,
        constituents: &[&Parse],
        index: usize,
        right_frontier: &[&Parse],
        rfi: usize,
        out: &mut Features,
    ) {
        out.clear();
        self.populate(constituents, index, right_frontier, rfi, out);
    }

    fn populate(
        &self,
        constituents: &[&Parse],
        index: usize,
        right_frontier: &[&Parse],
        rfi: usize,
        features: &mut impl FeatureSink,
    ) {
        let fnode = right_frontier[rfi];
        let fp = right_frontier.get(rfi + 1).copied();
        let p_1 = right_frontier.first().copied();
        let p0 = constituents[index];
        let p1 = constituents.get(index + 1).copied();

        let punct_1fs = punct(fnode.previous_punctuation());
        let punct_1s = punct(p0.previous_punctuation());
        let punct1s = punct(p0.next_punctuation());

        let cfp = cons_slot(fp, -3, features);
        let cf = cons_slot(Some(fnode), -2, features);
        let c_1 = cons_slot(p_1, -1, features);
        let c0 = cons_slot(Some(p0), 0, features);
        let c1 = cons_slot(p1, 1, features);

        features.emit(format_args!("default"));
        for c in [&cfp, &cf, &c_1, &c0, &c1] {
            features.emit(format_args!("{}", c.cons));
            features.emit(format_args!("{}", c.cons_bo));
        }

        let prod = features.prepare(production(fnode, false, &self.punct_set));
        features.emit(format_args!("pn={prod}"));
        features.emit(format_args!("pd={prod},{}", p0.kind()));
        features.emit(format_args!("ps={}->{},{}", fnode.kind(), fnode.kind(), p0.kind()));

        cons2(features, &cfp, &c0, punct_1s, true);
        cons2(features, &cf, &c0, punct_1s, true);
        cons2(features, &c_1, &c0, punct_1s, true);
        cons2(features, &c0, &c1, punct1s, true);
        cons3(features, &cf, &c_1, &c0, None, punct_1s, true, true, true);
        cons3(features, &cf, &c0, &c1, punct_1s, punct1s, true, true, true);
        cons3(features, &cfp, &cf, &c0, None, punct_1s, true, true, true);

        let head_distance = p0.head_index() - fnode.head_index();
        features.emit(format_args!("hd={head_distance}"));
        features.emit(format_args!("nd={rfi}"));
        features.emit(format_args!("nd={}.{rfi}", p0.kind()));
        features.emit(format_args!("hd={}.{head_distance}", p0.kind()));

        let contains = |set: Option<&[Parse]>, mark: &str| set.is_some_and(|s| s.iter().any(|p| p.kind() == mark));
        if contains(punct_1s, "''") && contains(punct_1fs, "``") {
            features.emit(format_args!("quotematch"));
        }
    }
}

/// Context generator for the tree-insertion check phase (`opennlp.tools.parser.treeinsert.CheckContextGenerator`).
#[derive(Clone, Debug)]
pub struct CheckContextGenerator {
    punct_set: HashSet<String>,
}

impl CheckContextGenerator {
    /// Creates a new [`CheckContextGenerator`] with `punct_set`.
    pub fn new(punct_set: HashSet<String>) -> Self {
        Self { punct_set }
    }

    /// Generates the features for deciding whether `parent` is complete
    /// (`CheckContextGenerator.getContext(Parse, Parse[], int, boolean)`). With `trim_frontier`,
    /// the right frontier of `constituents[0]` is cut below `parent`, which must be on it
    /// ([`TreeInsertError::ParentNotInFrontier`] otherwise). A `parent` with no children that are
    /// not punctuation fails with [`ParseError::TokenIndex`], where OpenNLP throws
    /// `ArrayIndexOutOfBoundsException`.
    ///
    /// # Panics
    ///
    /// When `index` is out of bounds, where OpenNLP throws `ArrayIndexOutOfBoundsException`.
    pub fn get_context(
        &self,
        parent: &Parse,
        constituents: &[&Parse],
        index: usize,
        trim_frontier: bool,
    ) -> Result<Vec<String>, TreeInsertError> {
        let mut features = Vec::with_capacity(100);
        self.populate(parent, constituents, index, trim_frontier, &mut features)?;
        Ok(features)
    }

    /// Replaces `out` with completeness features, reusing its byte storage.
    pub fn get_features(
        &self,
        parent: &Parse,
        constituents: &[&Parse],
        index: usize,
        trim_frontier: bool,
        out: &mut Features,
    ) -> Result<(), TreeInsertError> {
        out.clear();
        self.populate(parent, constituents, index, trim_frontier, out)
    }

    fn populate(
        &self,
        parent: &Parse,
        constituents: &[&Parse],
        index: usize,
        trim_frontier: bool,
        features: &mut impl FeatureSink,
    ) -> Result<(), TreeInsertError> {
        features.emit(format_args!("default"));
        let mut children = parent.children().iter().filter(|c| !self.punct_set.contains(c.kind()));
        // OpenNLP reads `children[0]` of the collapsed children, which fails when they are empty.
        let first = children.next().ok_or(TreeInsertError::Parse(ParseError::TokenIndex { index: 0, len: 0 }))?;
        let last = children.next_back().unwrap_or(first);
        let kind = parent.kind();
        check_cons(first, "begin", kind, features);
        check_cons(last, "last", kind, features);
        features.emit(format_args!("p={}", production(parent, false, &self.punct_set)));
        features.emit(format_args!("pp={}", production(parent, true, &self.punct_set)));

        let p1s = punct(constituents[index].next_punctuation());
        let p_1s = punct(constituents[index].previous_punctuation());

        let [p_1, p_2] = if index == 0 {
            [None, None]
        } else {
            frontier_nodes(constituents[0], &self.punct_set, trim_frontier.then_some(parent))?.0
        };
        let p_2s = p_1.and_then(|p| punct(p.previous_punctuation()));
        let p1 = constituents.get(index + 1).copied();
        let p2s = p1.and_then(|p| punct(p.next_punctuation()));
        let p2 = constituents.get(index + 2).copied();

        surround(p_1, -1, kind, p_1s, features);
        surround(p_2, -2, kind, p_2s, features);
        surround(p1, 1, kind, p1s, features);
        surround(p2, 2, kind, p2s, features);
        Ok(())
    }
}
