//! Bottom-up tree-insertion constituency parser and its context generators
//! (`opennlp.tools.parser.treeinsert.*`). The event stream and the trainer are
//! `turbo_train_components::parser::treeinsert`.
//!
//! A model trained by OpenNLP must decode here as it does on the JVM, so the feature strings, the
//! order of the derivations [`Parser::advance_parses`] returns, and the training events the event
//! stream makes with these generators follow `opennlp.tools.parser.treeinsert` exactly.
//!
//! OpenNLP shares nodes between derivations and changes their punctuation sets as a side effect of
//! `collapsePunctuation`, which its context generators call on the children of the nodes they
//! look at. A node copied by `Parse.clone` starts with no punctuation sets. Both effects change
//! later features, so the parser and the event stream repeat them on their own trees: the copies
//! along the path a derivation changes lose their punctuation sets, and the children of the nodes
//! a context generator collapses get the punctuation next to them added to their sets
//! ([`collapse_children`]). Sharing between sibling derivations in the beam is not repeated; a
//! node one derivation changes this way is seen changed by the other derivations that share it in
//! OpenNLP and not here.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use turbo_analysis::Span;
use turbo_jvm::strict_math;
use turbo_sequence::Features;

use crate::bottom_up::{
    DEFAULT_ADVANCE_PERCENTAGE, DEFAULT_BEAM_SIZE, Parser as ParserTrait, collapse_punctuation_in_place,
    parse_bottom_up,
};
use crate::context::{
    ConsRef, DisplayFn, EOS, FeatureSink, check_cons_into as check_cons, cons2_into as cons2, cons3_into as cons3,
    surround_into as surround,
};
use crate::head_rules::HeadRules;
use crate::model::ParserModel;
use crate::parse::{COMPLETE, INC_NODE, Parse, ParseError, TOP_NODE};

/// Outcome indicating no further unary build levels (`treeinsert.Parser.DONE = "d"`).
pub const DONE: &str = "d";

/// Outcome indicating sister adjunction (`treeinsert.Parser.ATTACH_SISTER = "s"`).
pub const ATTACH_SISTER: &str = "s";

/// Outcome indicating daughter attachment (`treeinsert.Parser.ATTACH_DAUGHTER = "d"`).
pub const ATTACH_DAUGHTER: &str = "d";

/// Outcome indicating non-attachment (`treeinsert.Parser.NON_ATTACH = "n"`).
pub const NON_ATTACH: &str = "n";

/// Label marking a constituent as already built (`treeinsert.Parser.BUILT = "built"`).
pub const BUILT: &str = "built";

/// An error raised by the tree-insertion parser or its event stream where OpenNLP throws or exits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeInsertError {
    /// Changing a parse tree failed (`Parse.add`, `Parse.adjoin`, `Parse.insert`, `Parse.pruneParse`).
    Parse(ParseError),
    /// The parser model has no `attach.model`.
    MissingAttachModel,
    /// The node a completeness check is asked about is not on the right frontier
    /// (`RuntimeException("Parent not found in right frontier")`).
    ParentNotInFrontier,
    /// The right frontier of the parse built so far and the gold frontier have different sizes,
    /// where OpenNLP logs `Frontiers mis-aligned` and calls `System.exit(1)`.
    FrontierMismatch {
        /// Size of the right frontier of the parse built so far.
        current: usize,
        /// Size of the gold right frontier.
        gold: usize,
    },
    /// No attachment site was found for a chunk (`RuntimeException("No Attachment")`).
    NoAttachment {
        /// Span of the gold node that could not be attached.
        span: Span,
    },
    /// A walk up the gold tree went past its root, where OpenNLP dereferences a missing parent.
    MissingParent,
}

impl fmt::Display for TreeInsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "{e}"),
            Self::MissingAttachModel => f.write_str("TREEINSERT parser model has no attach model"),
            Self::ParentNotInFrontier => f.write_str("parent not found in right frontier"),
            Self::FrontierMismatch { current, gold } => write!(f, "frontiers mis-aligned: {current} != {gold}"),
            Self::NoAttachment { span } => write!(f, "no attachment for the node at {span:?}"),
            Self::MissingParent => f.write_str("walked past the root of the gold parse"),
        }
    }
}

impl std::error::Error for TreeInsertError {}

impl From<ParseError> for TreeInsertError {
    fn from(e: ParseError) -> Self {
        Self::Parse(e)
    }
}

/// A punctuation set as OpenNLP sees it: `null` until a mark is added, so never empty.
fn punct(set: &[Parse]) -> Option<&[Parse]> {
    (!set.is_empty()).then_some(set)
}

/// `new Cons(cons(p, i), consbo(p, i), i, true)` with `useLabel` false.
fn cons_slot<'a>(p: Option<&'a Parse>, i: i32, sink: &impl FeatureSink) -> ConsRef<'a> {
    ConsRef::new(p, i, false, true, sink)
}

/// Formats a production without cloning subtrees or collecting child references.
fn production<'a>(p: &'a Parse, include_punctuation: bool, punct_set: &'a HashSet<String>) -> impl fmt::Display + 'a {
    DisplayFn(move |f: &mut fmt::Formatter<'_>| {
        write!(f, "{}->", p.kind())?;
        let mut children = p.children().iter().enumerate().filter(|(_, c)| !punct_set.contains(c.kind())).peekable();
        while let Some((ci, child)) = children.next() {
            f.write_str(child.kind())?;
            if let Some(&(next_index, _)) = children.peek() {
                f.write_str(",")?;
                if include_punctuation {
                    // TreeSet ordering and equality use probability, with the existing set's
                    // entries inserted first. Select each next entry without allocating a union.
                    let added = &p.children()[ci + 1..next_index];
                    let candidates = || child.next_punctuation().iter().chain(added);
                    let mut previous: Option<&Parse> = None;
                    loop {
                        let next = candidates()
                            .filter(|c| previous.is_none_or(|prev| prev.cmp_prob(c).is_lt()))
                            .min_by(|a, b| a.cmp_prob(b));
                        let Some(mark) = next else {
                            break;
                        };
                        write!(f, "{},", mark.kind())?;
                        previous = Some(mark);
                    }
                }
            }
        }
        Ok(())
    })
}

/// The two deepest frontier nodes with distinct heads. Walking down keeps only the
/// three most recent head runs; a deepest head of -1 is skipped just as on the JVM.
/// A trim selects the ancestors above the deepest matching parent.
fn frontier_nodes<'a>(
    root: &'a Parse,
    punct_set: &HashSet<String>,
    trim: Option<&Parse>,
) -> Result<([Option<&'a Parse>; 2], bool), TreeInsertError> {
    let mut node = if root.kind() == TOP_NODE || root.kind() == INC_NODE {
        root.children().iter().find(|c| !punct_set.contains(c.kind()))
    } else {
        Some(root)
    };
    let mut recent: [Option<&Parse>; 3] = [None; 3];
    let mut trimmed = None;
    let mut any = false;
    while let Some(current) = node.filter(|n| !n.is_pos_tag()) {
        any = true;
        if trim.is_some_and(|parent| current == parent) {
            trimmed = Some(recent);
        }
        if recent[0].is_none_or(|n| n.head_index() != current.head_index()) {
            recent[2] = recent[1];
            recent[1] = recent[0];
        }
        recent[0] = Some(current);
        node = current.children().last();
    }
    if trim.is_some() {
        recent = trimmed.ok_or(TreeInsertError::ParentNotInFrontier)?;
    }
    let offset = usize::from(recent[0].is_some_and(|n| n.head_index() == -1));
    Ok(([recent[offset], recent[offset + 1]], any))
}

/// `AbstractBottomUpParser.collapsePunctuation` run for its effect on the children of `node`:
/// each child gets the punctuation next to it added to its punctuation sets.
pub fn collapse_children(node: &mut Parse, punct_set: &HashSet<String>) {
    collapse_punctuation_in_place(node.children_mut(), punct_set);
}

/// Paths (child indices from `root`) of the right frontier of `root`, deepest first
/// (`treeinsert.Parser.getRightFrontier`). A `TOP` or `INC` root starts from its first child that
/// is not punctuation; any other root starts from itself. The walk follows the last child,
/// punctuation or not, and stops at a part-of-speech node or at a node without children.
pub fn right_frontier_paths(root: &Parse, punct_set: &HashSet<String>) -> Vec<Vec<usize>> {
    let mut path = Vec::new();
    let mut top = root;
    if root.kind() == TOP_NODE || root.kind() == INC_NODE {
        let Some(first) = root.children().iter().position(|c| !punct_set.contains(c.kind())) else {
            return Vec::new();
        };
        path.push(first);
        top = &root.children()[first];
    }
    let mut rf = Vec::new();
    while !top.is_pos_tag() {
        rf.push(path.clone());
        let Some(last) = top.children().len().checked_sub(1) else {
            break;
        };
        path.push(last);
        top = &top.children()[last];
    }
    rf.reverse();
    rf
}

/// An outcome index into `probs`, failing as OpenNLP's read of the index `-1` that `getIndex`
/// gives for a missing outcome does.
fn outcome_index(index: Option<usize>, probs: &[f64]) -> Result<usize, ParseError> {
    index.ok_or(ParseError::TokenIndex { index: -1, len: probs.len() })
}

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
    /// ([`TreeInsertError::ParentNotInFrontier`] otherwise).
    ///
    /// # Panics
    ///
    /// When `index` is out of bounds or `parent` has no children that are not punctuation, where
    /// OpenNLP throws `ArrayIndexOutOfBoundsException`.
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
        let first = children.next().expect("parent has a non-punctuation child");
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

/// Tree-insertion constituency parser (`opennlp.tools.parser.treeinsert.Parser`).
///
/// OpenNLP's `checkComplete` switch is a package-private static that is always `false`, so the
/// check model is never consulted while parsing and its branches are not ported.
#[derive(Debug)]
pub struct Parser {
    model: Arc<ParserModel>,
    beam_size: usize,
    advance_percentage: f64,
    build_cg: BuildContextGenerator,
    attach_cg: AttachContextGenerator,
    /// Outcome indices, `None` where OpenNLP's `getIndex` gives `-1`; reading one fails as
    /// OpenNLP's read of index `-1` does.
    done_index: Option<usize>,
    sister_attach_index: Option<usize>,
    daughter_attach_index: Option<usize>,
}

impl Parser {
    /// Creates a tree-insertion [`Parser`] with default beam size (`20`) and advance percentage (`0.95`).
    ///
    /// # Panics
    ///
    /// As [`Parser::with_params`].
    pub fn new(model: impl Into<Arc<ParserModel>>) -> Self {
        Self::with_params(model, DEFAULT_BEAM_SIZE, DEFAULT_ADVANCE_PERCENTAGE)
    }

    /// Creates a tree-insertion [`Parser`] with explicit `beam_size` and `advance_percentage`.
    ///
    /// # Panics
    ///
    /// When [`Parser::try_with_params`] fails.
    pub fn with_params(model: impl Into<Arc<ParserModel>>, beam_size: usize, advance_percentage: f64) -> Self {
        Self::try_with_params(model, beam_size, advance_percentage).expect("TREEINSERT parser model")
    }

    /// Creates a tree-insertion [`Parser`], failing when the model has no `attach.model`.
    pub fn try_with_params(
        model: impl Into<Arc<ParserModel>>,
        beam_size: usize,
        advance_percentage: f64,
    ) -> Result<Self, TreeInsertError> {
        let model = model.into();
        let attach_model = model.attach_model().ok_or(TreeInsertError::MissingAttachModel)?;
        let done_index = model.build_model().outcome_index(DONE);
        let sister_attach_index = attach_model.outcome_index(ATTACH_SISTER);
        let daughter_attach_index = attach_model.outcome_index(ATTACH_DAUGHTER);
        let punct_set = model.head_rules().punctuation_tags().clone();
        Ok(Self {
            model,
            beam_size,
            advance_percentage,
            build_cg: BuildContextGenerator::new(),
            attach_cg: AttachContextGenerator::new(punct_set),
            done_index,
            sister_attach_index,
            daughter_attach_index,
        })
    }

    /// Returns the underlying [`ParserModel`].
    pub fn model(&self) -> &ParserModel {
        &self.model
    }

    /// Returns the right frontier of `root`, deepest node first (`treeinsert.Parser.getRightFrontier`).
    ///
    /// A `TOP` or `INC` root starts from its first child that is not in `punct_set`, any other
    /// root from itself; the walk follows last children down to the first part-of-speech node,
    /// which is left out.
    pub fn get_right_frontier<'a>(root: &'a Parse, punct_set: &HashSet<String>) -> Vec<&'a Parse> {
        right_frontier_paths(root, punct_set).iter().map(|path| root.node_at_path(path)).collect()
    }

    fn is_built(p: &Parse) -> bool {
        p.label().is_some_and(|l| l.starts_with(BUILT))
    }

    fn is_complete(p: &Parse) -> bool {
        p.label().is_some_and(|l| l.ends_with(COMPLETE))
    }

    /// `treeinsert.Parser.setComplete`.
    fn set_complete(p: &mut Parse) {
        let label = if Self::is_built(p) { format!("{BUILT}.{COMPLETE}") } else { COMPLETE.to_owned() };
        p.set_label(Some(label));
    }

    /// Marks every top-level child of each parse complete, as `treeinsert.Parser.advanceChunks`
    /// does after the chunker has run.
    pub fn mark_chunks_complete(parses: &mut [Parse]) {
        for parse in parses {
            for chunk in parse.children_mut() {
                Self::set_complete(chunk);
            }
        }
    }

    /// Clears the punctuation sets of `p`, as a node copied by `Parse.clone` has none.
    fn clear_punctuation(p: &mut Parse) {
        p.set_previous_punctuation(Vec::new());
        p.set_next_punctuation(Vec::new());
    }

    /// Advances `p` one build or attach step (`treeinsert.Parser.advanceParses`).
    ///
    /// Returns `None` where OpenNLP returns `null`: when `p` has no constituent that is not
    /// punctuation, or a single part-of-speech node. The caller then completes `p` as it is. A
    /// single other constituent takes the punctuation around it as children and `p` is returned.
    /// Otherwise the derivations are returned in OpenNLP's order: one per build outcome taken,
    /// then one per daughter or sister attachment taken, frontier node by frontier node.
    ///
    /// Fails where OpenNLP throws: when the head rules give no head for a node an attachment
    /// changes, or when every top-level constituent is already built, where OpenNLP reads past
    /// the end of the constituents.
    pub fn advance_parses(&self, p: &mut Parse, prob_mass: f64) -> Result<Option<Vec<Parse>>, ParseError> {
        self.advance_parses_with_features(p, prob_mass, &mut Features::new())
    }

    fn advance_parses_with_features(
        &self,
        p: &mut Parse,
        prob_mass: f64,
        features: &mut Features,
    ) -> Result<Option<Vec<Parse>>, ParseError> {
        let q = 1.0 - prob_mass;
        let rules = self.model.head_rules();
        let punct_set = rules.punctuation_tags();
        let build_model = self.model.build_model();
        let attach_model = self.model.attach_model().expect("checked by Parser::try_with_params");

        let kept = collapse_punctuation_in_place(p.children_mut(), punct_set);
        let num_nodes = kept.len();
        if num_nodes == 0 {
            return Ok(None);
        }
        if num_nodes == 1 {
            if p.children()[kept[0]].is_pos_tag() {
                return Ok(None);
            }
            p.expand_top_node(kept[0])?;
            return Ok(Some(vec![p.clone()]));
        }
        let advance_node_index = (0..num_nodes)
            .find(|&i| !Self::is_built(&p.children()[kept[i]]))
            .ok_or(ParseError::TokenIndex { index: num_nodes as i64, len: num_nodes })?;
        let original_zero_index = kept[0];
        let original_advance_index = kept[advance_node_index];

        let mut bprobs = vec![0.0; build_model.num_outcomes()];
        let mut rf: Vec<Vec<usize>> = Vec::new();
        let mut aprobs: Vec<Vec<f64>> = Vec::new();
        {
            let children: Vec<&Parse> = kept.iter().map(|&i| &p.children()[i]).collect();
            self.build_cg.get_features(&children, advance_node_index, features);
            build_model.eval_features(features.iter(), &mut bprobs);
        }
        let done_prob = bprobs[outcome_index(self.done_index, &bprobs)?];

        if done_prob > q && advance_node_index != 0 {
            rf = right_frontier_paths(p, punct_set);
            {
                let children: Vec<&Parse> = kept.iter().map(|&i| &p.children()[i]).collect();
                let rf_nodes: Vec<&Parse> = rf.iter().map(|path| p.node_at_path(path)).collect();
                for fi in 0..rf.len() {
                    self.attach_cg.get_features(&children, advance_node_index, &rf_nodes, fi, features);
                    let mut probs = vec![0.0; attach_model.num_outcomes()];
                    attach_model.eval_features(features.iter(), &mut probs);
                    aprobs.push(probs);
                }
            }
            // `production(fn, false)` collapses the children of every frontier node.
            for path in &rf {
                collapse_children(p.node_at_path_mut(path), punct_set);
            }
        }

        let advance_node = p.children()[original_advance_index].clone();
        let mut new_parses = Vec::new();

        if 1.0 - done_prob > q {
            let mut bprob_sum = 0.0;
            while bprob_sum < prob_mass {
                let mut max = 0;
                for pi in 1..bprobs.len() {
                    if bprobs[pi] > bprobs[max] {
                        max = pi;
                    }
                }
                if bprobs[max] == 0.0 {
                    break;
                }
                let bprob = bprobs[max];
                bprobs[max] = 0.0;
                bprob_sum += bprob;
                let tag = build_model.outcome(max).to_str().unwrap_or_default();
                if tag != DONE {
                    let mut new_parse = p.clone();
                    let new_node = Parse::with_head(
                        Arc::clone(p.text_arc()),
                        advance_node.span(),
                        tag.as_ref(),
                        bprob,
                        Some(advance_node.head()),
                    );
                    new_parse.insert(new_node)?;
                    new_parse.add_prob(strict_math::log(bprob));
                    new_parses.push(new_parse);
                }
            }
        }

        if done_prob > q {
            let prepare_done = || {
                let mut branch = p.clone();
                let built = &mut branch.children_mut()[original_advance_index];
                built.set_label(Some(BUILT.to_owned()));
                Self::clear_punctuation(built);
                branch.add_prob(strict_math::log(done_prob));
                branch
            };
            if advance_node_index == 0 {
                new_parses.push(prepare_done());
            } else {
                for (fi, probs) in aprobs.iter().enumerate() {
                    let daughter = outcome_index(self.daughter_attach_index, probs)?;
                    let sister = outcome_index(self.sister_attach_index, probs)?;
                    let fn_complete = Self::is_complete(p.node_at_path(&rf[fi]));
                    for attachment in [daughter, sister] {
                        let prob = probs[attachment];
                        if !(prob > q && (attachment != daughter || !fn_complete)) {
                            continue;
                        }
                        // `cloneRoot(fn, originalZeroIndex)` copies the nodes from the top down to fn.
                        let mut new_parse = prepare_done();
                        for path in &rf[fi..] {
                            Self::clear_punctuation(new_parse.node_at_path_mut(path));
                        }
                        collapse_punctuation_in_place(new_parse.children_mut(), punct_set);
                        for _ in original_zero_index + 1..=original_advance_index {
                            new_parse.remove(original_zero_index + 1)?;
                        }
                        collapse_punctuation_in_place(new_parse.children_mut(), punct_set);
                        let crf = right_frontier_paths(&new_parse, punct_set);
                        if attachment == daughter {
                            new_parse.node_at_path_mut(&crf[fi]).add(advance_node.clone(), rules)?;
                        } else if fi + 1 < crf.len() {
                            new_parse.node_at_path_mut(&crf[fi + 1]).adjoin(advance_node.clone(), rules)?;
                        } else {
                            new_parse.adjoin_root(advance_node.clone(), rules, original_zero_index)?;
                        }
                        for path in &crf[fi + 1..] {
                            new_parse.node_at_path_mut(path).update_span();
                        }
                        new_parse.add_prob(strict_math::log(prob));
                        new_parses.push(new_parse);
                    }
                }
            }
        }
        Ok(Some(new_parses))
    }
}

impl ParserTrait for Parser {
    fn parse_n(&self, tokens: &Parse, num_parses: usize) -> Result<Vec<Parse>, ParseError> {
        let mut features = Features::new();
        parse_bottom_up(
            tokens,
            num_parses,
            self.beam_size,
            self.beam_size,
            self.advance_percentage,
            self.model.parser_tagger_model(),
            self.model.parser_chunker_model(),
            self.model.head_rules(),
            |p, q| self.advance_parses_with_features(p, q, &mut features),
            |p| p.set_kind(TOP_NODE),
            Self::mark_chunks_complete,
        )
    }
}

#[cfg(test)]
mod feature_tests {
    use super::*;

    #[test]
    fn compact_frontier_matches_path_walk_for_repeated_and_missing_heads() {
        let punct_set = HashSet::new();
        for depth in 1..=6 {
            for mut pattern in 0..3usize.pow(depth) {
                let mut root = None;
                for level in 0..depth {
                    let mut node = Parse::new("x", Span::new(0, 1), format!("N{level}"), 1.0, (pattern % 3) as i32 - 1);
                    pattern /= 3;
                    if let Some(child) = root {
                        node.insert(child).unwrap();
                    }
                    root = Some(node);
                }
                let root = root.unwrap();
                let paths = right_frontier_paths(&root, &punct_set);
                let nodes: Vec<&Parse> = paths.iter().map(|path| root.node_at_path(path)).collect();
                for trim_at in 0..=nodes.len() {
                    let trim = nodes.get(trim_at).copied();
                    let retained = if trim.is_some() { &nodes[trim_at + 1..] } else { &nodes[..] };
                    let mut previous = -1;
                    let expected: Vec<&str> = retained
                        .iter()
                        .filter_map(|node| {
                            if node.head_index() == previous {
                                None
                            } else {
                                previous = node.head_index();
                                Some(node.kind())
                            }
                        })
                        .take(2)
                        .collect();
                    let (actual, any) = frontier_nodes(&root, &punct_set, trim).unwrap();
                    assert!(any);
                    assert_eq!(actual.into_iter().flatten().map(Parse::kind).collect::<Vec<_>>(), expected);
                }
            }
        }
    }

    #[test]
    fn production_keeps_existing_punctuation_on_equal_probability() {
        let tree = Parse::parse("(TOP (S (NP (NN cat)) (, ,) (: :) (VP (VB runs))))").unwrap();
        let mut parent = tree.children()[0].clone();
        let existing = Parse::new(parent.text_arc().clone(), Span::new(0, 0), "EXISTING", 1.0, 0);
        parent.children_mut()[0].add_next_punctuation(existing);
        let punctuation = [",".to_owned(), ":".to_owned()].into();
        assert_eq!(production(&parent, true, &punctuation).to_string(), "S->NP,EXISTING,VP");
        assert_eq!(production(&parent, false, &punctuation).to_string(), "S->NP,VP");
    }
}
