//! Bottom-up tree-insertion constituency parser, context generators, event stream, and trainer
//! (`opennlp.tools.parser.treeinsert.*`).
//!
//! A model trained by OpenNLP must decode here as it does on the JVM, so the feature strings, the
//! order of the derivations [`Parser::advance_parses`] returns, and the training events
//! [`ParserEventStream`] emits follow `opennlp.tools.parser.treeinsert` exactly.
//!
//! OpenNLP shares nodes between derivations and changes their punctuation sets as a side effect of
//! `collapsePunctuation`, which its context generators call on the children of the nodes they
//! look at. A node copied by `Parse.clone` starts with no punctuation sets. Both effects change
//! later features, so the parser and the event stream repeat them on their own trees: the copies
//! along the path a derivation changes lose their punctuation sets, and the children of the nodes
//! a context generator collapses get the punctuation next to them added to their sets. Sharing
//! between sibling derivations in the beam is not repeated; a node one derivation changes this way
//! is seen changed by the other derivations that share it in OpenNLP and not here.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use turbo_analysis::Span;
use turbo_formats::{Field, PosSample};
use turbo_jvm::strict_math;
use turbo_train::{Event, TrainingConfiguration};
use turbo_train_components::package::Stamp;
use turbo_train_components::postag::{PosTaggerFactory, train_pos_model};

use crate::bottom_up::{
    DEFAULT_ADVANCE_PERCENTAGE, DEFAULT_BEAM_SIZE, Parser as ParserTrait, build_dictionary,
    collapse_punctuation_in_place, parse_bottom_up,
};
use crate::chunking::{
    ParserTrainingParameters, TrainingReports, collect_pos_samples, event_from_strings, train_parser_chunker_bytes,
};
use crate::context::{ChunkContextGenerator, Cons, EOS, check_cons, cons, cons_bo, cons2, cons3, surround};
use crate::head_rules::HeadRules;
use crate::model::{HeadRulesArtifact, ParserEventType, ParserModel, ParserModelError, ParserType};
use crate::parse::{COMPLETE, INC_NODE, INCOMPLETE, Parse, ParseError, TOP_NODE, insert_by_prob};
use crate::stream::{ParserChunkSample, ParserPosSample};

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
fn cons_slot(p: Option<&Parse>, i: i32) -> Cons {
    Cons::new(cons(p, i), cons_bo(p, i), i, true)
}

/// The non-punctuation children of a node, each with the next-punctuation set it has after
/// `collapsePunctuation` runs over them: its own set plus the punctuation that follows it.
fn collapsed_children<'a>(children: &'a [Parse], punct_set: &HashSet<String>) -> Vec<(&'a Parse, Vec<Parse>)> {
    let mut out: Vec<(&Parse, Vec<Parse>)> = Vec::with_capacity(children.len());
    for c in children {
        if punct_set.contains(c.kind()) {
            if let Some((_, next)) = out.last_mut() {
                insert_by_prob(next, c.clone());
            }
        } else {
            out.push((c, c.next_punctuation().to_vec()));
        }
    }
    out
}

/// `AbstractContextGenerator.production`, as [`crate::context::production_in`] computes it but
/// without copying the children, since the attach features need it for every frontier node.
fn production(p: &Parse, include_punctuation: bool, punct_set: &HashSet<String>) -> String {
    let mut prod = format!("{}->", p.kind());
    if include_punctuation {
        let children = collapsed_children(p.children(), punct_set);
        for (ci, (child, next)) in children.iter().enumerate() {
            prod.push_str(child.kind());
            if ci + 1 != children.len() {
                prod.push(',');
                for punct in next {
                    prod.push_str(punct.kind());
                    prod.push(',');
                }
            }
        }
    } else {
        let mut first = true;
        for child in p.children().iter().filter(|c| !punct_set.contains(c.kind())) {
            if !first {
                prod.push(',');
            }
            prod.push_str(child.kind());
            first = false;
        }
    }
    prod
}

/// `AbstractContextGenerator.getFrontierNodes` for two nodes: the first frontier nodes with
/// different head indices.
fn frontier_nodes<'a>(rf: &[&'a Parse]) -> [Option<&'a Parse>; 2] {
    let mut nodes = [None, None];
    let mut left = 0;
    let mut prev_head_index = -1;
    for fnode in rf {
        let head_index = fnode.head_index();
        if head_index != prev_head_index {
            nodes[left] = Some(*fnode);
            left += 1;
            prev_head_index = head_index;
            if left == nodes.len() {
                break;
            }
        }
    }
    nodes
}

/// `AbstractBottomUpParser.collapsePunctuation` run for its effect on the children of `node`:
/// each child gets the punctuation next to it added to its punctuation sets.
fn collapse_children(node: &mut Parse, punct_set: &HashSet<String>) {
    collapse_punctuation_in_place(node.children_mut(), punct_set);
}

/// Paths (child indices from `root`) of the right frontier of `root`, deepest first
/// (`treeinsert.Parser.getRightFrontier`). A `TOP` or `INC` root starts from its first child that
/// is not punctuation; any other root starts from itself. The walk follows the last child,
/// punctuation or not, and stops at a part-of-speech node or at a node without children.
fn right_frontier_paths(root: &Parse, punct_set: &HashSet<String>) -> Vec<Vec<usize>> {
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

fn path_parent(path: &[usize]) -> Option<&[usize]> {
    path.split_last().map(|(_, parent)| parent)
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
        let p0 = constituents[index];
        let p1 = constituents.get(index + 1).copied();
        let p2 = constituents.get(index + 2).copied();

        let punct_1s = punct(p0.previous_punctuation());
        let punct1s = punct(p0.next_punctuation());
        let punct2s = p1.and_then(|p| punct(p.next_punctuation()));

        let empty = HashSet::new();
        let rf_paths = if index == 0 { Vec::new() } else { right_frontier_paths(constituents[0], &empty) };
        let rf: Vec<&Parse> = rf_paths.iter().map(|path| constituents[0].node_at_path(path)).collect();
        let [p_1, p_2] = frontier_nodes(&rf);
        let punct_2s = p_1.and_then(|p| punct(p.previous_punctuation()));

        let c_2 = cons_slot(p_2, -2);
        let c_1 = cons_slot(p_1, -1);
        let c0 = cons_slot(Some(p0), 0);
        let c1 = cons_slot(p1, 1);
        let c2 = cons_slot(p2, 2);

        let mut features = Vec::with_capacity(64);
        features.push("default".to_owned());
        for c in [&c_2, &c_1, &c0, &c1, &c2] {
            features.push(c.cons.clone());
            features.push(c.cons_bo.clone());
        }

        cons2(&mut features, &c0, &c1, punct1s, true);
        cons2(&mut features, &c_1, &c0, punct_1s, true);
        cons3(&mut features, &c0, &c1, &c2, punct1s, punct2s, true, true, true);
        cons3(&mut features, &c_2, &c_1, &c0, punct_2s, punct_1s, true, true, true);
        cons3(&mut features, &c_1, &c0, &c1, punct_1s, punct_1s, true, true, true);

        if rf.is_empty() {
            features.push(format!("{EOS},{}", c0.cons));
            features.push(format!("{EOS},{}", c0.cons_bo));
        }
        features
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
        let fnode = right_frontier[rfi];
        let fp = right_frontier.get(rfi + 1).copied();
        let p_1 = right_frontier.first().copied();
        let p0 = constituents[index];
        let p1 = constituents.get(index + 1).copied();

        let punct_1fs = punct(fnode.previous_punctuation());
        let punct_1s = punct(p0.previous_punctuation());
        let punct1s = punct(p0.next_punctuation());

        let cfp = cons_slot(fp, -3);
        let cf = cons_slot(Some(fnode), -2);
        let c_1 = cons_slot(p_1, -1);
        let c0 = cons_slot(Some(p0), 0);
        let c1 = cons_slot(p1, 1);

        let mut features = Vec::with_capacity(100);
        features.push("default".to_owned());
        for c in [&cfp, &cf, &c_1, &c0, &c1] {
            features.push(c.cons.clone());
            features.push(c.cons_bo.clone());
        }

        let prod = production(fnode, false, &self.punct_set);
        features.push(format!("pn={prod}"));
        features.push(format!("pd={prod},{}", p0.kind()));
        features.push(format!("ps={}->{},{}", fnode.kind(), fnode.kind(), p0.kind()));

        cons2(&mut features, &cfp, &c0, punct_1s, true);
        cons2(&mut features, &cf, &c0, punct_1s, true);
        cons2(&mut features, &c_1, &c0, punct_1s, true);
        cons2(&mut features, &c0, &c1, punct1s, true);
        cons3(&mut features, &cf, &c_1, &c0, None, punct_1s, true, true, true);
        cons3(&mut features, &cf, &c0, &c1, punct_1s, punct1s, true, true, true);
        cons3(&mut features, &cfp, &cf, &c0, None, punct_1s, true, true, true);

        let head_distance = p0.head_index() - fnode.head_index();
        features.push(format!("hd={head_distance}"));
        features.push(format!("nd={rfi}"));
        features.push(format!("nd={}.{rfi}", p0.kind()));
        features.push(format!("hd={}.{head_distance}", p0.kind()));

        let contains = |set: Option<&[Parse]>, mark: &str| set.is_some_and(|s| s.iter().any(|p| p.kind() == mark));
        if contains(punct_1s, "''") && contains(punct_1fs, "``") {
            features.push("quotematch".to_owned());
        }
        features
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
        features.push("default".to_owned());
        let children = collapsed_children(parent.children(), &self.punct_set);
        let kind = parent.kind();
        check_cons(children[0].0, "begin", kind, &mut features);
        check_cons(children[children.len() - 1].0, "last", kind, &mut features);
        features.push(format!("p={}", production(parent, false, &self.punct_set)));
        features.push(format!("pp={}", production(parent, true, &self.punct_set)));

        let p1s = punct(constituents[index].next_punctuation());
        let p_1s = punct(constituents[index].previous_punctuation());

        let mut rf: Vec<&Parse> = Vec::new();
        if index != 0 {
            let root = constituents[0];
            rf = right_frontier_paths(root, &self.punct_set).iter().map(|path| root.node_at_path(path)).collect();
            if trim_frontier {
                let pi = rf.iter().position(|n| *n == parent).ok_or(TreeInsertError::ParentNotInFrontier)?;
                rf.drain(..=pi);
            }
        }
        let [p_1, p_2] = frontier_nodes(&rf);
        let p_2s = p_1.and_then(|p| punct(p.previous_punctuation()));
        let p1 = constituents.get(index + 1).copied();
        let p2s = p1.and_then(|p| punct(p.next_punctuation()));
        let p2 = constituents.get(index + 2).copied();

        surround(p_1, -1, kind, p_1s, &mut features);
        surround(p_2, -2, kind, p_2s, &mut features);
        surround(p1, 1, kind, p1s, &mut features);
        surround(p2, 2, kind, p2s, &mut features);
        Ok(features)
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
            p.expand_top_node(kept[0]);
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
            let ctx = self.build_cg.get_context(&children, advance_node_index);
            build_model.eval(&ctx.iter().map(String::as_str).collect::<Vec<_>>(), &mut bprobs);
        }
        let done_prob = bprobs[outcome_index(self.done_index, &bprobs)?];

        if done_prob > q && advance_node_index != 0 {
            rf = right_frontier_paths(p, punct_set);
            {
                let children: Vec<&Parse> = kept.iter().map(|&i| &p.children()[i]).collect();
                let rf_nodes: Vec<&Parse> = rf.iter().map(|path| p.node_at_path(path)).collect();
                for fi in 0..rf.len() {
                    let ctx = self.attach_cg.get_context(&children, advance_node_index, &rf_nodes, fi);
                    let mut probs = vec![0.0; attach_model.num_outcomes()];
                    attach_model.eval(&ctx.iter().map(String::as_str).collect::<Vec<_>>(), &mut probs);
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
            let mut done_parse = p.clone();
            let built = &mut done_parse.children_mut()[original_advance_index];
            built.set_label(Some(BUILT.to_owned()));
            Self::clear_punctuation(built);
            done_parse.add_prob(strict_math::log(done_prob));
            if advance_node_index == 0 {
                new_parses.push(done_parse);
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
                        let mut new_parse = done_parse.clone();
                        for path in &rf[fi..] {
                            Self::clear_punctuation(new_parse.node_at_path_mut(path));
                        }
                        collapse_punctuation_in_place(new_parse.children_mut(), punct_set);
                        for _ in original_zero_index + 1..=original_advance_index {
                            new_parse.remove(original_zero_index + 1);
                        }
                        collapse_punctuation_in_place(new_parse.children_mut(), punct_set);
                        let crf = right_frontier_paths(&new_parse, punct_set);
                        if attachment == daughter {
                            new_parse.node_at_path_mut(&crf[fi]).add(advance_node.clone(), rules)?;
                        } else if fi + 1 < crf.len() {
                            new_parse.node_at_path_mut(&crf[fi + 1]).adjoin(advance_node.clone(), rules)?;
                        } else {
                            new_parse.adjoin_root(advance_node.clone(), rules, original_zero_index);
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

    /// Trains a `TREEINSERT` [`ParserModel`] from `samples` (`treeinsert.Parser.train`).
    pub fn train(
        language: &str,
        samples: &[Parse],
        rules: HeadRulesArtifact,
        params: &ParserTrainingParameters,
    ) -> Result<ParserModel, ParserModelError> {
        let dict = build_dictionary(samples, &rules, params.dict_cutoff);

        let mut report = TrainingReports::default();

        // 1. Train build model
        let build_events = ParserEventStream::generate_events(samples, &rules, ParserEventType::Build, Some(&dict))?;
        let build_bytes = report.train("build", &build_events, &params.build)?;

        // 2. Train check model
        let check_events = ParserEventStream::generate_events(samples, &rules, ParserEventType::Check, None)?;
        let check_bytes = report.train("check", &check_events, &params.check)?;

        // 3. Train attach model
        let attach_events = ParserEventStream::generate_events(samples, &rules, ParserEventType::Attach, None)?;
        let attach_bytes = report.train("attach", &attach_events, &params.attach)?;

        // 4. Train POS tagger sub-model. Unlike the chunking parser, OpenNLP's tree-insertion
        // `train` leaves the tagger's beam size alone, so it is the tagger's default of 3 unless the
        // parameters give one.
        let pos_samples = collect_pos_samples(samples);
        let pos_factory = PosTaggerFactory::new(language, None, None)?;
        let trained_pos = train_pos_model(
            &pos_samples,
            &pos_factory,
            &params.tagger,
            &mut TrainingConfiguration::default(),
            Stamp::at(0),
        )?;
        let tagger_bytes = trained_pos.package.to_bytes();

        // 5. Train chunker sub-model with ParserChunkerFactory
        let chunker_bytes = train_parser_chunker_bytes(language, samples, &params.chunker)?;

        ParserModel::from_serialized_parts(
            language,
            build_bytes,
            check_bytes,
            Some(attach_bytes),
            tagger_bytes,
            chunker_bytes,
            rules,
            ParserType::TreeInsert,
            &report.manifest_entries(),
            Stamp::at(0),
        )
    }
}

impl ParserTrait for Parser {
    fn parse_n(&self, tokens: &Parse, num_parses: usize) -> Result<Vec<Parse>, ParseError> {
        parse_bottom_up(
            tokens,
            num_parses,
            self.beam_size,
            self.beam_size,
            self.advance_percentage,
            self.model.parser_tagger_model(),
            self.model.parser_chunker_model(),
            self.model.head_rules(),
            |p, q| self.advance_parses(p, q),
            |p| p.set_kind(TOP_NODE),
            Self::mark_chunks_complete,
        )
    }
}

/// Event stream for the tree-insertion constituency parser (`opennlp.tools.parser.treeinsert.ParserEventStream`).
pub struct ParserEventStream;

/// The context generators of one event stream.
struct Generators {
    build: BuildContextGenerator,
    attach: AttachContextGenerator,
    check: CheckContextGenerator,
}

impl ParserEventStream {
    /// Generates training [`Event`]s of type `etype` over `samples` (`AbstractParserEventStream`
    /// with `treeinsert.ParserEventStream.addParseEvents`). The dictionary is not used by the
    /// tree-insertion build features.
    pub fn generate_events(
        samples: &[Parse],
        rules: &dyn HeadRules,
        etype: ParserEventType,
        _dict: Option<&HashSet<String>>,
    ) -> Result<Vec<Event>, TreeInsertError> {
        let mut events = Vec::new();
        let punct_set = rules.punctuation_tags();
        let cgs = Generators {
            build: BuildContextGenerator::new(),
            attach: AttachContextGenerator::new(punct_set.clone()),
            check: CheckContextGenerator::new(punct_set.clone()),
        };
        let mut chunk_cg = ChunkContextGenerator::new();

        for orig in samples {
            let mut p = orig.clone();
            Parse::prune_parse(&mut p)?;
            p.update_heads(rules);

            match etype {
                ParserEventType::Tag => {
                    let pos = ParserPosSample::from_parse(&p);
                    if let Ok(factory) = PosTaggerFactory::new("en", None, None)
                        && let Ok(mut stream) = turbo_train_components::postag::PosEventStream::new(&factory)
                    {
                        let toks: Vec<Field<'_>> =
                            pos.sentence.iter().map(|s| Field::of(s.as_str(), Span::new(0, 0))).collect();
                        let tags: Vec<Field<'_>> =
                            pos.tags.iter().map(|s| Field::of(s.as_str(), Span::new(0, 0))).collect();
                        if let Ok(sample) = PosSample::new(toks, tags, Span::new(0, 0)) {
                            stream.events(&sample, &mut events);
                        }
                    }
                }
                ParserEventType::Chunk => {
                    let cs = ParserChunkSample::from_parse(&p);
                    let words: Vec<&str> = cs.sentence.iter().map(String::as_str).collect();
                    let tags: Vec<&str> = cs.tags.iter().map(String::as_str).collect();
                    let preds: Vec<&str> = cs.preds.iter().map(String::as_str).collect();
                    for i in 0..words.len() {
                        let ctx = chunk_cg.get_context(i, &words, &tags, &preds);
                        events.push(event_from_strings(preds[i], &ctx));
                    }
                }
                ParserEventType::Build | ParserEventType::Check | ParserEventType::Attach => {
                    Self::add_parse_events(&mut events, &mut p, rules, etype, &cgs)?;
                }
            }
        }
        Ok(events)
    }

    /// `treeinsert.ParserEventStream.addParseEvents` over the gold parse `gold`.
    ///
    /// Gold nodes are named by their paths in `gold`, whose shape never changes here. The parse
    /// built so far is `current[0]`, with the chunks still to attach after it; its nodes are named
    /// by their paths in `current[0]`.
    fn add_parse_events(
        events: &mut Vec<Event>,
        gold: &mut Parse,
        rules: &dyn HeadRules,
        etype: ParserEventType,
        cgs: &Generators,
    ) -> Result<(), TreeInsertError> {
        let punct_set = rules.punctuation_tags();
        let init_paths = gold.initial_chunk_paths();
        let mut init_nodes: Vec<Parse> = init_paths.iter().map(|path| gold.node_at_path(path).clone()).collect();
        let kept = collapse_punctuation_in_place(&mut init_nodes, punct_set);

        let mut chunks: Vec<Vec<usize>> = Vec::with_capacity(kept.len());
        let mut current: Vec<Parse> = Vec::with_capacity(kept.len());
        for idx in kept {
            let mut node = init_nodes[idx].clone();
            node.set_label(Some(COMPLETE.to_owned()));
            gold.node_at_path_mut(&init_paths[idx]).set_label(Some(COMPLETE.to_owned()));
            chunks.push(init_paths[idx].clone());
            current.push(node);
        }

        let kind_at = |gold: &Parse, path: &[usize]| gold.node_at_path(path).kind().to_owned();
        let mut right_frontier: Vec<Vec<usize>> = Vec::new();
        let mut built_nodes: Vec<Vec<usize>> = Vec::new();

        for ci in 0..chunks.len() {
            let mut parent = path_parent(&chunks[ci]).ok_or(TreeInsertError::MissingParent)?.to_vec();
            let mut prev_parent = chunks[ci].clone();
            let mut off = 0;
            if !gold.node_at_path(&chunks[ci]).is_pos_tag() {
                built_nodes.insert(off, chunks[ci].clone());
                off += 1;
            }

            // Build stages
            loop {
                let parent_node = gold.node_at_path(&parent);
                if parent_node.kind() == TOP_NODE || parent_node.label().is_some() {
                    break;
                }
                let parent_type = parent_node.kind().to_owned();
                if kind_at(gold, &prev_parent) != parent_type {
                    if etype == ParserEventType::Build {
                        let ctx = cgs.build.get_context(&current.iter().collect::<Vec<_>>(), ci);
                        events.push(event_from_strings(&parent_type, &ctx));
                    }
                    built_nodes.insert(off, parent.clone());
                    off += 1;

                    let cur = &current[ci];
                    let (prev_punct, next_punct) =
                        (cur.previous_punctuation().to_vec(), cur.next_punctuation().to_vec());
                    let mut new_parent = Parse::new(Arc::clone(cur.text_arc()), cur.span(), &parent_type, 1.0, 0);
                    new_parent.add(cur.clone(), rules)?;
                    new_parent.set_previous_punctuation(prev_punct);
                    new_parent.set_next_punctuation(next_punct);
                    new_parent.set_label(Some(BUILT.to_owned()));
                    current[ci] = new_parent;

                    let complete = Self::last_child(gold, &chunks[ci], &parent, punct_set);
                    if etype == ParserEventType::Check {
                        let refs: Vec<&Parse> = current.iter().collect();
                        let ctx = cgs.check.get_context(&current[ci], &refs, ci, false)?;
                        events.push(event_from_strings(if complete { COMPLETE } else { INCOMPLETE }, &ctx));
                        collapse_children(&mut current[ci], punct_set);
                    }
                    current[ci].set_label(Some(if complete { COMPLETE } else { INCOMPLETE }.to_owned()));
                    gold.node_at_path_mut(&parent).set_label(Some(COMPLETE.to_owned()));
                    chunks[ci] = parent.clone();
                }
                gold.node_at_path_mut(&parent).set_label(Some(BUILT.to_owned()));
                prev_parent = parent.clone();
                parent = path_parent(&parent).ok_or(TreeInsertError::MissingParent)?.to_vec();
            }

            if etype == ParserEventType::Build {
                let ctx = cgs.build.get_context(&current.iter().collect::<Vec<_>>(), ci);
                events.push(event_from_strings(DONE, &ctx));
            }

            if ci > 0 {
                Self::attach(events, gold, &mut current, &chunks, &mut right_frontier, ci, rules, etype, cgs)?;
            }

            built_nodes.append(&mut right_frontier);
            std::mem::swap(&mut right_frontier, &mut built_nodes);
        }
        Ok(())
    }

    /// The attachment part of `addParseEvents` for chunk `ci > 0`.
    #[allow(clippy::too_many_arguments)]
    fn attach(
        events: &mut Vec<Event>,
        gold: &mut Parse,
        current: &mut [Parse],
        chunks: &[Vec<usize>],
        right_frontier: &mut Vec<Vec<usize>>,
        ci: usize,
        rules: &dyn HeadRules,
        etype: ParserEventType,
        cgs: &Generators,
    ) -> Result<(), TreeInsertError> {
        let punct_set = rules.punctuation_tags();
        let crf = right_frontier_paths(&current[0], punct_set);
        if crf.len() != right_frontier.len() {
            return Err(TreeInsertError::FrontierMismatch { current: crf.len(), gold: right_frontier.len() });
        }
        let parents = Self::non_adjoined_parents(gold, &chunks[ci], punct_set)?;
        let mut attach: Option<(&'static str, usize)> = None;

        // Try daughters first.
        let mut collapsed = Vec::new();
        {
            let refs: Vec<&Parse> = current.iter().collect();
            let crf_nodes: Vec<&Parse> = crf.iter().map(|path| current[0].node_at_path(path)).collect();
            for (cfi, cfn) in crf_nodes.iter().enumerate() {
                let i = parents.get(&right_frontier[cfi]).copied();
                let non_punct = cfn.children().iter().filter(|k| !punct_set.contains(k.kind())).count();
                if attach.is_none() && i == Some(non_punct) {
                    attach = Some((ATTACH_DAUGHTER, cfi));
                    if etype == ParserEventType::Attach {
                        let ctx = cgs.attach.get_context(&refs, ci, &crf_nodes, cfi);
                        events.push(event_from_strings(ATTACH_DAUGHTER, &ctx));
                        collapsed.push(cfi);
                    }
                }
            }
        }
        for &cfi in &collapsed {
            collapse_children(current[0].node_at_path_mut(&crf[cfi]), punct_set);
        }

        // Try sisters, and generate non-attach events.
        collapsed.clear();
        {
            let refs: Vec<&Parse> = current.iter().collect();
            let crf_nodes: Vec<&Parse> = crf.iter().map(|path| current[0].node_at_path(path)).collect();
            for (cfi, frontier) in right_frontier.iter().enumerate() {
                let sister_site = path_parent(frontier).is_some_and(|fp| {
                    parents.contains_key(fp) && gold.node_at_path(frontier).kind() == gold.node_at_path(fp).kind()
                });
                let outcome = if attach.is_none() && sister_site {
                    attach = Some((ATTACH_SISTER, cfi));
                    Some(ATTACH_SISTER)
                } else if attach.is_some_and(|(_, idx)| idx == cfi) {
                    None
                } else {
                    Some(NON_ATTACH)
                };
                if let Some(outcome) = outcome
                    && etype == ParserEventType::Attach
                {
                    let ctx = cgs.attach.get_context(&refs, ci, &crf_nodes, cfi);
                    events.push(event_from_strings(outcome, &ctx));
                    collapsed.push(cfi);
                }
            }
        }
        for &cfi in &collapsed {
            collapse_children(current[0].node_at_path_mut(&crf[cfi]), punct_set);
        }
        if attach.is_some_and(|(kind, _)| kind == ATTACH_SISTER) {
            let chunk_parent = path_parent(&chunks[ci]).ok_or(TreeInsertError::MissingParent)?;
            gold.node_at_path_mut(chunk_parent).set_label(Some(BUILT.to_owned()));
        }

        let Some((kind, index)) = attach else {
            return Err(TreeInsertError::NoAttachment { span: gold.node_at_path(&chunks[ci]).span() });
        };
        let attachment = current[ci].clone();
        let updated = if kind == ATTACH_DAUGHTER {
            current[0].node_at_path_mut(&crf[index]).add(attachment, rules)?;
            crf[index].clone()
        } else {
            let frontier_parent = path_parent(&right_frontier[index]).ok_or(TreeInsertError::MissingParent)?.to_vec();
            right_frontier[index] = frontier_parent;
            match path_parent(&crf[index]) {
                None => {
                    // The parse built so far sits under a `TOP` node made for the first chunk.
                    let root = std::mem::replace(
                        &mut current[0],
                        Parse::new(Arc::clone(attachment.text_arc()), Span::new(0, 0), "", 0.0, 0),
                    );
                    let text = Arc::clone(root.text_arc());
                    let mut top = Parse::new(Arc::clone(&text), Span::new(0, text.len() as u32), TOP_NODE, 1.0, 0);
                    top.insert(root)?;
                    top.adjoin(attachment, rules)?;
                    current[0] = top.children()[0].clone();
                    Vec::new()
                }
                Some(site) => {
                    let site = site.to_vec();
                    let node = current[0].node_at_path_mut(&site);
                    node.adjoin(attachment, rules)?;
                    let mut path = site;
                    path.push(node.child_count() - 1);
                    path
                }
            }
        };

        let complete = Self::last_child(gold, &chunks[ci], &right_frontier[index], punct_set);
        if etype == ParserEventType::Check {
            let refs: Vec<&Parse> = current.iter().collect();
            let ctx = cgs.check.get_context(current[0].node_at_path(&updated), &refs, ci, true)?;
            events.push(event_from_strings(if complete { COMPLETE } else { INCOMPLETE }, &ctx));
            collapse_children(current[0].node_at_path_mut(&updated), punct_set);
        }
        let node = current[0].node_at_path_mut(&updated);
        if complete {
            node.set_label(Some(COMPLETE.to_owned()));
        } else if kind == ATTACH_SISTER {
            node.set_label(Some(INCOMPLETE.to_owned()));
        }
        right_frontier.drain(..index);
        Ok(())
    }

    /// `treeinsert.ParserEventStream.getNonAdjoinedParent`: the parents of `node` up through the
    /// ancestors of its own type, each with the position among its children that are not
    /// punctuation of the child on the way up.
    fn non_adjoined_parents(
        gold: &Parse,
        node: &[usize],
        punct_set: &HashSet<String>,
    ) -> Result<HashMap<Vec<usize>, usize>, TreeInsertError> {
        let mut parents = HashMap::new();
        let mut node = node.to_vec();
        let mut parent = path_parent(&node).ok_or(TreeInsertError::MissingParent)?.to_vec();
        parents.insert(parent.clone(), Self::index_of(gold, &node, punct_set));
        while gold.node_at_path(&parent).kind() == gold.node_at_path(&node).kind() {
            node = parent;
            parent = path_parent(&node).ok_or(TreeInsertError::MissingParent)?.to_vec();
            parents.insert(parent.clone(), Self::index_of(gold, &node, punct_set));
        }
        Ok(parents)
    }

    /// Position of the child at `child` among the children of its parent that are not punctuation.
    fn index_of(gold: &Parse, child: &[usize], punct_set: &HashSet<String>) -> usize {
        let (&idx, parent) = child.split_last().expect("child below the root");
        gold.node_at_path(parent).children()[..idx].iter().filter(|k| !punct_set.contains(k.kind())).count()
    }

    /// `AbstractParserEventStream.lastChild`: `child` is the last child of `parent` that is not
    /// punctuation.
    fn direct_last_child(gold: &Parse, child: &[usize], parent: &[usize], punct_set: &HashSet<String>) -> bool {
        let Some((&idx, child_parent)) = child.split_last() else {
            return false;
        };
        child_parent == parent
            && gold.node_at_path(parent).children()[idx + 1..].iter().all(|k| punct_set.contains(k.kind()))
    }

    /// `treeinsert.ParserEventStream.lastChild`: `child`, or an ancestor of its own type below
    /// `parent`, is the last child of `parent` that is not punctuation.
    fn last_child(gold: &Parse, child: &[usize], parent: &[usize], punct_set: &HashSet<String>) -> bool {
        let mut child = child;
        let mut lc = Self::direct_last_child(gold, child, parent, punct_set);
        while !lc {
            let Some(cp) = path_parent(child) else {
                break;
            };
            if cp != parent && gold.node_at_path(cp).kind() == gold.node_at_path(child).kind() {
                lc = Self::direct_last_child(gold, cp, parent, punct_set);
                child = cp;
            } else {
                break;
            }
        }
        lc
    }
}
