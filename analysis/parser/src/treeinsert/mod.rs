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

use turbo_analysis::Span;

use crate::bottom_up::collapse_punctuation_in_place;
use crate::context::{ConsRef, DisplayFn, FeatureSink};
use crate::parse::{INC_NODE, Parse, ParseError, TOP_NODE};

mod context;
#[cfg(test)]
mod feature_tests;
mod parser;

pub use context::{AttachContextGenerator, BuildContextGenerator, CheckContextGenerator};
pub use parser::Parser;

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
