//! spaCy 3.8.7 DependencyMatcher relations over a validated forest.
use super::{DependencyError, Work};
use turbo_analysis_layers::Dependencies;

/// Relation from an earlier pattern node to the current node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DependencyOp {
    Child,
    Parent,
    Descendant,
    Ancestor,
    Next,
    After,
    Previous,
    Before,
    NextSibling,
    PreviousSibling,
    RightSibling,
    LeftSibling,
    NextChild,
    PreviousChild,
    RightChild,
    LeftChild,
    NextParent,
    PreviousParent,
    RightParent,
    LeftParent,
}

impl DependencyOp {
    /// Parses the twenty spaCy dependency operators; unknown operators are refused.
    pub fn parse(value: &str) -> Result<Self, DependencyError> {
        Ok(match value {
            ">" => Self::Child,
            "<" => Self::Parent,
            ">>" => Self::Descendant,
            "<<" => Self::Ancestor,
            "." => Self::Next,
            ".*" => Self::After,
            ";" => Self::Previous,
            ";*" => Self::Before,
            "$+" => Self::NextSibling,
            "$-" => Self::PreviousSibling,
            "$++" => Self::RightSibling,
            "$--" => Self::LeftSibling,
            ">+" => Self::NextChild,
            ">-" => Self::PreviousChild,
            ">++" => Self::RightChild,
            ">--" => Self::LeftChild,
            "<+" => Self::NextParent,
            "<-" => Self::PreviousParent,
            "<++" => Self::RightParent,
            "<--" => Self::LeftParent,
            _ => return Err(DependencyError::Operator(value.into())),
        })
    }

    pub(super) fn holds(self, a: usize, b: usize, graph: &Graph<'_>, work: &mut Work) -> Result<bool, DependencyError> {
        use DependencyOp::*;
        if a == b || graph.roots[a] != graph.roots[b] {
            return Ok(false);
        }
        let head = |i: usize| graph.dependencies.head[i].map(|h| h.get());
        let child = head(b) == Some(a);
        let parent = head(a) == Some(b);
        // spaCy treats the root as its own head when enumerating siblings.
        let sibling = head(b) == Some(head(a).unwrap_or(a));
        Ok(match self {
            Child => child,
            Parent => parent,
            Descendant => graph.ancestor(a, b, work)?,
            Ancestor => graph.ancestor(b, a, work)?,
            Next => b == a + 1,
            After => a < b,
            Previous => a == b + 1,
            Before => b < a,
            NextSibling => sibling && b == a + 1,
            PreviousSibling => sibling && a == b + 1,
            RightSibling => sibling && a < b,
            LeftSibling => sibling && b < a,
            NextChild => child && b == a + 1,
            PreviousChild => child && a == b + 1,
            RightChild => child && a < b,
            LeftChild => child && b < a,
            NextParent => parent && b == a + 1,
            PreviousParent => parent && a == b + 1,
            RightParent => parent && a < b,
            LeftParent => parent && b < a,
        })
    }
}

pub(super) struct Graph<'a> {
    dependencies: &'a Dependencies,
    roots: Vec<usize>,
}
impl<'a> Graph<'a> {
    pub(super) fn new(dependencies: &'a Dependencies) -> Self {
        let mut roots = vec![usize::MAX; dependencies.head.len()];
        for i in 0..roots.len() {
            if roots[i] != usize::MAX {
                continue;
            }
            let mut at = i;
            while roots[at] == usize::MAX {
                if let Some(head) = dependencies.head[at] {
                    at = head.get();
                } else {
                    break;
                }
            }
            let root = if roots[at] == usize::MAX { at } else { roots[at] };
            at = i;
            while roots[at] == usize::MAX {
                roots[at] = root;
                if let Some(head) = dependencies.head[at] {
                    at = head.get();
                } else {
                    break;
                }
            }
        }
        Self { dependencies, roots }
    }
    fn ancestor(&self, ancestor: usize, mut child: usize, work: &mut Work) -> Result<bool, DependencyError> {
        while let Some(parent) = self.dependencies.head[child] {
            work.visit()?;
            child = parent.get();
            if child == ancestor {
                return Ok(true);
            }
        }
        Ok(false)
    }
}
