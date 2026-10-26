//! Arc-standard with Nivre's monotonic SWAP and canonical projective order.
//! https://aclanthology.org/P09-1040/
use crate::{ArcStandardOracle, ArcStandardState, DependencyGraph, Position, Result, Transition, error};
use turbo_analysis::Ix;
use turbo_analysis_layers::WordOrTokenAnchor;

pub struct SwapOracle;
impl SwapOracle {
    /// Retains the legacy oracle exactly on projective trees. For crossing
    /// trees, swaps original token indexes; text and source spans never move.
    pub fn transitions<A: WordOrTokenAnchor>(gold: &DependencyGraph<A>) -> Result<Vec<Transition>> {
        gold.check_tree()?;
        if gold.is_projective() {
            return ArcStandardOracle::transitions(gold);
        }
        let rank = projective_order(gold)?;
        let mut state = ArcStandardState::<A>::new(gold.len())?;
        let mut result = Vec::new();
        while !state.is_terminal() {
            let (top, below) = (state.stack(0), state.stack(1));
            let mut next = None;
            if let (Position::Node(t), Position::Node(b)) = (top, below) {
                let arc = gold.arc(b)?;
                if arc.head == Some(t) && state.assigned_dependents(b)? == gold.children(b)?.len() {
                    next = Some(Transition::LeftArc(arc.relation.into()));
                }
            }
            if next.is_none()
                && let Position::Node(t) = top
            {
                let arc = gold.arc(t)?;
                let matches = match below {
                    Position::Root => arc.head.is_none(),
                    Position::Node(b) => arc.head == Some(b),
                    Position::Missing => false,
                };
                let right = Transition::RightArc(arc.relation.into());
                if matches && state.assigned_dependents(t)? == gold.children(t)?.len() && state.can_apply(&right) {
                    next = Some(right);
                }
            }
            if next.is_none()
                && state.can_apply(&Transition::Swap)
                && let (Position::Node(t), Position::Node(b)) = (top, below)
                && rank[b.get()] > rank[t.get()]
            {
                next = Some(Transition::Swap);
            }
            let next = next
                .or_else(|| state.can_apply(&Transition::Shift).then_some(Transition::Shift))
                .ok_or_else(|| error("no oracle transition for dependency tree"))?;
            state.apply(&next)?;
            result.push(next);
        }
        Ok(result)
    }
}

/// Iterative inorder traversal avoids recursion on a long dependency chain.
fn projective_order<A: WordOrTokenAnchor>(gold: &DependencyGraph<A>) -> Result<Vec<usize>> {
    let mut rank = vec![0; gold.len()];
    let mut stack = vec![(gold.root().ok_or_else(|| error("one root required"))?, false)];
    let mut cursor = 0;
    while let Some((node, emit)) = stack.pop() {
        if emit {
            rank[node.get()] = cursor;
            cursor += 1;
        } else {
            let mut items: Vec<(Ix<A>, bool)> = gold.children(node)?.iter().map(|&child| (child, false)).collect();
            items.push((node, true));
            items.sort_by_key(|(id, _)| id.get());
            stack.extend(items.into_iter().rev());
        }
    }
    Ok(rank)
}
