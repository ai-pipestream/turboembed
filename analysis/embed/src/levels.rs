//! A local and a contextual vector for every unit of `Levels` (chunk,
//! section, document), from the same window planner and late pooling as
//! the sentence and chunk vectors.
//!
//! - The local vector reads the unit's own text and nothing else: its
//!   windows start at its first piece and stop at its last, so no piece
//!   outside it is in any row it is pooled from.
//! - The contextual vector reads the unit's parent (a chunk's section, a
//!   section's section or the document): the parent's windows, which
//!   stop at the parent's last piece, and the unit's pieces are pooled
//!   from them. The document has no parent, so its contextual vector is
//!   pooled from its own windows and equals its local one.
//!
//! Both are the planner's rules (`crate::window`) applied to one unit at
//! a time: for each unit, the unit lays out the windows over its own
//! pieces, its local vector is pooled from them whole, and each of its
//! children is placed in them for its contextual vector. A unit that
//! fits a window is one row; a longer one is cut into windows of the
//! full width from its first piece, the last one shorter, and a child is
//! pooled from the window that holds it whole, or in parts weighted by
//! pieces when it crosses from one to the next. So a child of a long
//! parent sees the context of its own window, not the whole parent.

use std::ops::Range;
use std::sync::Arc;

use turbo_analysis::{Annotator, Bound, Error, Execution, Ix, Layer, ModelRef, Refs};
use turbo_analysis_layers::{Levels, Subwords, VectorKind, Vectors};

use crate::vectors::{Input, VectorAnnotator};
use crate::window::{Plan, Target, Which, pieces_of};

/// The vectors of one kind in an output's pair.
pub(crate) fn of_kind(pair: &mut [Vectors<Levels>; 2], kind: VectorKind) -> &mut Vectors<Levels> {
    match kind {
        VectorKind::Local => &mut pair[0],
        VectorKind::Contextual => &mut pair[1],
    }
}

/// Places every unit of one document's levels: one call to the planner
/// per unit with children or none, each bounded by that unit's last
/// piece. A unit with no pieces is listed in both layers' `empty`. The
/// levels must pass their own check (`VectorAnnotator::embed` runs it).
pub(crate) fn plan(
    plan: &mut Plan,
    doc: u32,
    levels: &Levels,
    starts: &[u32],
    ends: &[u32],
    out: &mut [Vectors<Levels>; 2],
) -> Result<(), String> {
    let n = levels.len();
    let pieces: Vec<Range<u32>> = levels.spans.iter().map(|(_, s)| pieces_of(starts, ends, s.start, s.end)).collect();
    for (i, r) in pieces.iter().enumerate() {
        if r.is_empty() {
            for v in out.iter_mut() {
                v.empty.push(Ix::new(i as u32));
            }
        }
    }
    // Each unit's children, in row order: rows `first[u]..first[u + 1]`
    // of `children`.
    let mut first = vec![0u32; n + 1];
    for p in levels.parent.iter().flatten() {
        first[p.get() + 1] += 1;
    }
    for i in 0..n {
        first[i + 1] += first[i];
    }
    let mut children = vec![0u32; first[n] as usize];
    let mut fill = first.clone();
    for (i, p) in levels.parent.iter().enumerate() {
        if let Some(p) = p {
            children[fill[p.get()] as usize] = i as u32;
            fill[p.get()] += 1;
        }
    }
    let target = |kind, item| Target { doc, which: Which::Levels(kind), item };
    for u in 0..n {
        let own = pieces[u].clone();
        plan.layout(target(VectorKind::Local, u as u32), own.clone());
        if levels.parent[u].is_none() {
            plan.item(target(VectorKind::Contextual, u as u32), own.clone());
        }
        for &c in &children[first[u] as usize..first[u + 1] as usize] {
            plan.item(target(VectorKind::Contextual, c), pieces[c as usize].clone());
        }
        // Unreachable for levels that pass their check, where every child
        // is inside its parent and so are its pieces.
        plan.place(doc, own.end).map_err(|e| {
            format!(
                "unit {} ends at piece {}, past the end of unit {u} it is placed in, at piece {}",
                e.target.item, e.end, e.pieces
            )
        })?;
    }
    Ok(())
}

/// Writes the two `opennlp:embeddings<turbo:levels>` instances, `local`
/// and `contextual` (`VectorKind::key`), from one set of encoder runs.
/// Add it to a pipeline with `bound`, which names the two outputs; with
/// both under the default name the pipeline is refused.
#[derive(Clone)]
pub struct LevelVectors(pub Arc<VectorAnnotator>);

impl LevelVectors {
    /// The step, writing each kind under its own name.
    pub fn bound(self) -> Bound<LevelVectors> {
        Bound::new(self).write(0, VectorKind::Local.name()).write(1, VectorKind::Contextual.name())
    }
}

impl Annotator for LevelVectors {
    const VERSION: &'static str = "1";
    const ID: &'static str = "turbo:level-vectors";
    type Reads = (Subwords, Levels);
    type Writes = (Vectors<Levels>, Vectors<Levels>);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.annotate_batch(vec![input]).remove(0)
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        let inputs: Vec<Input<'_>> = inputs
            .into_iter()
            .map(|(sub, l)| Input { subwords: sub, sentences: None, chunks: None, levels: Some(l) })
            .collect();
        self.0
            .embed(Self::ID, &inputs)
            .into_iter()
            .map(|r| {
                r.map(|o| {
                    let [local, contextual] = o.levels.expect("asked for");
                    (local, contextual)
                })
            })
            .collect()
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.0.model().clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.0.execution().clone())
    }

    fn config_hash(&self) -> u64 {
        self.0.config(Which::Levels(VectorKind::Local))
    }
}
