//! Discovered dependency paths between entity heads: one row
//! per ordered entity pair whose heads connect, with no relation
//! type attached. This is the rule author's view of a corpus:
//! every row's [`RelationPaths::signature`] is a string that
//! parses unchanged as a [`crate::RelationPattern`] path.

use std::sync::Arc;

use turbo_analysis::{Anchor, Ix, Key, KeyId, Layer, Overlap, Spans, Tag, TagSet, TextSpans};
use turbo_analysis_layers::Entities;

/// One discovered path between two entity heads, read back from
/// a [`RelationPaths`] row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencyPath {
    pub subject: Ix<Entities>,
    pub object: Ix<Entities>,
    /// The lowest common ancestor's anchor word index.
    pub pivot: u32,
    /// The anchor word indexes walked, subject head first and
    /// object head last, the pivot included once between them.
    pub tokens: Vec<u32>,
    /// `(arrived-at word, goes up, relation label)` per step, in
    /// walk order: up steps first, then down steps.
    pub steps: Vec<(u32, bool, String)>,
}

impl DependencyPath {
    /// The canonical signature: the steps as `RelationPattern`
    /// path syntax, joined by single spaces.
    pub fn signature(&self) -> String {
        self.steps
            .iter()
            .map(|(_, up, label)| format!("{}{label}", if *up { '<' } else { '>' }))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Untyped dependency paths between entity heads:
/// `turbo:relation-paths`. Rows are in pair enumeration order
/// (subject in entity layer order, then object), one per ordered
/// pair whose heads connect within one sentence. Steps are
/// stored CSR-style: row `i`'s steps are
/// `step_row[i]..step_row[i + 1]` across the step columns.
#[derive(Clone, Debug)]
pub struct RelationPaths {
    /// The span covering both entity spans, per row.
    pub spans: Spans<RelationPaths>,
    pub subject: Vec<Ix<Entities>>,
    pub object: Vec<Ix<Entities>>,
    /// Subject and object head anchor word indexes.
    pub subject_head: Vec<u32>,
    pub object_head: Vec<u32>,
    /// The pivot (lowest common ancestor) anchor word index.
    pub pivot: Vec<u32>,
    /// Step offsets per row, `len() + 1` entries.
    pub step_row: Vec<u32>,
    /// The anchor word each step arrives at.
    pub step_token: Vec<u32>,
    /// Whether each step walks up (toward the pivot).
    pub step_up: Vec<bool>,
    /// Each step's relation label, into `relations`.
    pub step_relation: Vec<Tag>,
    pub relations: Arc<TagSet>,
}

impl RelationPaths {
    /// Row `i` as a [`DependencyPath`].
    pub fn path(&self, i: Ix<RelationPaths>) -> DependencyPath {
        let r = i.get();
        let steps: Vec<(u32, bool, String)> = (self.step_row[r]..self.step_row[r + 1])
            .map(|k| {
                let k = k as usize;
                (self.step_token[k], self.step_up[k], self.relations.label(self.step_relation[k]).to_owned())
            })
            .collect();
        let mut tokens = Vec::with_capacity(steps.len() + 1);
        tokens.push(self.subject_head[r]);
        tokens.extend(steps.iter().map(|s| s.0));
        DependencyPath { subject: self.subject[r], object: self.object[r], pivot: self.pivot[r], tokens, steps }
    }

    /// Row `i`'s canonical path signature (pattern syntax).
    pub fn signature(&self, i: Ix<RelationPaths>) -> String {
        self.path(i).signature()
    }
}

impl Layer for RelationPaths {
    const ID: &'static str = "turbo:relation-paths";
    const OVERLAP: Overlap = Overlap::Any;

    fn anchor() -> Anchor {
        Anchor::Text
    }

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        let bound = self.subject.iter().zip(&self.object).map(|(s, o)| s.get().max(o.get()) + 1).max().unwrap_or(0);
        vec![(Key::<Entities>::DEFAULT.id(), bound)]
    }

    fn check(&self) -> Result<(), String> {
        self.spans.check(Self::OVERLAP)?;
        let n = self.spans.len();
        if self.subject.len() != n
            || self.object.len() != n
            || self.subject_head.len() != n
            || self.object_head.len() != n
            || self.pivot.len() != n
            || self.step_row.len() != n + 1
        {
            return Err("a subject, an object, two heads, a pivot and step offsets for every path".into());
        }
        let steps = *self.step_row.last().unwrap_or(&0) as usize;
        if self.step_token.len() != steps || self.step_up.len() != steps || self.step_relation.len() != steps {
            return Err("a token, a direction and a relation for every step".into());
        }
        if self.step_row.windows(2).any(|w| w[0] > w[1]) {
            return Err("step offsets run backwards".into());
        }
        if self.step_relation.iter().any(|k| k.0 as usize >= self.relations.len()) {
            return Err(format!("a step relation outside {}", self.relations.name));
        }
        for r in 0..n {
            if self.subject[r] == self.object[r] {
                return Err(format!("path {r} has itself as subject and object"));
            }
            // The walk ends at the object head, starts from the
            // subject head, and the pivot is where it turns.
            let row = self.step_row[r] as usize..self.step_row[r + 1] as usize;
            if row.is_empty() {
                return Err(format!("path {r} has no steps"));
            }
            if self.step_token[row.end - 1] != self.object_head[r] {
                return Err(format!("path {r} does not end at its object head"));
            }
            let downs = self.step_up[row.clone()].iter().position(|up| !up).unwrap_or(row.len());
            if self.step_up[row.clone()][downs..].iter().any(|up| *up) {
                return Err(format!("path {r} goes up after going down"));
            }
            let turned = if downs == 0 { self.subject_head[r] } else { self.step_token[row.start + downs - 1] };
            if turned != self.pivot[r] {
                return Err(format!("path {r} does not turn at its pivot"));
            }
        }
        Ok(())
    }
}
