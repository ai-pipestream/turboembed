//! Joining documents: branch copies with [`Document::merge`], and two
//! independently built documents of one text with
//! [`Document::merge_with`].

use std::any::TypeId;
use std::collections::HashSet;
use std::sync::Arc;

use super::{Document, Entry, View};
use crate::error::Error;
use crate::text::Alignment;

/// What [`Document::merge_with`] does when both documents hold a layer
/// under one key.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DuplicateLayerPolicy {
    /// Refuse the merge, changing nothing.
    Refuse,
    /// The first document's layer wins. Layers of the second document
    /// that were made on its losing layer are dropped with it: their
    /// indexes name items of the layer that lost, not of the one kept.
    KeepFirst,
    /// Ask for both to be kept, the second under a new instance name.
    /// Instance names are static strings and a layer's indexes into
    /// other layers cannot be rewritten after the fact, so a duplicate
    /// under this policy is refused with that reason rather than kept
    /// under a wrong name. With no duplicates it behaves like
    /// [`DuplicateLayerPolicy::Refuse`].
    KeepBoth,
}

impl Document {
    /// Joins a copy grown on another branch of a pipeline: its new views
    /// and layers are added here. Both must be copies of one document, and
    /// the branches must not both have written a layer, both made a view,
    /// or written two layer types under one id, since each would leave two
    /// answers under one name. A refused merge changes nothing. For two
    /// documents built independently from one text, use
    /// [`Document::merge_with`].
    pub fn merge(&mut self, other: &Document) -> Result<(), Error> {
        if !Arc::ptr_eq(&self.views[0], &other.views[0]) {
            return Err(Error::Pipeline("merge needs two copies of one document".into()));
        }
        let shared = self.views.len().min(other.views.len());
        if !self.views[..shared].iter().zip(&other.views[..shared]).all(|(a, b)| Arc::ptr_eq(a, b)) {
            return Err(Error::Pipeline("both branches made a view".into()));
        }
        for e in &other.entries {
            match self.entries.iter().find(|m| m.key == e.key) {
                Some(mine) if !Arc::ptr_eq(mine, e) => return Err(Error::Duplicate(e.key)),
                None if self.clashes(e.key) => return Err(Error::Clash(e.key)),
                _ => {}
            }
        }
        for e in &other.entries {
            if !self.has(e.key) {
                self.entries.push(e.clone());
            }
        }
        self.views.extend(other.views[shared..].iter().cloned());
        Ok(())
    }

    /// Joins a second document built independently from the same source
    /// text: its layers are adopted here, their provenance unchanged,
    /// under `policy` where the two hold a layer under one key.
    ///
    /// The texts must be equal and the decoded bytes, when both keep
    /// them, must be the same bytes with the same alignment; a view at
    /// the same position in both must be the same view (same parent,
    /// text, and alignment), because adopted layers' spans name views
    /// by id and ids are positions. Views past the shared prefix are
    /// appended, keeping their ids. Two layer types under one id are a
    /// clash under every policy. The check runs whole before anything
    /// is adopted, so a refused merge changes nothing; `other` is never
    /// changed.
    pub fn merge_with(&mut self, other: &Document, policy: DuplicateLayerPolicy) -> Result<(), Error> {
        if self.source() != other.source() {
            return Err(Error::Pipeline("merge_with needs two documents of one text: the source texts differ".into()));
        }
        match (&self.bytes, &other.bytes) {
            (Some(a), Some(b)) if a.bytes() != b.bytes() || a.alignment() != b.alignment() => {
                return Err(Error::Pipeline("the documents were decoded from different bytes or alignments".into()));
            }
            (Some(a), None) if !(a.bytes() == other.source().as_bytes() && *a.alignment() == Alignment::identity()) => {
                return Err(Error::Pipeline("one document was decoded from bytes and the other made from text".into()));
            }
            (None, Some(b)) if !(b.bytes() == self.source().as_bytes() && *b.alignment() == Alignment::identity()) => {
                return Err(Error::Pipeline("one document was decoded from bytes and the other made from text".into()));
            }
            _ => {}
        }
        let shared = self.views.len().min(other.views.len());
        for i in 0..shared {
            let (a, b) = (&self.views[i], &other.views[i]);
            if a.parent != b.parent || a.text != b.text || a.alignment != b.alignment {
                return Err(Error::Pipeline(format!(
                    "view {i} differs between the documents, so spans naming it cannot be adopted"
                )));
            }
        }

        let view_ty = TypeId::of::<View>();
        let mut adopted: Vec<Arc<Entry>> = Vec::new();
        let mut adopted_keys: HashSet<_> = HashSet::new();
        let mut dropped: HashSet<_> = HashSet::new();
        for e in &other.entries {
            if self.entries.iter().chain(&adopted).any(|m| m.key.ty != e.key.ty && m.key.layer == e.key.layer) {
                return Err(Error::Clash(e.key));
            }
            if let Some(mine) = self.entries.iter().find(|m| m.key == e.key) {
                if Arc::ptr_eq(mine, e) {
                    continue;
                }
                if e.key.ty == view_ty {
                    // Views are matched by the position check above, not
                    // by policy, but only when the key names the same view
                    // position on both sides. A view key holding no view
                    // (a skipped or failed record) is an ordinary
                    // duplicate and follows the policy below.
                    fn view_of(entry: &Entry) -> Option<&View> {
                        entry.value.as_deref().and_then(|v| v.downcast_ref::<View>())
                    }
                    match (view_of(mine), view_of(e)) {
                        (Some(a), Some(b)) if a.id == b.id => continue,
                        (Some(a), Some(b)) => {
                            return Err(Error::Pipeline(format!(
                                "{} names view {} here and view {} in the other document",
                                e.key, a.id.0, b.id.0
                            )));
                        }
                        _ => {}
                    }
                }
                match policy {
                    DuplicateLayerPolicy::Refuse => return Err(Error::Duplicate(e.key)),
                    DuplicateLayerPolicy::KeepFirst => {
                        dropped.insert(e.key);
                        continue;
                    }
                    DuplicateLayerPolicy::KeepBoth => {
                        return Err(Error::Pipeline(format!(
                            "keep-both cannot rename {}: instance names are static and a layer's indexes cannot be rewritten",
                            e.key
                        )));
                    }
                }
            }
            if e.on.iter().any(|t| dropped.contains(t)) {
                dropped.insert(e.key);
                continue;
            }
            if let Some(missing) = e.on.iter().find(|t| !self.has(**t) && !adopted_keys.contains(*t)) {
                return Err(Error::Pipeline(format!(
                    "{} was made on {}, which the merge does not bring over",
                    e.key, missing
                )));
            }
            let entry = if e.key.ty == view_ty
                && let Some(view) = e.value.as_deref().and_then(|v| v.downcast_ref::<View>())
            {
                // Point the adopted view key at the view table entry at
                // the same position: the one already here, or the one
                // about to be appended from `other`. A view entry with no
                // view (skipped or failed) is adopted as the record it is.
                let idx = view.id.0 as usize;
                let shared_view: Arc<View> =
                    if idx < self.views.len() { self.views[idx].clone() } else { other.views[idx].clone() };
                let mut repointed = (**e).clone();
                repointed.value = Some(shared_view);
                Arc::new(repointed)
            } else {
                e.clone()
            };
            adopted_keys.insert(e.key);
            adopted.push(entry);
        }

        if other.views.len() > self.views.len() {
            self.views.extend(other.views[shared..].iter().cloned());
        }
        self.entries.extend(adopted);
        Ok(())
    }
}
