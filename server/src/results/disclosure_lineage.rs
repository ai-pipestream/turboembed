//! What a layer's data was made from, for the PII disclosure decision
//! (#1932). A layer's spans say which view it sits on, not which text its
//! strings came from: a `ProjectView` copy of the source's keywords sits
//! on the masked view with the source's phrases, and a lemma column
//! aligned to masked tokens has no view of its own. So a layer is judged
//! by its lineage: its own view, and every layer it read or was made on,
//! back to the views at the root. A view is a boundary: a PII rewrite view
//! whose text may leave is clean even though it read the source.
//!
//! A rewrite view's coverage is lineage too: the `turbo:pii` instances it
//! rewrote are those its step read, those a `ProjectView` copy it read was
//! projected from, and those its parent rewrite view covered.

use std::any::TypeId;
use std::collections::HashMap;

use turbo_analysis::{Document, Key, KeyId, View, ViewId};
use turbo_analysis_layers::Pii;

use super::{Disclosure, REWRITE_ANNOTATOR_PREFIX};

/// The judge of one disclosure over one document: `own` gives the view a
/// layer's spans sit on, for a layer that has spans of its own.
pub(crate) struct Lineage<'a, F> {
    doc: &'a Document,
    disclosure: &'a Disclosure,
    own: F,
    memo: HashMap<KeyId, bool>,
}

impl<'a, F: Fn(KeyId) -> Option<ViewId>> Lineage<'a, F> {
    pub(crate) fn new(doc: &'a Document, disclosure: &'a Disclosure, own: F) -> Self {
        Lineage { doc, disclosure, own, memo: HashMap::new() }
    }

    /// Whether the data of `key` may hold the text of a view the
    /// disclosure withholds. A layer with no view of its own and nothing
    /// read is of unknown origin and counts as the source's.
    pub(crate) fn withheld(&mut self, key: KeyId) -> bool {
        if let Some(&w) = self.memo.get(&key) {
            return w;
        }
        // Provenance is acyclic; a key revisited mid-walk is judged withheld.
        self.memo.insert(key, true);
        let w = self.judge(key);
        self.memo.insert(key, w);
        w
    }

    fn judge(&mut self, key: KeyId) -> bool {
        let (doc, d) = (self.doc, self.disclosure);
        if key.ty == TypeId::of::<View>() {
            return doc.get_key(&Key::<View>::named(key.name)).is_none_or(|v| !d.view_text(doc, v.id()));
        }
        let own = (self.own)(key);
        if own.is_some_and(|v| !d.view_text(doc, v)) {
            return true;
        }
        let Some(p) = doc.provenance(key) else { return own.is_none() };
        let inputs: Vec<KeyId> = p.reads.iter().chain(&p.on).copied().filter(|k| *k != key).collect();
        if own.is_none() && inputs.is_empty() {
            return true;
        }
        inputs.into_iter().any(|k| self.withheld(k))
    }
}

/// The `turbo:pii` instances the rewrite view `view_key` rewrote: read by
/// its step, projected into one it read, or covered by its parent rewrite.
pub(crate) fn rewritten_findings(doc: &Document, view_key: KeyId) -> Vec<KeyId> {
    let mut out = Vec::new();
    let mut at = Some(view_key);
    while let Some(v) = at.take() {
        let Some(p) = doc.provenance(v) else { break };
        if !p.annotator.starts_with(REWRITE_ANNOTATOR_PREFIX) {
            break;
        }
        for &r in &p.reads {
            if r.ty == TypeId::of::<Pii>() {
                projected_from(doc, r, &mut out);
            } else if r.ty == TypeId::of::<View>() && r != v {
                at = Some(r);
            }
        }
    }
    out
}

/// `pii` and, when it is a projected copy, the instances it was made from.
fn projected_from(doc: &Document, pii: KeyId, out: &mut Vec<KeyId>) {
    if out.contains(&pii) {
        return;
    }
    out.push(pii);
    if let Some(p) = doc.provenance(pii) {
        for &r in p.reads.iter().filter(|r| r.ty == TypeId::of::<Pii>()) {
            projected_from(doc, r, out);
        }
    }
}

/// Whether `key` is a `turbo:pii` layer a rewrite step wrote beside its
/// view (the replacement labels), not a finding over any text.
pub(crate) fn is_rewrite_output(doc: &Document, key: KeyId) -> bool {
    key.ty == TypeId::of::<Pii>()
        && doc.provenance(key).is_some_and(|p| {
            p.annotator.starts_with(REWRITE_ANNOTATOR_PREFIX) && p.reads.iter().any(|r| r.ty == TypeId::of::<View>())
        })
}
