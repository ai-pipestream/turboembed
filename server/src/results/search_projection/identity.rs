//! The replay identities of a search projection under a PII disclosure
//! (#2403). A digest leaves only with the text it is the digest of: that
//! rule, and what a withheld identity looks like, is
//! [`crate::results::withheld_identity`]'s, shared with the typed
//! projection. This answer carries no span of the source, so it asks for
//! [`WithheldLengths::Zeroed`]: when the document's graph may not leave
//! ([`Disclosure::graph`]), the input and result identities carry the
//! binding and versions, no digest and no length, and a view whose text
//! may not leave ([`Disclosure::view_text`]) is listed without its
//! `text_sha256` and with `utf8_bytes` 0. A rewrite that does not keep
//! length (a fixed replacement) would otherwise tell two values apart by
//! their size. A view whose text may leave, the rewrite view the answer
//! is on, keeps its digest and length.

use std::any::TypeId;

use turbo_analysis::{Document, Key, View};

use crate::proto::result::{InputBinding, InputIdentity, ResultIdentity, ViewAlignmentSummary};
use crate::results::disclosure::Disclosure;
use crate::results::withheld_identity::{self, WithheldLengths};

/// What this answer says of the size of a text it withholds: nothing.
const LENGTHS: WithheldLengths = WithheldLengths::Zeroed;

/// What `doc` was and what its result is, for replay: every digest and
/// length when the graph, whose TGRF bytes are `graph`, may leave, else
/// the binding and the versions alone.
pub(super) fn identities(
    doc: &Document,
    binding: Option<&InputBinding>,
    graph: &[u8],
    disclosure: &Disclosure,
) -> (InputIdentity, ResultIdentity) {
    withheld_identity::identities(doc, binding, graph, disclosure, LENGTHS)
}

/// Every view of `doc`, each with its text digest and length only when
/// its text may leave.
pub(super) fn views(doc: &Document, disclosure: &Disclosure) -> Vec<ViewAlignmentSummary> {
    let mut views = Vec::new();
    for k in doc.layers().filter(|k| k.ty == TypeId::of::<View>()) {
        if let Some(v) = doc.get_key(&Key::<View>::named(k.name)) {
            let (utf8_bytes, text_sha256) = withheld_identity::view(v, disclosure.view_text(doc, v.id()), LENGTHS);
            views.push(ViewAlignmentSummary {
                view_id: u32::from(v.id().0),
                name: k.name.to_owned(),
                parent_view: v.parent().map(|p| u32::from(p.0)),
                utf8_bytes,
                text_sha256,
                lossy: v.alignment().edits().iter().any(|e| !e.parent.is_empty()),
            });
        }
    }
    views
}
