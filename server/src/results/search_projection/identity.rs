//! The replay identities of a search projection under a PII disclosure
//! (#2403). An unkeyed SHA-256 names its input to anyone who can guess
//! it: for a short source (an address, a phone number, an id) the digest
//! of the source text, of the raw bytes, or of the graph bytes that hold
//! that text, gives the value back by hashing candidates. So a digest
//! leaves only with the text it is the digest of. When the document's
//! graph may not leave ([`Disclosure::graph`]), the input and result
//! identities carry their lengths and versions and no digest, and a view
//! whose text may not leave ([`Disclosure::view_text`]) is listed
//! without its `text_sha256`. A view whose text may leave, the rewrite
//! view the answer is on, keeps its digest.

use std::any::TypeId;

use turbo_analysis::{Document, Key, View};

use crate::proto::result::{InputBinding, InputIdentity, ResultIdentity, ViewAlignmentSummary};
use crate::results::disclosure::Disclosure;
use crate::results::typed::identity::{input_identity, result_identity};

/// What `doc` was, for replay: every digest when the graph may leave,
/// else the binding and lengths alone.
pub(super) fn input(doc: &Document, binding: Option<&InputBinding>, disclosure: &Disclosure) -> InputIdentity {
    let mut input = input_identity(doc, binding);
    if !disclosure.graph(doc) {
        input.raw_bytes_digest.clear();
        input.source_text_digest.clear();
    }
    input
}

/// The identity of the result whose TGRF bytes are `graph`: its digest
/// when the graph may leave, else the format and schema versions alone
/// (the graph holds the source text, so its digest and length stay).
pub(super) fn result(doc: &Document, graph: &[u8], disclosure: &Disclosure) -> ResultIdentity {
    let mut result = result_identity(graph);
    if !disclosure.graph(doc) {
        result.result_id.clear();
        result.content_digest.clear();
        result.graph_bytes_len = 0;
    }
    result
}

/// Every view of `doc`, each with its text digest only when its text may
/// leave.
pub(super) fn views(doc: &Document, disclosure: &Disclosure) -> Vec<ViewAlignmentSummary> {
    let mut views = Vec::new();
    for k in doc.layers().filter(|k| k.ty == TypeId::of::<View>()) {
        if let Some(v) = doc.get_key(&Key::<View>::named(k.name)) {
            let text_sha256 =
                if disclosure.view_text(doc, v.id()) { v.text_digest().as_bytes().to_vec() } else { Vec::new() };
            views.push(ViewAlignmentSummary {
                view_id: u32::from(v.id().0),
                name: k.name.to_owned(),
                parent_view: v.parent().map(|p| u32::from(p.0)),
                utf8_bytes: v.text().len() as u32,
                text_sha256,
                lossy: v.alignment().edits().iter().any(|e| !e.parent.is_empty()),
            });
        }
    }
    views
}
