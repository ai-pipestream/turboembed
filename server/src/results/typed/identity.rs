//! The result envelope's identities (issue #1907): what a result was
//! computed from (`InputIdentity`), what it is (`ResultIdentity`) and how
//! each layer and view was made (`LayerProvenance`), on a
//! `TypedDocumentResult`. Every digest is one of the document's own
//! SHA-256 digests (`ViewTextDigest`, `SourceBytesDigest`,
//! `ProvenanceDigest`, `ContentDigest`, `InputSnapshotDigest`) or the
//! SHA-256 of the TGRF bytes; nothing new is hashed here. The consumer
//! side, [`verify_input`] and [`verify_result`], refuses a result whose
//! input or graph bytes are not the ones it names.
//!
//! None of these digests is keyed, so each names its input to anyone who
//! can guess it: a short source (an address, a phone number, an id) comes
//! back by hashing candidates (#2403). A digest therefore leaves only with
//! the text it is a digest of. When the document's graph may not leave
//! ([`Disclosure::graph`]), [`disclosed_input`] and [`disclosed_result`]
//! give the lengths and versions alone, and no layer or view entry carries
//! a provenance, content or input snapshot digest: every one of them
//! chains the source text digest.

use std::any::TypeId;
use std::collections::HashMap;

use turbo_analysis::export::{DocumentGraph, FORMAT_VERSION, graph_content_digest};
use turbo_analysis::{Document, Key, KeyId, Layer, View, ViewTextDigest};

use super::{ProjectionCx, bad};
use crate::api::Failure;
use crate::proto::result::{
    InputBinding, InputIdentity, LayerInstanceRef, LayerProvenance, ResultIdentity, TypedDocumentResult,
};
use crate::results::disclosure::Disclosure;

/// The version of the envelope's messages and rules.
pub const ENVELOPE_SCHEMA_VERSION: u32 = 1;

/// The exact input of `doc`, with the caller's `binding` copied as given.
pub fn input_identity(doc: &Document, binding: Option<&InputBinding>) -> InputIdentity {
    let source = doc.get::<View>().expect("every document has its source view");
    let (raw_bytes_digest, raw_bytes_len) = match doc.source_bytes() {
        Some(b) => (b.digest().as_bytes().to_vec(), b.bytes().len() as u64),
        // A document made from text: its bytes are the source view's UTF-8.
        None => (source.text_digest().as_bytes().to_vec(), source.text().len() as u64),
    };
    InputIdentity {
        binding: binding.cloned(),
        raw_bytes_digest,
        raw_bytes_len,
        source_text_digest: source.text_digest().as_bytes().to_vec(),
        source_text_len: source.text().len() as u32,
    }
}

/// [`input_identity`] as an answer under `disclosure` may carry it: with
/// both digests when the graph of `doc` may leave, else the binding and
/// the lengths alone.
pub fn disclosed_input(doc: &Document, binding: Option<&InputBinding>, disclosure: &Disclosure) -> InputIdentity {
    input_identity_with(doc, binding, disclosure.graph(doc))
}

/// [`input_identity`], with its two digests only when `digests`.
pub fn input_identity_with(doc: &Document, binding: Option<&InputBinding>, digests: bool) -> InputIdentity {
    let mut input = input_identity(doc, binding);
    if !digests {
        input.raw_bytes_digest.clear();
        input.source_text_digest.clear();
    }
    input
}

/// [`result_identity`] as an answer under `disclosure` may carry it: of
/// `graph` when the graph of `doc` may leave, else the format and schema
/// versions alone.
pub fn disclosed_result(doc: &Document, graph: &[u8], disclosure: &Disclosure) -> ResultIdentity {
    result_identity_with(graph, disclosure.graph(doc))
}

/// [`result_identity`] when `digests`, else the format and schema versions
/// alone: the graph holds the source text, so its digest, the id that is
/// its hexadecimal, and its length stay with it.
pub fn result_identity_with(graph: &[u8], digests: bool) -> ResultIdentity {
    if digests {
        return result_identity(graph);
    }
    ResultIdentity {
        graph_format_version: u32::from(FORMAT_VERSION),
        envelope_schema_version: ENVELOPE_SCHEMA_VERSION,
        ..ResultIdentity::default()
    }
}

/// Refuses a caller's `binding` that names a parse artifact digest and no
/// parse artifact id; `what` names the field. Absent is accepted.
pub fn check_binding(binding: Option<&InputBinding>, what: &str) -> Result<(), Failure> {
    match binding {
        Some(b) if !b.parse_artifact_digest.is_empty() && b.parse_artifact_id.is_empty() => {
            Err(bad(format!("{what} has a parse_artifact_digest and no parse_artifact_id")))
        }
        _ => Ok(()),
    }
}

/// `typed` with the caller's `binding` on its input identity and on its
/// receipt's, so the two still agree: for a projection built without it,
/// such as the findings of a withheld graph.
pub fn bound(mut typed: TypedDocumentResult, binding: Option<&InputBinding>) -> TypedDocumentResult {
    for input in [typed.input.as_mut(), typed.receipt.as_mut().and_then(|r| r.input.as_mut())].into_iter().flatten() {
        input.binding = binding.cloned();
    }
    typed
}

/// The identity of the result whose TGRF bytes are `graph`.
pub fn result_identity(graph: &[u8]) -> ResultIdentity {
    let digest = graph_content_digest(graph);
    ResultIdentity {
        result_id: digest.to_hex(),
        content_digest: digest.as_bytes().to_vec(),
        graph_bytes_len: graph.len() as u64,
        graph_format_version: u32::from(FORMAT_VERSION),
        envelope_schema_version: ENVELOPE_SCHEMA_VERSION,
    }
}

fn layer_ref(key: KeyId) -> LayerInstanceRef {
    LayerInstanceRef { layer: key.layer.to_string(), instance: key.name.to_owned() }
}

/// The provenance of every view and layer instance on `cx.doc`, in
/// document order, with model references registered on `cx` and each
/// layer's codec schema read from `graph`, the document's export. With
/// `digests` false, when the graph may not leave, every entry is listed
/// without its three digests.
pub(crate) fn layer_provenances(
    cx: &mut ProjectionCx<'_>,
    graph: &DocumentGraph,
    digests: bool,
) -> Vec<LayerProvenance> {
    let doc = cx.doc;
    let schemas: HashMap<(&str, &str), u16> =
        graph.layers.iter().map(|l| ((l.layer.as_str(), l.instance.as_str()), l.schema)).collect();
    let mut out = Vec::new();
    for key in doc.layers() {
        let model_ref = doc.provenance(key).map(|p| cx.register_provenance(p)).unwrap_or_default();
        let (reads, on) = match doc.provenance(key) {
            Some(p) => {
                (p.reads.iter().copied().map(layer_ref).collect(), p.on.iter().copied().map(layer_ref).collect())
            }
            None => (Vec::new(), Vec::new()),
        };
        let layer = key.layer.to_string();
        let codec_schema = if key.ty == TypeId::of::<View>() {
            u32::from(<View as Layer>::SCHEMA)
        } else {
            schemas.get(&(layer.as_str(), key.name)).copied().map(u32::from).unwrap_or_default()
        };
        let provenance_digest = doc.provenance_digest(key).filter(|_| digests);
        // Computed from the read layers' content digests here, not read
        // from the entry, so a document and its re-imported graph agree.
        let input_snapshot_digest = if !digests || key == Key::<View>::DEFAULT.id() {
            None
        } else {
            doc.provenance(key).and_then(|p| doc.compute_upstream_content_hash(&p.reads))
        };
        out.push(LayerProvenance {
            layer,
            instance: key.name.to_owned(),
            model_ref,
            provenance_digest: provenance_digest.map(|d| d.as_bytes().to_vec()).unwrap_or_default(),
            content_digest: doc.content_digest(key).filter(|_| digests).map(|d| d.as_bytes().to_vec()),
            input_snapshot_digest: input_snapshot_digest.map(|d| d.as_bytes().to_vec()),
            codec_schema,
            reads,
            on,
        });
    }
    out
}

fn digest_differs(what: &str, got: &[u8], want: &[u8; 32]) -> Result<(), Failure> {
    if got != want {
        return Err(bad(format!("the result's {what} is not the digest of this document")));
    }
    Ok(())
}

/// Refuses `input` unless it names exactly `doc`: the same raw bytes and
/// the same source text, by digest and length. An old result is never
/// read against changed input.
pub fn verify_input(input: Option<&InputIdentity>, doc: &Document) -> Result<(), Failure> {
    let Some(input) = input else {
        return Err(bad("the typed document carries no input identity"));
    };
    let expected = input_identity(doc, None);
    if input.raw_bytes_len != expected.raw_bytes_len {
        return Err(bad(format!(
            "the result was computed from {} raw bytes and this document has {}",
            input.raw_bytes_len, expected.raw_bytes_len
        )));
    }
    digest_differs("raw_bytes_digest", &input.raw_bytes_digest, expected.raw_bytes_digest[..].try_into().expect("32"))?;
    if input.source_text_len != expected.source_text_len {
        return Err(bad(format!(
            "the result was computed from {} source text bytes and this document has {}",
            input.source_text_len, expected.source_text_len
        )));
    }
    digest_differs(
        "source_text_digest",
        &input.source_text_digest,
        expected.source_text_digest[..].try_into().expect("32"),
    )
}

/// Refuses `result` unless it is the identity of exactly `graph`, the
/// TGRF bytes a consumer holds.
pub fn verify_result(result: Option<&ResultIdentity>, graph: &[u8]) -> Result<(), Failure> {
    let Some(result) = result else {
        return Err(bad("the typed document carries no result identity"));
    };
    if result.graph_bytes_len != graph.len() as u64 {
        return Err(bad(format!(
            "the result is of {} graph bytes and these are {}",
            result.graph_bytes_len,
            graph.len()
        )));
    }
    digest_differs("content_digest", &result.content_digest, graph_content_digest(graph).as_bytes())
}

/// Refuses a view digest that is not the digest of `text`.
pub fn verify_view_text(what: &str, digest: &[u8], text: &str) -> Result<(), Failure> {
    digest_differs(what, digest, ViewTextDigest::of_text(text).as_bytes())
}
