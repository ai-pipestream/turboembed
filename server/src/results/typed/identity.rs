//! The result envelope's identities (issue #1907): what a result was
//! computed from (`InputIdentity`), what it is (`ResultIdentity`) and how
//! each layer and view was made (`LayerProvenance`), on a
//! `TypedDocumentResult`. Every digest is one of the document's own
//! SHA-256 digests (`ViewTextDigest`, `SourceBytesDigest`,
//! `ProvenanceDigest`, `ContentDigest`, `InputSnapshotDigest`) or the
//! SHA-256 of the TGRF bytes; nothing new is hashed here. The consumer
//! side, [`verify_input`] and [`verify_result`], refuses a result whose
//! input or graph bytes are not the ones it names.

use std::any::TypeId;
use std::collections::HashMap;

use turbo_analysis::export::{DocumentGraph, FORMAT_VERSION, graph_content_digest};
use turbo_analysis::{Document, Key, KeyId, Layer, View, ViewTextDigest};

use super::{ProjectionCx, bad};
use crate::api::Failure;
use crate::proto::result::{InputBinding, InputIdentity, LayerInstanceRef, LayerProvenance, ResultIdentity};

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
/// layer's codec schema read from `graph`, the document's export.
pub(crate) fn layer_provenances(cx: &mut ProjectionCx<'_>, graph: &DocumentGraph) -> Vec<LayerProvenance> {
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
        out.push(LayerProvenance {
            layer,
            instance: key.name.to_owned(),
            model_ref,
            provenance_digest: doc.provenance_digest(key).map(|d| d.as_bytes().to_vec()).unwrap_or_default(),
            content_digest: doc.content_digest(key).map(|d| d.as_bytes().to_vec()),
            // Computed from the read layers' content digests here, not read
            // from the entry, so a document and its re-imported graph agree.
            input_snapshot_digest: if key == Key::<View>::DEFAULT.id() {
                None
            } else {
                doc.provenance(key)
                    .and_then(|p| doc.compute_upstream_content_hash(&p.reads))
                    .map(|d| d.as_bytes().to_vec())
            },
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
