//! The result envelope's identities (issue #1907): golden digests for
//! every digest input, a changed input or view refused, the result bound
//! to its exact TGRF bytes, per-layer provenance, the caller's binding
//! copied and never invented, scoped references, and the envelope
//! refusals of RoundTripTypedDocument.

#[path = "typed_envelope/fixtures.rs"]
mod fixtures;

use turbo_analysis::export::{FORMAT_VERSION, export, graph_content_digest};
use turbo_analysis::{Document, Key, Provenance, Span, Spans, View, ViewId, ViewTextDigest};
use turbo_analysis_layers::Tokens;
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{InputBinding, InspectDocumentGraphRequest};
use turbo_kserve::results::document::codecs;
use turbo_kserve::results::typed::identity::{ENVELOPE_SCHEMA_VERSION, verify_input, verify_result};
use turbo_kserve::results::typed::scoped::{resolve, scoped_ref};
use turbo_kserve::results::typed::{document_to_typed, document_to_typed_bound, inspect_document_graph};

use fixtures::*;

const TEXT: &str = "Café 😀 北京";
/// `sha256sum` of the UTF-8 of TEXT, computed outside this crate.
const TEXT_SHA256: &str = "57c37aa82d516cf92c091b77d4d125d3cabc9afa652577734a01f8dcd0d84921";
/// `sha256sum` of the UTF-8 of "Cafe 😀 北京", the plain view's text.
const PLAIN_SHA256: &str = "5b8bd1292329a784bacf5fdf239e5514673470493af87f98ea7d623f65f5d472";

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// TEXT with a "plain" view that replaces "é" with "e" and tokens on it.
fn golden_document() -> Document {
    let mut doc = Document::new(TEXT);
    derive(&mut doc, ViewId::SOURCE, "plain", |b| {
        b.keep(3);
        b.replace(3 + 'é'.len_utf8(), "e");
    });
    let mut spans = Spans::<Tokens>::new(ViewId(1));
    spans.push(Span::new(0, 4));
    doc.add(&Key::<Tokens>::named("plain"), Tokens { spans, sentence: None }, Provenance::by("test:tokens")).unwrap();
    doc
}

#[test]
fn input_and_view_digests_are_the_sha256_of_the_exact_bytes() {
    let doc = golden_document();
    let typed = document_to_typed("golden", &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let input = typed.input.as_ref().unwrap();
    // A document made from text: its raw bytes are the source UTF-8.
    assert_eq!(hex(&input.raw_bytes_digest), TEXT_SHA256);
    assert_eq!(input.raw_bytes_len, TEXT.len() as u64);
    assert_eq!(hex(&input.source_text_digest), TEXT_SHA256);
    assert_eq!(input.source_text_len, TEXT.len() as u32);
    assert!(input.binding.is_none(), "no binding is invented");
    assert_eq!(ViewTextDigest::of_text(TEXT).to_hex(), TEXT_SHA256);

    let chain = turbo_kserve::results::typed::chain::WireChain::from_payload(typed.mapping.as_ref()).unwrap();
    assert_eq!(hex(&chain.view(ViewId::SOURCE).unwrap().text_digest), TEXT_SHA256);
    assert_eq!(hex(&chain.view(ViewId(1)).unwrap().text_digest), PLAIN_SHA256);
    assert_eq!(chain.view(ViewId(1)).unwrap().instance, "plain");
}

#[test]
fn the_result_identity_is_the_sha256_of_the_tgrf_bytes() {
    let doc = golden_document();
    let typed = document_to_typed("golden", &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let bytes = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let result = typed.result.as_ref().unwrap();
    assert_eq!(result.content_digest, turbo_sha256_of(&bytes));
    assert_eq!(result.result_id, hex(&result.content_digest));
    assert_eq!(result.graph_bytes_len, bytes.len() as u64);
    assert_eq!(result.graph_format_version, u32::from(FORMAT_VERSION));
    assert_eq!(result.envelope_schema_version, ENVELOPE_SCHEMA_VERSION);
    // The same bytes through InspectDocumentGraph give the same identity,
    // and the identity does not depend on the view or offsets asked for.
    let req = InspectDocumentGraphRequest {
        doc_id: "golden".into(),
        graph: bytes.clone(),
        view: 1,
        offsets: 1,
        binding: None,
        operation_key: String::new(),
    };
    let inspected = inspect_document_graph(&req).unwrap();
    assert_eq!(inspected.result, typed.result);
    assert_eq!(inspected.input, typed.input);
    assert_eq!(inspected.mapping, typed.mapping);
    assert_eq!(inspected.provenance, typed.provenance);
    verify_result(typed.result.as_ref(), &bytes).unwrap();
    let mut other = bytes.clone();
    other.push(0);
    assert!(verify_result(typed.result.as_ref(), &other).unwrap_err().message.contains("graph bytes"));
    let mut flipped = bytes;
    let last = flipped.len() - 1;
    flipped[last] ^= 1;
    assert!(verify_result(typed.result.as_ref(), &flipped).unwrap_err().message.contains("content_digest"));
    assert!(verify_result(None, &flipped).is_err());
}

fn turbo_sha256_of(bytes: &[u8]) -> Vec<u8> {
    graph_content_digest(bytes).as_bytes().to_vec()
}

#[test]
fn a_changed_input_is_detected_and_refused() {
    let doc = golden_document();
    let typed = document_to_typed("golden", &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    verify_input(typed.input.as_ref(), &doc).unwrap();
    verify_input(typed.input.as_ref(), &Document::new(TEXT)).unwrap();
    let same_len = Document::new("Café 😀 北亰");
    assert!(verify_input(typed.input.as_ref(), &same_len).unwrap_err().message.contains("raw_bytes_digest"));
    let other = Document::new("Cafe 😀 北京");
    assert!(verify_input(typed.input.as_ref(), &other).unwrap_err().message.contains("raw bytes"));
    assert!(verify_input(None, &doc).is_err());
    // Bytes decoded to the same text are a different input.
    let decoded = Document::decoded(
        b"Caf\xe9 \xf0\x9f\x98\x80 \xe5\x8c\x97\xe4\xba\xac".to_vec(),
        TEXT,
        turbo_analysis::Alignment::from_edits(vec![turbo_analysis::Edit {
            parent: Span::new(3, 4),
            child: Span::new(3, 5),
        }])
        .unwrap(),
    )
    .unwrap();
    assert!(verify_input(typed.input.as_ref(), &decoded).unwrap_err().message.contains("raw bytes"));
    let decoded_typed = document_to_typed("decoded", &decoded, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let input = decoded_typed.input.as_ref().unwrap();
    assert_eq!(input.raw_bytes_len, TEXT.len() as u64 - 1);
    assert_ne!(hex(&input.raw_bytes_digest), TEXT_SHA256);
    assert_eq!(hex(&input.source_text_digest), TEXT_SHA256);
    verify_input(decoded_typed.input.as_ref(), &decoded).unwrap();
}

#[test]
fn the_callers_binding_is_copied_as_given_and_never_invented() {
    let doc = golden_document();
    let binding = InputBinding {
        source_revision: "rev-7".into(),
        parse_artifact_id: "parse/42".into(),
        parse_artifact_digest: vec![9; 32],
    };
    let typed = document_to_typed_bound("b", &doc, ViewId::SOURCE, OffsetMode::Utf8, Some(&binding), "").unwrap();
    assert_eq!(typed.input.as_ref().unwrap().binding.as_ref(), Some(&binding));
    let bytes = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let req = InspectDocumentGraphRequest {
        doc_id: "b".into(),
        graph: bytes.clone(),
        view: 0,
        offsets: 0,
        binding: Some(binding.clone()),
        operation_key: String::new(),
    };
    assert_eq!(inspect_document_graph(&req).unwrap().input.unwrap().binding, Some(binding));
    let unbound = InspectDocumentGraphRequest {
        doc_id: "b".into(),
        graph: bytes,
        view: 0,
        offsets: 0,
        binding: None,
        operation_key: String::new(),
    };
    assert_eq!(inspect_document_graph(&unbound).unwrap().input.unwrap().binding, None);
    let round = turbo_kserve::results::typed::round_trip_typed_document(&typed).unwrap();
    assert_eq!(round, typed);
}

#[test]
fn every_view_and_layer_has_its_provenance_digests() {
    let doc = golden_document();
    let typed = document_to_typed("p", &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let names: Vec<(&str, &str)> = typed.provenance.iter().map(|p| (p.layer.as_str(), p.instance.as_str())).collect();
    assert_eq!(names, vec![("turbo:view", ""), ("turbo:view", "plain"), ("turbo:tokens", "plain")]);
    let view_key = Key::<View>::named("plain").id();
    let tok_key = Key::<Tokens>::named("plain").id();
    let plain = &typed.provenance[1];
    assert_eq!(plain.provenance_digest, doc.provenance_digest(view_key).unwrap().as_bytes().to_vec());
    assert_eq!(plain.content_digest, doc.content_digest(view_key).map(|d| d.as_bytes().to_vec()));
    assert_eq!(plain.codec_schema, 1);
    let tok = &typed.provenance[2];
    assert_eq!(tok.provenance_digest, doc.provenance_digest(tok_key).unwrap().as_bytes().to_vec());
    assert_eq!(tok.content_digest, doc.content_digest(tok_key).map(|d| d.as_bytes().to_vec()));
    assert_eq!(tok.input_snapshot_digest, doc.upstream_content_hash(tok_key).map(|d| d.as_bytes().to_vec()));
    assert_eq!(tok.codec_schema, u32::from(<Tokens as turbo_analysis::Layer>::SCHEMA));
    let model = typed.models.iter().find(|m| m.model_ref == tok.model_ref).unwrap();
    assert_eq!(model.annotator, "test:tokens");
    assert!(typed.provenance.iter().all(|p| p.provenance_digest.len() == 32));
    assert_eq!(typed.provenance[0].input_snapshot_digest, None, "the source view reads nothing");
    // The token layer is natively on the plain view, whatever view it is shown in.
    let layer = typed.layers.iter().find(|l| l.id == "turbo:tokens").unwrap();
    assert_eq!(layer.native_view, Some(1));
    assert_eq!(typed.view, 0);
}

#[test]
fn scoped_references_survive_the_message_and_refuse_another_result() {
    let doc = golden_document();
    let typed = document_to_typed("s", &doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let layer = typed.layers.iter().find(|l| l.id == "turbo:tokens").unwrap();
    let r = scoped_ref(&typed, layer, 0).unwrap();
    assert_eq!(r.result_id, typed.result.as_ref().unwrap().result_id);
    assert_eq!((r.layer.as_str(), r.instance.as_str(), r.index), ("turbo:tokens", "plain", 0));
    assert_eq!((r.view_instance.as_str(), hex(&r.view_text_digest).as_str()), ("plain", PLAIN_SHA256));
    let (found, ann) = resolve(&typed, &r).unwrap();
    assert_eq!(found.id, "turbo:tokens");
    assert_eq!(ann, &layer.annotations[0]);
    assert!(scoped_ref(&typed, layer, 1).unwrap_err().message.contains("outside"));

    // Another document, even with the same layers, is another result.
    let mut other_doc = Document::new("Café 😀 北亰");
    derive(&mut other_doc, ViewId::SOURCE, "plain", |b| {
        b.keep(3);
        b.replace(3 + 'é'.len_utf8(), "e");
    });
    let mut spans = Spans::<Tokens>::new(ViewId(1));
    spans.push(Span::new(0, 4));
    other_doc
        .add(&Key::<Tokens>::named("plain"), Tokens { spans, sentence: None }, Provenance::by("test:tokens"))
        .unwrap();
    let other = document_to_typed("s", &other_doc, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    assert!(resolve(&other, &r).unwrap_err().message.contains("another result"));
    let mut same_result = r.clone();
    same_result.result_id = other.result.as_ref().unwrap().result_id.clone();
    assert!(resolve(&other, &same_result).unwrap_err().message.contains("not the text"));
    let mut wrong_view = r.clone();
    wrong_view.view_instance = "nfc".into();
    assert!(resolve(&typed, &wrong_view).unwrap_err().message.contains("on view"));
}

#[test]
fn the_envelope_refusals_of_round_trip() {
    let doc = sibling_views();
    let typed = document_to_typed("r", &doc, ViewId(2), OffsetMode::Utf16).unwrap();
    let round = |edit: &dyn Fn(&mut turbo_kserve::proto::result::TypedDocumentResult)| {
        let mut m = typed.clone();
        edit(&mut m);
        turbo_kserve::results::typed::round_trip_typed_document(&m).unwrap_err().message
    };
    turbo_kserve::results::typed::round_trip_typed_document(&typed).unwrap();
    assert!(round(&|m| m.input = None).contains("input identity"));
    assert!(round(&|m| m.result = None).contains("result identity"));
    assert!(round(&|m| m.input.as_mut().unwrap().raw_bytes_digest.truncate(31)).contains("not the 32"));
    assert!(round(&|m| m.result.as_mut().unwrap().result_id.push('0')).contains("hexadecimal"));
    assert!(round(&|m| m.result.as_mut().unwrap().envelope_schema_version = 9).contains("envelope schema"));
    assert!(round(&|m| m.result.as_mut().unwrap().graph_format_version = 9).contains("graph format"));
    assert!(round(&|m| m.provenance.remove(0).layer.clear()).contains("no provenance entry"));
    assert!(round(&|m| m.provenance[2].model_ref = "m99".into()).contains("unknown model_ref"));
    assert!(round(&|m| m.provenance[2].content_digest = Some(vec![1; 2])).contains("content_digest"));
    assert!(round(&|m| m.provenance.push(m.provenance[0].clone())).contains("duplicate provenance"));
    assert!(round(&|m| m.layers[0].native_view = Some(7)).contains("native_view"));
    // raw_text that is not the target view's text, by length and by digest.
    assert!(round(&|m| m.raw_text.push('x')).contains("bytes and the mapping"));
    assert!(round(&|m| m.raw_text = m.raw_text.replacen('H', "h", 1)).contains("text_digest"));
    assert!(round(&|m| m.views.pop().map(|_| ()).unwrap()).contains("not in views"));
    // The views list and the mapping must agree on every parent.
    assert!(round(&|m| m.views[2].parent = "ascii".into()).contains("has parent"));
    assert!(round(&|m| m.views[0].parent = "ascii".into()).contains("has parent"));
}

#[test]
fn the_input_identity_is_tied_to_the_source_view_whatever_view_is_shown() {
    let doc = sibling_views();
    for view in [0u16, 1, 2, 3] {
        let typed = document_to_typed("r", &doc, ViewId(view), OffsetMode::Utf8).unwrap();
        turbo_kserve::results::typed::round_trip_typed_document(&typed).unwrap();
        let round = |edit: &dyn Fn(&mut turbo_kserve::proto::result::TypedDocumentResult)| {
            let mut m = typed.clone();
            edit(&mut m);
            turbo_kserve::results::typed::round_trip_typed_document(&m).unwrap_err().message
        };
        assert!(round(&|m| m.input.as_mut().unwrap().source_text_digest = vec![0; 32]).contains("source_text_digest"));
        assert!(round(&|m| m.input.as_mut().unwrap().source_text_len = 42).contains("source_text_len"));
    }
    let simple = document_to_typed("abc", &Document::new("abc"), ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let mut zeroed = simple.clone();
    zeroed.input.as_mut().unwrap().source_text_digest = vec![0; 32];
    assert!(turbo_kserve::results::typed::round_trip_typed_document(&zeroed).is_err());
    let mut forty_two = simple;
    forty_two.input.as_mut().unwrap().source_text_len = 42;
    assert!(turbo_kserve::results::typed::round_trip_typed_document(&forty_two).is_err());
}
