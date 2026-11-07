//! What the Rust consumer does with wire content it does not know
//! (issue #1910): an unknown enum number that a decision reads is
//! refused by `RoundTripTypedDocument`, never downgraded to a known value
//! or to success, and an unknown field is dropped by the `prost` decode
//! and absent from the re-encode, so a TurboEmbed endpoint never
//! forwards what it did not understand. Generated Java keeps both; the
//! Java conformance cases in clients/java show that side.

#[path = "typed_envelope/fixtures.rs"]
mod fixtures;

use prost::Message;
use turbo_analysis::ViewId;
use turbo_kserve::api::TURBO_E_INVALID_ARGUMENT;
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{LayerError, LayerErrorKind, LayerWireStatus, TypedDocumentResult};
use turbo_kserve::results::typed::{document_to_typed, round_trip_typed_document};

use fixtures::sibling_views;

const UNKNOWN: i32 = 99;

type Edit = dyn Fn(&mut TypedDocumentResult);

fn typed() -> TypedDocumentResult {
    document_to_typed("u", &sibling_views(), ViewId(2), OffsetMode::Utf8).unwrap()
}

/// The refusal of `edit` applied to a message the server accepts as is.
fn refused(edit: &Edit) -> String {
    let mut m = typed();
    round_trip_typed_document(&m).expect("the untouched message is accepted");
    edit(&mut m);
    let e = round_trip_typed_document(&m).expect_err("the unknown value is refused");
    assert_eq!(e.code, TURBO_E_INVALID_ARGUMENT);
    e.message
}

#[test]
fn an_unknown_enum_number_is_refused_wherever_a_decision_reads_it() {
    let cases: [(&str, &Edit); 8] = [
        ("offsets", &|m| m.offsets = UNKNOWN),
        ("view status", &|m| m.views[0].status = UNKNOWN),
        ("layer status", &|m| m.layers[0].status = UNKNOWN),
        ("layer scope", &|m| m.layers[0].scope = UNKNOWN),
        ("layer error kind", &|m| {
            m.layers[0].status = LayerWireStatus::Failed as i32;
            m.layers[0].annotations.clear();
            m.layers[0].error =
                Some(LayerError { kind: UNKNOWN, turbo_code: 0, layer: None, annotator: String::new() });
        }),
        ("receipt status", &|m| m.receipt.as_mut().unwrap().status = UNKNOWN),
        ("receipt layer status", &|m| m.receipt.as_mut().unwrap().layers[0].status = UNKNOWN),
        ("receipt truncation", &|m| m.receipt.as_mut().unwrap().layers[0].truncations.push(UNKNOWN)),
    ];
    for (what, edit) in cases {
        let message = refused(edit);
        assert!(message.contains("99"), "{what}: the refusal names the number: {message}");
    }
    // A known kind with another kind's code is refused too: the code is
    // never taken over the kind or the kind over the code.
    let message = refused(&|m| {
        m.layers[0].status = LayerWireStatus::Failed as i32;
        m.layers[0].annotations.clear();
        m.layers[0].error = Some(LayerError {
            kind: LayerErrorKind::Declined as i32,
            turbo_code: 1,
            layer: None,
            annotator: String::new(),
        });
    });
    assert!(message.contains("turbo code"), "{message}");
}

/// Field 999, varint 7: tag `(999 << 3) | 0` as a two-byte varint, then
/// the value.
const UNKNOWN_FIELD: [u8; 3] = [0xb8, 0x3e, 0x07];

#[test]
fn an_unknown_field_is_dropped_on_decode_and_never_re_encoded() {
    let original = typed();
    let bytes = original.encode_to_vec();
    let mut with_unknown = bytes.clone();
    with_unknown.extend_from_slice(&UNKNOWN_FIELD);
    let decoded = TypedDocumentResult::decode(with_unknown.as_slice()).unwrap();
    assert_eq!(decoded, original, "the unknown field changes nothing the consumer sees");
    assert_eq!(decoded.encode_to_vec(), bytes, "the re-encode is the original bytes, without field 999");
    let answered = round_trip_typed_document(&decoded).unwrap();
    assert_eq!(answered.encode_to_vec(), bytes);
    assert!(!contains(&answered.encode_to_vec(), &UNKNOWN_FIELD));
}

/// An annotation whose value is an arm this build does not know (field
/// 200, length-delimited) decodes as an annotation without a value, and
/// a layer that carries one is refused: the unknown arm is never read
/// as some known value or as nothing to decide.
#[test]
fn an_unknown_value_arm_is_refused_not_read_as_nothing() {
    let mut m = typed();
    let layer = m.layers.iter_mut().find(|l| !l.annotations.is_empty()).unwrap();
    let mut bytes = layer.annotations[0].encode_to_vec();
    // tag (200 << 3) | 2 as a varint, length 2, then a two-byte message.
    bytes.extend_from_slice(&[0xc2, 0x0c, 0x02, 0x08, 0x01]);
    let mut decoded = turbo_kserve::proto::result::TypedAnnotation::decode(bytes.as_slice()).unwrap();
    assert!(decoded.value.is_some(), "the known arm is still there beside the unknown one");
    decoded.value = None;
    layer.annotations[0] = decoded;
    let e = round_trip_typed_document(&m).unwrap_err();
    assert_eq!(e.code, TURBO_E_INVALID_ARGUMENT);
    assert!(e.message.contains("missing its typed annotation value"), "{}", e.message);
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}
