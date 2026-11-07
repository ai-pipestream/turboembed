//! The typed projection and validation of PII mentions (#1906): spans map
//! between views through their lowest common ancestor, as
//! `Document::map_span` maps them; a mention whose text the target view
//! deletes keeps an empty anchor; a PII payload needs a kind the layer can
//! reference the layer can resolve and a known confidence tier; and `document_to_typed` refuses to
//! emit a message its own validator would refuse.

use turbo_analysis::{
    Alignment, Analyzer, Document, Key, Provenance, Span, Spans, Tag, TagSet, View, ViewBuilder, ViewId,
};
use turbo_analysis_layers::{Categories, QualityMetrics, TextQuality, Tokens};
use turbo_kserve::api::TURBO_E_INVALID_ARGUMENT;
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::pii_kind_ref::Kind;
use turbo_kserve::proto::result::pii_value::Form;
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::proto::result::{
    PiiKeyRef, PiiKindRef, PiiMention as WirePii, PiiValue, PiiValueCiphertext, PiiValueToken, TypedDocumentResult,
    TypedLayer,
};
// The projection itself is under test, so the text is asked for (#1904).
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::results::typed::{document_to_typed, document_to_typed_with, validate_typed_document};
use turbo_pii::{Pii, PiiConfidence, PiiDetector, PiiKind, PiiMention};

/// The wire spans of every PII annotation, in layer order.
fn pii_spans(wire: &TypedDocumentResult) -> Vec<Vec<(u32, u32)>> {
    wire.layers
        .iter()
        .flat_map(|l| &l.annotations)
        .filter(|a| matches!(&a.value, Some(Value::Pii(_))))
        .map(|a| a.spans.iter().map(|s| (s.start, s.end)).collect())
        .collect()
}

/// Adds the view `name` derived from `parent` as a builder finished it.
fn add_view(doc: &mut Document, name: &'static str, parent: &View, (text, a): (String, Alignment)) -> ViewId {
    doc.add(&Key::<View>::named(name), View::derived(parent, text, a), Provenance::by("test:view")).unwrap();
    doc.get_key(&Key::<View>::named(name)).unwrap().id()
}

/// Source "n=<3 Arabic-Indic digits>;" (the digits are bytes 2..8); `v1`
/// replaces all three digits with "123" in one edit; `v2` and `v3` are
/// siblings under `v1` that each add a bracket. A mention over the middle
/// digit of `v2` lies inside the lossy source edit but not inside any edit
/// between `v1` and either sibling.
fn sibling_views() -> (Document, ViewId, ViewId) {
    let mut doc = Document::new("n=\u{661}\u{662}\u{663};");
    let v0 = doc.get::<View>().unwrap().clone();
    let mut b1 = ViewBuilder::new(v0.text());
    b1.keep(2);
    b1.replace(8, "123");
    add_view(&mut doc, "v1", &v0, b1.finish());
    let v1 = doc.get_key(&Key::<View>::named("v1")).unwrap().clone();
    let mut b2 = ViewBuilder::new(v1.text());
    b2.replace(0, "[");
    let v2 = add_view(&mut doc, "v2", &v1, b2.finish());
    let mut b3 = ViewBuilder::new(v1.text());
    b3.keep(5);
    b3.replace(6, "];");
    let v3 = add_view(&mut doc, "v3", &v1, b3.finish());
    (doc, v2, v3)
}

#[test]
fn typed_projection_maps_sibling_views_through_their_lowest_common_ancestor() {
    let (mut doc, v2, v3) = sibling_views();
    assert_eq!((doc.text(v2), doc.text(v3)), ("[n=123;", "n=123];"));
    let mut pii = Pii::new(v2);
    pii.push(PiiMention {
        span: Span::new(4, 5),
        kind: PiiKind::Phone,
        confidence: PiiConfidence::Format,
        normalized: "2",
    });
    doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("test:pii")).unwrap();
    assert_eq!(doc.map_span(v2, v3, Span::new(4, 5)).unwrap(), Span::new(3, 4));

    let wire = document_to_typed_with("lca", &doc, v3, OffsetMode::Utf8, &Disclosure::FULL).unwrap();
    assert_eq!(pii_spans(&wire), vec![vec![(3, 4)]], "the exact digit, not the whole source edit");
    assert_eq!(&wire.raw_text[3..4], "2");
    // Every offset mode agrees: the text before the digit is ASCII.
    for mode in [OffsetMode::Utf16, OffsetMode::CodePoint] {
        let wire = document_to_typed_with("lca", &doc, v3, mode, &Disclosure::FULL).unwrap();
        assert_eq!(pii_spans(&wire), vec![vec![(3, 4)]], "{mode:?}");
    }
}

#[test]
fn a_layer_on_a_child_view_projects_onto_its_parent_without_a_source_detour() {
    // Tokens over v2's "123" (bytes 3..6) land on v1's "123" (2..5) through
    // v1 itself, and each single digit of v2 stays a single digit of v1.
    let (mut doc, v2, _) = sibling_views();
    let v1 = doc.get_key(&Key::<View>::named("v1")).unwrap().id();
    let mut spans = Spans::new(v2);
    for s in [Span::new(3, 4), Span::new(4, 5), Span::new(5, 6)] {
        spans.push(s);
    }
    doc.add(&Key::<Tokens>::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("test:tokens")).unwrap();
    let wire = document_to_typed_with("parent", &doc, v1, OffsetMode::Utf8, &Disclosure::FULL).unwrap();
    let tokens = wire.layers.iter().find(|l| l.id == "turbo:tokens").unwrap();
    let got: Vec<Vec<(u32, u32)>> =
        tokens.annotations.iter().map(|a| a.spans.iter().map(|s| (s.start, s.end)).collect()).collect();
    assert_eq!(got, vec![vec![(2, 3)], vec![(3, 4)], vec![(4, 5)]]);
}

/// A detected e-mail address in the source, and a view that deletes it.
fn deleted_mention() -> (Document, ViewId) {
    let analyzer = Analyzer::builder().add(PiiDetector::cursor()).build().unwrap();
    let mut docs = [Document::new("mail jane@example.com end")];
    for result in analyzer.run(&mut docs) {
        result.unwrap();
    }
    let [mut doc] = docs;
    let v0 = doc.get::<View>().unwrap().clone();
    assert_eq!(v0.id(), ViewId::SOURCE);
    let mut b = ViewBuilder::new(v0.text());
    b.keep(5);
    b.replace(21, "");
    let deleted = add_view(&mut doc, "deleted", &v0, b.finish());
    assert_eq!(doc.text(deleted), "mail  end");
    (doc, deleted)
}

#[test]
fn a_deleted_mention_projects_to_an_empty_anchor_that_validates() {
    let (doc, deleted) = deleted_mention();
    let wire = document_to_typed_with("anchor", &doc, deleted, OffsetMode::Utf8, &Disclosure::FULL).unwrap();
    assert_eq!(wire.raw_text, "mail  end");
    assert_eq!(pii_spans(&wire), vec![vec![(5, 5)]], "an empty anchor where the mention was");
    validate_typed_document(&wire).unwrap();
}

/// A change to one wire PII payload.
type Edit = fn(&mut WirePii);

/// The source projection of the detected address, with its one PII
/// annotation's kind and tier set by `edit`.
fn pii_message(edit: impl FnOnce(&mut WirePii)) -> TypedDocumentResult {
    let (doc, _) = deleted_mention();
    let mut wire = document_to_typed_with("pii", &doc, ViewId::SOURCE, OffsetMode::Utf8, &Disclosure::FULL).unwrap();
    let ann = wire.layers.iter_mut().flat_map(|l| &mut l.annotations).find(|a| matches!(a.value, Some(Value::Pii(_))));
    let Some(Value::Pii(m)) = &mut ann.expect("a pii annotation").value else { unreachable!() };
    edit(m);
    wire
}

/// A kind reference to built-in id `id`.
fn builtin(id: i32) -> Option<PiiKindRef> {
    Some(PiiKindRef { kind: Some(Kind::Builtin(id)) })
}

/// A kind reference to custom index `index`.
fn custom(index: u32) -> Option<PiiKindRef> {
    Some(PiiKindRef { kind: Some(Kind::Custom(index)) })
}

/// The pii layer of `wire`.
fn pii_layer(wire: &mut TypedDocumentResult) -> &mut TypedLayer {
    wire.layers.iter_mut().find(|l| l.id == "turbo:pii").expect("a pii layer")
}

/// The error of `wire`, held to carry no planted value.
#[track_caller]
fn refused(what: &str, wire: &TypedDocumentResult, secrets: &[&str]) -> String {
    let err = validate_typed_document(wire).expect_err(what);
    assert_eq!(err.code, TURBO_E_INVALID_ARGUMENT, "{what}");
    for secret in secrets.iter().chain(&["jane@example.com"]) {
        assert!(!err.message.contains(secret), "{what}: the error echoes input: {}", err.message);
    }
    err.message
}

/// Sets the retired string kind, which is never populated.
#[allow(deprecated)]
fn retired_kind(m: &mut WirePii) {
    m.kind = "email".to_owned();
}

/// Sets the retired string tier, which is never populated.
#[allow(deprecated)]
fn retired_tier(m: &mut WirePii) {
    m.confidence_tier = "format".to_owned();
}

#[test]
fn pii_payloads_need_a_known_kind_ref_and_a_known_confidence_tier() {
    validate_typed_document(&pii_message(|_| {})).unwrap();
    for id in [1, 5, 30] {
        validate_typed_document(&pii_message(|m| m.kind_ref = builtin(id))).unwrap_or_else(|e| panic!("{id}: {e:?}"));
    }
    for tier in PiiConfidence::ALL {
        let tier = tier.id() as i32;
        validate_typed_document(&pii_message(|m| m.confidence = tier)).unwrap();
    }
    let refusals: [(&str, Edit); 10] = [
        ("no kind", |m| m.kind_ref = None),
        ("an empty kind ref", |m| m.kind_ref = Some(PiiKindRef { kind: None })),
        ("unspecified kind", |m| m.kind_ref = builtin(0)),
        ("kind past the table", |m| m.kind_ref = builtin(31)),
        ("negative kind", |m| m.kind_ref = builtin(-7)),
        ("custom kind with no table", |m| m.kind_ref = custom(0)),
        ("unspecified tier", |m| m.confidence = 0),
        ("unknown tier", |m| m.confidence = 5),
        ("a retired string kind", retired_kind),
        ("a retired string tier", retired_tier),
    ];
    for (what, edit) in refusals {
        let message = refused(what, &pii_message(edit), &[]);
        // The error names the layer, the row and its span, never the value.
        assert!(message.contains("turbo:pii[0]") && message.contains("5..21"), "{what}: {message}");
    }
}

/// A value of `form`, or `None` for a value with no form.
fn value(form: Option<Form>) -> Option<PiiValue> {
    Some(PiiValue { form })
}

/// A key reference no test resolves.
fn key() -> Option<PiiKeyRef> {
    Some(PiiKeyRef { id: "planted-key-id".to_owned(), version: 3 })
}

#[test]
fn a_pii_value_is_withheld_a_keyed_token_or_a_clear_value() {
    // The projection writes `withheld`, which validates (#1904).
    let wire = pii_message(|_| {});
    let mut layer = wire.layers.iter().filter(|l| l.id == "turbo:pii").flat_map(|l| &l.annotations);
    let Some(Value::Pii(m)) = &layer.next().expect("a pii annotation").value else { panic!("a pii annotation") };
    assert!(matches!(m.value, Some(PiiValue { form: Some(Form::Withheld(_)) })), "{:?}", m.value);
    validate_typed_document(&wire).unwrap();
    // A release (#1909) writes a 32-byte token naming its key, or the
    // value in clear; both validate.
    let token = |m: &mut WirePii| m.value = value(Some(Form::Token(PiiValueToken { token: vec![7; 32], key: key() })));
    validate_typed_document(&pii_message(token)).unwrap();
    let clear = |m: &mut WirePii| m.value = value(Some(Form::Clear("planted.clear@example.com".to_owned())));
    validate_typed_document(&pii_message(clear)).unwrap();
    // Everything else is refused, and the error names the form, never the
    // token, the ciphertext or the key.
    let secrets = ["planted-token", "planted-cipher", "planted-key-id", "planted-scheme"];
    let refusals: [(&str, Edit); 6] = [
        ("no value", |m| m.value = None),
        ("a value with no form", |m| m.value = value(None)),
        ("an empty clear value", |m| m.value = value(Some(Form::Clear(String::new())))),
        ("a short token", |m| {
            m.value = value(Some(Form::Token(PiiValueToken { token: b"planted-token".to_vec(), key: key() })))
        }),
        ("a token without a key", |m| {
            m.value = value(Some(Form::Token(PiiValueToken { token: vec![7; 32], key: None })))
        }),
        ("a ciphertext", |m| {
            m.value = value(Some(Form::Ciphertext(PiiValueCiphertext {
                ciphertext: b"planted-cipher".to_vec(),
                key: key(),
                scheme: "planted-scheme".to_owned(),
            })))
        }),
    ];
    for (what, edit) in refusals {
        let message = refused(what, &pii_message(edit), &secrets);
        assert!(message.contains("turbo:pii[0]") && message.contains("5..21"), "{what}: {message}");
        assert!(message.contains("pii mention value"), "{what}: {message}");
    }
}

#[test]
fn custom_kind_refs_index_the_layers_own_table() {
    let with_table = |names: &[&str], edit: Edit| {
        let mut wire = pii_message(edit);
        pii_layer(&mut wire).custom_pii_kinds = names.iter().map(|n| (*n).to_owned()).collect();
        wire
    };
    validate_typed_document(&with_table(&["badge", "ticket"], |m| m.kind_ref = custom(1))).unwrap();
    refused("index past the table", &with_table(&["badge", "ticket"], |m| m.kind_ref = custom(2)), &["badge"]);
    let long = "q".repeat(65);
    let tables: [(&str, Vec<&str>); 4] = [
        ("an empty name", vec!["badge", ""]),
        ("a name over 64 bytes", vec![long.as_str()]),
        ("a built-in name", vec!["email"]),
        ("a duplicate name", vec!["secret-badge", "secret-badge"]),
    ];
    for (what, names) in tables {
        let message = refused(what, &with_table(&names, |_| {}), &["secret-badge", "qqqq"]);
        assert!(message.contains("custom_pii_kinds"), "{what}: {message}");
    }
    let full: Vec<String> = (0..257).map(|i| format!("k{i}")).collect();
    let mut wire = pii_message(|_| {});
    pii_layer(&mut wire).custom_pii_kinds = full;
    refused("257 names", &wire, &["k256"]);
}

#[test]
fn pii_payloads_and_custom_tables_belong_to_the_pii_layer() {
    // Another layer may neither carry a custom-kind table nor a pii mention.
    let elsewhere = |table: bool| {
        let mut wire = pii_message(|_| {});
        let mut other = pii_layer(&mut wire).clone();
        other.id = "turbo:entities".to_owned();
        if table {
            other.annotations.clear();
            other.custom_pii_kinds = vec!["badge".to_owned()];
        }
        // Its own provenance entry, so the envelope (#1907) lists it.
        let mut entry = wire.provenance.iter().find(|p| p.layer == "turbo:pii").unwrap().clone();
        entry.layer = other.id.clone();
        wire.provenance.push(entry);
        wire.layers.push(other);
        wire
    };
    assert!(refused("a table on another layer", &elsewhere(true), &["badge"]).contains("custom_pii_kinds"));
    assert!(refused("a mention on another layer", &elsewhere(false), &[]).contains("outside a turbo:pii layer"));
    // A pii mention needs a span.
    let mut wire = pii_message(|_| {});
    let ann = pii_layer(&mut wire).annotations.first_mut().unwrap();
    ann.spans.clear();
    assert!(refused("no span", &wire, &[]).contains("no span"));
}

#[test]
fn document_to_typed_refuses_to_emit_a_message_its_validator_refuses() {
    // A provenance with no annotator projects to a model table entry that
    // the wire validator refuses; with debug assertions on, as in tests,
    // the projection refuses it itself. A release build skips that check,
    // and the message is refused where it is validated.
    let mut doc = Document::new("mail jane@example.com end");
    let mut pii = Pii::new(ViewId::SOURCE);
    pii.push(PiiMention {
        span: Span::new(5, 21),
        kind: PiiKind::Email,
        confidence: PiiConfidence::Format,
        normalized: "jane@example.com",
    });
    doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("")).unwrap();
    let projected = document_to_typed("anon", &doc, ViewId::SOURCE, OffsetMode::Utf8);
    let err = if cfg!(debug_assertions) {
        projected.map(|_| ()).unwrap_err()
    } else {
        validate_typed_document(&projected.unwrap()).unwrap_err()
    };
    assert_eq!(err.code, TURBO_E_INVALID_ARGUMENT);
    assert!(err.message.contains("empty annotator"), "{}", err.message);
}

#[test]
fn a_findings_only_result_lists_other_layers_bare_and_still_validates() {
    // A text quality summary (one annotation required) and a view-aligned
    // category layer (one per view) beside the PII layer (#1904).
    let mut doc = Document::new("mail jane@example.com end");
    let mut pii = Pii::new(ViewId::SOURCE);
    pii.push(PiiMention {
        span: Span::new(5, 21),
        kind: PiiKind::Email,
        confidence: PiiConfidence::Format,
        normalized: "jane@example.com",
    });
    doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("turbo-pii:pii")).unwrap();
    let quality = TextQuality::new(ViewId::SOURCE, doc.text(ViewId::SOURCE).len() as u32, QualityMetrics::default());
    doc.add(&Key::DEFAULT, quality, Provenance::by("test:quality")).unwrap();
    let mut cats = Categories::<View>::new(TagSet::new("sentiment", &["positive", "negative"]));
    cats.push(Tag(0), &[0.9, 0.1]);
    doc.add(&Key::DEFAULT, cats, Provenance::by("test:doccat")).unwrap();

    let full = document_to_typed_with("pii", &doc, ViewId::SOURCE, OffsetMode::Utf8, &Disclosure::FULL).unwrap();
    let findings =
        document_to_typed_with("pii", &doc, ViewId::SOURCE, OffsetMode::Utf8, &Disclosure::FINDINGS).unwrap();
    validate_typed_document(&findings).unwrap();
    for id in ["turbo:text_quality", "turbo:categories"] {
        let listed =
            |w: &TypedDocumentResult| w.layers.iter().find(|l| l.id.starts_with(id)).map(|l| l.annotations.len());
        assert!(listed(&full).is_some_and(|n| n > 0), "{id}: positive control");
        assert_eq!(listed(&findings), Some(0), "{id} is listed bare");
    }
    // A withheld result that still carries another layer's annotations is
    // refused: those annotations are the withheld text's words.
    let mut leaky = findings.clone();
    let donor = full.layers.iter().find(|l| l.id.starts_with("turbo:text_quality")).unwrap();
    let bare = leaky.layers.iter_mut().find(|l| l.id == donor.id).unwrap();
    bare.annotations = donor.annotations.clone();
    let message = refused("annotations beside a withheld text", &leaky, &[]);
    assert!(message.contains("while the text is withheld"), "{message}");
}
