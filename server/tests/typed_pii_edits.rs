//! The typed projection of `turbo:pii-edits` (#1902): a document with a
//! redacted view projects every edit with its replacement span, source
//! span in canonical UTF-8 bytes, covered mention rows, rule, strategy and
//! policy revision, and the validator refuses a broken edit.

use turbo_analysis::{Analyzer, Document};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::pii_edit_rule::Rule;
use turbo_kserve::proto::result::pii_kind_ref::Kind;
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::proto::result::{PiiEditAnnotation, PiiStrategy, TypedDocumentResult, TypedLayer};
use turbo_kserve::results::disclosure::Disclosure;
use turbo_kserve::results::typed::{document_to_typed, document_to_typed_with, validate_typed_document};
use turbo_pii::rewrite::{Pseudonymizer, REDACTED, RedactedView, RewritePolicy, StaticReplacer};
use turbo_pii::{PiiDetector, PiiEdits, PiiKind};

const TEXT: &str = "Zoë: jane@example.com, card 4111 1111 1111 1111.";

fn redacted_doc() -> Document {
    let policy = RewritePolicy::new(Pseudonymizer::default())
        .with(PiiKind::Card, StaticReplacer::new("[CARD]"))
        .with_revision(7);
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(RedactedView::new(policy).bound()).build().unwrap();
    let mut docs = vec![Document::new(TEXT)];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs.pop().unwrap()
}

fn edits(wire: &TypedDocumentResult) -> Vec<(Vec<(u32, u32)>, PiiEditAnnotation)> {
    let layer = wire.layers.iter().find(|l| l.id == "turbo:pii-edits").expect("the edits layer");
    layer
        .annotations
        .iter()
        .map(|a| match &a.value {
            Some(Value::PiiEdit(e)) => (a.spans.iter().map(|s| (s.start, s.end)).collect(), e.clone()),
            other => panic!("not an edit: {other:?}"),
        })
        .collect()
}

#[test]
fn edits_project_onto_the_redacted_view_and_the_source() {
    let doc = redacted_doc();
    let view = doc.get_key(&REDACTED).unwrap();
    assert_eq!(view.text(), "Zoë: EMAIL-1, card [CARD].");
    let native = doc.get_key(&turbo_analysis::Key::<PiiEdits>::named(REDACTED.name)).unwrap();
    let wire = document_to_typed("edits", &doc, view.id(), OffsetMode::Utf8).unwrap();
    let on_view = edits(&wire);
    assert_eq!(on_view.len(), 2);
    for ((spans, e), n) in on_view.iter().zip(native.edits()) {
        assert_eq!(spans, &vec![(n.span.start, n.span.end)]);
        let src = e.source.clone().unwrap();
        assert_eq!((src.start, src.end), (n.source.start, n.source.end));
        assert_eq!(e.members.len(), n.members.len());
        assert_eq!(e.policy_revision, 7);
        assert_eq!(&TEXT[src.start as usize..src.end as usize].is_empty(), &false);
    }
    assert_eq!(on_view[0].1.strategy, PiiStrategy::Pseudonym as i32);
    assert!(matches!(on_view[0].1.rule.as_ref().unwrap().rule, Some(Rule::DefaultRule(true))));
    assert_eq!(on_view[1].1.strategy, PiiStrategy::Static as i32);
    match &on_view[1].1.rule.as_ref().unwrap().rule {
        Some(Rule::Kind(k)) => assert_eq!(k.kind, Some(Kind::Builtin(i32::from(PiiKind::Card.builtin_id().unwrap())))),
        other => panic!("{other:?}"),
    }
    // On the source, each replacement covers exactly the mention it
    // replaced; UTF-16 offsets move the spans, never the source span.
    let on_source = edits(&document_to_typed("edits", &doc, turbo_analysis::ViewId::SOURCE, OffsetMode::Utf8).unwrap());
    for (spans, e) in &on_source {
        let src = e.source.clone().unwrap();
        assert_eq!(spans, &vec![(src.start, src.end)]);
    }
    let utf16 = edits(&document_to_typed("edits", &doc, view.id(), OffsetMode::Utf16).unwrap());
    assert_eq!(utf16[0].0[0].0 + 1, on_view[0].0[0].0, "the e with diaeresis is one UTF-16 unit, two bytes");
    assert_eq!(utf16[0].1.source, on_view[0].1.source);
}

#[test]
fn broken_edits_are_refused() {
    let doc = redacted_doc();
    let view = doc.get_key(&REDACTED).unwrap();
    let wire = document_to_typed("edits", &doc, view.id(), OffsetMode::Utf8).unwrap();
    validate_typed_document(&wire).unwrap();
    let at = wire.layers.iter().position(|l| l.id == "turbo:pii-edits").unwrap();
    let broken = |f: &dyn Fn(&mut PiiEditAnnotation)| {
        let mut w = wire.clone();
        if let Some(Value::PiiEdit(e)) = &mut w.layers[at].annotations[0].value {
            f(e);
        }
        validate_typed_document(&w).unwrap_err().message
    };
    assert!(broken(&|e| e.members.clear()).contains("covers no mention"));
    assert!(broken(&|e| e.members = vec![1, 0]).contains("out of order"));
    assert!(broken(&|e| e.members = vec![99]).contains("outside the covered"));
    assert!(broken(&|e| e.strategy = 0).contains("not a known strategy"));
    assert!(broken(&|e| e.rule = None).contains("rule is missing"));
    assert!(broken(&|e| e.source = None).contains("no source span"));
}

#[test]
fn edits_are_checked_across_the_layer() {
    let doc = redacted_doc();
    let view = doc.get_key(&REDACTED).unwrap();
    let wire = document_to_typed("edits", &doc, view.id(), OffsetMode::Utf8).unwrap();
    let at = wire.layers.iter().position(|l| l.id == "turbo:pii-edits").unwrap();
    assert!(!wire.layers[at].indexes_layer.is_empty(), "the projection names the covered pii layer");
    let broken = |f: &dyn Fn(&mut TypedLayer)| {
        let mut w = wire.clone();
        f(&mut w.layers[at]);
        validate_typed_document(&w).unwrap_err().message
    };
    let edit = |l: &mut TypedLayer, i: usize| match &mut l.annotations[i].value {
        Some(Value::PiiEdit(e)) => e.clone(),
        _ => unreachable!(),
    };
    let set = |l: &mut TypedLayer, i: usize, e: PiiEditAnnotation| l.annotations[i].value = Some(Value::PiiEdit(e));
    assert!(
        broken(&|l| {
            let mut e = edit(l, 1);
            e.policy_revision = 8;
            set(l, 1, e);
        })
        .contains("policy_revision differs")
    );
    assert!(
        broken(&|l| {
            let mut e = edit(l, 1);
            e.source = edit(l, 0).source;
            set(l, 1, e);
        })
        .contains("overlaps or precedes")
    );
    assert!(broken(&|l| l.annotations[1].value = None).contains("is not a pii edit"));
    // Without indexes_layer the pii layer of the same instance bounds the
    // members, so an out-of-range row is still refused.
    assert!(
        broken(&|l| {
            l.indexes_layer.clear();
            let mut e = edit(l, 1);
            e.members = vec![99];
            set(l, 1, e);
        })
        .contains("outside the covered")
    );
}

#[test]
fn tampered_source_spans_are_refused_against_the_chain() {
    let doc = redacted_doc();
    let view = doc.get_key(&REDACTED).unwrap();
    for (target, offsets) in [
        (view.id(), OffsetMode::Utf8),
        (view.id(), OffsetMode::Utf16),
        (turbo_analysis::ViewId::SOURCE, OffsetMode::Utf8),
        (turbo_analysis::ViewId::SOURCE, OffsetMode::CodePoint),
    ] {
        // The chain is checked against the text, so the text is asked for
        // (#1904): a withheld text in UTF-16 or code points cannot stand in.
        let wire = document_to_typed_with("edits", &doc, target, offsets, &Disclosure::FULL).unwrap();
        validate_typed_document(&wire).unwrap();
        let at = wire.layers.iter().position(|l| l.id == "turbo:pii-edits").unwrap();
        let tampered = |i: usize, start: u32, end: u32| {
            let mut w = wire.clone();
            if let Some(Value::PiiEdit(e)) = &mut w.layers[at].annotations[i].value {
                let s = e.source.as_mut().unwrap();
                (s.start, s.end) = (start, end);
            }
            validate_typed_document(&w).unwrap_err().message
        };
        let first = edits(&wire)[0].1.source.clone().unwrap();
        let second = edits(&wire)[1].1.source.clone().unwrap();
        let err = tampered(0, 1, u32::MAX);
        assert!(
            err.contains("disagrees with") || err.contains("overlaps or precedes"),
            "{target:?} {offsets:?}: {err}"
        );
        assert!(tampered(0, first.start + 1, first.end).contains("disagrees with"), "{target:?} {offsets:?}");
        if target == turbo_analysis::ViewId::SOURCE {
            // Only the source shows the parent: a longer last source span
            // keeps every gap but no longer matches its span there.
            assert!(tampered(1, second.start, second.end + 1).contains("disagrees with"));
        }
    }
}

#[test]
fn members_are_checked_against_the_mentions_on_the_source() {
    let doc = redacted_doc();
    let wire = document_to_typed("edits", &doc, turbo_analysis::ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let at = wire.layers.iter().position(|l| l.id == "turbo:pii-edits").unwrap();
    let mut w = wire.clone();
    let swap = |w: &mut TypedDocumentResult, i: usize, members: Vec<u32>| {
        if let Some(Value::PiiEdit(e)) = &mut w.layers[at].annotations[i].value {
            e.members = members;
        }
    };
    let m0 = edits(&wire)[0].1.members.clone();
    let m1 = edits(&wire)[1].1.members.clone();
    swap(&mut w, 0, m1.clone());
    swap(&mut w, 1, m0.clone());
    let err = validate_typed_document(&w).unwrap_err().message;
    assert!(err.contains("exactly the mentions"), "{err}");
}

#[test]
fn edits_stored_under_another_name_than_their_view_are_checked() {
    let policy = RewritePolicy::new(Pseudonymizer::default());
    let analyzer = Analyzer::builder()
        .add(PiiDetector::cursor())
        .step(RedactedView::new(policy).bound().write(1, "audit"))
        .build()
        .unwrap();
    let mut docs = vec![Document::new(TEXT)];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let doc = docs.pop().unwrap();
    let view = doc.get_key(&REDACTED).unwrap();
    for target in [view.id(), turbo_analysis::ViewId::SOURCE] {
        let wire = document_to_typed("audit", &doc, target, OffsetMode::Utf8).unwrap();
        validate_typed_document(&wire).unwrap();
        let at = wire.layers.iter().position(|l| l.id == "turbo:pii-edits").unwrap();
        assert_eq!(wire.layers[at].instance, "audit");
        let mut w = wire.clone();
        if let Some(Value::PiiEdit(e)) = &mut w.layers[at].annotations[0].value {
            e.source = Some(turbo_kserve::proto::result::SpanFragment::exact(1, u32::MAX));
        }
        let err = validate_typed_document(&w).unwrap_err().message;
        assert!(err.contains("disagrees with") || err.contains("overlaps or precedes"), "{target:?}: {err}");
        // The first source moved by one keeps the layer ordered; only the
        // chain sees it.
        let mut w = wire.clone();
        if let Some(Value::PiiEdit(e)) = &mut w.layers[at].annotations[0].value {
            e.source.as_mut().unwrap().start += 1;
        }
        assert!(validate_typed_document(&w).unwrap_err().message.contains("disagrees with"), "{target:?}");
    }
}

#[test]
fn many_edits_are_matched_to_their_mentions() {
    let text = "a@b.io ".repeat(5000);
    let policy = RewritePolicy::new(StaticReplacer::new("[E]"));
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(RedactedView::new(policy).bound()).build().unwrap();
    let mut docs = vec![Document::new(text.as_str())];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let doc = docs.pop().unwrap();
    let wire = document_to_typed("many", &doc, turbo_analysis::ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    let at = wire.layers.iter().position(|l| l.id == "turbo:pii-edits").unwrap();
    assert_eq!(wire.layers[at].annotations.len(), 5000);
    validate_typed_document(&wire).unwrap();
    let mut w = wire.clone();
    if let Some(Value::PiiEdit(e)) = &mut w.layers[at].annotations[4999].value {
        e.members = vec![4998];
    }
    assert!(validate_typed_document(&w).unwrap_err().message.contains("exactly the mentions"));
}
