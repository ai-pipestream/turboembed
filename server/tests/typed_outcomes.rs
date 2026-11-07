//! Fragment outcomes on the wire (issue #1907, item 7): every projected
//! `SpanFragment` says whether it is the native span exactly, widened
//! over a lossy edit, or an empty anchor on deleted text, coalesced
//! fragments list their native members, the outcomes are the native
//! projection's and a consumer reproduces them from the mapping with
//! `WireChain` alone, outcomes stay on the right view with two derived
//! views, and the validator refuses an outcome that lies.

#[path = "typed_envelope/fixtures.rs"]
mod fixtures;

use turbo_analysis::{
    Analyzer, Document, EvidenceStep, IxRange, Key, Layer, Provenance, Span, SpanOutcome, Spans, Tag, TagSet, View,
    ViewId,
};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::typed_annotation::Value;
use turbo_kserve::proto::result::{FragmentMember, SpanFragment, SpanOutcome as Wire, TypedDocumentResult};
use turbo_kserve::results::MAX_FRAGMENTS;
use turbo_kserve::results::typed::chain::WireChain;
use turbo_kserve::results::typed::{document_to_typed, validate_typed_document};
use turbo_pii::rewrite::{MASKED, MaskedView, Pseudonymizer, REDACTED, RedactedView, RewritePolicy};
use turbo_pii::{PiiDetector, PiiEdits};

use fixtures::*;

const NAMES: [&str; 4] = ["o0", "o1", "o2", "o3"];

fn views_of(doc: &Document) -> Vec<ViewId> {
    let mut ids: Vec<ViewId> = doc
        .layers()
        .filter(|k| k.ty == std::any::TypeId::of::<View>())
        .filter_map(|k| doc.get_key(&Key::<View>::named(k.name)))
        .map(|v| v.id())
        .collect();
    ids.sort_by_key(|v| v.0);
    ids
}

/// A tokens layer on every view of `doc`: every span between two char
/// boundaries, every empty span at one, and two-fragment items of
/// adjacent characters with one between them, so that onto another view
/// fragments widen, delete, coalesce and touch in every way the view
/// allows.
fn tokens_everywhere(doc: &mut Document) {
    for v in views_of(doc) {
        let bounds: Vec<u32> =
            doc.text(v).char_indices().map(|(i, _)| i as u32).chain([doc.text(v).len() as u32]).collect();
        let mut spans = Spans::<Tokens>::new(v);
        for (i, &a) in bounds.iter().enumerate() {
            for &b in &bounds[i..] {
                spans.push(Span::new(a, b));
            }
        }
        for w in bounds.windows(4) {
            spans.push_fragments(&[Span::new(w[0], w[1]), Span::new(w[2], w[3])]);
        }
        let key = Key::<Tokens>::named(NAMES[v.0 as usize]);
        doc.add(&key, Tokens { spans, sentence: None }, Provenance::by("test:tokens")).unwrap();
    }
}

fn wire(o: SpanOutcome) -> i32 {
    match o {
        SpanOutcome::Exact => Wire::Exact as i32,
        SpanOutcome::Widened => Wire::Widened as i32,
        SpanOutcome::Deleted => Wire::Deleted as i32,
    }
}

/// The outcome a consumer derives from the chain alone: the native span
/// mapped to the target and back, stepwise.
fn from_chain(chain: &WireChain, from: ViewId, to: ViewId, s: Span) -> (Span, i32) {
    let mapped = chain.map_span(from, to, s).unwrap();
    let back = chain.map_span(to, from, mapped).unwrap();
    let outcome = if !s.is_empty() && mapped.is_empty() {
        Wire::Deleted
    } else if back == s {
        Wire::Exact
    } else {
        Wire::Widened
    };
    (mapped, outcome as i32)
}

/// Checks every fragment of every `spans` item in `msg`'s layer
/// `instance` against the native projection and against the chain.
fn check_layer<L: turbo_analysis::Layer>(
    doc: &Document,
    msg: &TypedDocumentResult,
    id: &str,
    instance: &str,
    spans: &Spans<L>,
) -> usize {
    let chain = WireChain::from_payload(msg.mapping.as_ref()).unwrap();
    let target = ViewId(msg.view as u16);
    let layer = msg.layers.iter().find(|l| l.id == id && l.instance == instance).expect("the layer");
    assert_eq!(layer.native_view, Some(u32::from(spans.view().0)), "{id}@{instance} native_view");
    assert_eq!(layer.annotations.len(), spans.len());
    let off = turbo_kserve::offsets::Offsetter::new(&msg.raw_text, OffsetMode::from_proto(msg.offsets).unwrap());
    let mut checked = 0;
    for (ix, ann) in layer.annotations.iter().enumerate() {
        let native: Vec<Span> = spans.fragments(turbo_analysis::Ix::new(ix as u32)).collect();
        let expected: Vec<SpanFragment> = if spans.view() == target {
            native
                .iter()
                .map(|f| {
                    let (start, end) = off.to_wire(&msg.raw_text, *f);
                    SpanFragment::exact(start, end)
                })
                .collect()
        } else {
            doc.project_fragments(spans.view(), target, native.iter().copied())
                .unwrap()
                .into_iter()
                .map(|p| {
                    let (start, end) = off.to_wire(&msg.raw_text, p.span);
                    let members = if p.is_coalesced() {
                        p.members.iter().map(|&(index, o)| FragmentMember { index, outcome: wire(o) }).collect()
                    } else {
                        Vec::new()
                    };
                    SpanFragment { start, end, outcome: wire(p.outcome), members }
                })
                .collect()
        };
        assert_eq!(ann.spans, expected, "{id}@{instance}[{ix}] onto view {}", msg.view);
        // The chain alone reproduces each native fragment's outcome and
        // position, member by member for a coalesced fragment.
        for (f_ix, frag) in ann.spans.iter().enumerate() {
            let wire_span = off.from_wire(&msg.raw_text, "frag", frag.start, frag.end).unwrap();
            if frag.members.is_empty() {
                // Not coalesced: exactly one native fragment, the one that
                // lands here (fragments stay in text order).
                let hits: Vec<(Span, i32)> = native
                    .iter()
                    .map(|&n| from_chain(&chain, spans.view(), target, n))
                    .filter(|(m, _)| *m == wire_span)
                    .collect();
                assert!(
                    hits.iter().any(|(_, o)| *o == frag.outcome),
                    "{id}[{ix}] fragment {f_ix}: {hits:?} vs {frag:?}"
                );
            } else {
                assert!(frag.members.len() > 1);
                let mut union: Option<Span> = None;
                for m in &frag.members {
                    let (mapped, outcome) = from_chain(&chain, spans.view(), target, native[m.index as usize]);
                    assert_eq!(outcome, m.outcome, "{id}[{ix}] member {}", m.index);
                    union = Some(match union {
                        None => mapped,
                        Some(u) => Span::new(u.start.min(mapped.start), u.end.max(mapped.end)),
                    });
                }
                assert_eq!(union, Some(wire_span), "{id}[{ix}] coalesced span");
            }
            checked += 1;
        }
    }
    checked
}

#[test]
fn every_fragment_outcome_is_the_native_projection_and_the_chain_reproduces_it() {
    let mut seen = [0usize; 3];
    for (name, mut doc) in fixture_documents() {
        tokens_everywhere(&mut doc);
        let views = views_of(&doc);
        for &target in &views {
            for mode in [OffsetMode::Utf8, OffsetMode::Utf16] {
                let msg = document_to_typed(name, &doc, target, mode).unwrap();
                validate_typed_document(&msg).unwrap();
                for &v in &views {
                    let key = Key::<Tokens>::named(NAMES[v.0 as usize]);
                    let tokens = doc.get_key(&key).unwrap();
                    check_layer(&doc, &msg, "turbo:tokens", NAMES[v.0 as usize], &tokens.spans);
                    let layer = msg.layers.iter().find(|l| l.instance == NAMES[v.0 as usize]).unwrap();
                    for frag in layer.annotations.iter().flat_map(|a| &a.spans) {
                        seen[frag.outcome as usize - 1] += 1;
                        if v == target {
                            assert_eq!(frag.outcome, Wire::Exact as i32, "{name}: own view is exact");
                            assert!(frag.members.is_empty());
                        }
                    }
                }
            }
        }
    }
    // Every outcome and coalescing occurred somewhere in the fixtures.
    assert!(seen.iter().all(|&n| n > 0), "exact, widened, deleted all seen: {seen:?}");
}

#[test]
fn coalesced_fragments_list_their_members_with_their_own_outcomes() {
    // "ab--cd" -> spaced "a b-c\u{200b}d" -> tight "ab-c\u{200b}d": a
    // two-fragment item of "a" and "b" on the source widens and joins.
    let mut doc = insertion_and_deletion();
    tokens_everywhere(&mut doc);
    let msg = document_to_typed("ins", &doc, ViewId(1), OffsetMode::Utf8).unwrap();
    let layer = msg.layers.iter().find(|l| l.instance == "o0").unwrap();
    let coalesced: Vec<&SpanFragment> =
        layer.annotations.iter().flat_map(|a| &a.spans).filter(|f| !f.members.is_empty()).collect();
    assert!(!coalesced.is_empty(), "something coalesces onto the spaced view");
    for f in &coalesced {
        assert!(f.members.windows(2).all(|w| w[0].index < w[1].index));
        let any = |o: Wire| f.members.iter().any(|m| m.outcome == o as i32);
        let merged = if any(Wire::Widened) {
            Wire::Widened
        } else if any(Wire::Exact) {
            Wire::Exact
        } else {
            Wire::Deleted
        };
        assert_eq!(f.outcome, merged as i32);
    }
    // A deleted anchor stays in a multi-fragment span, as a deletion.
    let deleted = layer.annotations.iter().flat_map(|a| &a.spans).filter(|f| f.outcome == Wire::Deleted as i32).count();
    assert!(deleted > 0, "the dashes deleted by the spaced view leave anchors");
    validate_typed_document(&msg).unwrap();
}

fn masked_and_redacted() -> Document {
    let text = "Mail jane@example.com or bob@example.org today.";
    let policy = RewritePolicy::new(Pseudonymizer::default());
    let analyzer = Analyzer::builder()
        .add(PiiDetector::cursor())
        .step(MaskedView::by_kind('*').unwrap().bound())
        .step(RedactedView::new(policy).bound())
        .build()
        .unwrap();
    let mut docs = vec![Document::new(text)];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let mut doc = docs.pop().unwrap();
    tokens_everywhere(&mut doc);
    doc
}

#[test]
fn two_derived_views_keep_their_outcomes_on_the_right_view() {
    let doc = masked_and_redacted();
    let masked = doc.get_key(&MASKED).unwrap().id();
    let redacted = doc.get_key(&REDACTED).unwrap().id();
    assert_ne!(masked, redacted);
    let edits = doc.get_key(&Key::<PiiEdits>::named(REDACTED.name)).unwrap();
    for target in [ViewId::SOURCE, masked, redacted] {
        let msg = document_to_typed("two", &doc, target, OffsetMode::Utf16).unwrap();
        validate_typed_document(&msg).unwrap();
        for v in views_of(&doc) {
            let tokens = doc.get_key(&Key::<Tokens>::named(NAMES[v.0 as usize])).unwrap();
            check_layer(&doc, &msg, "turbo:tokens", NAMES[v.0 as usize], &tokens.spans);
        }
        check_layer(&doc, &msg, "turbo:pii-edits", REDACTED.name, &edits.spans);
        let edits_layer = msg.layers.iter().find(|l| l.id == "turbo:pii-edits").unwrap();
        assert_eq!(edits_layer.native_view, Some(u32::from(redacted.0)));
        for a in &edits_layer.annotations {
            let Some(Value::PiiEdit(e)) = &a.value else { panic!("an edit") };
            // The source span is not projected: it is exact by definition.
            assert_eq!(e.source.as_ref().unwrap().outcome, Wire::Exact as i32);
            if target == redacted {
                assert!(a.spans.iter().all(|f| f.outcome == Wire::Exact as i32));
            }
        }
        // The edits' view is resolved through native_view, so the chain
        // checks hold with two derived views: a tampered source span is
        // refused on the redacted view and on the source alike.
        if target != masked {
            let mut bad = msg.clone();
            let l = bad.layers.iter_mut().find(|l| l.id == "turbo:pii-edits").unwrap();
            let Some(Value::PiiEdit(e)) = &mut l.annotations[0].value else { panic!("an edit") };
            e.source.as_mut().unwrap().start += 1;
            let err = validate_typed_document(&bad).unwrap_err().message;
            assert!(err.contains("disagrees with"), "onto view {}: {err}", msg.view);
        }
        // The redacted view's own tokens are exact only on the redacted
        // view: its replacements change lengths. The masked view's are
        // exact on the source too, since a mask keeps every byte's width,
        // and widen only through the redacted view's edits.
        let on_redacted = msg.layers.iter().find(|l| l.instance == NAMES[redacted.0 as usize]).unwrap();
        let all_exact = on_redacted.annotations.iter().flat_map(|a| &a.spans).all(|f| f.outcome == Wire::Exact as i32);
        assert_eq!(all_exact, target == redacted, "onto view {}", msg.view);
        let on_masked = msg.layers.iter().find(|l| l.instance == NAMES[masked.0 as usize]).unwrap();
        let masked_exact = on_masked.annotations.iter().flat_map(|a| &a.spans).all(|f| f.outcome == Wire::Exact as i32);
        assert_eq!(masked_exact, target != redacted, "onto view {}", msg.view);
    }
}

fn refused(msg: &TypedDocumentResult, edit: impl FnOnce(&mut TypedDocumentResult)) -> String {
    let mut m = msg.clone();
    edit(&mut m);
    validate_typed_document(&m).unwrap_err().message
}

#[test]
fn the_validator_refuses_an_outcome_that_lies() {
    let mut doc = insertion_and_deletion();
    tokens_everywhere(&mut doc);
    // Onto the spaced view: tokens on the source coalesce where a deleted
    // dash touches a kept letter; tokens on the spaced view are exact.
    let msg = document_to_typed("v", &doc, ViewId(1), OffsetMode::Utf8).unwrap();
    validate_typed_document(&msg).unwrap();
    let own = msg.layers.iter().position(|l| l.instance == "o1").unwrap();
    let other = msg.layers.iter().position(|l| l.instance == "o0").unwrap();
    let coalesced = msg.layers[other]
        .annotations
        .iter()
        .position(|a| a.spans.iter().any(|f| f.members.len() > 1))
        .expect("a coalesced fragment");
    fn frag(m: &mut TypedDocumentResult, l: usize, i: usize) -> &mut SpanFragment {
        let spans = &mut m.layers[l].annotations[i].spans;
        let at = spans.iter().position(|f| !f.members.is_empty()).unwrap_or(0);
        &mut spans[at]
    }
    assert!(refused(&msg, |m| frag(m, own, 1).outcome = 0).contains("has no outcome"));
    assert!(refused(&msg, |m| frag(m, own, 1).outcome = 99).contains("unknown outcome"));
    assert!(refused(&msg, |m| frag(m, own, 1).outcome = Wire::Widened as i32).contains("own view and not exact"));
    assert!(refused(&msg, |m| frag(m, own, 1).outcome = Wire::Deleted as i32).contains("deleted and not empty"));
    let one = |m: &mut TypedDocumentResult| {
        frag(m, own, 1).members = vec![FragmentMember { index: 0, outcome: Wire::Exact as i32 }];
    };
    assert!(refused(&msg, one).contains("lists one member"));
    assert!(refused(&msg, |m| frag(m, other, coalesced).members.reverse()).contains("ascending native order"));
    assert!(refused(&msg, |m| frag(m, other, coalesced).members[0].outcome = 0).contains("member 0 has no outcome"));
    // Members that merge to widened under a fragment that says exact.
    let merge = |m: &mut TypedDocumentResult| {
        let f = frag(m, other, coalesced);
        for member in &mut f.members {
            member.outcome = Wire::Widened as i32;
        }
        f.outcome = Wire::Exact as i32;
    };
    assert!(refused(&msg, merge).contains("not the one its members merge to"));
    // An empty fragment in a multi-fragment span is a deletion anchor or
    // nothing; and no native fragment belongs to two fragments.
    let empty_exact = |m: &mut TypedDocumentResult| {
        let end = frag(m, own, 1).end;
        m.layers[own].annotations[1].spans.push(SpanFragment::exact(end, end));
    };
    assert!(refused(&msg, empty_exact).contains("empty in a multi-fragment span"));
    let twice = |m: &mut TypedDocumentResult| {
        let f = frag(m, other, coalesced).clone();
        let end = f.end;
        let mut second = f;
        second.start = end;
        second.end = end;
        second.outcome = Wire::Deleted as i32;
        for member in &mut second.members {
            member.outcome = Wire::Deleted as i32;
        }
        m.layers[other].annotations[coalesced].spans.push(second);
    };
    assert!(refused(&msg, twice).contains("member of two fragments"));
    // Nested spans obey the same rules.
    let edits = masked_and_redacted();
    let redacted = edits.get_key(&REDACTED).unwrap().id();
    let msg = document_to_typed("e", &edits, redacted, OffsetMode::Utf8).unwrap();
    let unspecified = |m: &mut TypedDocumentResult| {
        let l = m.layers.iter_mut().find(|l| l.id == "turbo:pii-edits").unwrap();
        let Some(Value::PiiEdit(e)) = &mut l.annotations[0].value else { panic!("an edit") };
        e.source.as_mut().unwrap().outcome = 0;
    };
    assert!(refused(&msg, unspecified).contains("source has no outcome"));
}

/// One entity over the whole text with one evidence vote on the source.
fn entity_votes_doc() -> Document {
    let mut doc = Document::new("abc");
    let mut tokens = Spans::new(ViewId::SOURCE);
    tokens.push(Span::new(0, 3));
    doc.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: None }, Provenance::by("test:tokens")).unwrap();
    let key = Key::<Entities>::named("v");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 3));
    let entities = Entities {
        spans,
        tokens: vec![IxRange::new(0, 1)],
        kind: vec![Tag(0)],
        kinds: TagSet::new("ner", &["THING"]),
        score: vec![0.75],
    };
    let on = Provenance { on: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test:entity") };
    doc.add(&key, entities, on).unwrap();
    let step = EvidenceStep::new(key);
    let evidence = step.build(&doc).unwrap();
    doc.add(&step.output(), evidence, Provenance { on: vec![key.id()], ..Provenance::by("test:evidence") }).unwrap();
    doc
}

/// Codex on #1921: member lists reach the validator from an untrusted
/// `RoundTripTypedDocument` caller, so they are budgeted before they are
/// read and checked in linear time once they are.
#[test]
fn member_lists_are_budgeted_and_checked_in_linear_time() {
    let mut doc = insertion_and_deletion();
    tokens_everywhere(&mut doc);
    let msg = document_to_typed("v", &doc, ViewId(1), OffsetMode::Utf8).unwrap();
    let other = msg.layers.iter().position(|l| l.instance == "o0").unwrap();
    let coalesced = msg.layers[other]
        .annotations
        .iter()
        .position(|a| a.spans.iter().any(|f| f.members.len() > 1))
        .expect("a coalesced fragment");
    let members = |n: u32| -> Vec<FragmentMember> {
        (0..n).map(|index| FragmentMember { index, outcome: Wire::Exact as i32 }).collect()
    };
    let with = |m: &mut TypedDocumentResult, n: u32| {
        let f = m.layers[other].annotations[coalesced].spans.iter_mut().find(|f| !f.members.is_empty()).unwrap();
        f.members = members(n);
        f.outcome = Wire::Exact as i32;
    };
    // Over the fragment budget: refused by the count alone, before any member is read.
    let over = u32::try_from(MAX_FRAGMENTS + 1).unwrap();
    let message = refused(&msg, |m| with(m, over));
    assert!(message.contains("max_fragments"), "{message}");
    // Within it: a long well-formed list validates in time linear in its length.
    let mut big = msg.clone();
    with(&mut big, 200_000);
    let started = std::time::Instant::now();
    validate_typed_document(&big).unwrap();
    let elapsed = started.elapsed();
    assert!(elapsed < std::time::Duration::from_secs(5), "200k members took {elapsed:?}");
    // Votes carry fragments of their own and count too.
    let voted = entity_votes_doc();
    let mut votes = document_to_typed("w", &voted, ViewId::SOURCE, OffsetMode::Utf8).unwrap();
    validate_typed_document(&votes).unwrap();
    let l = votes.layers.iter_mut().find(|l| l.id == Entities::NAME.to_string()).expect("an entities layer");
    let Some(Value::Entity(ent)) = &mut l.annotations[0].value else { panic!("an entity") };
    ent.votes.first_mut().expect("a vote").spans[0].members = members(over);
    let message = validate_typed_document(&votes).unwrap_err().message;
    assert!(message.contains("max_fragments"), "{message}");
    // A single projected span never coalesces, so members on one are refused outright.
    let edits = masked_and_redacted();
    let redacted = edits.get_key(&REDACTED).unwrap().id();
    let msg = document_to_typed("e", &edits, redacted, OffsetMode::Utf8).unwrap();
    let listed = |m: &mut TypedDocumentResult| {
        let l = m.layers.iter_mut().find(|l| l.id == "turbo:pii-edits").unwrap();
        let Some(Value::PiiEdit(e)) = &mut l.annotations[0].value else { panic!("an edit") };
        e.source.as_mut().unwrap().members = members(2);
    };
    assert!(refused(&msg, listed).contains("source lists members"));
}
